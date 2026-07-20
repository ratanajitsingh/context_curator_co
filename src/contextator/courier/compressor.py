"""
this is used to compress the fetched content into something manageable for the slm to use
i am using a linear and static loss scorer which basically will compress the information
then compare it against what the original document was to get a round about score which will
give an idea about how much information is actually lost
with the loss score, depending on how severe or good it is itll either
re try compression till a ceiling checking at each stage if it is within the request boundary
or if the compression score is too high then itll straight away increase the request boundary

adding the loss scorer and compression strat as params so we can change it later, just tryna test the logic rn
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable
from ollama_backend import *


#placeholder tokenizer
#test strategy not used
def estimate_tokens(text:str) -> int:
    if not text:
        return 0
    return max(1, round(len(text)/4))

#created buckets for the loss scores. lower the better
class LossBucket(Enum):
    LOW = "low" #compression works
    LOW_MID = "low_mid" #its kinda working try a slightly higher boundary
    MID = "mid" #not really working
    HIGH = "high" #it shit the bed

BUCKETS = {
    LossBucket.LOW: 1.10,
    LossBucket.LOW_MID: 1.15,
    LossBucket.MID: 1.25,
    LossBucket.HIGH: 1.50,
}

def classify_loss(score:float) -> LossBucket:
    if score <= 3:
        return LossBucket.LOW
    if score <= 5:
        return LossBucket.LOW_MID
    if score <= 7:
        return LossBucket.MID
    return LossBucket.HIGH

#placeholder strategy

CompressFn = Callable[[str,int], Awaitable[str]]
ScoreFn = Callable[[str,str], Awaitable[float]]

#compression strat func
#test strategy not used
async def naive_truncate_compress(text: str, target_tokens:int) -> str:

    target_chars = target_tokens * 4
    if len(text) <= target_chars:
        return text
    return text[:target_chars]

async def loss_scorer(original:str,compressed:str) -> float:
    #in real would use 9b to score the loss between original and compressed
    #currently using length-ratio heuristic so i can test the retry loop

    if not original:
        return 0.0
    ratio_kept = len(compressed)/len(original)
    #assuming more that is cut the more that is lost
    return round((1-ratio_kept)*10,1)


@dataclass
class CompressionAttempt:
    attempt_number: int
    text: str
    token_count: int
    loss_score: float | None
    bucket: LossBucket | None


@dataclass
class CompressionResult:
    final_text: str
    final_token_count: int
    boundary: int
    boundary_broken: bool
    suggested_boundary_multiplier: float
    attempts: list[CompressionAttempt] = field(default_factory=list)

    @property
    def final_loss_score(self) -> float | None:
        return self.attempts[-1].loss_score if self.attempts else None


#this is the main loop now
MAX_LOW_LOSS_RE = 3 #will tune this based on token usage

'''
the logic im tryna implement - compress if fits in boundary perfect no need to score loss
if it doesnt fit, score how much meaning was lost  
if low loss the retry compression until ceiling hit 
anything else stop immediately and increase the boundary 
if even after retrying it doesnt fit then fall back on low mid 
'''
async def compress_to_boundary(
        text:str,
        boundary_tokens: int,
        compress_fn: CompressFn = qwen_compress,
        score_fn: ScoreFn = qwen_loss_score,
        max_loss_re:int = MAX_LOW_LOSS_RE
) -> CompressionResult:
    attempts: list[CompressionAttempt] = []
    current_target = boundary_tokens
    low_loss_retry_count = 0
    attempt_no = 0

    while True:
        attempt_no += 1
        compressed = await compress_fn(text,current_target)
        token_count = estimate_tokens(compressed)

        if token_count <= boundary_tokens:
            attempts.append(
                CompressionAttempt(
                    attempt_number= attempt_no,
                    text = compressed,
                    token_count = token_count,
                    loss_score = None,
                    bucket = None,
                )
            )

            return CompressionResult(
                final_text=compressed,
                final_token_count = token_count,
                boundary=boundary_tokens,
                boundary_broken=False,
                suggested_boundary_multiplier=1.0,
                attempts=attempts,
            )

        loss_score = await score_fn(text,compressed)
        bucket = classify_loss(loss_score)
        attempts.append(
            CompressionAttempt(
                attempt_number=attempt_no,
                text = compressed,
                token_count = token_count,
                loss_score= loss_score,
                bucket = bucket
            )
        )

        if bucket == LossBucket.LOW and low_loss_retry_count < max_loss_re:
            low_loss_retry_count += 1
            current_target = max(1, round(current_target*0.85))
            continue


        multiplier = BUCKETS[bucket]
        return CompressionResult(
            final_text = compressed,
            final_token_count=token_count,
            boundary=boundary_tokens,
            boundary_broken=True,
            suggested_boundary_multiplier=multiplier,
            attempts=attempts
        )






