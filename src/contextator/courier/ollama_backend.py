'''
This is the file which I am using to control the ollama backend
ollama is being run locally the model is QWEN 3.5 4b and 9b
Using 4b as the talking agent and 9b as the librarian
'''

from __future__ import annotations
import re
import httpx

DEFAULT_HOST = "http://localhost:11434"
COMPRESS_MODEL = "qwen3.5:4b"
SCORE_MODEL = "qwen3.5:9b"
LIBRARIAN_MODEL = SCORE_MODEL

#setting a timeout score
REQUEST_TIMEOUT_SECONDS = 120.0


class OllamaError(RuntimeError):
    pass
        #handling ollama error cases

async def _generate(
        prompt: str,
        model: str,
        host: str = DEFAULT_HOST,
        client: httpx.AsyncClient | None = None,
) -> str:
    #non streamed single prompt, raw text output, if fail calls ollama error so empty output doesnt mean either model said nothing or request failed silently
    payload = {"model": model, "prompt": prompt, "stream": False, "think": False}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

    try:
        try:
            resp = await client.post(f"{host}/api/generate", json = payload)
        except httpx.ConnectError as e:
            raise OllamaError(
                f"Could not reach at {host}, make sure ollama serve is running and {model} is pulled"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaError(
            f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s for {model}"
            ) from e

        if resp.status_code != 200:
            raise OllamaError(
                f"Ollama returned HTTP {resp.status_code} for model={model}: "
                f"{resp.text[:300]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise OllamaError(f" Ollama response was invalid json: {e}") from e
        if "response" not in data:
            raise OllamaError(f" Ollama is missing response field : {data}")

        return data["response"]
    finally:
        if owns_client:
            await client.aclose()



_COMPRESS_PROMPT_TEMPLATE = """You are compressing text to fit within a strict token budget, \
for delivery to a smaller downstream model. Preserve every fact, entity, number, and relationship \
in the original -- rephrase and shorten, do not summarize away specifics.

Target length: approximately {target_tokens} tokens.

Output ONLY the compressed text. No preamble, no explanation, no markdown formatting.

--- ORIGINAL TEXT --- 
{text}
--- END ORIGINAL TEXT ---"""

async def qwen_compress(
        text: str,
        target_tokens: int,
        host: str = DEFAULT_HOST,
        client: httpx.AsyncClient | None = None,
) -> str:
    prompt = _COMPRESS_PROMPT_TEMPLATE.format(target_tokens = target_tokens, text = text)
    result = await _generate(prompt, model=COMPRESS_MODEL, host = host, client=client)
    return result.strip()


_SCORE_PROMPT_TEMPLATE = """Compare the ORIGINAL text to the COMPRESSED text below. Rate how much \
meaning, facts, or important detail was LOST in the compression, on a scale of 0 to 10:
  0  = nothing lost, fully faithful
  10 = severe loss, critical information missing or meaning changed

Respond with ONLY a single number from 0 to 10. No words, no explanation.

--- ORIGINAL ---
{original}
--- COMPRESSED ---
{compressed}
--- END ---"""

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")

#if model's response cannot be parsed, we default to 10 (max loss)

UNPARSEABLE_FALLBACK_SCORE = 10.0

async def qwen_loss_score(
        original: str,
        compressed: str,
        host: str = DEFAULT_HOST,
        client: httpx.AsyncClient | None = None,
) -> float:
    prompt = _SCORE_PROMPT_TEMPLATE.format(original = original, compressed = compressed)
    raw = await _generate(prompt, model =SCORE_MODEL, host = host, client = client)

    match = _NUMBER_RE.search(raw)
    if not match:
        return UNPARSEABLE_FALLBACK_SCORE

    try:
        score = float(match.group(1))
    except ValueError:
        return UNPARSEABLE_FALLBACK_SCORE

    return max(0.0, min(10.0, score))


async def ask_librarian(
        prompt: str,
        host: str = DEFAULT_HOST,
        client: httpx.AsyncClient | None = None,
) -> str:
    '''
    asks librarian model for planner reasoning
    includes task decompostion, sufficiency checks, etc.
    '''
    return await _generate(prompt, model=LIBRARIAN_MODEL, host=host, client=client)