'''
session level token budget tracking and per req boundary revision
Design I went for:
- session has total fixed budget (tokens it can use) and is known
- amount of requests per session is not known
-- since per request boundary is not known lets start optimistic with an even split with estimated requests (total/estimated_reqs)
-- each req gets compressed, if boundary is help nothing changes
-- if not, using the suggest boundary multiplier which uses loss score as driver to revise new per req boundary
-- is reactive, so if the boundary never breaks, the budget will never change
-- maybe later for optimisations, what we can do is create a function that checks if under budget and changes
-- actually maybe not it is so much extra computation it will have to recompute every prompt then
-- idk this is a future problem to think on
-- well with a hard ceiling atleast overflow is unlikely
--- this basically does how much can i put to the next request given this much has been spent
--- compared ot the compress_to_boundary() which does can I fit this in what I am given
'''

from __future__ import annotations

from dataclasses import dataclass, field

from src.contextator.courier.compressor import CompressionResult


#giving the hard ceiling estimate for now !!!!NOT TESTED!!!!!
HARD_CEILING_FRACTION = 0.4

@dataclass
class RequestRecord:
    #outcome of one request, needed for session history
    request_no: int
    boundary_used: int
    tokens_spent: int
    boundary_broken: bool
    suggest_multiplier: float
    boundary_after: int
    hit_hard_ceiling: bool


@dataclass
class SessionBudget:
    #live updating of session tokens budget
    total_budget_token: int
    estimated_requests: int
    current_boundary: int
    tokens_spent: int = 0
    requests_completed: int = 0
    history: list[RequestRecord] = field(default_factory=list)

    @property
    def remaining_tokens(self):
        return max(0,self.total_budget_token - self.tokens_spent)

    @property
    def hard_ceiling_tokens(self) -> int:
        return round(self.total_budget_token * HARD_CEILING_FRACTION)

    @property
    def estimated_requests_remaining(self) -> int:
        #this is purely informational for user like itll tell the user 4 reqs left at this budget
        if self.current_boundary <= 0:
            return 0
        return self.remaining_tokens // self.current_boundary


def start_session(total_budget_tokens: int, estimated_requests:int) -> SessionBudget:
    #this creates a new session
    #per req boundary starts at even split and is a starting guess, no way it should be spot on
    #gets corrected as user interacts

    if total_budget_tokens <= 0:
        raise ValueError("total budget is negative?")
    if estimated_requests <= 0:
        raise ValueError("estimated requests is negative?")

    initial_boundary = max(1, total_budget_tokens//estimated_requests)
    return SessionBudget(
        total_budget_token=total_budget_tokens,
        estimated_requests=estimated_requests,
        current_boundary=initial_boundary
    )



def apply_request_result(session: SessionBudget, result: CompressionResult) -> RequestRecord:
    #input a compress_to_boundary() output into session, spends tokens actually used
    #if boundary broken revises current_boundary for future reqs
    #mutates session in place - it serves as the live sessions state
    #returns RequestRecord of what happened, used for logging

    boundary_used = session.current_boundary
    tokens_spent = result.final_token_count

    session.tokens_spent += tokens_spent
    session.requests_completed += 1

    hit_hard_ceiling = False
    new_boundary = session.current_boundary

    if result.boundary_broken:
        proposed = round(boundary_used * result.suggested_boundary_multiplier)
        ceiling = session.hard_ceiling_tokens
        capped_by_ceiling = min(proposed,ceiling)

        capped = min(capped_by_ceiling, max(1, session.remaining_tokens))
        hit_hard_ceiling = capped < proposed
        new_boundary = max(1,capped)
        session.current_boundary = new_boundary

    record = RequestRecord(
        request_no=session.requests_completed,
        boundary_used = boundary_used,
        tokens_spent=tokens_spent,
        boundary_broken=result.boundary_broken,
        suggest_multiplier=result.suggested_boundary_multiplier,
        boundary_after= new_boundary,
        hit_hard_ceiling= hit_hard_ceiling,
    )

    session.history.append(record)
    return record
