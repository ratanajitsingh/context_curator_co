'''
librarian - reads tasks, decomposes into sub-queries, starts workflow
workflow is: fetch doc -> compress -> budget, and judges when enough context has been gathered
connects fetcher, compressor and payload/budget into a request loop
reasoning is so clutch
current scope:
- only local files
- sub-query plans supports fetch and synthesize
- independent sub-queries run concurrently
- synthesis waits for dependencies
- if insufficient, report rather than auto retry - not its job
- insufficiency loop is an optimisation for later
'''

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

from torch.distributions.constraints import dependent

from fetcher import fetch_files,split_successes
from compressor import compress_to_boundary,CompressionResult
from ollama_backend import ask_librarian,OllamaError
from prompt_templates.reasoning import (DECOMPOSE_PROMPT_TEMPLATE,SUFFICIENCY_PROMPT_TEMPLATE)
from src.contextator.payload.budget import SessionBudget,apply_request_result


_JSON_BLOCK_RE = re.compile(r"\{.*\}",re.DOTALL)

def extract_json(raw:str) -> dict|None:
    #pull json out of model resp that might have stray text around it.
    #tries full string, if not does the biggest {..} block
    #return none if nothing parses
    #never assume it succeeds

    raw = raw.strip()
    try:
        return json.loads(raw)
    except ValueError:
        pass

    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None



@dataclass
class SubQuery:
    id: str
    description: str
    kind: str
    target_files: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

@dataclass
class QueryPlan:
    subqueries: list[SubQuery]

@dataclass
class SubQueryResult:
    subquery_id: str
    compressed_text: str
    compression_result: CompressionResult
    fetch_failures: list[str] = field(default_factory=list)

@dataclass
class PlannerResult:
    task: str
    plan: QueryPlan
    subquery_results: dict[str, SubQueryResult]
    final_text: str
    sufficient: bool
    sufficiency_reason: str



#if decomp fails then this is the fallback plan

def _fallback_single(task:str, available_files: list[str]) -> QueryPlan:
    #safe plan
    #check against every available file
    #dont break cuz of malformed json response

    return QueryPlan(
        subqueries=[
            SubQuery(
                id="q1",
                description=task,
                kind="fetch",
                target_files=list(available_files),
                depends_on=[]
            )
        ]
    )


#decompose
async def decompose_task(task:str, available_files:list[str]) -> QueryPlan:
    #ask librarian to break task into subqueries
    #goes to fallback if response cant be parsed

    prompt = DECOMPOSE_PROMPT_TEMPLATE.format(
        available_files="\n".join(available_files),
        task=task,
    )
    try:
        raw = await ask_librarian(prompt)
    except OllamaError:
        return _fallback_single(task,available_files)

    data = extract_json(raw)
    if data is None or "subqueries" not in data:
        return _fallback_single(task, available_files)
    try:
        subqueries = [
            SubQuery(
                id=sq["id"],
                description=sq.get("description",""),
                kind = sq["kind"],
                target_files = sq.get("target_files",[]),
                depends_on=sq.get("depends_on",[]),
            )
            for sq in data["subqueries"]
        ]
    except (KeyError, TypeError):
        return _fallback_single(task,available_files)

    if not subqueries:
        return _fallback_single(task,available_files)

    return QueryPlan(subqueries=subqueries)


#sufficiency check

async def check_sufficiency(task:str, context:str) -> tuple[bool,str]:
    #ask librarian whether context is enough to answer task
    #if cannot be parsed then treat as insufficient

    prompt = SUFFICIENCY_PROMPT_TEMPLATE.format(task= task, context=context)
    try:
        raw = await ask_librarian(prompt)
    except OllamaError as e:
        return False, f"Librarian gone : {e}"

    data = extract_json(raw)
    if data is None or "sufficient" not in data:
        return False, "could not parse sufficiency judgement"

    return bool(data["sufficient"]), str(data.get("reason",""))


#sub-query execute

async def _run_fetch_subquery(subquery: SubQuery, session: SessionBudget) -> SubQueryResult:
    fetch_results = await fetch_files(subquery.target_files)
    ok, failed = split_successes(fetch_results)

    combined = "\n\n".join(r.content for r in ok if r.content)
    compression_result = await compress_to_boundary(combined, session.current_boundary)
    apply_request_result(session, compression_result)

    return SubQueryResult(
        subquery_id=subquery.id,
        compressed_text=compression_result.final_text,
        compression_result=compression_result,
        fetch_failures=[r.src_path for r in failed]
    )

async def _run_synthesize_subquery(
        subquery: SubQuery,
        session: SessionBudget,
        completed: dict[str, SubQueryResult],
) -> SubQueryResult:
    dependency_text = "\n\n".join(
        completed[dep_id].compressed_text for dep_id in subquery.depends_on if dep_id in completed
    )
    combined = f"{subquery.description}\n\n{dependency_text}"
    compression_result = await compress_to_boundary(combined, session.current_boundary)
    apply_request_result(session,compression_result)

    return SubQueryResult(
        subquery_id=subquery.id,
        compressed_text=compression_result.final_text,
        compression_result=compression_result,
        fetch_failures=[]
    )

def _resolve_waves(subqueries:list[SubQuery]) -> list[list[SubQuery]]:
    #group sub-queries into dependency waves
    #this way independent sub-queries can run concurrently
    #synthesis steps only run once every dependency is done

    by_id = {sq.id: sq for sq in subqueries}
    remaining = dict(by_id)
    completed_ids: set[str] = set()
    waves: list[list[SubQuery]] = []

    while remaining:
        wave = [
            sq for sq in remaining.values()
            if all(dep in completed_ids for dep in sq.depends_on)
        ]
        if not wave:
            unresolved = list(remaining.keys())
            raise ValueError(
                f"could not resolve dependency order for sub-queries: {unresolved}"
                f"possible cycle or missing dependency id"
            )
        waves.append(wave)
        for sq in wave:
            completed_ids.add(sq.id)
            del remaining[sq.id]

    return waves


async def execute_plan(plan: QueryPlan, session: SessionBudget) -> dict[str,SubQueryResult]:
    #run every subquery in plan respecting dependency order
    #sub queries within the same wave run concurrently

    waves = _resolve_waves(plan.subqueries)
    completed: dict[str, SubQueryResult] = {}

    for wave in waves:
        async def run_one(sq: SubQuery) -> SubQueryResult:
            if sq.kind == "fetch":
                return await _run_fetch_subquery(sq,session)
            return await _run_synthesize_subquery(sq, session, completed)

        results = await asyncio.gather(*(run_one(sq) for sq in wave))
        for result in results:
            completed[result.subquery_id] = result

    return completed


async def run_planner(
        task:str,
        available_files: list[str],
        session: SessionBudget,
) -> PlannerResult:
    '''
    full single request planner loop
    1- decompose task into query plan
    2- execute the plan
    3- combine the final waves output and check if sufficient

    no auto retry yet
    '''

    plan = await decompose_task(task, available_files)
    results = await execute_plan(plan,session)

    #final context is whatever last dependency wave made
    #for synth plan its synth output
    #for flat plan its all fetch results
    waves = _resolve_waves(plan.subqueries)
    final_waves_ids = [sq.id for sq in waves[-1]]
    final_text = "\n\n".join(results[qid].compressed_text for qid in final_waves_ids)

    sufficient, reason = await check_sufficiency(task,final_text)

    return PlannerResult(
        task=task,
        plan=plan,
        subquery_results=results,
        final_text=final_text,
        sufficient=sufficient,
        sufficiency_reason=reason,
    )





