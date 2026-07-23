'''
Checking if the pipeline works
score and librarian model = ollama qwen3.5:9b
compress model = ollama qwen3.5:4b

first using a hardcoded corpus (not pulling from files on the laptop yet) got claude to generate the corpus
just checking if the pipeline is fully working as intended
should be: decompose -> fetch -> compress -> budget -> sufficiency

if this works will check with an actual input path

'''

import asyncio
import sys
import time
from pathlib import  Path

#dont even ask what is up with my file path. I dont know why everything opens in onedrive then goes to pictures??? then goes to onedrive again its so fucking weird ive deleted onedrive idk cba looking into it
REPO_ROOT = Path(r"C:\Users\Ratan\OneDrive\Pictures\OneDrive\Desktop\Personal\Projects\context_slm")

sys.path.insert(0,str(REPO_ROOT))
sys.path.insert(0,str(REPO_ROOT/ "src" / "contextator" / "courier"))

USE_SAMPLE_CORPUS = True

#will be used when use_sample_corpus is false
SOURCE_DIR = REPO_ROOT / "doc_testing"

TEST_DIR = REPO_ROOT / "scripts" / "_live_test_scratch" / "docs"

TASK = (
    "What caused the Project Aurora Q3 outage, and what mitigation steps were recommended afterward?"
)

#claudes cooking ->
# Deliberately small and a bit dense, and deliberately includes one
# irrelevant file (unrelated_notes.txt) to check the sufficiency check
# doesn't get fooled by noise sitting in the source set.
CORPUS = {
    "incident_report.txt": """Project Aurora Incident Report - Q3
Incident ID: AUR-2024-Q3-014
Date: September 14. Duration: 2 hours 47 minutes. Severity: SEV-1.

Root cause: a misconfigured connection pool on the payments-gateway
service reduced max connections from 200 to 20 during a routine config
sync. Under normal Q3 traffic this pool exhausted within 90 seconds of
peak load. The exhaustion triggered aggressive client-side retries,
producing a retry storm that saturated the upstream database's
connection limit, cascading the outage to checkout, notifications, and
the reporting pipeline.

Detection: automated alerting on connection pool saturation fired 4
minutes after the config sync, but the on-call engineer misread it as a
transient blip and did not escalate for 22 minutes.
""",
    "mitigation_plan.txt": """Project Aurora - Mitigation Plan
Following outage AUR-2024-Q3-014, these mitigations were approved:
1. Hard floor on connection pool sizing in config validation, so a sync
   cannot silently drop max connections below 100. Owner: platform-infra.
2. Circuit breaker on payments-gateway's outbound DB calls to fail fast
   under saturation instead of retrying. Owner: payments-team.
3. Tune alert thresholds so saturation alerts page directly instead of
   routing through a dashboard the on-call has to interpret. Owner: SRE.
4. Runbook entry for "connection pool saturation" with explicit
   escalation criteria. Owner: SRE.
""",
    "unrelated_notes.txt": """Q4 Marketing Planning Notes
Draft ideas for the Q4 campaign refresh: pricing tier messaging, the
onboarding email sequence, and a partner co-marketing push before the
holiday freeze. Needs design review before it goes to legal.
""",
}


def write_test_corpus() -> list[str]:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in CORPUS.items():
        p = TEST_DIR / name
        p.write_text(content)
        paths.append(str(p))
    return paths


def print_plan(plan) -> None:
    print(f"  {len(plan.subqueries)} sub-quer{'y' if len(plan.subqueries) == 1 else 'ies'}:")
    for sq in plan.subqueries:
        print(f"    [{sq.id}] kind={sq.kind}  depends_on={sq.depends_on}")
        print(f"        desc: {sq.description}")
        if sq.target_files:
            print(f"        files: {[Path(f).name for f in sq.target_files]}")


def print_results(results: dict) -> None:
    for qid, r in results.items():
        cr = r.compression_result
        print(f"  [{qid}] boundary={cr.boundary}  broken={cr.boundary_broken}  "
              f"final_tokens={cr.final_token_count}  multiplier={cr.suggested_boundary_multiplier}")
        for a in cr.attempts:
            bucket = a.bucket.value if a.bucket else "n/a (fit first try)"
            loss = a.loss_score if a.loss_score is not None else "n/a"
            print(f"        attempt {a.attempt_number}: tokens={a.token_count} "
                  f"loss={loss} bucket={bucket}")
        if r.fetch_failures:
            print(f"        fetch_failures: {r.fetch_failures}")
        preview = r.compressed_text[:160].replace("\n", " ")
        print(f"        preview: {preview}...")


def print_budget(session) -> None:
    print(f"  total={session.total_budget_token}  spent={session.tokens_spent}  "
          f"remaining={session.remaining_tokens}  current_boundary={session.current_boundary}  "
          f"requests_completed={session.requests_completed}  "
          f"hard_ceiling={session.hard_ceiling_tokens}")
    for rec in session.history:
        print(f"    req {rec.request_no}: boundary_used={rec.boundary_used} "
              f"tokens_spent={rec.tokens_spent} broken={rec.boundary_broken} "
              f"-> boundary_after={rec.boundary_after} hit_hard_ceiling={rec.hit_hard_ceiling}")


async def main():
    print("=== ContextKube live pipeline smoke test (Phase 1: sample text) ===\n")

    if USE_SAMPLE_CORPUS:
        sources = write_test_corpus()
    else:
        sources = [str(p) for p in SOURCE_DIR.glob("*") if p.is_file()]
        if not sources:
            print(f"No files found in {SOURCE_DIR}, aborting.")
            return

    print(f"Sources ({len(sources)}):")
    for s in sources:
        print(f"  - {s}")
    print(f"\nTask: {TASK}\n")

    from src.contextator.payload.budget import start_session
    from src.contextator.courier import planner

    # Deliberately tight so the compression retry/boundary-expansion path
    # actually gets exercised on this first run, not just the happy path.
    session = start_session(total_budget_tokens=600, estimated_requests=4)
    print("--- Initial session budget ---")
    print_budget(session)

    t0 = time.time()
    result = await planner.run_planner(TASK, sources, session)
    elapsed = time.time() - t0

    print("\n--- Plan ---")
    print_plan(result.plan)

    print("\n--- Sub-query results ---")
    print_results(result.subquery_results)

    print("\n--- Final session budget ---")
    print_budget(session)

    print(f"\n--- Sufficiency ---")
    print(f"  sufficient={result.sufficient}")
    print(f"  reason: {result.sufficiency_reason}")

    print(f"\n--- Final text handed to answering SLM ---")
    print(result.final_text[:1000])

    print(f"\n=== TOTAL WALL TIME: {elapsed:.1f}s ===")
    if elapsed > 120:
        print("NOTE: exceeded REQUEST_TIMEOUT_SECONDS (120s) somewhere -- "
              "check ollama serve isn't stalled.")


if __name__ == "__main__":
    asyncio.run(main())