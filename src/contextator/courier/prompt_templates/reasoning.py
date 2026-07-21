"""
prompt templates for librarians(planner.py) reasoning
it decomposes tasks and checks if it is sufficient
Asked for json output so planner can parse them reliably
"""

from __future__ import annotations

DECOMPOSE_PROMPT_TEMPLATE = """You are a librarian planning how to answer a question using only \
the local files listed below. Break the task into sub-queries.

Each sub-query is one of two kinds:
  - "fetch": needs specific file(s) to answer part of the task. List which of the AVAILABLE FILES \
are relevant.
  - "synthesize": combines the results of earlier sub-queries (no files of its own). Use this for \
comparison/combination steps that depend on other sub-queries finishing first.

If the task is simple and only needs one lookup, output a single "fetch" sub-query -- do not \
invent unnecessary steps.

AVAILABLE FILES:
{available_files}

TASK:
{task}

Respond with ONLY a JSON object in this exact shape, no other text:
{{
  "subqueries": [
    {{
      "id": "q1",
      "description": "short description of what this sub-query is for",
      "kind": "fetch",
      "target_files": ["path/to/file.py"],
      "depends_on": []
    }},
    {{
      "id": "q2",
      "description": "compare results of q1 against ...",
      "kind": "synthesize",
      "target_files": [],
      "depends_on": ["q1"]
    }}
  ]
}}"""

SUFFICIENCY_PROMPT_TEMPLATE = """You are a librarian checking whether the CONTEXT below is \
enough to answer the TASK. Do not answer the task itself -- only judge whether enough \
information is present.

TASK:
{task}

CONTEXT:
{context}

Respond with ONLY a JSON object in this exact shape, no other text:
{{
  "sufficient": true,
  "reason": "one short sentence explaining why"
}}"""
