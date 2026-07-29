# Rule: frontend-backend-boundary

**When this applies:** You are working on frontend work (anything under
`frontend/`) — a frontend agent, or the main agent doing a frontend task.

## The boundary

Writable:

- `frontend/**` — yours, edit freely.
- `backend/api/**` — route/handler layer, edit allowed.
- `backend/schemas/**` — request/response models, edit allowed.

Read-only. Read as much as you want, **never write**:

- `backend/algorithms/**`, `backend/io/**`, `backend/main.py`,
  `backend/tests/**`, and every other path under `backend/`.
- `scripts/**`, `pyproject.toml`, `uv.lock`, `Dockerfile`,
  `docker-compose.yml`.

## If a frontend change needs a read-only file changed

**STOP. Do not edit it. Do not work around it either** — no monkey-patching
from the API layer, no duplicating the function into `backend/api/`, no
"temporary" shim. Both count as touching the file.

Halt all coding and report to the user in this shape:

```
BOUNDARY STOP — backend change needed

File:    backend/algorithms/foo.py
Why:     <what the frontend needs that this file blocks>
Change:  <the smallest edit that would unblock it>
Blast:   <codegraph_impact result — what else calls it>

Frontend work done so far: <list>
Blocked on your approval: <list>
```

Then wait. Do not resume until the user explicitly approves the file by name.
Approval for one file is not approval for the next one.

## Before proposing any backend edit

Run codegraph first (see [codegraph-first](codegraph-first.md)) — the `Blast`
line above is `codegraph_impact`, not a guess.

## Finish what you can

A boundary stop blocks one thread of work, not the whole task. Ship every
frontend change that does not depend on the blocked file, then report. Do not
sit idle waiting for approval with unfinished work you could have done.

## Note

Edits inside `backend/api/` and `backend/schemas/` are still real backend
edits — keep them surgical, match the existing FastAPI/pydantic style, and do
not refactor neighbouring routes or models while you are in there.
