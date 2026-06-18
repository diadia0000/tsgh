# Rule: codegraph-first

**When this applies:** Before you write/edit code or look for anything in the
project (symbols, definitions, call relationships), query codegraph first.
Don't jump straight into edits, and don't reach for grep. Query *before*
coding, not while coding.

## Intent → tool
- Find a symbol by name → `codegraph_search`
- Understand a feature/area as a whole → `codegraph_context` (primary; one call
  composes search + node + callers + callees)
- What calls this → `codegraph_callers`
- What this calls → `codegraph_callees`
- What changing this would break (blast radius) → `codegraph_impact`
- See a symbol's source / signature / docstring → `codegraph_node`
- What's in a directory → `codegraph_files`
- Survey an unfamiliar module broadly → `codegraph_explore` (heavier; only when
  genuinely new to the area)

## Where grep is allowed
Use grep / text search only for literal strings, comments, and log messages.
Symbols, definitions, and callers always go through codegraph.

## Notes
- Don't query right after editing a file: the watcher needs ~1s to sync — wait
  for the next turn.
- Unsure of a symbol's name? Start with `codegraph_search`, then `codegraph_node`
  / `codegraph_context` as needed.
