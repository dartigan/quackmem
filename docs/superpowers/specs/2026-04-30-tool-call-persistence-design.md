# Tool-call persistence — design

**Date:** 2026-04-30
**Status:** approved (Approach 1)

## Problem

Persona-style agents (named assistant + system prompt + allowed tools) call tools — often several in parallel per turn — and receive tool results that the LLM consumes on the next turn. Quackmem currently extracts `tool_calls` into `CanonicalMessage` but **drops them silently before persistence**: there's no `tool_calls` column on `tracked_messages`, no `MessageFinalization.tool_calls` field, and no first-class API for tool-result rows. Reading a session back via `read_messages` cannot reconstruct a tool-using conversation.

Persona name / prompt / allowed-tools list do **not** need first-class storage and stay in `metadata` (per the project's "just the conversation" philosophy).

## Goals

1. Persist tool calls — including multiple calls in one assistant turn — on the assistant message row.
2. Persist tool results as their own `role=tool` rows, linked back to the call they answered.
3. Stay framework-agnostic: the columns hold the tool-call shape verbatim so a future Google ADK / etc. wrapper plugs in without schema changes.
4. Preserve the existing decorator ergonomics; users opt into tool-result persistence via one new public helper.

## Non-goals

- No new `personas` table. Persona attributes ride on `metadata`.
- No automatic detection of tool results in the *next* turn's input — too magical, too fragile.
- No multi-step (confirm → execute → result) granularity. The single `tool_calls` array is enough; if a future framework demands it, a child table can be grafted on later without breaking storage.

## Architecture

### Schema (migration 0004)

Two columns added to `tracked_messages`:

| Column         | Type        | Populated on                       | Purpose                                          |
| -------------- | ----------- | ---------------------------------- | ------------------------------------------------ |
| `tool_calls`   | `JSONB`     | `role=assistant` rows that called tools | Array of tool-call dicts; supports parallel calls |
| `tool_call_id` | `TEXT`      | `role=tool` rows                   | Identifies which call this row answers            |

`parent_message_id` (already exists) on a `role=tool` row points to the assistant message that issued the call. Combined with `tool_call_id`, multi-tool turns reconstruct unambiguously.

Indexes:
- `tracked_messages_tool_calls_gin` — GIN on `tool_calls` for "messages that called tool X" queries.
- `tracked_messages_tool_call_id` — partial btree (`WHERE tool_call_id IS NOT NULL`) for tool-result lookups.

### Pydantic models

In `quackmem/schema/canonical.py`:

```python
class ToolCall(BaseModel):
    """A single tool/function call. Permissive — extra framework-specific fields pass through."""
    model_config = ConfigDict(extra="allow")
    id: str | None = None        # OpenAI 'id' / Anthropic 'tool_use_id' / ADK call id
    name: str | None = None      # function name (top-level OR nested under .function for OpenAI)
    arguments: dict | str | None = None  # OpenAI: JSON string; Anthropic/ADK: dict
```

In `quackmem/schema/models.py`:

- `TrackedMessage`: add `tool_calls: list[dict] | None = None`, `tool_call_id: str | None = None`.
- `MessageFinalization`: add `tool_calls: list[dict] | None = None`.
- `MessageReservation`: gains `tool_call_id: str | None = None` so tool-result rows can be reserved/inserted via the same path.

We deliberately keep the storage-facing shapes as `list[dict] | None` so framework-emitted call dicts pass through without coercion. `ToolCall` is a typed convenience users can construct in their own code.

### Decorator changes

In `_post_write`:
- After `wrapper.extract_response(result)`, forward `response.tool_calls` into `MessageFinalization.tool_calls`.
- Backend's `finalize_message` writes the column.

No new code paths in the decorator — just one additional field carried through.

### Wrapper changes

`extract_response` already returns `CanonicalMessage`, which has a `tool_calls` field. The change is per-wrapper extraction logic so the field is populated on the response:

- **GenericWrapper:** if `result` is a dict with `tool_calls`, copy them.
- **LangGraphWrapper:** if the last message in `result["messages"]` is an `AIMessage` with `tool_calls`, copy them. (`_normalize_message` already does this — just make sure `extract_response` doesn't drop them.)
- **OpenAIAgentsWrapper:** RunResult's `final_output` is post-tool-loop text in the SDK, so a typical run won't have outstanding tool calls in the response. Best-effort: read `getattr(result, "tool_calls", None)` if present.

### New public API: `record_tool_result`

```python
async def record_tool_result(
    session_id: UUID | str,
    tool_call_id: str,
    content: str | list[dict],
    *,
    parent_message_id: UUID | str | None = None,
    conversation_id: UUID | str | None = None,
    name: str | None = None,
    metadata: dict | None = None,
) -> UUID:
    """Insert a role=tool row that answers a previous tool call.

    Returns the inserted message id. Caller is responsible for invoking this
    once per tool result; quackmem does not auto-detect them.
    """
```

Behavior:
- If `parent_message_id` is omitted, it's looked up: most recent assistant row in the session whose `tool_calls` array contains this `tool_call_id`. Fast via the GIN index. If no parent is found, the row is still inserted with `parent_message_id=NULL` (don't fail just because the lookup missed).
- If `conversation_id` is omitted, derived from the session row.
- `name` (the tool name) goes into `metadata.tool_name` for convenience — it's not a separate column.

### Read path

`read_messages` already returns full row dicts; with the new columns, callers automatically get `tool_calls` and `tool_call_id` fields on each `TrackedMessage`. No API change.

## Error handling

- Migration is additive (NULL columns + indexes); rollback drops columns.
- `record_tool_result` failures (DB down, etc.) raise to the caller — unlike the decorator, this is a deliberate user action and silent failure would be worse than loud failure.
- The decorator's existing best-effort error swallowing is unchanged.

## Test strategy

1. **Schema/migration** — apply 0004 against a real DB; verify columns + indexes exist; round-trip through `tables.py`.
2. **Pydantic models** — `ToolCall` accepts/rejects expected shapes; `MessageFinalization.tool_calls` round-trips.
3. **Decorator** — assistant turn with `tool_calls` in response gets persisted; `finalize_message` is called with them.
4. **Wrappers** — each wrapper's `extract_response` surfaces `tool_calls` from its native shape (LangGraph AIMessage, dict, etc.).
5. **`record_tool_result`** — happy path inserts row; parent lookup works; missing parent still inserts; round-trip via `read_messages` returns the right `tool_call_id`.
6. **Multi-tool turn (integration)** — assistant emits 3 tool_calls in one turn; 3 `record_tool_result` calls; `read_messages` returns 1 assistant row with 3 calls + 3 tool rows linked correctly.
