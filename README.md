# quackmem

Persist AI agent conversations to Postgres. One decorator, zero opinions.

## What it does

- Wraps any AI agent function with a decorator
- Saves every message (inputs and outputs) to Postgres, losslessly
- Works with LangGraph, OpenAI Agents SDK, or any LLM framework
- Writes synchronously inline — no background workers, no broker
- Schema migrations bundled — one function call to set up the database

## What it doesn't do

- No retrieval, no semantic search, no memory injection
- No summarization, no embeddings
- No opinions on how you structure your agents

## Installation

```bash
# Core
pip install quackmem

# With LangGraph support
pip install "quackmem[langgraph]"

# With OpenAI Agents SDK support
pip install "quackmem[openai-agents]"
```

## Quickstart — standalone script

```python
import asyncio
from quackmem import TrackerConfig, init_tracker, upgrade_db, get_tracking_context
from quackmem.wrappers import track_conversation

async def main():
    config = TrackerConfig(
        database_url="postgresql://user:pass@localhost/myapp",
    )

    upgrade_db()       # runs Alembic migrations, safe to call every startup
    init_tracker(config)

    @track_conversation(user_id="u123", conversation_id="sess-abc")
    async def call_llm(messages: list[dict]) -> str:
        # Replace with a real LLM call:
        # response = await client.chat.completions.create(model="gpt-4o", messages=messages)
        # return response.choices[0].message.content
        user_content = messages[-1]["content"] if messages else ""
        return f"Response to: {user_content}"

    messages = [{"role": "user", "content": "What is the capital of France?"}]
    response = await call_llm(messages)

    ctx = get_tracking_context()
    print(f"Response:        {response}")
    print(f"Session ID:      {ctx.session_id if ctx else 'not tracked'}")
    print(f"Conversation ID: {ctx.conversation_id if ctx else 'not tracked'}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Quickstart — FastAPI + LangGraph

```python
from contextlib import asynccontextmanager
from uuid import uuid4
from fastapi import FastAPI
from pydantic import BaseModel
from quackmem import TrackerConfig, init_tracker, upgrade_db, register_metadata, get_tracking_context
from quackmem.wrappers import langgraph_mem


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = TrackerConfig(
        database_url="postgresql://user:pass@localhost/myapp",
    )
    upgrade_db()
    init_tracker(config)
    register_metadata({"user_id": str, "agent_id": str})
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    user_message: str
    user_id: str
    agent_id: str = "default"
    conversation_id: str | None = None


@app.post("/chat")
async def chat(req: ChatRequest):
    conversation_id = req.conversation_id or str(uuid4())

    @langgraph_mem(user_id=req.user_id, agent_id=req.agent_id, conversation_id=conversation_id)
    async def agent_node(state: dict) -> dict:
        user_msg = state["messages"][-1]["content"] if state["messages"] else ""
        # Replace with real LangGraph node logic
        return {"messages": [{"role": "assistant", "content": f"Echo: {user_msg}"}]}

    state = {"messages": [{"role": "user", "content": req.user_message}]}
    result = await agent_node(state)
    ctx = get_tracking_context()

    return {
        "response": result["messages"][-1]["content"],
        "conversation_id": conversation_id,
        "session_id": str(ctx.session_id) if ctx else None,
    }
```

## Decorators

### `@langgraph_mem(**kwargs)`

For LangGraph node functions.

| Detail | Value |
|---|---|
| Input | State dict with a `messages` key containing LangChain `BaseMessage` objects or OpenAI-format dicts |
| Output | State dict with a `messages` key; last message is recorded as the assistant response |

```python
from quackmem.wrappers import langgraph_mem

@langgraph_mem(user_id="u123", conversation_id="conv-456")
async def my_node(state: dict) -> dict:
    ...
```

### `@openai_agents_mem(**kwargs)`

For OpenAI Agents SDK runner functions.

| Detail | Value |
|---|---|
| Input | A string or list of OpenAI-format message dicts |
| Output | A `RunResult` object; `result.final_output` is recorded as the assistant response |

```python
from quackmem.wrappers import openai_agents_mem

@openai_agents_mem(user_id="u123")
async def run_agent(input: str):
    ...
```

### `@track_conversation(**kwargs)`

Generic fallback for any LLM call.

| Detail | Value |
|---|---|
| Input | List of OpenAI-format dicts or a plain string |
| Output | String or dict; string content (or `content`/`output` key from dict) is recorded |

```python
from quackmem.wrappers import track_conversation

@track_conversation(user_id="u123")
async def call_llm(messages: list[dict]) -> str:
    ...
```

All three decorators accept the same keyword arguments: `conversation_id` (optional; auto-generated if omitted), plus any metadata keys registered via `register_metadata` (e.g. `user_id`, `agent_id`). `session_id` and `message_id` are always library-managed.

## Configuration — `TrackerConfig`

```python
from quackmem import TrackerConfig

config = TrackerConfig(
    database_url="postgresql://user:pass@host/db",
    schema_name="public",
    table_prefix="",
    pool_size=5,
    max_overflow=10,
    echo=False,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `database_url` | `str` | — | PostgreSQL connection URL. `postgres://` and `postgresql://` schemes are auto-corrected to use the `asyncpg` driver. |
| `schema_name` | `str` | `"public"` | Postgres schema in which tables are created. |
| `table_prefix` | `str` | `""` | Optional prefix applied to both table names (e.g. `"ai_"` → `ai_tracked_sessions`). |
| `pool_size` | `int` | `5` | Number of persistent connections kept in the pool. |
| `max_overflow` | `int` | `10` | Additional connections allowed beyond `pool_size` under load. |
| `echo` | `bool` | `False` | When `True`, SQLAlchemy logs all SQL statements — useful for debugging. |

## Metadata registry

Call `register_metadata` once at startup to declare which metadata keys are valid. After registration, any unknown kwarg passed to a decorator raises `MetadataValidationError` at decoration time rather than silently being ignored.

```python
from quackmem import register_metadata

register_metadata({"user_id": str, "agent_id": str})

# Later, at decoration time:
@langgraph_mem(user_id="u1", agent_id="bot-v2")   # OK
@langgraph_mem(user_id="u1", unknown_key="x")      # raises MetadataValidationError
```

If `register_metadata` is never called, all extra kwargs are accepted without validation.

## Conversation threading

`conversation_id` groups multiple sessions over time — it represents the full conversation history between a user and an agent across multiple requests. `session_id` identifies a single agent invocation (one decorated function call).

- **`session_id`** — always auto-generated per decorated call. Never pass to the decorator. Access via `get_tracking_context().session_id` after the call.
- **`conversation_id`** — developer-supplied (or auto-generated if omitted). Pass the same `conversation_id` across requests to stitch sessions into a continuous conversation thread.

Pass the same `conversation_id` across requests to stitch sessions into a continuous conversation thread:

```python
# First request
ctx = get_tracking_context()
conversation_id = ctx.conversation_id  # auto-generated on first call
client_stores_this = conversation_id

# Subsequent requests — pass the stored conversation_id back
@langgraph_mem(user_id="u1", conversation_id=client_stores_this)
async def node(state): ...
```

Use `get_tracking_context()` after a decorated call returns to retrieve the IDs assigned to that invocation:

```python
from quackmem import get_tracking_context

result = await node(state)
ctx = get_tracking_context()

print(ctx.session_id)       # UUID of this specific invocation (auto-generated)
print(ctx.conversation_id)  # UUID grouping all related sessions (developer-provided or auto-generated)
```

## Database schema

Two tables are created (names respect `table_prefix` and `schema_name`):

**`tracked_sessions`** — one row per decorated function call.

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Primary key; the `session_id` returned in context. |
| `conversation_id` | `UUID` | Groups sessions belonging to the same conversation thread. |
| `created_at` | `timestamptz` | When the session was created. |
| `metadata` | `JSONB` | All extra kwargs passed to the decorator (e.g. `user_id`, `agent_id`). Indexed with GIN. |

**`tracked_messages`** — one row per message (inputs and the assistant response).

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Primary key. |
| `session_id` | `UUID` | Foreign key to `tracked_sessions`. |
| `conversation_id` | `UUID` | Denormalized for efficient per-conversation queries. |
| `parent_message_id` | `UUID` | Self-referential FK for threading within a session. |
| `role` | `text` | One of `system`, `user`, `assistant`, `tool`. |
| `content` | `JSONB` | Full message content. |
| `token_count` | `integer` | Token count if the model returned usage data; otherwise `NULL`. |
| `status` | `text` | `completed` or `failed`. |
| `created_at` | `timestamptz` | When the message was written. |
| `metadata` | `JSONB` | Inherited from the session kwargs. Indexed with GIN. |

## Startup pattern

```python
from quackmem import TrackerConfig, init_tracker, upgrade_db

config = TrackerConfig(database_url="postgresql://user:pass@host/db")
upgrade_db()          # runs Alembic migrations, safe to call every startup
init_tracker(config)  # initializes connection pool and table metadata
```

`upgrade_db` is idempotent — it checks the current migration state before applying anything, so calling it on every startup is safe and recommended.

## Error handling

Tracking failures (database errors, serialization errors) are caught internally and never propagate to the caller — your agent function always gets its result back. Two exceptions do propagate:

- `TrackerConfigError` — raised at `init_tracker()` time when the configuration is invalid (e.g. unreachable database, bad URL).
- `MetadataValidationError` — raised at decoration time when an unknown metadata key is passed and a registry has been set via `register_metadata`.

## Roadmap

### Near-term

- [ ] Google ADK wrapper (`google_adk_mem`)
- [ ] CrewAI wrapper (`crewai_mem`)
- [ ] Streaming: per-chunk row storage option
- [ ] `get_messages(session_id)` public helper

### Future

- [ ] SQLite backend for local development
- [ ] Retrieval helpers (semantic search, keyword search) as an optional add-on
- [ ] Dashboard / viewer UI
- [ ] S3 offloading for large multimodal payloads
- [ ] Per-message token budget alerts
