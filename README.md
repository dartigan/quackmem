# 🦆 QuackMem

> A simple, reliable memory layer for AI systems.

Capture everything. Decide later.

---

## ✨ Overview

QuackMem is a lightweight memory layer designed for AI applications.

It captures conversations, links messages, and stores them reliably — without forcing how memory should be used.

No abstractions. No magic. Just clean, structured memory.

---

## 🚀 Why QuackMem?

Most memory systems try to be *smart*.

QuackMem focuses on being **correct, simple, and composable**.

* 🧠 Store full conversation history
* 🔗 Link inputs and outputs
* ⚡ Minimal overhead
* 🧩 Works with any framework (LangGraph, FastAPI, custom agents)

---

## 🧱 Core Concepts

### Session

A session represents a single conversation or workflow.

### Message

Each input/output is stored as a message with a unique ID.

### Links

Messages are connected via `parent_message_id` to form chains.

---

## 🔌 Quick Start

```python
@quackmem.track(session_id="abc123")
async def chat():
    return await llm.generate("Hello")
```

That’s it.

QuackMem automatically:

* captures input
* captures output
* links them together
* stores them

---

## ⚙️ Configuration

`TrackerConfig` reads `QUACKMEM_*` environment variables (or accepts kwargs directly):

| Variable                | Default    | Notes                                   |
| ----------------------- | ---------- | --------------------------------------- |
| `QUACKMEM_DATABASE_URL` | _required_ | Postgres URL; driver auto-rewritten     |
| `QUACKMEM_SCHEMA_NAME`  | `public`   |                                         |
| `QUACKMEM_TABLE_PREFIX` | `""`       |                                         |
| `QUACKMEM_POOL_SIZE`    | `5`        |                                         |
| `QUACKMEM_MAX_OVERFLOW` | `10`       |                                         |
| `QUACKMEM_ECHO`         | `false`    | Log emitted SQL                         |

---

## 🚦 Production: graceful shutdown

QuackMem schedules tracking writes from sync code paths as background tasks on
the running event loop. Drain them — and close the connection pool — on
shutdown so messages aren't lost when your worker stops:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from quackmem import (
    TrackerConfig, init_tracker, upgrade_db, verify_tracker, shutdown_tracker,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = TrackerConfig()                       # reads QUACKMEM_* env vars
    upgrade_db(database_url=config.database_url)
    init_tracker(config)
    await verify_tracker()                         # use as your /health check
    yield
    await shutdown_tracker(drain_timeout=10)       # drain writes + close pool

app = FastAPI(lifespan=lifespan)
```

`shutdown_tracker` is the convenience entry point. If you need finer control,
call `wait_pending_writes(timeout=...)` and `dispose_engine()` separately.

For containerised deployments, use `dumb-init` (or another init that forwards
SIGTERM) so the lifespan shutdown actually fires — see the bundled
`Dockerfile` for a production-ready template.

---

## 🧠 How It Fits

```text
User → LLM / Agent → QuackMem → Database
```

Optional:

```text
Database → Retrieval → LLM Context
```

---

## 🧩 Works With

* LangGraph
* LangChain
* Custom agents
* RAG pipelines
* Internal AI tools

---

## 🗺️ Roadmap

### Core

* [x] Message storage
* [x] Session tracking
* [x] Message linking

### Next

* [ ] Retrieval APIs (last N, session replay)
* [ ] Basic filtering

### Wrappers

* [ ] Chat wrapper
* [ ] RAG wrapper
* [ ] Agent wrapper
* [ ] Vision wrapper

---

## ⚠️ What This Is Not

* Not an agent framework
* Not a vector database
* Not an automated memory system

QuackMem does not decide what matters — you do.

---

## 🔮 Vision

A universal memory layer that works across all AI workflows.

---

## 📄 License

MIT

