"""
FastAPI + LangGraph example.

Demonstrates integrating quackmem with LangGraph nodes in a FastAPI application.
Shows how to track conversations with per-request metadata.

Install: pip install quackmem[langgraph] fastapi uvicorn
Run:     uvicorn examples.fastapi_langgraph:app --reload
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from quackmem import (
    TrackerConfig,
    TrackerConfigError,
    init_tracker,
    upgrade_db,
    register_metadata,
    get_tracking_context,
    shutdown_tracker,
    verify_tracker,
)
from quackmem.wrappers import langgraph_mem


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize quackmem on application startup; drain pending writes on shutdown.

    Configuration is loaded from environment variables prefixed with ``QUACKMEM_``,
    e.g. ``QUACKMEM_DATABASE_URL=postgresql://user:pass@host/db``. See
    ``quackmem.core.config.TrackerConfig`` for the full list.
    """
    config = TrackerConfig()  # reads QUACKMEM_DATABASE_URL etc. from env
    # upgrade_db calls asyncio.run() under the hood — push it to a thread so
    # it doesn't conflict with the running ASGI event loop.
    await asyncio.to_thread(upgrade_db, database_url=config.database_url)
    init_tracker(config)
    await verify_tracker()
    register_metadata({"user_id": str, "agent_id": str})
    yield
    # Drain in-flight tracking writes scheduled from sync code paths and
    # close the Postgres connection pool.
    await shutdown_tracker(drain_timeout=10)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """Liveness + DB connectivity check.

    ``verify_tracker()`` issues a ``SELECT 1`` against the configured database
    and raises ``TrackerConfigError`` if the engine is uninitialised or the
    database is unreachable. Use this as the readiness probe for your
    orchestrator (Kubernetes ``readinessProbe``, ECS health check, etc.).
    """
    try:
        await verify_tracker()
    except TrackerConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


class ChatRequest(BaseModel):
    user_message: str
    user_id: str
    agent_id: str = "default_agent"
    conversation_id: str | None = None


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Process a chat message through a tracked LangGraph node.

    Returns the response, conversation ID, and session ID for tracking.
    """
    conversation_id = req.conversation_id or str(uuid4())

    @langgraph_mem(user_id=req.user_id, conversation_id=conversation_id)
    async def tracked_node(state: dict) -> dict:
        """Simulated LangGraph node that echoes the user message."""
        user_msg = state["messages"][-1]["content"] if state["messages"] else ""
        return {
            "messages": [
                {"role": "assistant", "content": f"Echo: {user_msg}"}
            ]
        }

    state = {"messages": [{"role": "user", "content": req.user_message}]}
    result = await tracked_node(state)
    ctx = get_tracking_context()

    return {
        "response": result["messages"][-1]["content"],
        "conversation_id": conversation_id,
        # session_id is auto-generated per request; store conversation_id to link threads
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
