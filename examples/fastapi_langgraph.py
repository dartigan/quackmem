"""
FastAPI + LangGraph example.

Demonstrates integrating quackmem with LangGraph nodes in a FastAPI application.
Shows how to track conversations with per-request metadata.

Install: pip install quackmem[langgraph] fastapi uvicorn
Run:     uvicorn examples.fastapi_langgraph:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel

from quackmem import (
    TrackerConfig,
    init_tracker,
    upgrade_db,
    register_metadata,
    get_tracking_context,
)
from quackmem.wrappers import langgraph_mem


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize quackmem on application startup."""
    config = TrackerConfig(
        database_url="postgresql+asyncpg://postgres:password@localhost/myapp",
    )
    upgrade_db()
    init_tracker(config)
    register_metadata({"user_id": str, "agent_id": str})
    yield


app = FastAPI(lifespan=lifespan)


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
