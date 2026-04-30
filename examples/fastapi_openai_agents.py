"""
FastAPI + OpenAI Agents example.

Demonstrates integrating quackmem with OpenAI Agents SDK in a FastAPI application.
Shows how to wrap an agent runner with conversation tracking.

Install: pip install quackmem fastapi uvicorn openai
Run:     uvicorn examples.fastapi_openai_agents:app --reload
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
)
from quackmem.wrappers import openai_agents_mem


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


class AgentRequest(BaseModel):
    user_message: str
    user_id: str
    agent_id: str = "default_agent"
    conversation_id: str | None = None


class MockRunResult:
    """Mock OpenAI Agents RunResult for testing without openai package."""

    def __init__(self, final_output: str):
        self.final_output = final_output
        self.usage = None


@app.post("/agent")
async def agent(req: AgentRequest):
    """
    Run a message through a tracked OpenAI agent.

    Returns the agent response, conversation ID, and session ID for tracking.
    """
    conversation_id = req.conversation_id or str(uuid4())

    @openai_agents_mem(user_id=req.user_id, conversation_id=conversation_id)
    async def run_tracked_agent(user_input: str) -> MockRunResult:
        """
        Simulated OpenAI agent runner.

        In real usage, this would call:
            result = client.agents.run(...)
        """
        # Simulate agent processing
        response_text = f"Agent processed: {user_input}"
        return MockRunResult(final_output=response_text)

    result = await run_tracked_agent(req.user_message)

    return {
        "response": result.final_output,
        "conversation_id": conversation_id,
        # session_id is auto-generated per request; store conversation_id to link threads
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
