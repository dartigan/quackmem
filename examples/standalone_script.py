"""
Standalone script example.

Demonstrates convo-tracker in a simple standalone script with sync_mode=True.
No Celery or Redis required. Useful for scripts, CLIs, and simple applications.

Install: pip install convo-tracker
Run:     python examples/standalone_script.py
"""
from __future__ import annotations

import asyncio

from convo_tracker import (
    TrackerConfig,
    init_tracker,
    upgrade_db,
    get_tracking_context,
)
from convo_tracker.wrappers import track_conversation


async def main():
    """Initialize tracker and run a tracked conversation."""
    # Configure tracker for sync mode (no Celery needed)
    config = TrackerConfig(
        database_url="postgresql+asyncpg://postgres:password@localhost/myapp",
        sync_mode=True,  # Write directly to DB, no broker needed
    )

    # Run migrations
    upgrade_db()

    # Initialize tracker
    init_tracker(config)

    # Define a tracked LLM call
    @track_conversation(user_id="script_user", conversation_id="example-1")
    async def call_llm(messages: list[dict]) -> str:
        """
        Simulated LLM call.

        In real usage, you'd call an actual LLM:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=messages,
            )
            return response.choices[0].message.content
        """
        user_content = messages[-1]["content"] if messages else ""
        return f"Response to: {user_content}"

    # Run the tracked function
    messages = [
        {"role": "user", "content": "Hello, what is the capital of France?"},
    ]
    response = await call_llm(messages)

    # Access tracking context
    ctx = get_tracking_context()

    print(f"User:     {messages[0]['content']}")
    print(f"Assistant: {response}")
    print(f"Session ID: {ctx.session_id if ctx else 'not tracked'}")
    print(f"Conversation ID: {ctx.conversation_id if ctx else 'not tracked'}")


if __name__ == "__main__":
    asyncio.run(main())
