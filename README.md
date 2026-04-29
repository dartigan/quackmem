# convo-tracker

Persist AI agent conversations to Postgres. One decorator, zero opinions.

## Install

```bash
pip install convo-tracker
# with LangGraph support
pip install "convo-tracker[langgraph]"
# with OpenAI Agents support
pip install "convo-tracker[openai-agents]"
```

## Quick Start

```python
from convo_tracker import track_conversation

@track_conversation
async def my_agent(user_input: str) -> str:
    # your agent logic here
    ...
```

See `examples/` for full FastAPI and standalone script usage.
