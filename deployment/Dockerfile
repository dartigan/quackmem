# syntax=docker/dockerfile:1.7
#
# Production image template for an app that uses quackmem.
#
# This file builds the bundled FastAPI/LangGraph example as a representative
# service. Swap the final CMD module path for your own ASGI app — the rest of
# the layout (multi-stage build, uv-based install, non-root user, dumb-init,
# healthcheck) is reusable as-is.
#
# Build:   docker build -t quackmem-app:latest .
# Run:     docker run --rm -p 8000:8000 --env-file quackmem.env quackmem-app:latest

# ─── Builder stage ────────────────────────────────────────────────────────────
# Pinned uv image bundles a CPython interpreter and the uv installer. The
# `-bookworm-slim` tag gives glibc compatibility (asyncpg ships wheels for it)
# at a fraction of the full debian size.
FROM ghcr.io/astral-sh/uv:0.5.11-python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first, in a separate layer that caches as long as
# pyproject.toml + uv.lock are unchanged.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra langgraph --extra examples

# Now copy the source and install the project itself. README.md is referenced
# by pyproject.toml's `readme =` field, so hatchling needs it at build time.
COPY quackmem ./quackmem
COPY examples ./examples
COPY py.typed README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra langgraph --extra examples

# ─── Runtime stage ────────────────────────────────────────────────────────────
# Slim runtime image with only what we need at run time. Pinning the digest
# isn't done here for readability — pin in CI before promoting to production.
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# dumb-init forwards signals correctly so SIGTERM hits the ASGI worker rather
# than getting swallowed by PID 1, which is what makes graceful shutdown
# (and `wait_pending_writes`) actually work in containers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. Running as root in production widens blast radius if the
# process is compromised.
RUN groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

# Copy the resolved virtualenv and the application code from the builder.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/quackmem /app/quackmem
COPY --from=builder --chown=appuser:appuser /app/examples /app/examples

USER appuser

EXPOSE 8000

# verify_tracker() (wired into /health on the example app) issues a SELECT 1
# against the configured database. Failures surface as HTTP 503 and Docker
# marks the container unhealthy after three consecutive failures.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["dumb-init", "--"]
CMD ["uvicorn", "examples.fastapi_langgraph:app", "--host", "0.0.0.0", "--port", "8000"]
