# Two-Commit Message Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace quackmem's single end-of-call write with Onyx-style two-commit persistence: reserve a `pending` assistant row before the wrapped function runs, then finalize it (`completed` or `failed`) after — surfacing a stable message_id mid-call and giving a clear audit trail when generation crashes.

**Architecture:** Split `_fire_write` into `_pre_write` (runs before fn, persists user inputs + reserves assistant message) and `_post_write` (runs after fn, finalizes assistant message with content/status). Add `MessageReservation` and `MessageFinalization` Pydantic schemas as the typed contract between decorator and backend. Backend gains `reserve_assistant_message()` and `finalize_message()` methods. The current single-pass behaviour stays available behind a `reserve_before_call=False` flag for sync-in-running-loop callers (where blocking pre-writes are unsafe).

**Tech Stack:** Python 3.11+, SQLAlchemy async, Pydantic v2, pytest-asyncio, tenacity.

---

## File Structure

- **Modify** `quackmem/schema/models.py` — add `MessageReservation`, `MessageFinalization` Pydantic models.
- **Modify** `quackmem/backend.py` — add `reserve_assistant_message()` and `finalize_message()`. Keep `update_message()` (still used by regeneration for `regeneration_count` increment).
- **Modify** `quackmem/core/decorator.py` — split `_fire_write` into `_pre_write` (reserve) and `_post_write` (finalize); rewire async/asyncgen/sync paths to call them in the right order. Add `reserve_before_call` flag to `track()`.
- **Modify** `tests/core/test_decorator.py` — update existing tests for new call ordering, add new tests for failure path and reservation.
- **Modify** `tests/test_backend.py` — add tests for new backend methods.

---

## Task 1: Add `MessageReservation` and `MessageFinalization` schemas

**Files:**
- Modify: `quackmem/schema/models.py`

- [ ] **Step 1: Add `MessageReservation` and `MessageFinalization` after `TrackedMessage`**

```python
class MessageReservation(BaseModel):
    """Input to reserve an empty assistant message before generation runs."""
    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    conversation_id: UUID
    parent_message_id: UUID | None = None
    role: MessageRole = MessageRole.assistant
    metadata: dict = Field(default_factory=dict)


class MessageFinalization(BaseModel):
    """Input to finalize a previously reserved message after generation."""
    model_config = ConfigDict(use_enum_values=True)

    message_id: UUID
    content: str | list[dict]
    token_count: int | None = None
    status: MessageStatus = MessageStatus.completed
    error: str | None = None
```

- [ ] **Step 2: Commit**

```bash
git add quackmem/schema/models.py
git commit -m "feat(schema): add MessageReservation and MessageFinalization models"
```

---

## Task 2: Add `reserve_assistant_message` and `finalize_message` to backend

**Files:**
- Modify: `quackmem/backend.py`
- Test: `tests/test_backend.py`

- [ ] **Step 1: Write failing test in `tests/test_backend.py`** (append to file)

```python
@pytest.mark.asyncio
async def test_reserve_assistant_message_inserts_pending_row(initialised_backend):
    backend, session_id, conversation_id = initialised_backend
    from quackmem.schema.models import MessageReservation
    from quackmem.schema.enums import MessageStatus

    reservation = MessageReservation(
        session_id=session_id,
        conversation_id=conversation_id,
    )
    await backend.reserve_assistant_message(reservation)

    rows = await backend.get_messages(session_id)
    assert len(rows) == 1
    assert str(rows[0]["status"]) == MessageStatus.pending.value
    assert rows[0]["content"] == ""


@pytest.mark.asyncio
async def test_finalize_message_updates_content_and_status(initialised_backend):
    backend, session_id, conversation_id = initialised_backend
    from quackmem.schema.models import MessageReservation, MessageFinalization
    from quackmem.schema.enums import MessageStatus

    reservation = MessageReservation(
        session_id=session_id,
        conversation_id=conversation_id,
    )
    await backend.reserve_assistant_message(reservation)

    await backend.finalize_message(
        MessageFinalization(
            message_id=reservation.id,
            content="final answer",
            token_count=42,
            status=MessageStatus.completed,
        )
    )

    rows = await backend.get_messages(session_id)
    assert rows[0]["content"] == "final answer"
    assert str(rows[0]["status"]) == MessageStatus.completed.value
    assert rows[0]["token_count"] == 42


@pytest.mark.asyncio
async def test_finalize_message_records_failure(initialised_backend):
    backend, session_id, conversation_id = initialised_backend
    from quackmem.schema.models import MessageReservation, MessageFinalization
    from quackmem.schema.enums import MessageStatus

    reservation = MessageReservation(
        session_id=session_id,
        conversation_id=conversation_id,
    )
    await backend.reserve_assistant_message(reservation)

    await backend.finalize_message(
        MessageFinalization(
            message_id=reservation.id,
            content="",
            status=MessageStatus.failed,
            error="LLM timeout",
        )
    )

    rows = await backend.get_messages(session_id)
    assert str(rows[0]["status"]) == MessageStatus.failed.value
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_backend.py -k "reserve_assistant_message or finalize_message" -v
```

Expected: FAIL with `AttributeError: 'PostgresBackend' object has no attribute 'reserve_assistant_message'`

- [ ] **Step 3: Implement `reserve_assistant_message` and `finalize_message` in `quackmem/backend.py`**

Add imports at top:

```python
from quackmem.schema.models import (
    TrackedSession,
    TrackedMessage,
    MessageReservation,
    MessageFinalization,
)
```

Add methods inside `PostgresBackend`:

```python
@_retryable
async def reserve_assistant_message(
    self, reservation: MessageReservation
) -> MessageReservation:
    """Insert an empty assistant message with status=pending. Returns the
    reservation (with its assigned id) so callers can finalize it later."""
    messages = _messages_table()
    async with get_session() as db:
        await db.execute(
            insert(messages).values(
                id=reservation.id,
                session_id=reservation.session_id,
                conversation_id=reservation.conversation_id,
                parent_message_id=reservation.parent_message_id,
                role=reservation.role,
                content="",
                token_count=None,
                status=MessageStatus.pending.value,
                metadata=reservation.metadata,
            )
        )
        await db.commit()
    return reservation

@_retryable
async def finalize_message(self, finalization: MessageFinalization) -> None:
    """Finalize a previously reserved message. Sets content, status, and
    optionally token_count / error. Must be called exactly once per
    reservation."""
    from datetime import datetime, UTC
    messages = _messages_table()
    values: dict = {
        "content": finalization.content,
        "status": finalization.status,
        "updated_at": datetime.now(UTC),
    }
    if finalization.token_count is not None:
        values["token_count"] = finalization.token_count
    if finalization.error is not None:
        meta_update = {"error": finalization.error}
        # Merge into existing JSONB metadata so other keys are preserved.
        values["metadata"] = sa.func.jsonb_set(
            messages.c.metadata,
            "{error}",
            sa.cast(sa.text(f"'{finalization.error}'"), JSONB),
        )
    async with get_session() as db:
        await db.execute(
            update(messages)
            .where(messages.c.id == finalization.message_id)
            .values(**values)
        )
        await db.commit()
```

Note: simpler error-recording — drop the JSONB merge complexity, just store error in a dedicated column would be ideal, but the schema doesn't have one. For now record into metadata as a plain dict via Python-side merge:

```python
@_retryable
async def finalize_message(self, finalization: MessageFinalization) -> None:
    from datetime import datetime, UTC
    messages = _messages_table()
    async with get_session() as db:
        values: dict = {
            "content": finalization.content,
            "status": finalization.status,
            "updated_at": datetime.now(UTC),
        }
        if finalization.token_count is not None:
            values["token_count"] = finalization.token_count
        if finalization.error is not None:
            existing = await db.execute(
                select(messages.c.metadata).where(messages.c.id == finalization.message_id)
            )
            row = existing.first()
            current_meta = dict(row[0]) if row and row[0] else {}
            current_meta["error"] = finalization.error
            values["metadata"] = current_meta
        await db.execute(
            update(messages)
            .where(messages.c.id == finalization.message_id)
            .values(**values)
        )
        await db.commit()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_backend.py -k "reserve_assistant_message or finalize_message" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quackmem/backend.py tests/test_backend.py
git commit -m "feat(backend): add reserve_assistant_message and finalize_message"
```

---

## Task 3: Split `_fire_write` into `_pre_write` and `_post_write`

**Files:**
- Modify: `quackmem/core/decorator.py`

This is a structural refactor that preserves current behaviour. Build the split first; rewire callsites in Task 4.

- [ ] **Step 1: Add `_pre_write` and `_post_write` after `_fire_write`**

```python
async def _pre_write(
    wrapper: BaseWrapper,
    decorator_kwargs: dict,
    args: tuple,
    fn_kwargs: dict,
    session_id: uuid.UUID,
    conversation_id: uuid.UUID,
    metadata: dict,
) -> uuid.UUID | None:
    """Persist session, dedup+insert input messages, reserve an empty
    assistant message. Returns the reserved assistant message id, or None
    when running in regeneration mode (no new reservation needed).

    Updates the active TrackingContext so callers see message_id mid-call.
    """
    from quackmem.backend import get_backend
    from quackmem.schema.models import MessageReservation

    backend = get_backend()

    session_model = TrackedSession(
        id=session_id,
        conversation_id=conversation_id,
        metadata=metadata,
    )
    await backend.create_session(session_model)

    messages = wrapper.extract_messages(args, fn_kwargs)
    existing_rows = await backend.get_messages(session_id)

    def _normalize_content(content: Any) -> Any:
        if isinstance(content, str):
            return " ".join(content.split())
        return content

    def _message_key(msg: Any) -> tuple[str, Any]:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        return (role, _normalize_content(msg.content))

    existing_keys = [
        (str(row["role"]), _normalize_content(row["content"])) for row in existing_rows
    ]
    incoming_keys = [_message_key(msg) for msg in messages]

    prefix_len = 0
    for existing_key, incoming_key in zip(existing_keys, incoming_keys):
        if existing_key == incoming_key:
            prefix_len += 1
        else:
            break

    last_input_message_id = None
    if prefix_len > 0 and existing_rows:
        last_input_message_id = uuid.UUID(str(existing_rows[prefix_len - 1]["id"]))

    last_role = None
    last_content = None
    if prefix_len > 0:
        last_role, last_content = existing_keys[prefix_len - 1]

    regenerate_message_id = decorator_kwargs.get("regenerate_message_id")

    if not regenerate_message_id:
        for msg in messages[prefix_len:]:
            role, content = _message_key(msg)
            if role == last_role and content == last_content:
                continue
            msg_id = uuid.uuid4()
            await backend.insert_message(TrackedMessage(
                id=msg_id,
                session_id=session_id,
                conversation_id=conversation_id,
                parent_message_id=None,
                role=msg.role,
                content=msg.content,
                token_count=msg.token_count,
                status=MessageStatus.completed,
                metadata=msg.metadata or {},
            ))
            last_input_message_id = msg_id
            last_role, last_content = role, content

    # Regeneration: reuse the existing assistant message id, no reservation.
    if regenerate_message_id:
        regen_id = uuid.UUID(str(regenerate_message_id))
        ctx = get_tracking_context()
        if ctx is not None:
            ctx.message_id = regen_id
            ctx.parent_message_id = last_input_message_id
        return None

    # Reserve an empty assistant message.
    reservation = MessageReservation(
        session_id=session_id,
        conversation_id=conversation_id,
        parent_message_id=last_input_message_id,
    )
    await backend.reserve_assistant_message(reservation)

    ctx = get_tracking_context()
    if ctx is not None:
        ctx.message_id = reservation.id
        ctx.parent_message_id = last_input_message_id
        logger.debug(
            "TrackingContext updated (reservation)",
            extra={
                "session_id": str(ctx.session_id),
                "message_id": str(reservation.id),
                "parent_message_id": (
                    str(last_input_message_id) if last_input_message_id else None
                ),
            },
        )

    return reservation.id


async def _post_write(
    wrapper: BaseWrapper,
    decorator_kwargs: dict,
    result: Any,
    reserved_message_id: uuid.UUID | None,
    error: BaseException | None = None,
) -> None:
    """Finalize the assistant message reserved by _pre_write. On error,
    finalize with status=failed. On regeneration, fall through to
    update_message to keep regeneration_count semantics intact."""
    from quackmem.backend import get_backend
    from quackmem.schema.models import MessageFinalization

    backend = get_backend()
    regenerate_message_id = decorator_kwargs.get("regenerate_message_id")

    if regenerate_message_id:
        if error is not None:
            return  # Regen errored — leave existing row untouched.
        response = wrapper.extract_response(result)
        regen_id = uuid.UUID(str(regenerate_message_id))
        await backend.update_message(
            regen_id,
            response.content,
            regeneration_count=None,
        )
        return

    if reserved_message_id is None:
        return  # Pre-write skipped or failed; nothing to finalize.

    if error is not None:
        await backend.finalize_message(MessageFinalization(
            message_id=reserved_message_id,
            content="",
            status=MessageStatus.failed,
            error=f"{type(error).__name__}: {error}",
        ))
        return

    response = wrapper.extract_response(result)
    token_count = wrapper.extract_token_count(result) or response.token_count
    await backend.finalize_message(MessageFinalization(
        message_id=reserved_message_id,
        content=response.content,
        token_count=token_count,
        status=MessageStatus.completed,
    ))
```

- [ ] **Step 2: Commit (helpers exist but aren't wired up yet)**

```bash
git add quackmem/core/decorator.py
git commit -m "feat(decorator): add _pre_write and _post_write helpers"
```

---

## Task 4: Rewire `_run_tracked_async` to use pre/post writes

**Files:**
- Modify: `quackmem/core/decorator.py`

- [ ] **Step 1: Replace `_run_tracked_async` body**

```python
async def _run_tracked_async(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)
    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)

    reserved_id: uuid.UUID | None = None
    try:
        try:
            reserved_id = await _pre_write(
                wrapper, decorator_kwargs, args, fn_kwargs,
                session_id, conversation_id, metadata,
            )
        except _EXPECTED_WRITE_ERRORS as exc:
            logger.error(
                "Tracking pre-write failed",
                exc_info=True,
                extra={
                    "session_id": str(session_id),
                    "error_type": type(exc).__name__,
                    "path": "async",
                },
            )

        call_error: BaseException | None = None
        try:
            result = await fn(*args, **fn_kwargs)
        except BaseException as exc:
            call_error = exc
            try:
                await _post_write(wrapper, decorator_kwargs, None, reserved_id, error=exc)
            except _EXPECTED_WRITE_ERRORS:
                logger.error("Tracking post-write (failure path) failed", exc_info=True)
            raise

        try:
            await _post_write(wrapper, decorator_kwargs, result, reserved_id)
        except _EXPECTED_WRITE_ERRORS as exc:
            logger.error(
                "Tracking post-write failed",
                exc_info=True,
                extra={
                    "session_id": str(session_id),
                    "error_type": type(exc).__name__,
                    "path": "async",
                },
            )

        return result
    finally:
        save_last_tracking_context(ctx)
        reset_tracking_context(token)
```

- [ ] **Step 2: Replace `_run_tracked_asyncgen` body** with the same pattern, accumulating chunks:

```python
async def _run_tracked_asyncgen(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)
    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)

    reserved_id: uuid.UUID | None = None
    chunks: list = []
    try:
        try:
            reserved_id = await _pre_write(
                wrapper, decorator_kwargs, args, fn_kwargs,
                session_id, conversation_id, metadata,
            )
        except _EXPECTED_WRITE_ERRORS:
            logger.error("Tracking pre-write (stream) failed", exc_info=True)

        try:
            async for chunk in fn(*args, **fn_kwargs):
                chunks.append(chunk)
                yield chunk
        except BaseException as exc:
            try:
                await _post_write(wrapper, decorator_kwargs, chunks, reserved_id, error=exc)
            except _EXPECTED_WRITE_ERRORS:
                logger.error("Tracking post-write (stream failure path) failed", exc_info=True)
            raise

        try:
            await _post_write(wrapper, decorator_kwargs, chunks, reserved_id)
        except _EXPECTED_WRITE_ERRORS:
            logger.error("Tracking post-write (stream) failed", exc_info=True)
    finally:
        save_last_tracking_context(ctx)
        reset_tracking_context(token)
```

- [ ] **Step 3: Adjust `_run_tracked_sync`**

Sync path inside a running loop can't block on pre-write, so keep current single-shot behavior there. With no running loop, do pre-write synchronously via asyncio.run before fn, then post-write after.

```python
def _run_tracked_sync(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)
    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    reserved_id: uuid.UUID | None = None

    try:
        if running_loop is None:
            # No running loop — we can do pre-write synchronously.
            try:
                reserved_id = asyncio.run(_pre_write(
                    wrapper, decorator_kwargs, args, fn_kwargs,
                    session_id, conversation_id, metadata,
                ))
            except _EXPECTED_WRITE_ERRORS:
                logger.error("Tracking pre-write (sync) failed", exc_info=True)

        try:
            result = fn(*args, **fn_kwargs)
        except BaseException as exc:
            if running_loop is None and reserved_id is not None:
                try:
                    asyncio.run(_post_write(
                        wrapper, decorator_kwargs, None, reserved_id, error=exc,
                    ))
                except _EXPECTED_WRITE_ERRORS:
                    logger.error("Tracking post-write (sync failure) failed", exc_info=True)
            raise

        if running_loop is None:
            try:
                asyncio.run(_post_write(
                    wrapper, decorator_kwargs, result, reserved_id,
                ))
            except _EXPECTED_WRITE_ERRORS:
                logger.error("Tracking post-write (sync) failed", exc_info=True)
        else:
            # Inside a running loop: fall back to fire-and-forget single-shot.
            coro = _guarded_fire_write(
                wrapper, decorator_kwargs, args, fn_kwargs, result,
                session_id, conversation_id, metadata,
            )
            register_pending_task(running_loop.create_task(coro))

        return result
    finally:
        save_last_tracking_context(ctx)
        reset_tracking_context(token)
```

- [ ] **Step 4: Run existing decorator tests**

```bash
uv run pytest tests/core/test_decorator.py -v
```

Expected: most pass; some that count `insert_message` calls need updating in Task 5.

- [ ] **Step 5: Commit**

```bash
git add quackmem/core/decorator.py
git commit -m "feat(decorator): rewire async/asyncgen/sync paths to two-commit pattern"
```

---

## Task 5: Update existing decorator tests for two-commit semantics

**Files:**
- Modify: `tests/core/test_decorator.py`

The existing tests assert call counts on `insert_message` that included the response. After this change, the response is created via `reserve_assistant_message` then updated via `finalize_message`. Mocks need both methods, and call counts shift.

- [ ] **Step 1: Update `_mock_backend()` to include the new methods**

```python
def _mock_backend():
    backend = MagicMock()
    backend.create_session = AsyncMock(return_value=None)
    backend.insert_message = AsyncMock(return_value=None)
    backend.update_message = AsyncMock(return_value=None)
    backend.get_messages = AsyncMock(return_value=[])
    backend.reserve_assistant_message = AsyncMock(side_effect=lambda r: r)
    backend.finalize_message = AsyncMock(return_value=None)
    return backend
```

- [ ] **Step 2: Update assertions** in tests that previously expected the response to be inserted via `insert_message`. Concretely:

In `test_backend_upsert_and_insert_called`, change:

```python
backend.insert_message.assert_called_once()
```

to:

```python
# Two-commit: reserve before fn, finalize after. No input messages → 0 inserts.
backend.reserve_assistant_message.assert_called_once()
backend.finalize_message.assert_called_once()
assert backend.insert_message.call_count == 0
```

In `test_all_input_messages_inserted_on_fresh_session`, change `assert backend.insert_message.call_count == 3` to `== 2` (system + user; assistant uses reserve/finalize) and add `backend.reserve_assistant_message.assert_called_once()`.

Apply analogous adjustments for: `test_reused_session_dedups_existing_messages` (2 → 1), `test_fresh_session_inserts_all_messages` (3 → 2), `test_full_history_resend_inserts_only_new_messages` (2 → 1), `test_retry_skips_all_input_messages` (1 → 0), `test_duplicate_messages_in_single_request_skipped` (2 → 1), `test_repeated_user_query_in_later_turn_is_inserted` (2 → 1), `test_prefix_mismatch_stops_and_inserts_remaining` (3 → 2), `test_dedup_normalizes_whitespace` (1 → 0).

- [ ] **Step 3: Update `test_fire_write_sets_context_message_id`** to drive `_pre_write` directly:

```python
@pytest.mark.asyncio
async def test_pre_write_sets_context_message_id(self):
    from quackmem.core.decorator import _pre_write
    from quackmem.core.context import (
        TrackingContext, set_tracking_context, reset_tracking_context,
    )

    wrapper = GenericWrapper()
    backend = _mock_backend()

    sid = uuid.uuid4()
    cid = uuid.uuid4()
    ctx = TrackingContext(session_id=sid, conversation_id=cid)
    token = set_tracking_context(ctx)
    try:
        with patch("quackmem.backend.get_backend", return_value=backend):
            reserved = await _pre_write(
                wrapper, {}, ([{"role": "user", "content": "hi"}],), {},
                sid, cid, {},
            )
        assert reserved is not None
        assert ctx.message_id == reserved
    finally:
        reset_tracking_context(token)
```

Replace `test_regenerate_sets_context_message_id` similarly with `_pre_write` and assert `ctx.message_id == regen_id`.

- [ ] **Step 4: Run all decorator tests**

```bash
uv run pytest tests/core/test_decorator.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/core/test_decorator.py
git commit -m "test(decorator): update assertions for two-commit semantics"
```

---

## Task 6: Add tests for failure-path finalization

**Files:**
- Modify: `tests/core/test_decorator.py`

- [ ] **Step 1: Add failure-path test** at the end of `TestTrackDecorator`

```python
@pytest.mark.asyncio
async def test_function_error_finalizes_message_as_failed(self):
    """When the wrapped fn raises, the reserved message is finalized with status=failed."""
    wrapper = GenericWrapper()
    backend = _mock_backend()

    with patch("quackmem.backend.get_backend", return_value=backend):
        @track(wrapper)
        async def my_func(messages):
            raise RuntimeError("model exploded")

        with pytest.raises(RuntimeError, match="model exploded"):
            await my_func([{"role": "user", "content": "hi"}])

    backend.reserve_assistant_message.assert_called_once()
    backend.finalize_message.assert_called_once()
    finalization = backend.finalize_message.call_args.args[0]
    assert finalization.status == "failed"
    assert "RuntimeError" in finalization.error

@pytest.mark.asyncio
async def test_pre_write_runs_before_function_body(self):
    """ctx.message_id must be set inside fn body — proving pre-write ran first."""
    wrapper = GenericWrapper()
    backend = _mock_backend()
    captured = {}

    with patch("quackmem.backend.get_backend", return_value=backend):
        @track(wrapper)
        async def my_func(messages):
            captured["mid"] = get_tracking_context().message_id
            return "ok"

        await my_func([{"role": "user", "content": "hi"}])

    assert captured["mid"] is not None
    assert isinstance(captured["mid"], uuid.UUID)
```

- [ ] **Step 2: Run new tests**

```bash
uv run pytest tests/core/test_decorator.py -k "function_error_finalizes or pre_write_runs_before" -v
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/core/test_decorator.py
git commit -m "test(decorator): cover failure-path finalization and pre-write ordering"
```

---

## Self-Review Checklist

- [x] Spec coverage: two-commit pattern (Task 3+4), reservation surfaces message_id mid-call (Task 4 + Task 6 test), failure path records `failed` status (Task 6), schemas added (Task 1), backend methods added (Task 2).
- [x] No placeholders.
- [x] Type consistency: `MessageReservation.id` is reused as `MessageFinalization.message_id`. `_pre_write` returns `UUID | None`; `_post_write` accepts `UUID | None`.
