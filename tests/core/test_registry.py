"""Tests for register_metadata and validate_metadata."""
from __future__ import annotations

import pytest

from quackmem.core.exceptions import MetadataValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_registry(monkeypatch):
    """Clear the global _metadata_registry dict between tests."""
    import quackmem.core.registry as reg_module
    monkeypatch.setattr(reg_module, "_metadata_registry", {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_empty_registry_allows_any_kwargs(self, monkeypatch):
        _reset_registry(monkeypatch)
        from quackmem.core.registry import validate_metadata
        # Should not raise
        validate_metadata({"user_id": "abc", "foo": 123, "bar": None})

    def test_empty_registry_allows_empty_kwargs(self, monkeypatch):
        _reset_registry(monkeypatch)
        from quackmem.core.registry import validate_metadata
        validate_metadata({})  # no exception

    def test_registered_key_passes_validation(self, monkeypatch):
        _reset_registry(monkeypatch)
        from quackmem.core.registry import register_metadata, validate_metadata
        register_metadata({"user_id": str})
        # known key — no exception
        validate_metadata({"user_id": "alice"})

    def test_unknown_key_raises_metadata_validation_error(self, monkeypatch):
        _reset_registry(monkeypatch)
        from quackmem.core.registry import register_metadata, validate_metadata
        register_metadata({"user_id": str})
        with pytest.raises(MetadataValidationError):
            validate_metadata({"user_id": "alice", "unknown_key": "x"})

    def test_error_message_contains_unknown_key_name(self, monkeypatch):
        _reset_registry(monkeypatch)
        from quackmem.core.registry import register_metadata, validate_metadata
        register_metadata({"user_id": str})
        with pytest.raises(MetadataValidationError, match="bad_key"):
            validate_metadata({"bad_key": "value"})

    def test_register_metadata_merges_not_replaces(self, monkeypatch):
        _reset_registry(monkeypatch)
        from quackmem.core.registry import register_metadata, validate_metadata
        register_metadata({"user_id": str})
        register_metadata({"org_id": str})
        # Both keys should now be valid
        validate_metadata({"user_id": "alice", "org_id": "acme"})

    def test_multiple_unknown_keys_all_reported(self, monkeypatch):
        _reset_registry(monkeypatch)
        from quackmem.core.registry import register_metadata, validate_metadata
        register_metadata({"user_id": str})
        with pytest.raises(MetadataValidationError) as exc_info:
            validate_metadata({"alpha": 1, "beta": 2})
        msg = str(exc_info.value)
        assert "alpha" in msg
        assert "beta" in msg

    def test_validate_metadata_no_known_keys_passed(self, monkeypatch):
        """Registry is active but caller passes no kwargs — still valid."""
        _reset_registry(monkeypatch)
        from quackmem.core.registry import register_metadata, validate_metadata
        register_metadata({"user_id": str})
        validate_metadata({})  # empty dict — no unknown keys, no exception

    def test_type_is_stored_but_not_enforced(self, monkeypatch):
        """Registry only checks key presence, not type correctness."""
        _reset_registry(monkeypatch)
        from quackmem.core.registry import register_metadata, validate_metadata
        register_metadata({"count": int})
        # Passing string for an int key — still passes (type not enforced in v1)
        validate_metadata({"count": "not_an_int"})
