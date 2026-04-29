from __future__ import annotations

from convo_tracker.core.exceptions import MetadataValidationError

_metadata_registry: dict[str, type] = {}


def register_metadata(schema: dict[str, type]) -> None:
    """Register valid metadata keys and their expected types. Global, applies to all wrappers."""
    _metadata_registry.update(schema)


def validate_metadata(kwargs: dict) -> None:
    """Validate kwargs against the registry. No-op if registry is empty.
    Raises MetadataValidationError for unknown keys."""
    if not _metadata_registry:
        return

    unknown_keys = [key for key in kwargs if key not in _metadata_registry]
    if unknown_keys:
        raise MetadataValidationError(
            f"Unknown metadata key(s): {', '.join(sorted(unknown_keys))}. "
            f"Registered keys: {', '.join(sorted(_metadata_registry.keys()))}"
        )
