from __future__ import annotations


class TrackerConfigError(Exception):
    """Raised for misconfiguration caught at init_tracker time."""
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


class MetadataValidationError(Exception):
    """Raised when an unknown metadata key is passed and a registry is active."""
    def __init__(self, message: str, unknown_keys: list[str] | None = None):
        super().__init__(message)
        self.unknown_keys = unknown_keys or []
