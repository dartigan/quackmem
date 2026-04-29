from __future__ import annotations


class TrackerConfigError(Exception):
    """Raised for misconfiguration caught at init_tracker time."""


class MetadataValidationError(Exception):
    """Raised when an unknown metadata key is passed and a registry is active."""
