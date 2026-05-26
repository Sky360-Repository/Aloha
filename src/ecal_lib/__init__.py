"""
ecal_lib: eCAL middleware utilities and core functionality.

This package provides:
- Messaging (publisher/subscriber)
- Runtime utilities
- Protobuf message definitions

Entry-point scripts (nodes/tests) are intentionally not part of the public API.
"""

from __future__ import annotations

__all__ = [
    "src",
    "utils",
    "proto_files",
]