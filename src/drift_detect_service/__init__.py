"""Drift Detect Service package.

Exports the FastAPI app at package level for convenience.
"""

from .api import app  # noqa: F401

__all__ = ["app"]
