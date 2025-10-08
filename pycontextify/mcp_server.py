"""Compatibility shim for :mod:`pycontextify.mcp`."""

from __future__ import annotations

import sys
import warnings

from . import mcp as _mcp

warnings.warn(
    "pycontextify.mcp_server is deprecated; use pycontextify.mcp instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _mcp
