"""Data-loading public API with lazy exports.

Importing a submodule such as ``src.data.can_bindings`` must not eagerly import
pandas, NumPy loaders, MDF and BLF implementations during GUI startup.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FastDataLoader": ("src.data.loader", "FastDataLoader"),
    "DataLoadThread": ("src.data.loader", "DataLoadThread"),
    "MDFLazyLoader": ("src.data.mdf_lazy_loader", "MDFLazyLoader"),
    "BLFLazyLoader": ("src.data.blf_loader", "BLFLazyLoader"),
    "VarMetadata": ("src.data.metadata", "VarMetadata"),
    "VALID": ("src.data.metadata", "VALID"),
    "CONST": ("src.data.metadata", "CONST"),
    "INVALID": ("src.data.metadata", "INVALID"),
    "UNKNOWN": ("src.data.metadata", "UNKNOWN"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
