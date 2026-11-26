"""Core package with lazy public exports.

Importing a lightweight submodule such as ``src.core.startup_metrics`` must not
initialize settings, templates, logging or Qt.  Public package attributes stay
backwards-compatible through PEP 562 lazy resolution.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.auto_save_manager import AutoSaveManager
    from src.core.plot_config import (
        PlotConfig,
        PlotSessionConfig,
        TemplateError,
        TemplateNameConflictError,
        TemplateNotFoundError,
        TemplateStorageError,
        TemplateValidationError,
    )
    from src.core.settings import AppSettings, ConfigKey
    from src.core.storage import TemplateStorage
    from src.core.template_manager import TemplateManager
    from src.core.template_models import PlotTemplate, TemplateMetadata

_EXPORT_MODULES = {
    "AppSettings": "src.core.settings",
    "ConfigKey": "src.core.settings",
    "PlotSessionConfig": "src.core.plot_config",
    "PlotConfig": "src.core.plot_config",
    "TemplateError": "src.core.plot_config",
    "TemplateNotFoundError": "src.core.plot_config",
    "TemplateNameConflictError": "src.core.plot_config",
    "TemplateValidationError": "src.core.plot_config",
    "TemplateStorageError": "src.core.plot_config",
    "PlotTemplate": "src.core.template_models",
    "TemplateMetadata": "src.core.template_models",
    "TemplateStorage": "src.core.storage",
    "TemplateManager": "src.core.template_manager",
    "AutoSaveManager": "src.core.auto_save_manager",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
