"""Core module for CSV Plot"""

from src.core.plot_config import (
    PlotSessionConfig,
    PlotConfig,
    TemplateError,
    TemplateNotFoundError,
    TemplateNameConflictError,
    TemplateValidationError,
    TemplateStorageError,
)
from src.core.template_models import PlotTemplate, TemplateMetadata
from src.core.storage import TemplateStorage
from src.core.template_manager import TemplateManager
from src.core.auto_save_manager import AutoSaveManager

__all__ = [
    "PlotSessionConfig",
    "PlotConfig",
    "TemplateError",
    "TemplateNotFoundError",
    "TemplateNameConflictError",
    "TemplateValidationError",
    "TemplateStorageError",
    "PlotTemplate",
    "TemplateMetadata",
    "TemplateStorage",
    "TemplateManager",
    "AutoSaveManager",
]
