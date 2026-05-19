"""
VarMetadata dataclass and utility functions

Provides unified variable metadata representation for CSV and MDF data sources,
along with classification and enumeration helpers.
"""

from dataclasses import dataclass
from typing import Optional
import itertools

import numpy as np

VALID = 1
CONST = 0
INVALID = -1
UNKNOWN = -2


@dataclass(slots=True)
class VarMetadata:
    name: str
    unit: str
    group_index: int = 0
    channel_index: int = 0
    time_min: float = 0.0
    time_max: float = 0.0
    sample_count: int = 0
    sampling_rate_hz: Optional[float] = None
    is_enum: bool = False
    is_time_channel: bool = False
    is_date: bool = False
    is_time_of_day: bool = False
    validity: int = UNKNOWN
    enum_map: Optional[dict[int, str]] = None


def classify_validity(values: np.ndarray) -> int:
    if values.size == 0:
        return INVALID
    if not np.issubdtype(values.dtype, np.number):
        return VALID
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return INVALID
    finite_vals = values[finite_mask]
    if np.allclose(finite_vals, finite_vals[0]):
        return CONST
    return VALID


def make_var_name_unique(name: str, group_index: int, conflict_set: set[str]) -> str:
    if name in conflict_set:
        return f"{name}_G{group_index}"
    return name


def is_enum_conversion(conversion) -> bool:
    if conversion is None:
        return False
    ct = getattr(conversion, "conversion_type", None)
    return ct in (9, 10, 11)


def extract_enum_map(conversion) -> Optional[dict[int, str]]:
    if conversion is None:
        return None

    result: dict[int, str] = {}

    if hasattr(conversion, "val_to_text") and conversion.val_to_text:
        for int_key, label in conversion.val_to_text.items():
            if isinstance(label, bytes):
                label = label.decode("utf-8", errors="replace").rstrip("\x00")
            result[int(int_key)] = str(label)
        if result:
            return result

    for i in itertools.count():
        text_attr = f"text_{i}"
        val_attr = f"param_val_{i}"
        if not hasattr(conversion, text_attr):
            break
        try:
            text_raw = getattr(conversion, text_attr)
            param_val = getattr(conversion, val_attr)
        except Exception:
            break
        if text_raw is None or param_val is None:
            break
        label = (
            text_raw.rstrip(b"\x00").decode("utf-8", errors="replace")
            if isinstance(text_raw, bytes)
            else str(text_raw)
        )
        result[int(param_val)] = label

    return result if result else None
