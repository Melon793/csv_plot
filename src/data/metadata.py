"""
VarMetadata dataclass and utility functions

Provides unified variable metadata representation for CSV and MDF data sources,
along with classification and enumeration helpers.
"""

from dataclasses import dataclass
from typing import Optional

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
    return ct in (7, 9, 10, 11)


def _get_enum_entry_count(conversion) -> int:
    """获取枚举转换的实际条目数。

    优先使用 asammdf 的 ref_param_nr 属性（MDF3/MDF4 均支持），
    若不可用则回退到保守上限，避免大型枚举表被截断。
    """
    nr = getattr(conversion, "ref_param_nr", None)
    if nr is not None and isinstance(nr, int) and nr > 0:
        return nr
    # 回退：保守上限（覆盖绝大多数场景）
    return 2048


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

    max_entries = _get_enum_entry_count(conversion)

    for i in range(max_entries):
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
        try:
            result[int(float(param_val))] = label
        except (ValueError, TypeError):
            break

    if not result:
        # 策略三：CAN db 指针型枚举 (conv_type=7)
        # text_i 为 int（CAN 数据库字符串地址），实际值在 val_i 中
        # 通过 conversion.convert() 利用 asammdf 内置 CAN db 解析还原文本
        can_vals = []
        for i in range(max_entries):
            text_attr = f"text_{i}"
            val_attr = f"val_{i}"
            if not hasattr(conversion, text_attr):
                break
            if not hasattr(conversion, val_attr):
                continue
            try:
                text_val = getattr(conversion, text_attr)
                raw_val = getattr(conversion, val_attr)
            except Exception:
                break
            if text_val is None or raw_val is None:
                break
            # text_i 必须是 int 才走 CAN db 指针策略
            if not isinstance(text_val, int):
                can_vals.clear()
                break
            can_vals.append(float(raw_val))

        if can_vals:
            try:
                raw_arr = np.asarray(can_vals, dtype=np.float64)
                converted = conversion.convert(raw_arr)
                for j, raw_val in enumerate(can_vals):
                    label = converted[j]
                    if isinstance(label, bytes):
                        label = label.decode("utf-8", errors="replace").rstrip("\x00")
                    result[int(raw_val)] = str(label)
            except Exception:
                pass  # convert() 失败时不阻塞，返回空 map

    return result if result else None
