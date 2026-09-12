"""
VarMetadata dataclass and utility functions

Provides unified variable metadata representation for CSV and MDF data sources,
along with classification and enumeration helpers.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

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
    # 标称采样间隔（秒）。MDF3 的 ch.sampling_rate 语义即为"秒"，
    # MDF4 的 raster 藏在 CN comment XML 的 <raster> 标签内，两者统一存为秒。
    nominal_raster_s: Optional[float] = None
    # 平均有效采样率（Hz），由所属组 master 首尾时间戳推算：
    # (cycles_nr - 1) / (time_max - time_min)。对变速/事件型采样会失真，
    # 且与 nominal_raster_s 可能不一致（实测标称 1 kHz 而实际 100 Hz）。
    effective_rate_hz: Optional[float] = None
    # 跨通道组重名时 name 会被改写为 f"{original_name}_G{gi}"，
    # 此字段保留改写前的原始通道名；非重名场景与 name 相同。
    original_name: str = ""
    is_enum: bool = False
    is_time_channel: bool = False
    is_date: bool = False
    is_time_of_day: bool = False
    validity: int = UNKNOWN
    enum_map: Optional[dict[int, str]] = None


# ---------------------------------------------------------------------------
# conversion_type 版本感知判定
#
# MDF 2.x/3.x 与 4.x 的 conversion_type 数值语义完全不同，同一数值可能指向
# 相反类别，必须按文件版本取表：
#   ct=7  -> MDF4: TABX（文本表，枚举） / MDF3: EXPO（指数，纯数值）
#   ct=9  -> MDF4: TTAB（文本表，枚举） / MDF3: RAT（有理，纯数值）
#   ct=11 -> MDF4: BITFIELD（位域）     / MDF3: TABX（文本表，枚举）
# 此处内联 frozenset 而不 import asammdf：本模块仅依赖 numpy，硬 import 会
# 拖慢 CSV-only 场景的启动并增大打包体积（项目有启动性能优化历史）。
# ---------------------------------------------------------------------------
# 判定依据为 asammdf 的 CONVERSION_TYPE_TO_STRING 常量表（实测 8.8.9）：
#   MDF3: 0 LINEAR / 1 TABI / 2 TAB / 6 POLY / 7 EXPO / 8 LOGH / 9 RAT /
#         10 FORMULA / 11 TABX / 12 RTABX / 65535 NONE
#   MDF4: 0 NON / 1 LIN / 2 RAT / 3 ALG / 4 TABI / 5 TAB / 6 RTAB / 7 TABX /
#         8 RTABX / 9 TTAB / 10 TRANS / 11 BITFIELD
# 只有"输出为文本"的转换才算枚举：TABX/RTABX/TTAB/TRANS/BITFIELD。
# RTAB(6) 与 TAB(5)/TABI(4) 虽然也是查表，但输出仍是**数值**，其转换块内
# 没有 text_i 字段，extract_enum_map() 必定返回 None；若误判为枚举会让
# get_series() 走 raw=True 显示原始码值，且无任何文本标签可用。
# ---------------------------------------------------------------------------
_ENUM_CT_MDF3 = frozenset({11, 12})  # TABX 文本表 / RTABX 范围文本表
_ENUM_CT_MDF4 = frozenset({7, 8, 9, 10, 11})  # TABX/RTABX/TTAB/TRANS/BITFIELD


def is_mdf3_version(version: Optional[str]) -> bool:
    """判断是否为 MDF 2.x/3.x（其 conversion_type 语义与 4.x 完全不同）。"""
    return str(version or "").startswith(("2", "3"))


def enum_conversion_types(version: Optional[str]) -> frozenset:
    """按 MDF 版本返回属于枚举/文本转换的 conversion_type 集合。"""
    return _ENUM_CT_MDF3 if is_mdf3_version(version) else _ENUM_CT_MDF4


# RTABX 是「区间 → 文本」表，与 TABX 的「码值 → 文本」结构完全不同，
# extract_enum_map() 的 dict[int, str] 契约无法表达（详见该函数 docstring）。
# 单独识别出来，是为了让 UI 能给出准确提示：“范围文本表暂不支持展示”
# 而不是误导用户的 “文本表提取失败”——后者会让人以为是解析出了 bug。
# 注意 RTAB（MDF4 ct=6）输出的是**数值**而非文本，不属于范围文本表。
_RANGE_TEXT_CT_MDF3 = frozenset({12})  # RTABX
_RANGE_TEXT_CT_MDF4 = frozenset({8})  # RTABX


def is_range_text_conversion(version: Optional[str], ct) -> bool:
    """判断转换类型码是否为「区间 → 文本」的范围文本表 (RTABX)。"""
    if ct is None:
        return False
    table = _RANGE_TEXT_CT_MDF3 if is_mdf3_version(version) else _RANGE_TEXT_CT_MDF4
    return ct in table


def is_enum_conversion(conversion, mdf_version: Optional[str]) -> bool:
    """判定转换块是否为枚举/文本类型。

    mdf_version 为必填：缺少版本信息会把 MDF3 的 RAT(9)/EXPO(7)/FORMULA(10)
    误判为枚举，导致 get_series() 走 raw=True 返回原始码值而非物理值
    （实测某 MDF3 通道因此显示 2731 而真实物理值为 0.12，差 4 个数量级）。
    """
    if conversion is None:
        return False
    ct = getattr(conversion, "conversion_type", None)
    if ct is None:
        return False
    return ct in enum_conversion_types(mdf_version)


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
    """提取「码值 → 文本」映射，无法表达时返回 None。

    三个策略依次尝试：asammdf 已解析好的 ``val_to_text``；``param_val_i`` +
    ``text_i`` 配对遍历；CAN db 指针型（``text_i`` 为 int 地址 + ``val_i``）。

    **RTABX（范围文本表）刻意不支持，必定返回 None** —— 它映射的是区间而不是
    码值，与本函数的 ``dict[int, str]`` 契约根本不兼容。实测 asammdf 8.8.9 的
    存储结构（MDF3 与 MDF4 一致，见 v2_v3_blocks.py / v4_blocks.py）：

    - 键值属性：TABX 用 ``param_val_i``；RTABX 用 ``lower_i`` + ``upper_i``
    - 文本位置：TABX 的 ``text_i`` 直接是 bytes；RTABX 的 ``text_i`` 是 int
      （TX 块地址），真文本在 ``referenced_blocks["text_i"]``
    - 条目数：TABX 是 ``ref_param_nr``；RTABX 是 ``ref_param_nr - 1``

    于是策略二因缺 ``param_val_i`` 而 break；策略三又因 RTABX 的 ``text_i``
    恰好是 int 而误入 CAN db 分支，但 RTABX 无 ``val_i``，最终收集为空。
    两条路都落空是结构差异导致，不是解析缺陷。

    补充：asammdf 的 ``Conversion.convert()`` **完整支持** RTABX（含 default
    文本带 ``{X}`` 时退化为线性公式 ``a*X+b`` 的特殊分支），因此 ``raw=False``
    本可拿到正确文本标签。当前仍按 ``is_enum`` 走 ``raw=True`` 取码值是刻意为之：
    避免字符串数组进入绘图路径。UI 侧用 ``is_range_text_conversion`` 给出准确提示。
    """
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
            except Exception as e:
                logger.debug("CAN db convert() 失败，返回空枚举映射: %s", e)

    return result if result else None
