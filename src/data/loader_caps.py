"""loader 能力谓词：判断 loader 是否为惰性加载型。

与格式谓词（`LOADER_TYPE`）分离（决策 D5）：
- `is_lazy_loader(loader)` —— 能力谓词：数据不在内存里，取数必须走
  `get_series` / `get_value_from_name`，`loader.df` 为 None。
  `MDFLazyLoader` 与 `ParquetLazyLoader` 为 True。
- `LOADER_TYPE == "mdf"` —— 格式谓词：MDF 专有特性（通道组 / TABX 枚举 /
  元数据措辞 / 分组标签页），与「是否惰性」无关。

UI 判据点应按能力选择谓词：凡是「数据怎么取」的分支（取 Series、取列名、
数值表添加列）必须用 `is_lazy_loader()`；凡是「MDF 专有行为」的分支
（tab 模式、通道组措辞）继续用 `LOADER_TYPE == "mdf"`。
"""

from __future__ import annotations

from typing import Any

# 未定义 IS_LAZY 的第三方/测试替身 loader 默认视为内存型（False），
# 与 BaseDataLoader 的默认保持一致。
_DEFAULT = False


def is_lazy_loader(loader: Any) -> bool:
    """返回 loader 是否为惰性加载型（数据不在内存，df 恒为 None）。

    依据类属性 ``IS_LAZY`` 判断；BaseDataLoader（FastDataLoader /
    ExcelDataLoader 靠继承）为 False，MDFLazyLoader / ParquetLazyLoader
    为 True。loader 为 None 时返回 False。
    """
    if loader is None:
        return False
    return bool(getattr(loader, "IS_LAZY", _DEFAULT))
