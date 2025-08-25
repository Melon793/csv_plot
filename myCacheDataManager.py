from __future__ import annotations
from typing import Dict, Tuple

import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal


class CacheManager(QObject):

    # 参数：列名
    data_changed = pyqtSignal(str)
    
    # 参数：None，整表被重载或清空
    all_cleared  = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._cache: Dict[str, Tuple[pd.Series, str]] = {}
        self._df: pd.DataFrame = pd.DataFrame()   # 当前完整数据

    # --------------------------------------------------
    # 对外接口
    # --------------------------------------------------
    def load_new_data(self, df: pd.DataFrame) -> None:
        """整表重载：清空缓存 + 替换数据 + 广播"""
        self._cache.clear()
        self._df = df.copy()
        self.all_cleared.emit()

    def append_rows(self, new_df: pd.DataFrame) -> None:
        """增量追加：更新内部 DataFrame，只把受影响的列标记为脏"""
        old_cols = set(self._df.columns)
        self._df = pd.concat([self._df, new_df], ignore_index=True)

        # 新列也需要加入缓存
        for col in new_df.columns:
            if col in self._cache:           # 旧列：删掉，稍后再按需重建
                del self._cache[col]
            self.data_changed.emit(col)      # 通知刷新

        # 如果新增了列，也广播一下（可选）
        for col in set(self._df.columns) - old_cols:
            self.data_changed.emit(col)

    def clear(self) -> None:
        """手动清空缓存 & 数据"""
        self._cache.clear()
        self._df = pd.DataFrame()
        self.all_cleared.emit()

    # --------------------------------------------------
    # 内部：按需转换并缓存
    # --------------------------------------------------
    def get_value_and_format(
        self,
        var_name: str,
        *,
        time_channels_info: dict[str, str]
    ) -> Tuple[pd.Series, str] | None:
        """
        与之前逻辑一致，只是多了 self._df 作为数据源
        """
        if var_name not in self._df.columns:
            return None

        if var_name in self._cache:          # 命中
            return self._cache[var_name]

        raw = self._df[var_name]
        kind = raw.dtype.kind

        # ---- 数值 ----
        if kind in "iuf":
            values = raw.astype("int64", copy=False)
            fmt = "number"

        # ---- 时间 ----
        elif var_name in time_channels_info:
            fmt_str = time_channels_info[var_name]
            try:
                dt = pd.to_datetime(raw, errors="coerce")
                if "%H:%M:%S" in fmt_str:          # 毫秒
                    values = (
                        dt.dt.hour * 3_600_000
                        + dt.dt.minute * 60_000
                        + dt.dt.second * 1_000
                        + dt.dt.microsecond // 1_000
                    ).astype("int64")
                    fmt = "ms"
                else:                               # 天数
                    epoch = pd.Timestamp("1970-01-01")
                    values = (dt.dt.normalize() - epoch).dt.days.astype("int64")
                    fmt = "date"
            except Exception:
                return None
        else:
            return None

        self._cache[var_name] = (values, fmt)
        return values, fmt