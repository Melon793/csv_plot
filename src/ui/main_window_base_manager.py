"""MainWindow 管理器基类"""

from __future__ import annotations
import weakref
from typing import Any


class MainWindowBaseManager:
    """所有 MainWindow 管理器的基类，提供统一的弱引用管理"""

    def __init__(self, main_window: Any):
        self._mw_ref = weakref.ref(main_window)

    @property
    def mw(self) -> Any:
        mw = self._mw_ref()
        if mw is None:
            raise RuntimeError(
                f"{type(self).__name__}: MainWindow has been garbage collected"
            )
        return mw
