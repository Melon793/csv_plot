"""独立窗口几何持久化入口的回归测试（数值表 / 变量信息）。

Bug 场景：用户打开变量数值表并手动调整位置/大小后，对一个新变量
触发数值表入口（双击行 → popup else 分支、拖拽/右键添加 →
add_variables），窗口会被重置回旧快照——用户视角"跳回默认"。

根因：restoreGeometry 对**已显示**窗口不是幂等操作（按保存时的
frame 偏移重摆位置，offscreen 实测连续 restore 逐次内缩漂移），
而窗口开着时当前几何就是最新状态，本不需要恢复。修复为 load_geom
加 isVisible 守卫；关闭后重开（新构造窗口）仍走 blob 恢复。
"""

import time

import pytest

from PySide6.QtWidgets import QApplication, QMessageBox

USER_GEOM = (152, 122, 900, 600)


def _pump(ms: int = 60) -> None:
    end = time.time() + ms / 1000
    while time.time() < end:
        QApplication.processEvents()


def _geom(win) -> tuple[int, int, int, int]:
    g = win.geometry()
    return (g.x(), g.y(), g.width(), g.height())


@pytest.fixture(autouse=True)
def _close_singleton_windows():
    """两个类级单例窗口跨用例保活，不显式收尾会污染后续用例的 popup 复用。"""
    yield
    from src.ui.dialogs.variable_info_dialog import VariableInfoDialog
    from src.ui.table_dialog import DataTableDialog

    for cls in (DataTableDialog, VariableInfoDialog):
        dlg = cls._instance
        cls._instance = None
        if dlg is None:
            continue
        try:
            if hasattr(dlg, "set_skip_close_confirmation"):
                dlg.set_skip_close_confirmation(True)
            dlg.close()
        except RuntimeError:
            pass  # 已随主窗口销毁
    _pump(50)


def test_popup_new_var_keeps_user_geometry(loaded_window, monkeypatch):
    """数值表开着时 popup 新变量（else 分支 load_geom）不得重置用户调好的几何。"""
    mw = loaded_window
    _pump(200)
    from src.ui.table_dialog import DataTableDialog

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    dlg = DataTableDialog.popup("speed", mw.loader.df["speed"], parent=mw)
    _pump(200)
    dlg.setGeometry(*USER_GEOM)
    _pump(50)

    DataTableDialog.popup("flag", mw.loader.df["flag"], parent=mw)
    _pump(200)
    assert _geom(dlg) == USER_GEOM, "对新变量 popup 后几何被 load_geom 重置"


def test_add_variables_keeps_user_geometry(loaded_window):
    """批量添加入口（拖拽/变量列表添加）同样不得重置已开窗口的几何。"""
    mw = loaded_window
    _pump(200)
    from src.ui.table_dialog import DataTableDialog

    dlg = DataTableDialog.popup("speed", mw.loader.df["speed"], parent=mw)
    _pump(200)
    dlg.setGeometry(*USER_GEOM)
    _pump(50)

    DataTableDialog.add_variables(["rpm"], parent=mw)
    _pump(200)
    assert _geom(dlg) == USER_GEOM, "add_variables 后几何被 load_geom 重置"


def test_variable_info_reopen_keeps_user_geometry(loaded_window):
    """变量信息窗口开着时再次 popup（右键另一个变量）不得重置其几何。"""
    from src.ui.dialogs.variable_info_dialog import VariableInfoDialog

    mw = loaded_window
    _pump(200)
    vi = VariableInfoDialog.popup(["rpm"], parent=mw)
    _pump(100)
    vi.setGeometry(*USER_GEOM)
    _pump(50)

    VariableInfoDialog.popup(["flag"], parent=mw)
    _pump(100)
    assert _geom(vi) == USER_GEOM, "第二次打开变量信息时几何被 load_geom 重置"


def test_geometry_restored_after_close_and_reopen(loaded_window, monkeypatch):
    """关闭后重开必须从快照恢复（防 isVisible 守卫过度屏蔽恢复路径）。

    offscreen 下首次 restore 会按保存的 frame 偏移内缩（main 基线一致，
    真实 WM 无此怪癖），故只断言"明显不是 __init__ 默认的 600x400 居中"
    且纵向位置被 blob 带回，不钉死漂移后的具体值。
    """
    mw = loaded_window
    _pump(200)
    from src.ui.table_dialog import DataTableDialog

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    dlg = DataTableDialog.popup("speed", mw.loader.df["speed"], parent=mw)
    _pump(200)
    dlg.setGeometry(*USER_GEOM)
    _pump(50)

    dlg.set_skip_close_confirmation(True)
    dlg.close()
    _pump(100)
    assert DataTableDialog._instance is None

    dlg2 = DataTableDialog.popup("speed", mw.loader.df["speed"], parent=mw)
    _pump(200)
    x, y, w, h = _geom(dlg2)
    assert (w, h) != (600, 400), "重开走了 __init__ 默认尺寸 → data_table_geometry 快照链路断了"
    assert w > 700 and h == USER_GEOM[3], f"重开未从快照恢复尺寸: {_geom(dlg2)}"
    assert y == USER_GEOM[1], f"重开未从快照恢复纵向位置: {_geom(dlg2)}"
