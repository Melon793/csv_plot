"""关窗竞态：收尾路径不得摸已销毁的控件（file_loader_manager 的 BLF / reload 收尾）。

现象来自真实样本实测（655 万帧 BLF 重载进行中关窗、F 路径回环途中关窗）：
``MainWindowBaseManager.mw`` 是**弱引用**，而 ``close() → deleteLater()`` 之后
Python 包装器往往还活着 —— ``getattr(mw, "_is_loading_new_data")`` 这类属性读照样
返回旧值，守卫于是放行，代码一路走到控件调用才抛 ``RuntimeError``。异常落在 Qt
事件循环里只会被 ``sys.excepthook`` 记成一条无人认领的错误；更糟的是
``_post_reload_ui_refresh`` 的 except 恢复块自己也直接摸控件，二次抛出会把
``reload_btn`` 的恢复整段带走。

沿用 ``test_xlink_sync_dead_containers.py`` 的两步写法：先用
``pytest.raises(RuntimeError)`` 证明陷阱真实存在，再证明守卫把它吞掉。
"""

from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow, QPushButton

from src.ui.file_loader_manager import FileLoaderManager, _qt_alive


class _Host(QMainWindow):
    """带收尾路径会用到的那几个真控件的最小宿主（必须是真 QWidget 家族）。"""

    def __init__(self):
        super().__init__()
        self.load_btn = QPushButton("导入数据文件", self)
        self.reload_btn = QPushButton("重载", self)
        self._data_version = 3
        self._is_loading_new_data = True
        self.plot_widgets: list = []


def _drop(widget):
    """销毁 C++ 侧、保留 Python 包装器（正是线上残留引用的形态）。

    ``deleteLater()`` 的 DeferredDelete 默认不在 ``processEvents()`` 的处理范围里，
    必须 ``sendPostedEvents`` 主动推一把 —— 真窗口关闭时这是 Qt 自己做的。
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    widget.deleteLater()
    QCoreApplication.sendPostedEvents(widget, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()
    return widget


def _manager(host):
    m = FileLoaderManager(host)
    host._kept = m  # 基类只被弱引用指向会立刻 GC，反向撑住
    return m


@pytest.fixture()
def spy_excepthook(monkeypatch):
    """捕获 Qt 回调里抛出的异常（不挂钩就会把崩溃类缺陷测成「通过」）。"""
    caught: list = []
    monkeypatch.setattr(
        sys, "excepthook",
        lambda t, v, tb: caught.append(f"{t.__name__}: {v}"),
    )
    return caught


# ---------------------------------------------------------------------------
# 前提：包装器活着 ≠ 控件能用
# ---------------------------------------------------------------------------

def test_a_dropped_wrapper_still_reads_attributes_but_raises_on_widgets(qapp):
    """守卫必须判 C++ 侧：只读 Python 属性探不到窗口已经销毁。"""
    host = _drop(_Host())

    assert _qt_alive(host) is False, "前提：C++ 侧确实没了"
    assert host._is_loading_new_data is True, "属性读不到销毁 —— 旧守卫的漏洞就在这"
    with pytest.raises(RuntimeError):
        host.reload_btn.setEnabled(True)


# ---------------------------------------------------------------------------
# P3：F 路径重开导入的存活守卫
# ---------------------------------------------------------------------------

def test_restart_gives_up_quietly_once_the_window_is_gone(qapp, spy_excepthook):
    """窗口已销毁时 ``_restart_blf_import`` 不得再走到 ``load_csv_file``。"""
    host = _Host()
    manager = _manager(host)
    reached = []
    manager.load_csv_file = lambda path: reached.append(path)  # type: ignore[assignment]
    manager._blf_force_dialog = True
    _drop(host)

    manager._restart_blf_import("/tmp/whatever.blf")

    assert reached == [], "守卫放行就会在按钮调用上抛 RuntimeError"
    assert manager._blf_force_dialog is False
    assert manager._blf_preset_db_paths is None
    assert spy_excepthook == []


def test_restart_drops_the_stashed_loader_when_giving_up(qapp):
    """放弃重开时，F 路径暂存的扫描结果也要一起释放（它可能占几百 MB）。"""

    class _Loader:
        path = "/tmp/a.blf"
        closed = False

        def close(self):
            self.closed = True

    host = _Host()
    manager = _manager(host)
    pending = _Loader()
    manager._blf_pending_loader = pending

    manager._give_up_blf_restart()

    assert pending.closed is True
    assert manager._blf_pending_loader is None


# ---------------------------------------------------------------------------
# P2：_post_reload_ui_refresh 的收尾与兜底
# ---------------------------------------------------------------------------

def test_post_reload_skips_dead_plot_widgets_and_still_unlocks(qapp, spy_excepthook):
    """绘图区已销毁、窗口还活着：跳过死控件，但锁与按钮必须照常恢复。"""
    host = _Host()
    manager = _manager(host)
    dead_container = type("C", (), {"plot_widget": _drop(_Host())})()
    host.plot_widgets = [dead_container]
    manager._post_reload_pending_version = host._data_version
    manager._safety_timer = _drop(QTimer())  # 真 QTimer：销毁后 .stop() 就是二次抛出

    manager._post_reload_ui_refresh()

    assert host._is_loading_new_data is False, "锁必须放开，否则下一次导入点不动"
    assert host.reload_btn.isEnabled() is True
    assert manager._safety_timer is None
    assert spy_excepthook == [], f"收尾路径不许抛到事件循环外: {spy_excepthook}"


def test_force_unlock_is_skipped_when_the_window_is_gone(qapp, spy_excepthook):
    """3 s 兜底解锁落在已销毁窗口上：原地收口，不许清曲线也不许外抛。"""
    host = _Host()
    host._is_loading_new_data = True
    manager = _manager(host)
    manager._safety_unlock_version = host._data_version
    cleared = []

    class _Container:
        class _W:
            def _safe_clear_plot_items(self_inner):
                cleared.append(1)

            def setUpdatesEnabled(self_inner, _on):
                cleared.append(2)

        plot_widget = _W()

    host.plot_widgets = [_Container()]
    _drop(host)

    manager._force_unlock_all()

    assert cleared == [], "窗口已销毁时不该再动绘图区"
    assert host._is_loading_new_data is True, "跳过即不写回：写它也会抛 RuntimeError"
    assert spy_excepthook == []


def test_load_csv_file_stays_quiet_when_the_window_died_mid_load(
    qapp, monkeypatch, tmp_path
):
    """加载途中关窗：失败弹窗与 ``raise_()`` 都要跳过，不许外抛异常。"""
    target = tmp_path / "a.blf"
    target.write_bytes(b"\x00")
    host = _Host()
    host._is_loading_new_data = False
    manager = _manager(host)
    caught: list = []
    boxes = []

    def _boom(_path):
        raise RuntimeError("模拟：加载途中控件被销毁")

    monkeypatch.setattr(manager, "_load_file", _boom)
    monkeypatch.setattr(manager, "_release_old_data", lambda: None)
    monkeypatch.setattr(
        "src.ui.file_loader_manager.QMessageBox.critical",
        staticmethod(lambda *a, **k: boxes.append(a[1])),
    )
    monkeypatch.setattr(
        sys, "excepthook", lambda t, v, tb: caught.append(f"{t.__name__}: {v}")
    )
    _drop(host)

    manager.load_csv_file(str(target))

    assert boxes == [], "窗口没了还弹错误框只会堵住退出流程"
    assert caught == [], f"load_csv_file 的收尾不许外抛: {caught}"


def test_load_csv_file_still_reports_on_a_live_window(qapp, monkeypatch, tmp_path):
    """反向兜底：窗口活着时失败必须照旧弹窗 + 恢复按钮（守卫不能写成静默）。"""
    target = tmp_path / "a.blf"
    target.write_bytes(b"\x00")
    host = _Host()
    host._is_loading_new_data = False
    manager = _manager(host)
    host.load_btn.setEnabled(False)
    boxes = []

    def _boom(_path):
        raise ValueError("坏了")

    monkeypatch.setattr(manager, "_load_file", _boom)
    monkeypatch.setattr(manager, "_release_old_data", lambda: None)
    monkeypatch.setattr(
        "src.ui.file_loader_manager.QMessageBox.critical",
        staticmethod(lambda *a, **k: boxes.append((a[1], a[2]))),
    )

    manager.load_csv_file(str(target))

    assert boxes and boxes[0][0] == "加载错误"
    assert host.load_btn.isEnabled() is True


def test_restart_still_runs_when_the_window_is_alive(qapp, monkeypatch):
    """守卫的另一侧：窗口好好的时候必须照常重开，不能被判死。"""
    host = _Host()
    host._is_loading_new_data = False
    manager = _manager(host)
    reached = []
    manager.load_csv_file = lambda path: reached.append(path)  # type: ignore[assignment]

    manager._restart_blf_import("/tmp/whatever.blf")
    assert reached == ["/tmp/whatever.blf"]
