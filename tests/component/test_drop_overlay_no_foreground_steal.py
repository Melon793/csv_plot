"""拖入文件「悬停」阶段不得抢前台：``show_drop_overlay`` 只做视觉反馈。

现场：在资源管理器里拖动 csv 去移动文件，光标经过 csv_plot 时窗口跳到前台、
拖拽被打断。根因是 DragEnter 分支对**子控件**调了 ``activateWindow()``
（Qt 语义 = 激活该控件所属的顶层窗口）。本用例钉住两点：

1. 遮罩照常 ``show()``（视觉反馈不能少）；
2. 遮罩与宿主窗口**都不**发起激活/抬升请求 —— 断言同时覆盖宿主，
   防止有人把置前改写成 ``self.mw.activateWindow()`` 仍然绿。

不建真 MainWindow：沿用 ``test_event_filter_entry_guard.py`` 的轻量做法
（``LayoutManager.__new__`` + 手放 ``_mw_ref``），宿主用真 ``QMainWindow``
（``obj.window() is mw`` 需要真窗口关系），但从不 ``show()``。
"""

from types import SimpleNamespace

from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent
from PySide6.QtWidgets import QMainWindow, QWidget

from src.ui.layout_manager import LayoutManager

# offscreen 下手工构造的 QMimeData 不受 Qt 事件系统接管，局部变量被 GC 后
# 事件内访问 mimeData 会 SIGSEGV（tests/README 陷阱 #2）
_keep_alive: list = []


class _SpyOverlay(QWidget):
    """记下每个公开调用；activateWindow 只记录、绝不真调。"""

    def __init__(self, parent):
        super().__init__(parent)
        self.calls: list[str] = []

    def show(self):
        self.calls.append("show")
        super().show()

    def raise_(self):
        self.calls.append("raise")

    def activateWindow(self):
        self.calls.append("activate")

    def adjust_text(self, file_type_supported=True):
        self.calls.append(f"text:{file_type_supported}")


class _SpyHostWindow(QMainWindow):
    """宿主也用记录器：置前若被挪到窗口上，这里会留下痕迹。"""

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def raise_(self):
        self.calls.append("raise")

    def activateWindow(self):
        self.calls.append("activate")


def _build(qapp, *, supported_extractor=None):
    mw = _SpyHostWindow()
    host = QWidget(mw)
    mw.setCentralWidget(host)

    overlay = _SpyOverlay(host)
    mw.drop_overlay = overlay
    mw.file_loader_manager = SimpleNamespace(
        _extract_file_extension=supported_extractor,
        load_csv_file=lambda path: (_ for _ in ()).throw(
            AssertionError(f"DragEnter 不该触发加载: {path}")
        ),
    )

    lm = LayoutManager.__new__(LayoutManager)
    lm._mw_ref = lambda: mw
    return lm, host, overlay, mw


def _drag_enter(host, path):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(path)])
    _keep_alive.append(mime)
    # QDragEnterEvent 的位置参数只收 QPoint（QDropEvent 才是 QPointF）
    ev = QDragEnterEvent(
        QPoint(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    return ev


def _no_activation(overlay: _SpyOverlay, mw: _SpyHostWindow):
    assert "activate" not in overlay.calls, f"遮罩请求了激活: {overlay.calls}"
    assert "activate" not in mw.calls, f"宿主窗口请求了激活: {mw.calls}"


def test_supported_file_drag_enter_shows_overlay_without_activation(qapp):
    lm, host, overlay, mw = _build(qapp)

    ev = _drag_enter(host, "/tmp/demo.csv")

    assert lm._handle_event_filter(host, ev) is True
    assert ev.isAccepted()
    assert "show" in overlay.calls
    assert overlay.calls[-1] == "text:True"
    _no_activation(overlay, mw)


def test_unsupported_file_drag_enter_shows_overlay_without_activation(qapp):
    # 不受支持的类型同样显示遮罩（既定行为），因此同样不能置前
    lm, host, overlay, mw = _build(qapp, supported_extractor=lambda _p: None)

    ev = _drag_enter(host, "/tmp/demo.zip")

    assert lm._handle_event_filter(host, ev) is True
    assert "show" in overlay.calls
    assert overlay.calls[-1] == "text:False"
    _no_activation(overlay, mw)


def test_text_drag_enter_is_left_to_child_widgets(qapp):
    # 变量名拖拽走子控件自己的 dragEnterEvent，遮罩不该出现
    lm, host, overlay, mw = _build(qapp)

    mime = QMimeData()
    mime.setText("speed")
    _keep_alive.append(mime)
    ev = QDragEnterEvent(
        QPoint(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    assert lm._handle_event_filter(host, ev) is False
    assert overlay.calls == []
    _no_activation(overlay, mw)