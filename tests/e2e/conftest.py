"""e2e 共享夹具：真实 MainWindow（offscreen）+ 模态对话框替身。

e2e 层职责：验证「加载 → 浏览 → 绘图 → 交互」纵向链路在真实
MainWindow 上跑通，断言保持冒烟级（深断言下沉 component 层）。

关键隔离点（见 tests/README.md 陷阱清单）：
1. sys.argv 必须在构造 MainWindow 前重置——_handle_cli_args 会把
   pytest 的命令行参数当作数据文件路径去加载，触发 QMessageBox
   在 offscreen 永久阻塞；
2. QFileDialog/QMessageBox 静态方法统一替身化，防止任何模态弹窗阻塞；
3. 配置/日志路径由顶层 conftest 的环境变量注入隔离。
"""

import sys
import traceback

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox

# 已知的清理期竞态异常（非功能缺陷）：加载链路会调度
# QTimer.singleShot 延迟回调（如 _post_reload_ui_refresh），窗口关闭/
# 销毁后这些定时器仍可能触发，src 内部已自行容错，但 PySide6 会把
# 槽函数异常交给 sys.excepthook → pytest-qt 记入异常池并错误归因给
# 其他用例（随机失败）。在 main_window 夹具内对 pytest-qt 的钩子
# 包一层过滤（本夹具 teardown 早于 qtbot，钩子此时仍在生效）。
_KNOWN_TEARDOWN_RACE_MARKERS = (
    "_post_reload_ui_refresh",
    "_deferred_cursor_refresh_all",
    "_safety_force_unlock",
    "_restore_cursor_state_after_reload",
)


def _install_race_filter():
    """包装当前 sys.excepthook（pytest-qt 捕获钩子）：
    已知清理竞态异常直接吞掉，其余照常交给原钩子。"""
    wrapped = sys.excepthook

    def filtered(exc_type, exc_value, exc_tb):
        tb_text = "".join(
            traceback.format_exception(exc_type, exc_value, exc_tb)
        )
        if any(m in tb_text for m in _KNOWN_TEARDOWN_RACE_MARKERS):
            return  # 已知清理竞态：src 已自行处理，静默即可
        wrapped(exc_type, exc_value, exc_tb)

    sys.excepthook = filtered
    return wrapped


@pytest.fixture()
def dialog_stubs(monkeypatch):
    """模态对话框统一替身：记录调用 + 返回安全默认值。

    返回 dict：{"file": (path, filter), "warning": [...], ...}，
    用例可断言特定路径未触发错误弹窗。
    """
    records = {
        "open_file": ("", ""),          # 默认取消选择
        "warning": [],
        "critical": [],
        "information": [],
        "question": [],
    }

    def fake_get_open_file_name(parent=None, caption="", directory="", filter="", *args, **kwargs):
        return records["open_file"]

    def make_recorder(kind, ret):
        def fake(parent, title, text, *args, **kwargs):
            records[kind].append((title, text))
            return ret
        return fake

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(fake_get_open_file_name)
    )
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        make_recorder("warning", QMessageBox.StandardButton.Ok)))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(
        make_recorder("critical", QMessageBox.StandardButton.Ok)))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(
        make_recorder("information", QMessageBox.StandardButton.Ok)))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        make_recorder("question", QMessageBox.StandardButton.No)))
    return records


@pytest.fixture()
def main_window(qapp, app_settings, dialog_stubs, monkeypatch, qtbot):
    """真实 MainWindow 夹具（offscreen）。

    构造前重置 sys.argv（防止 pytest 参数进入 _handle_cli_args），
    收尾 close() 触发正常退出路径（closeEvent → 自动保存/清理）。
    """
    monkeypatch.setattr("sys.argv", ["csv_plot_e2e"])

    from src.ui.main_window import MainWindow

    mw = MainWindow()
    qtbot.addWidget(mw)
    mw.show()
    qapp.processEvents()
    yield mw
    # 收尾窗口才安装竞态过滤器（仅覆盖清理期，测试体内的真实异常照常上报）：
    # 先向 monkeypatch 注册还原点（恢复原钩子，不泄漏到后续用例），再包装。
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    _install_race_filter()
    # 先让 pending 定时器尽量跑完再 close，减少清理期竞态窗口。
    qtbot.wait(80)
    qapp.processEvents()
    try:
        mw.close()
    except RuntimeError:
        pass  # WA_DeleteOnClose：可能已被销毁
    qapp.processEvents()


@pytest.fixture()
def loaded_window(main_window, qapp, qtbot, dialog_stubs, tmp_path):
    """已加载合成 CSV 的 MainWindow（模拟 help.md 1️⃣ 加载数据）。

    通过点击「导入数据文件」按钮 + QFileDialog 替身注入路径，
    走完整用户路径（load_btn_click → load_csv_file → 建图）。
    """
    from tests.fixtures.data_factory import make_simple_rows, write_csv

    csv = write_csv(
        tmp_path / "e2e_demo.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(50),
    )
    dialog_stubs["open_file"] = (str(csv), "CSV/TXT Files (*.csv)")

    qtbot.mouseClick(main_window.load_btn, Qt.MouseButton.LeftButton)

    # 等待加载完成：小文件同步路径，但仍以防抖/延迟刷新为准确认点
    qtbot.waitUntil(lambda: main_window.loader is not None, timeout=5000)
    qtbot.waitUntil(lambda: len(main_window.plot_widgets) > 0, timeout=5000)
    qapp.processEvents()

    assert not dialog_stubs["critical"], f"加载出现错误弹窗: {dialog_stubs['critical']}"
    return main_window
