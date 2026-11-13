"""曲线选中联动的生命周期收口（P3，真实 MainWindow，offscreen）。

补齐验收 §6 中 flow 文件未覆盖的四条缺口（见 acceptance-audit-checklist）：

- §6.2 plot 侧连续 Ctrl 加选到 3 条 → 再 Ctrl 减选回 2 条，三条 pen/tint 同步
- §6.3 批量菜单「清除选择」真实分派 → 全部曲线恢复 + 真实状态栏播报
- §6.5 加载期（``_status_busy``）点击曲线：投影照常、不崩溃、秒表不被误清
- §6.7 真实 reload → 选择集 prune + 两投影同步修剪 + 系统路径静默

几何等待、点击投递、量测口径一律复用 test_curve_selection_flow 的助手，
避免两个文件各写一套导致断言口径分叉。
"""

import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from src.core.config import (
    HIGHLIGHT_CURVE_Z,
    HIGHLIGHT_LINE_WIDTH,
    HIGHLIGHT_THIN_LINE_WIDTH,
    THICK_LINE_WIDTH,
    THIN_LINE_WIDTH,
)
from src.core.data_types import SYMBOL_POLICY_OFF
from src.ui.widgets.multi_curve_manager import hit_test_curve, resolve_symbol_show
from tests.component.test_legend_context_menu import _fake_menu_env
from tests.e2e.test_curve_selection_flow import (
    _click_plot,
    _curve_pos,
    _is_tinted,
    _pw,
    _width,
)

# 夹具跨模块复用：import 进本模块命名空间即被 pytest 注册为可用夹具。
# pytest 8 起禁止直接调用夹具函数，故用"别名夹具 + 参数请求"转发，
# 既复用 flow 的几何等待口径，又不与本地夹具名冲突（ruff F811）。
from tests.e2e.test_curve_selection_flow import (  # noqa: F401
    plotted_window as _flow_plotted_window,
)
from tests.fixtures.data_factory import write_csv
from tests.fixtures.waits import settle

_SELECTION_PREFIXES = ("已选中", "已取消选中", "已清除选择")


@pytest.fixture(autouse=True)
def _guard_app_exit(monkeypatch):
    """防毒化：任何路径若调用 QApplication.exit，同进程后续用例会被 quitNow 拖死。"""
    monkeypatch.setattr(QApplication, "exit", staticmethod(lambda *a, **k: None))


@pytest.fixture()
def plotted_window(request):
    """转发 flow 文件的绘图夹具（几何等待 + symbol 策略钉死）。

    用 getfixturevalue 按名请求：夹具名不进本函数签名，避免与 import
    进来的同名夹具构成 ruff F811 重定义。
    """
    return request.getfixturevalue("_flow_plotted_window")


@pytest.fixture()
def three_curve_window(plotted_window, qapp, qtbot):
    """三条曲线：speed(10~35) / rpm(800~1290) / flag(恒 1)，三个分离带。

    flag 来自 data_factory.make_simple_rows 的常数列，与另两条在 Y 上
    完全分离，Ctrl 链式点击不会因 8px 容差互相串扰。
    """
    mw = plotted_window
    pw = _pw(mw)
    assert pw.plot_variable("flag")
    pw.curves["flag"].symbol_policy = SYMBOL_POLICY_OFF
    pw._refresh_selection_projections()
    qapp.processEvents()
    qtbot.waitUntil(
        lambda: pw.view_box.sceneBoundingRect().height() > 100, timeout=5000
    )
    settle()
    return mw


def _expected_width(pw, name, *, selected):
    """按当前 symbol 判定算出应有 pen 宽（与 _apply_curve_style 同口径）。

    reload 后曲线重建，symbol 策略回到 auto；用本函数代替硬编码 4/2，
    断言既不受模式影响，也不会因为"先刷一遍再断言"而掩盖投影缺失。
    """
    ci = pw.curves[name]
    x_min, x_max = pw.view_box.viewRange()[0]
    if resolve_symbol_show(ci, x_max - x_min):
        return HIGHLIGHT_THIN_LINE_WIDTH if selected else THIN_LINE_WIDTH
    return HIGHLIGHT_LINE_WIDTH if selected else THICK_LINE_WIDTH


def _spy_status(mw, monkeypatch):
    """记录真实状态栏收到的全部播报（保留原实现，不改通道行为）。"""
    messages: list[str] = []
    original = mw._show_status_message
    monkeypatch.setattr(
        mw,
        "_show_status_message",
        lambda message: (messages.append(message), original(message))[1],
    )
    return messages


def _reload_with(mw, qtbot, tmp_path, *, header, units, rows):
    """改写源 CSV 后真实 reload，等数据版本推进 + 加载态落定。"""
    write_csv(tmp_path / "e2e_demo.csv", header=header, units=units, rows=rows)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    version_before = mw._data_version
    mw.file_loader_manager.reload_data()
    qtbot.waitUntil(lambda: mw._data_version > version_before, timeout=5000)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    settle()


def _two_col_rows(n: int = 50) -> list[list]:
    return [[f"{i * 0.1:.1f}", f"{10.0 + i * 0.5:.2f}"] for i in range(n)]


# ---------------------------------------------------------------------------
# §6.2 三条 Ctrl 链式加选 → 减选
# ---------------------------------------------------------------------------


def test_ctrl_chain_three_curves_then_deselect_back_to_two(three_curve_window):
    """plot 侧 Ctrl 连点到 3 条全加粗，再 Ctrl 点其中一条 → 剩 2 条。"""
    mw = three_curve_window
    pw = _pw(mw)
    names = ("speed", "rpm", "flag")
    # 落点前提：三条曲线在三个分离带，命中判定必须各自归位
    for name in names:
        pos = _curve_pos(pw, name)
        assert hit_test_curve(pw, pw.mapToScene(pos)) == name, name

    _click_plot(pw, _curve_pos(pw, "speed"))
    for name in ("rpm", "flag"):
        _click_plot(pw, _curve_pos(pw, name), Qt.KeyboardModifier.ControlModifier)

    assert pw._legend_selection == set(names)
    for name in names:
        assert _width(pw, name) == HIGHLIGHT_LINE_WIDTH, name
        assert _is_tinted(pw, name), name
    assert mw._message_label.text() == "已选中：flag"

    # 减选 rpm：它恢复细线 + 去 tint，另两条保持加粗
    _click_plot(pw, _curve_pos(pw, "rpm"), Qt.KeyboardModifier.ControlModifier)

    assert pw._legend_selection == {"speed", "flag"}
    assert _width(pw, "rpm") == THICK_LINE_WIDTH
    assert not _is_tinted(pw, "rpm")
    for name in ("speed", "flag"):
        assert _width(pw, name) == HIGHLIGHT_LINE_WIDTH, name
        assert _is_tinted(pw, name), name
    assert mw._message_label.text() == "已取消选中：rpm"


# ---------------------------------------------------------------------------
# §6.3 批量菜单「清除选择」真实分派
# ---------------------------------------------------------------------------


def test_batch_menu_clear_restores_all_and_broadcasts(plotted_window, qapp, monkeypatch):
    """真实 MainWindow 上走批量菜单分派：两投影全恢复 + 状态栏「已清除选择」。"""
    mw = plotted_window
    pw = _pw(mw)
    _click_plot(pw, _curve_pos(pw, "speed"))
    _click_plot(pw, _curve_pos(pw, "rpm"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"speed", "rpm"}

    _fake_menu_env(monkeypatch, choose_text="清除选择")
    pw.legend_label._exec_batch_var_menu(QPoint(0, 0))
    qapp.processEvents()

    assert pw._legend_selection == set()
    for name in ("speed", "rpm"):
        assert _width(pw, name) == THICK_LINE_WIDTH, name
        assert not _is_tinted(pw, name), name
    assert mw._message_label.text() == "已清除选择"
    assert mw._message_label.toolTip() == "已清除选择"


# ---------------------------------------------------------------------------
# §6.5 加载期（_status_busy）播报不崩溃
# ---------------------------------------------------------------------------


def test_click_during_loading_busy_keeps_projections_and_stopwatch(
    plotted_window, monkeypatch
):
    """加载期点击：选择/两投影照常落地，busy 标志不被翻转，秒表不被误清。

    §6.5 口径：播报被秒表覆盖属可接受行为，故这里只钉"不崩溃 + 投影正确
    + busy 语义完好"，不把覆盖与否写成硬断言。
    """
    mw = plotted_window
    pw = _pw(mw)
    monkeypatch.setattr(mw, "_status_busy", True)
    mw._broadcast("正在加载数据…", "info")
    assert mw._message_label.text()

    _click_plot(pw, _curve_pos(pw, "speed"))

    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert _is_tinted(pw, "speed")
    assert mw._status_busy is True
    # busy 期 clear_status_message 是 no-op：秒表文案不该被自动回收抹掉
    mw.clear_status_message()
    assert mw._message_label.text()


# ---------------------------------------------------------------------------
# §6.7 真实 reload → prune + 两投影同步 + 静默
# ---------------------------------------------------------------------------


def test_reload_prunes_selection_and_both_projections(
    plotted_window, qapp, qtbot, tmp_path, monkeypatch
):
    """reload 后 rpm 不在新数据：选择集 prune 到 speed，加粗/tint 同步修剪。"""
    mw = plotted_window
    pw = _pw(mw)
    _click_plot(pw, _curve_pos(pw, "speed"))
    _click_plot(pw, _curve_pos(pw, "rpm"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"speed", "rpm"}
    messages = _spy_status(mw, monkeypatch)

    _reload_with(
        mw,
        qtbot,
        tmp_path,
        header=["time", "speed"],
        units=["s", "km/h"],
        rows=_two_col_rows(),
    )

    pw = _pw(mw)  # reload 可能重建容器，重新取引用
    assert "rpm" not in pw.curves
    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == _expected_width(pw, "speed", selected=True)
    assert _is_tinted(pw, "speed")
    assert not _is_tinted(pw, "rpm")
    # 系统路径静默：reload 全程不得出现选择类播报
    assert not any(m.startswith(_SELECTION_PREFIXES) for m in messages), messages


def test_reload_emphasis_survives_skipped_style_refresh(
    plotted_window, qapp, qtbot, tmp_path, monkeypatch
):
    """BL-2（评审 r2 自查）：reload 收尾的样式刷新被跳过时，加粗仍须在。

    真实缺陷时序：刚点过曲线就 reload，``_is_interacting`` 尚未落定 →
    收尾唯一那次 ``update_plot_style`` 被门控整个跳过 → 重建出的曲线停在
    基础 pen（``_cached_pen_key is None``），此后视图范围不变就永不重刷，
    选中曲线永久不加粗。负载下实测 6 次 reload 命中 2 次、pump 1.2 s 不收敛
    （tmp/2026-11-14/probe_reload_gates.py 门控轨迹）。

    确定性复现：把 ``update_plot_style`` 换成空实现并计数，等价于"那次刷新
    被跳过"，不靠 CPU 负载碰运气。断言加粗只能来自独立收口
    ``_refresh_selection_after_rebuild``；同时钉住系统路径静默。
    """
    mw = plotted_window
    pw = _pw(mw)
    _click_plot(pw, _curve_pos(pw, "speed"))
    assert pw._legend_selection == {"speed"}
    messages = _spy_status(mw, monkeypatch)

    style_calls: list[int] = []
    monkeypatch.setattr(
        pw._plot_ui_manager,
        "update_plot_style",
        lambda *a, **k: style_calls.append(1),
    )

    _reload_with(
        mw,
        qtbot,
        tmp_path,
        header=["time", "speed"],
        units=["s", "km/h"],
        rows=_two_col_rows(),
    )

    pw = _pw(mw)
    assert style_calls, "前提：reload 收尾确实尝试过样式刷新（否则本例空转）"
    assert pw._legend_selection == {"speed"}
    curve = pw.curves["speed"].curve
    assert getattr(curve, "_cached_pen_key", None) is not None, (
        "曲线重建后没有任何一次 _apply_curve_style 落地"
    )
    assert curve._cached_pen_key.endswith("_hl"), curve._cached_pen_key
    assert _width(pw, "speed") == _expected_width(pw, "speed", selected=True)
    assert curve.zValue() == HIGHLIGHT_CURVE_Z
    assert _is_tinted(pw, "speed")
    assert not any(m.startswith(_SELECTION_PREFIXES) for m in messages), messages


def test_reload_empty_data_keeps_selection_consistent_and_silent(
    plotted_window, qapp, qtbot, tmp_path, monkeypatch
):
    """空数据 reload：replots_after_loading 早退，曲线资产原样保留。

    实测（offscreen）：早退路径不重建曲线，speed/rpm 仍在 pw.curves、
    legend 仍显示旧图例。prune 的契约是"只修剪不在 curves 里的条目"，
    故此时选择集**不该**被清空——这与暂存区按"数据变量是否存在"置灰
    是两套口径。本例钉住：选择集仍是 curves 子集、两投影不脱节、全程静默。
    """
    mw = plotted_window
    pw = _pw(mw)
    _click_plot(pw, _curve_pos(pw, "speed"))
    _click_plot(pw, _curve_pos(pw, "rpm"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"speed", "rpm"}
    messages = _spy_status(mw, monkeypatch)

    _reload_with(
        mw,
        qtbot,
        tmp_path,
        header=["time", "other"],
        units=["s", "-"],
        rows=[],
    )

    pw = _pw(mw)
    assert pw._legend_selection <= set(pw.curves)
    for name in pw.curves:
        selected = name in pw._legend_selection
        assert _width(pw, name) == _expected_width(pw, name, selected=selected), name
        assert _is_tinted(pw, name) == selected, name
    assert not any(m.startswith(_SELECTION_PREFIXES) for m in messages), messages