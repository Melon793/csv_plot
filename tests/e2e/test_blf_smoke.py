"""e2e 冒烟（§8）：真实夹具 BLF + 其配对 DBC → 导入 → 绘图 → 游标 → 重载不再弹窗。

与 component 层的分工：``tests/component/test_blf_import_flow.py`` 用替身线程把
§5.1 的 A–G 七条路径逐条铺开、逐句核对文案；这里只用**真 MainWindow + 真后台
扫描线程 + 真第三方样本**证明整条纵向链路能跑通，断言保持冒烟级。

样本是 ``mrafatpanah/blf-viewer`` 的 ``two-channel.blf`` / ``two-channel.dbc``
（§6.2 下载入库）：204 帧、2 条总线、7 个报文桶。选它而不是合成样本，正是为了
避免「自产自验」—— 二进制由别的工具产出、DBC 也是别人写的。

QSettings 的 BLF 记忆是**文件级**的（``app_settings`` 只重置单例、不删值），所
以本文件用 autouse 夹具逐条清空 —— 否则第一个用例写下的记忆会让后面的用例
莫名其妙地跳过对话框。
"""

from __future__ import annotations

import copy
import os

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QDialog

from src.core.config import BLF_TEXT_AUTO_USED
from src.core.settings import AppSettings
from src.ui.dialogs.can_db_selector import CanDatabaseSelectorDialog
from src.ui.dialogs.can_unmatched_dialog import CanUnmatchedDialog
from src.ui.drag_drop import build_var_mimedata
from tests.fixtures import bus_data_access as bd
from tests.fixtures.data_factory import CAN_FRAMES, write_blf_pair, write_dbc

# 保活容器：offscreen 下手工构造的 QMimeData 需防 GC（见 tests/README.md）
_keep_alive: list = []

TIMEOUT = 30_000


@pytest.fixture(autouse=True)
def _isolated_blf_memory(app_settings):
    """逐用例清空 BLF 数据库记忆（跨用例泄漏 = 测试顺序敏感）。"""
    app_settings.set_blf_last_db_paths([])
    app_settings.set_blf_auto_use_last(False)
    yield app_settings


@pytest.fixture(scope="module")
def two_channel_pair():
    """真实 BLF + 真实 DBC（未下载则整文件 skip）。"""
    return (bd.require("blfviewer_two-channel.blf"),
            bd.require("blfviewer_two-channel.dbc"))


@pytest.fixture()
def can_dialogs(monkeypatch):
    """两个 CAN 对话框的替身：记录弹出次数，选择框一律 Accepted。

    ``paths`` 由用例设定，等价于用户在框里点了「添加文件…」再按「导入」。
    """
    state = {"selector": 0, "selector_rows": [], "paths": [], "unmatched": [],
             "replace": False}

    def fake_selector_exec(self, *args, **kwargs):
        state["selector"] += 1
        if state["replace"]:
            # 真实用户会先把旧行移除再「添加文件…」：不清行则 first-match-wins
            # 还是旧库赢，测不出换库效果
            for row in list(self._rows):
                self._rows.remove(row)
        if state["paths"]:
            self.add_paths(list(state["paths"]))
        state["selector_rows"].append(list(self.get_selected_paths()))
        return QDialog.DialogCode.Accepted

    def fake_unmatched_exec(self, *args, **kwargs):
        state["unmatched"].append(self)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CanDatabaseSelectorDialog, "exec", fake_selector_exec)
    monkeypatch.setattr(CanUnmatchedDialog, "exec", fake_unmatched_exec)
    return state


def _blf_error(stubs):
    """waitUntil 回调里的快速失败：加载已经炸了就别再干等 30 s。"""
    if stubs["critical"]:
        raise RuntimeError(f"BLF 后台加载失败: {stubs['critical']}")
    if stubs["warning"]:
        raise RuntimeError(f"BLF 后台加载被警告打断: {stubs['warning']}")


def _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc):
    """点「导入数据文件」→（需要时）选库 → 等后台扫描与 post-load 全部落地。"""
    loader_before = mw.loader
    dialog_stubs["open_file"] = (str(blf), "BLF Files (*.blf)")
    can_dialogs["paths"] = [str(dbc)]
    qtbot.mouseClick(mw.load_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: _ready(mw, dialog_stubs, loader_before, expect_path=str(blf)),
        timeout=TIMEOUT,
    )
    qapp.processEvents()


def _ready(mw, stubs, loader_before, expect_path=None):
    _blf_error(stubs)
    loader = mw.loader
    return (
        loader is not None
        and loader is not loader_before
        and loader.LOADER_TYPE == "blf"
        and (expect_path is None
             or os.path.abspath(loader.path) == os.path.abspath(expect_path))
        and not mw._is_loading_new_data
        and mw.loaded_path == expect_path
    )


def _drag_to_plot(mw, qapp, names):
    """把变量从列表拖到第一个绘图区（help.md 3️⃣ 的语义）。"""
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata(list(names))
    _keep_alive.append(mime)
    pw.dropEvent(QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    ))
    qapp.processEvents()
    return pw


# ---------------------------------------------------------------------------
# §5.1 B：首次导入 —— 问一次库，出信号，banner 变绿
# ---------------------------------------------------------------------------

def test_importing_a_real_blf_yields_signals_through_the_ui(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, two_channel_pair
):
    mw = main_window
    blf, dbc = two_channel_pair
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)

    loader = mw.loader
    assert can_dialogs["selector"] == 1, "首次导入必须问一次数据库（§5.1 B）"
    assert can_dialogs["selector_rows"][0] == [str(dbc)]
    assert loader.var_names, "真实配对样本不该 0 信号"
    assert loader.group_count >= 2, "group = (总线通道, CAN ID) 桶，双通道样本应 ≥2"
    assert {b.channel for b in loader._buckets} >= {0, 1}
    # 变量列表与 loader 同步，且单位也带过来了
    assert mw.list_widget.rowCount() == len(loader.var_names)
    assert mw.units["BattVoltage"] == "V"
    assert not dialog_stubs["warning"], "正常导入不该弹任何模态框"


def test_the_can_segment_appears_with_the_source_and_opens_the_drawer(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, two_channel_pair
):
    """§6.2/§6.3：常驻信息在状态栏 CAN 段里，细节与操作在段点开的抽屉里。

    左栏那条 44px 横幅已整体撤走（§6.1）；这里钉住新锚点：非 BLF 时段与
    分隔符一起隐藏、导入后出现，点段弹抽屉且 banner 是其固定头部行。
    """
    mw = main_window
    blf, dbc = two_channel_pair
    # 拉宽窗口：段可见性有宽度阈值（窄窗收纳），这里要的是"常态下可见"
    mw.resize(1400, 800)
    qapp.processEvents()
    assert mw._can_segment.isHidden() is True, "非 BLF 数据源时段整组隐藏"
    assert mw._can_segment_separator.isHidden() is True
    assert mw._can_db_drawer.isHidden() is True, "开机只有一扇隐身抽屉"
    assert mw.can_db_banner.window() is mw._can_db_drawer, "§6.4：banner 归属抽屉"

    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)

    assert mw._can_segment.isHidden() is False
    assert mw._can_segment_separator.isHidden() is False
    # two-channel 的 7 个桶全在 dbc 里有定义 → 状态①（绿）
    assert mw.can_db_banner.state == "normal"
    assert mw._can_segment.text() == (
        f"CAN：1 库 · {mw.loader.database.signal_count} 信号"
    )
    text = mw.can_db_banner.info_label.text()
    assert "1 个数据库" in text and "two-channel.dbc" in text

    # 点段 → 抽屉贴状态栏打开，表里列出刚用的库
    qtbot.mouseClick(mw._can_segment, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    drawer = mw._can_db_drawer
    assert drawer.isHidden() is False
    assert mw.can_db_banner.isHidden() is False, "banner 是抽屉里的固定头部行"
    assert drawer.table.rowCount() == len(mw.loader.database.reports)
    assert drawer.table.item(0, 1).text() == os.path.basename(str(dbc))
    assert drawer.table.item(0, 2).text() == "DBC"
    assert drawer.table.item(0, 5).text() == "✔ 已使用"


def test_the_can_segment_is_folded_when_the_window_is_too_narrow(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, two_channel_pair
):
    """§6.3：Qt 不会替我们收纳段 —— 窄窗下自行隐藏，宽窗下回来。

    判据是显式求和的宽度阈值（始终含 CAN 段自身，避免显隐互相触发抖动），
    所以这里用「最小宽度 + 一条顶到 elide 上限的长播报」制造真实拥挤。
    """
    mw = main_window
    blf, dbc = two_channel_pair
    mw.resize(1400, 800)
    qapp.processEvents()
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)
    assert mw._can_segment.isHidden() is False, "前提：宽窗下段可见"

    mw._broadcast("重载 " + "很长的现场播报文案，" * 12)
    mw.resize(mw.minimumWidth(), 700)
    qapp.processEvents()
    assert mw.minimumWidth() >= 700, "窗口的最小宽度由主窗口自己定"
    assert mw._can_segment.isHidden() is True, "窄窗先丢 CAN 段（不是硬裁文字）"
    assert mw._can_segment_separator.isHidden() is True

    mw.resize(1600, 800)
    qapp.processEvents()
    assert mw._can_segment.isHidden() is False, "宽回来段必须回来"
    assert mw._can_segment.text().startswith("CAN：")


def test_the_choice_is_remembered_for_the_next_blf(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, two_channel_pair
):
    """§5.2「记忆写入时机：导入成功后写」——取消不写，成功必须写。"""
    mw = main_window
    blf, dbc = two_channel_pair
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)

    settings = AppSettings()
    assert settings.get_blf_last_db_paths() == [str(dbc)]
    assert settings.is_blf_auto_use_last() is True, "对话框里「记住…」默认勾选"


# ---------------------------------------------------------------------------
# help.md 3️⃣ / 4️⃣：绘图与游标
# ---------------------------------------------------------------------------

def test_dragging_a_can_signal_onto_the_plot_draws_its_frames(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, two_channel_pair
):
    mw = main_window
    blf, dbc = two_channel_pair
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)
    loader = mw.loader

    # 挑样本最多的那个组：只有几个点的曲线看不出「取的是不是这个桶」
    gi = max(range(loader.group_count),
             key=lambda g: loader.get_group_time_array(g).size)
    name = loader.get_group_variables(gi)[0]
    times = loader.get_group_time_array(gi)

    pw = _drag_to_plot(mw, qapp, [name])
    assert name in pw.curves, f"BLF 信号应能画进图里: {list(pw.curves)}"
    curve = pw.curves[name]
    assert curve.y_data.size == times.size == loader.get_metadata(name).sample_count
    # 时间轴是相对秒且首帧归零（§2 关键语义决策）
    assert curve.x_data.size == times.size
    assert curve.x_data[0] == pytest.approx(float(times[0]), abs=1e-6)
    assert loader.global_time_range[0] == pytest.approx(0.0, abs=1e-6)
    assert not dialog_stubs["critical"]


def test_the_cursor_can_be_toggled_over_a_blf_curve(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, two_channel_pair
):
    """游标是分析主入口，grouped loader 走的是「按组时间轴」分支，必须实测。"""
    mw = main_window
    blf, dbc = two_channel_pair
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)
    name = max(
        (n for g in range(mw.loader.group_count)
         for n in mw.loader.get_group_variables(g)),
        key=lambda n: mw.loader.get_series(n).size,
    )
    pw = _drag_to_plot(mw, qapp, [name])

    mw.activateWindow()
    qapp.processEvents()
    assert not mw.cursor_btn.isChecked()
    qtbot.keyClick(mw, Qt.Key_R, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert mw.cursor_btn.isChecked(), "Ctrl+R 应打开游标"
    assert mw.cursor_btn.text() == "隐藏游标"
    assert len(pw.curves) == 1, "游标开关不该动曲线"
    assert not dialog_stubs["critical"]

    qtbot.keyClick(mw, Qt.Key_R, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert not mw.cursor_btn.isChecked()


# ---------------------------------------------------------------------------
# §5.1 A + §5.4 重载：绝不再弹窗
# ---------------------------------------------------------------------------

def test_reload_reuses_the_database_without_asking_again(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, two_channel_pair
):
    mw = main_window
    blf, dbc = two_channel_pair
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)
    assert can_dialogs["selector"] == 1

    loader_before = mw.loader
    names_before = list(mw.loader.var_names)
    qtbot.mouseClick(mw.reload_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: _ready(mw, dialog_stubs, loader_before, expect_path=str(blf)),
        timeout=TIMEOUT,
    )
    qapp.processEvents()

    assert can_dialogs["selector"] == 1, "「重载」绝不主动弹窗（§5.1 末行）"
    assert mw.loader is not loader_before, "重载应真的重新扫描一遍"
    assert mw.loader.db_paths == [str(dbc)]
    assert list(mw.loader.var_names) == names_before
    assert mw.can_db_banner.isHidden() is False


def test_a_second_import_uses_the_memory_and_says_so_on_the_banner(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, two_channel_pair
):
    """§5.1 A：有记忆且文件都在 → 不弹窗，banner 上说明来源。"""
    mw = main_window
    blf, dbc = two_channel_pair
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)
    assert can_dialogs["selector"] == 1

    loader_before = mw.loader
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)

    assert can_dialogs["selector"] == 1, "A 路径不该弹框"
    assert mw.loader is not loader_before
    assert mw.can_db_banner.state == "normal"
    assert BLF_TEXT_AUTO_USED in mw.can_db_banner.info_label.text()


# ---------------------------------------------------------------------------
# §5.3 状态③ + §5.5 未匹配详情
# ---------------------------------------------------------------------------

def test_unmatched_ids_surface_in_the_banner_and_open_a_detail_list(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, tmp_path
):
    """合成样本自带一个「数据库里没定义」的 ID —— 非阻塞反馈的真实形态。"""
    mw = main_window
    blf, dbc = write_blf_pair(tmp_path, n=30, buses=(0,))
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)

    assert mw.can_db_banner.state == "unmatched"
    assert mw.can_db_banner.view_btn.isHidden() is False
    assert len(can_dialogs["unmatched"]) == 0, "扫描结果不许弹模态（§5.1 原则）"

    mw.can_db_banner.view_btn.click()
    qapp.processEvents()

    assert len(can_dialogs["unmatched"]) == 1, "「查看」应打开 §5.5 详情框"
    dialog = can_dialogs["unmatched"][0]
    assert dialog.table.columnCount() == 4
    assert dialog.table.rowCount() == len(mw.loader.unmatched_ids)
    shown = {dialog.table.item(r, 1).text() for r in range(dialog.table.rowCount())}
    expected = {f"0x{can_id:X}" + ("E" if extended else "")
                for _ch, can_id, extended, _n in mw.loader.unmatched_ids}
    assert shown == expected
    assert "%" in dialog.table.item(0, 3).text(), "占比列必须是百分比"


# ---------------------------------------------------------------------------
# §5.6 重选数据库：换库后的下游收敛必须真的走到控件
# ---------------------------------------------------------------------------

def test_reselecting_pushes_the_new_units_into_the_plot_widgets(
    main_window, qapp, qtbot, dialog_stubs, can_dialogs, tmp_path
):
    """per-widget 的 ``units`` 只有 ``_apply_loader`` 会下发。

    ``reselect_can_database`` 历史上手写过一份精简版收敛逻辑，漏了单位下发、
    数值表重建、绘图变量编辑器/变量信息窗刷新四项（表现是「换了库但图上还是
    老单位」）；现在尾巴交回唯一入口，这条用真 MainWindow 钉住它真的跑到了控件。
    """
    mw = main_window
    blf, dbc = write_blf_pair(tmp_path, n=30, buses=(0,))
    _import(mw, qtbot, qapp, dialog_stubs, can_dialogs, blf, dbc)
    # 拉宽窗口：段可见性有宽度阈值，这里要保证点的到
    mw.resize(1400, 800)
    qapp.processEvents()

    loader = mw.loader
    frames_before = loader.total_messages
    buckets_before = len(loader._buckets)
    name = max(loader.var_names, key=lambda n: loader.get_series(n).size)
    pw = _drag_to_plot(mw, qapp, [name])
    assert pw.curves, "前提：图上先有一条曲线"

    alt_frames = copy.deepcopy(list(CAN_FRAMES))
    for fr in alt_frames:
        for sig in fr["signals"]:
            sig["unit"] = "xalt"
    alt = write_dbc(tmp_path / "alt.dbc", alt_frames)
    can_dialogs["replace"] = True
    can_dialogs["paths"] = [str(alt)]

    # 入口搬进抽屉（§6.5）：点段开抽屉 → 页脚「重选数据库…」
    qtbot.mouseClick(mw._can_segment, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert mw._can_db_drawer.isHidden() is False, "先开抽屉"
    qtbot.mouseClick(mw._can_db_drawer.reselect_btn, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert mw._can_db_drawer.isHidden() is True, "先收抽屉再开模态选库框"

    assert not dialog_stubs["critical"], dialog_stubs["critical"]
    assert mw.loader is loader, "换库只重绑数据库，不该重建 loader"
    assert loader.db_paths == [str(alt)]
    assert loader.total_messages == frames_before, "「无需重新扫描报文」的承诺"
    assert len(loader._buckets) == buckets_before
    assert loader.units[name] == "xalt"
    assert mw.units[name] == "xalt"
    assert pw.units[name] == "xalt", "控件级单位没跟着换 = 图上仍是老单位"
    assert not pw.curves, "§5.6：旧曲线引用的是作废名字，一律清除"
    assert mw.list_widget.rowCount() == len(loader.var_names)
