"""数值变量表 MDF tab 模式改进的 component 测试（offscreen）。

对应 tmp/mdf_table_tab_improvement_plan.md v2 的验收点：
- P0 单例：closeEvent 必须经类名清除 DataTableDialog._instance（旧写法
  ``self._instance = None`` 只创建遮蔽类属性的实例属性 → 跨文件数据串显）
- P1 重置链：clear_all_columns 快照 → update_data 按快照重建（问题②
  「MDF 表残留在新 CSV 数据界面」的核心回归）
- T2 锚点：_nearest_time_row 端点、切 tab 按时间同步、端点钳制不回写锚点、
  locate_time 跳转
- T3 行号：tab 视图行号表头可见且定宽
- T4 定位器：条目与表内变量一致、只定位不添加、MatchContains
- T0 守卫：tab 右键分析自带 df/model、复制透传 tab df（旧版直接崩溃）

替身策略（同 test_table_dialog_blink.py）：直接构造 DataTableDialog 并
实例级替换 _resolve_loader，不走 popup/add_variables，避免污染类级单例。
"""

import time

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import Qt, QItemSelectionModel, QCoreApplication
from PySide6.QtWidgets import QAbstractItemView, QApplication, QMenu

from tests.fixtures.data_factory import write_mdf
from src.data.mdf_lazy_loader import MDFLazyLoader
from src.ui.table_dialog import DataTableDialog, _nearest_time_row
from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget


def pump(ms: int = 50) -> None:
    """驱动事件循环，让 QTimer.singleShot 的延后回调落地。"""
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.002)


@pytest.fixture()
def mdf_loader(qapp, tmp_path):
    """合成 MDF4：G0 Press_G0(bar) 12 点 @0.1s；G1 Press_G1(degC) 6 点 @0.2s。"""
    path = write_mdf(tmp_path / "tab.mf4", version="4.10", n=12)
    return MDFLazyLoader(str(path))


@pytest.fixture()
def tab_dialog(qapp, mdf_loader, monkeypatch):
    monkeypatch.setattr(DataTableDialog, "_instance", None)
    dlg = DataTableDialog()
    dlg._resolve_loader = lambda: mdf_loader
    yield dlg
    dlg.hide()
    dlg.deleteLater()
    pump(20)


class FakeCsvLoader:
    """最小 CSV loader 替身：update_data 非 MDF 分支只用 df/units。"""

    LOADER_TYPE = "csv"
    path = "fake.csv"
    var_names = ()

    def __init__(self, df):
        self.df = df
        self.units = {c: "" for c in df.columns}


# ---------- T2(纯函数) _nearest_time_row：中点/端点钳制 ----------

def test_nearest_time_row():
    t = np.arange(10, dtype=np.float64)
    assert _nearest_time_row(t, 2.4) == 2
    assert _nearest_time_row(t, 2.6) == 3
    assert _nearest_time_row(t, 5.0) == 5  # 精确命中
    assert _nearest_time_row(t, -5.0) == 0  # 下越界钳到首行
    assert _nearest_time_row(t, 99.0) == 9  # 上越界钳到末行


# ---------- P0 单例回归：closeEvent 清除类级 _instance ----------

def test_close_clears_class_singleton(qapp, app_settings):
    monkey_dlg = DataTableDialog()
    DataTableDialog._instance = monkey_dlg
    monkey_dlg._add_variable_to_table("a", pd.Series([1.0, 2.0, 3.0], name="a"))
    DataTableDialog._saved_scroll_pos = 7
    monkey_dlg.set_skip_close_confirmation(True)

    monkey_dlg.close()

    # 旧写法 self._instance = None 只设实例属性，这里必须看到类属性被清空
    assert DataTableDialog._instance is None
    assert DataTableDialog._saved_scroll_pos is None
    monkey_dlg.deleteLater()
    pump(20)


# ---------- T3 + has_table_content：tab 模式判据 ----------

def test_tab_mode_row_header_and_content(tab_dialog, mdf_loader):
    state = tab_dialog._add_variable_to_tab("Press_G0", 0)

    assert state is not None
    assert tab_dialog._tab_mode is True
    assert tab_dialog._df.empty, "tab 模式下 _df 恒空（旧判据据此误判）"
    assert tab_dialog.has_table_content() is True
    # 行号表头：可见 + 定宽 48（PandasTableModel.headerData 提供 1-based 行号）
    vh = state.view.verticalHeader()
    assert not vh.isHidden()
    assert vh.minimumWidth() == 48 and vh.maximumWidth() == 48


# ---------- P1 重置链：clear_all_columns 快照 + update_data 重建 ----------

def test_clear_all_columns_leaves_snapshots(tab_dialog):
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    tab_dialog._add_variable_to_tab("Press_G1", 1)

    tab_dialog.clear_all_columns()

    assert tab_dialog._tab_vars_snapshot == {0: ["Press_G0"], 1: ["Press_G1"]}
    assert set(tab_dialog._table_vars_snapshot) == {"Press_G0", "Press_G1"}
    # tab 状态整体释放
    assert tab_dialog._tab_mode is False
    assert tab_dialog._group_tabs == {}
    assert tab_dialog._tab_widget.count() == 0
    assert tab_dialog._var_locator.count() == 0
    assert not tab_dialog.has_table_content()


def test_update_data_mdf_to_csv_resets_tabs(tab_dialog, monkeypatch):
    """问题②核心回归：MDF tab 表拖入 CSV 重载后不得残留旧 tab。"""
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    tab_dialog.clear_all_columns()  # file_loader_manager 的真实调用顺序

    called = {}

    def _info(parent, title, text, *a, **k):
        called["text"] = text

    monkeypatch.setattr(
        "src.ui.table_dialog.QMessageBox.information", staticmethod(_info)
    )
    # 新 CSV 里没有旧变量 → removed 提示 + 空表自动关闭
    tab_dialog.update_data(FakeCsvLoader(pd.DataFrame({" unrelated": [1, 2, 3] })))

    assert tab_dialog._tab_mode is False
    assert tab_dialog._group_tabs == {}
    assert "Press_G0" in called.get("text", "")
    assert not tab_dialog.has_table_content()
    assert tab_dialog.isVisible() is False, "空表应自动关闭"


def test_update_data_rebuilds_matching_columns(tab_dialog):
    """快照中的变量在新数据里同名存在 → 单表重建并继续显示。"""
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    tab_dialog.clear_all_columns()

    df = pd.DataFrame({"Press_G0": [1.0, 2.0, 3.0], "z": [4.0, 5.0, 6.0]})
    tab_dialog.update_data(FakeCsvLoader(df))

    assert tab_dialog._tab_mode is False
    assert tab_dialog.get_column_names() == ["Press_G0"]
    assert tab_dialog.has_table_content()
    assert tab_dialog.model is not None


# ---------- T4 变量定位器 ----------

def test_var_locator_items_and_activate(tab_dialog):
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    state1 = tab_dialog._add_variable_to_tab("Press_G1", 1)

    combo = tab_dialog._var_locator
    assert combo.count() == 2, "time 列不得入条目"
    datas = {combo.itemData(i) for i in range(combo.count())}
    assert datas == {("Press_G0", 0), ("Press_G1", 1)}
    texts = " | ".join(combo.itemText(i) for i in range(combo.count()))
    assert "[bar]" in texts and "[degC]" in texts  # 单位展示
    # 子串过滤必须 MatchContains（默认 MatchStartsWith 只能前缀匹配）
    assert combo.completer().filterMode() == Qt.MatchFlag.MatchContains

    # 选中 G1 条目：仅切 tab + 定位，不新增变量
    idx = next(i for i in range(combo.count()) if combo.itemData(i)[0] == "Press_G1")
    combo.lineEdit().setText("Press_G1")
    tab_dialog._on_var_locator_activated(idx)

    assert tab_dialog._tab_widget.currentIndex() == state1.widget_index
    assert combo.lineEdit().text() == ""
    assert combo.count() == 2, "定位不得向表里添加变量"


# ---------- T2 时间锚点同步 ----------

def test_tab_switch_syncs_time_anchor(tab_dialog):
    state0 = tab_dialog._add_variable_to_tab("Press_G0", 0)
    state1 = tab_dialog._add_variable_to_tab("Press_G1", 1)
    # 新 tab 会自动切页（已在 state1）：先切回 G0 模拟用户视角，
    # 否则 setCurrentIndex(state1) 不触发 currentChanged
    tab_dialog._tab_widget.setCurrentIndex(state0.widget_index)

    # 模拟用户在 G0（0.1s 步长）滚到第 6 行 → 锚点 0.6s
    tab_dialog._update_time_anchor(state0, 6)
    assert tab_dialog._current_time_anchor == pytest.approx(0.6)

    calls = []
    state1.view.scrollTo = lambda idx, hint: calls.append((idx.row(), hint))
    tab_dialog._tab_widget.setCurrentIndex(state1.widget_index)

    # G1 是 0.2s 步长：0.6s → 第 3 行（行号相同反而错）
    assert calls and calls[-1] == (
        3,
        QAbstractItemView.ScrollHint.PositionAtTop,
    )


def test_tab_switch_out_of_range_does_not_jump(tab_dialog):
    """锚点超出目标表时间轴覆盖 → 不跳端点（旧实现钳到末行，用户视角
    即“莫名跳到表末尾”），保持该表历史位置且锚点不动。"""
    state0 = tab_dialog._add_variable_to_tab("Press_G0", 0)
    state1 = tab_dialog._add_variable_to_tab("Press_G1", 1)
    # 让当前页回到 G0，保证切到 G1 能触发 currentChanged
    tab_dialog._tab_widget.setCurrentIndex(state0.widget_index)

    tab_dialog._current_time_anchor = 100.0  # 超出 G1 [0, 1.0]s
    calls = []
    state1.view.scrollTo = lambda idx, hint: calls.append((idx.row(), hint))
    tab_dialog._tab_widget.setCurrentIndex(state1.widget_index)

    assert calls == [], "越界不得 scrollTo 到端点"
    assert tab_dialog._current_time_anchor == 100.0, "锚点不得被回写"


def test_locate_time(tab_dialog):
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    state1 = tab_dialog._add_variable_to_tab("Press_G1", 1)

    assert tab_dialog.locate_time(999, 0.0) is False, "未知 group 返回 False"
    assert tab_dialog.locate_time(1, 0.9) is True

    pump(80)
    sm = state1.view.selectionModel()
    rows = {idx.row() for idx in sm.selectedIndexes()}
    assert rows == {5}, "0.9s 在 G1(0.2s 步长) 上最近行为末行 5"
    assert tab_dialog._current_time_anchor == pytest.approx(0.9)
    assert tab_dialog._tab_widget.currentIndex() == state1.widget_index


# ---------- T0 右键/复制守卫 ----------

def test_tab_selection_analysis_and_copy(tab_dialog):
    state = tab_dialog._add_variable_to_tab("Press_G0", 0)
    sm = state.view.selectionModel()
    sm.select(
        state.model.index(0, 1), QItemSelectionModel.SelectionFlag.ClearAndSelect
    )
    sm.select(state.model.index(1, 1), QItemSelectionModel.SelectionFlag.Select)
    sm.select(state.model.index(2, 1), QItemSelectionModel.SelectionFlag.Select)

    analysis = tab_dialog._analyze_selection(state.view)

    assert analysis is not None
    assert analysis["can_copy"] is True
    assert analysis["df"] is state.df, "tab 分析结果必须自带 df（透传守卫）"
    assert analysis["model"] is state.model

    tab_dialog._copy_selected_to_clipboard(
        analysis["ordered_cols"], analysis["rows_order"], df=analysis["df"]
    )
    lines = QApplication.clipboard().text().splitlines()
    assert lines[0] == "Press_G0"
    assert lines[1] == "bar"  # 单位行
    assert len(lines) == 5  # 头 + 单位 + 3 数据行
    assert float(lines[2]) == pytest.approx(float(state.df["Press_G0"].iloc[0]))


def test_plot_menu_uses_tab_model(tab_dialog):
    """选 time+变量两列：绘图菜单用 tab 自带 model 取列名（旧版用空单表 model 崩溃）"""
    state = tab_dialog._add_variable_to_tab("Press_G0", 0)
    sm = state.view.selectionModel()
    first = True
    for col in (0, 1):
        for row in (0, 1):
            flag = (
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                if first
                else QItemSelectionModel.SelectionFlag.Select
            )
            sm.select(state.model.index(row, col), flag)
            first = False

    analysis = tab_dialog._analyze_selection(state.view)
    assert analysis["plot_enabled"] is True

    menu = QMenu()
    added = tab_dialog._build_plot_menu(menu, analysis)
    assert added is True
    texts = [a.text() for a in menu.actions()]
    assert any("绘制x/y图" in t and "time" in t for t in texts)
    menu.deleteLater()


# ---------- 入口守卫：数据源变更后不得往旧 tab 里加新数据 ----------

def test_ensure_loader_consistent_resets_stale_tabs(tab_dialog):
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    assert tab_dialog._group_tabs

    # 模拟 loader 已被替换（路径/变量数变化），而入口未走过置路径
    tab_dialog._tab_loader_key = ("mdf", "other.mf4", 999)
    tab_dialog._ensure_loader_consistent()

    assert tab_dialog._group_tabs == {}
    assert tab_dialog._tab_mode is False
    # 一致时不误伤：key 与当前 loader 相符则保留 tab
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    tab_dialog._tab_loader_key = tab_dialog._loader_identity(tab_dialog._resolve_loader())
    tab_dialog._ensure_loader_consistent()
    assert len(tab_dialog._group_tabs) == 1


# ---------- jump_to_data 委托链：plot_widget → dlg.locate_time ----------

def test_jump_to_data_delegates_to_locate_time(tab_dialog, mdf_loader):
    """_jump_to_data_mdf_tab 只做反算+查 group，定位全部委托 locate_time。"""
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    tab_dialog._add_variable_to_tab("Press_G1", 1)

    class FakePlot:
        factor = 1.0
        offset = 0.0

    # 不构造重量级 widget，直接非绑定调用；x=0.55s → G1(0.2s 步长) 最近行 3
    DraggableGraphicsLayoutWidget._jump_to_data_mdf_tab(
        FakePlot(), tab_dialog, ["Press_G1"], 0.55, mdf_loader
    )
    pump(60)

    assert tab_dialog._current_time_anchor == pytest.approx(0.55)
    assert tab_dialog._tab_widget.currentIndex() == tab_dialog._group_tabs[1].widget_index
    sm = tab_dialog._group_tabs[1].view.selectionModel()
    rows = {i.row() for i in sm.selectedIndexes()}
    assert rows == {3}  # t1[3]=0.6 距 0.55 最近

    # factor=0 防护：不得碰锚点
    anchor_before = tab_dialog._current_time_anchor
    FakePlot.factor = 0.0
    DraggableGraphicsLayoutWidget._jump_to_data_mdf_tab(
        FakePlot(), tab_dialog, ["Press_G0"], 1.0, mdf_loader
    )
    assert tab_dialog._current_time_anchor == anchor_before


# ---------- 锚点污染回归（线上反馈：切换后跳表末尾 / 定位后跳 0s） ----------
# 需要真实布局：show + 小窗口让 G0（12 行）的滚动条有非零范围，
# 否则 setValue 被钳到 0、发不出 valueChanged，测试退化成空转。

@pytest.fixture()
def shown_tab_dialog(tab_dialog):
    tab_dialog.resize(420, 200)
    tab_dialog.show()
    pump(150)
    return tab_dialog


def test_anchor_ignores_hidden_tab_scroll(shown_tab_dialog):
    dlg = shown_tab_dialog
    state0 = dlg._add_variable_to_tab("Press_G0", 0)
    state1 = dlg._add_variable_to_tab("Press_G1", 1)
    pump(100)
    # 新 tab 自动切页：当前在 G1，G0 处于隐藏页
    assert dlg._tab_widget.currentIndex() == state1.widget_index

    sb0 = state0.view.verticalScrollBar()
    assert sb0.maximum() > 0, "窗口太小/行太高导致无滚动范围，测试无效"

    before = dlg._current_time_anchor
    sb0.setValue(sb0.maximum())  # 隐藏页 G0 滚到末尾
    pump(100)
    assert dlg._current_time_anchor == before, "隐藏 tab 的滚动不得污染全局锚点"

    # 当前可见 tab 的滚动仍要正常回写（守卫不能一刀切）
    dlg._tab_widget.setCurrentIndex(state0.widget_index)
    pump(100)
    sb0.setValue(3)
    pump(100)
    assert dlg._current_time_anchor == pytest.approx(0.3)


def test_scroll_to_column_preserves_vertical(shown_tab_dialog):
    dlg = shown_tab_dialog
    state0 = dlg._add_variable_to_tab("Press_G0", 0)
    pump(100)

    sb0 = state0.view.verticalScrollBar()
    assert sb0.maximum() > 0
    sb0.setValue(6)
    pump(100)
    before = sb0.value()

    # 旧实现用 index(0, col)+PositionAtCenter 做水平滚动，垂直被连带
    # 拉到第 0 行居中（实测 12345→0）：定位新变量后时刻丢失跳回 0s
    dlg._scroll_tab_to_column(state0, "Press_G0", blink=False)
    pump(100)

    assert abs(sb0.value() - before) <= 1, "水平定位列不得改变垂直位置"


def test_tab_switch_out_of_range_keeps_own_position(shown_tab_dialog):
    """越界切换保持目标表历史位置（真实布局，端到端验证 value 层面行为）。"""
    dlg = shown_tab_dialog
    state0 = dlg._add_variable_to_tab("Press_G0", 0)  # G0 时轴 [0, 1.1]s
    state1 = dlg._add_variable_to_tab("Press_G1", 1)  # 新 tab 自动切页 → 当前在 G1
    pump(100)

    sb0 = state0.view.verticalScrollBar()
    assert sb0.maximum() > 0

    # 用户在 G0 滚到第 4 行后离开 → scroll_pos 链记住 4
    dlg._tab_widget.setCurrentIndex(state0.widget_index)
    pump(100)
    sb0.setValue(4)
    pump(100)
    dlg._tab_widget.setCurrentIndex(state1.widget_index)
    pump(100)

    # 锚点拉到 G0 时间轴覆盖不到的大时刻（模拟长时组里的视线）
    dlg._current_time_anchor = 50.0
    dlg._tab_widget.setCurrentIndex(state0.widget_index)
    pump(100)

    assert sb0.value() == 4, "越界时 G0 应停在历史位置而非跳端点"
    assert dlg._current_time_anchor == 50.0
