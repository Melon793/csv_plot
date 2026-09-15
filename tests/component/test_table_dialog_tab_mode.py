"""变量数值表 MDF tab 模式改进的 component 测试（offscreen）。

对应 tmp/mdf_table_tab_improvement_plan.md v2 的验收点：
- P0 单例：closeEvent 必须经类名清除 DataTableDialog._instance（旧写法
  ``self._instance = None`` 只创建遮蔽类属性的实例属性 → 跨文件数据串显）
- P1 重置链：clear_all_columns 快照 → update_data 按快照重建（问题②
  「MDF 表残留在新 CSV 数据界面」的核心回归）
- T2 锚点：_nearest_time_row 端点、切 tab 按时间同步、端点钳制不回写锚点、
  locate_time 跳转
- T3 行号：tab 视图行号表头可见且定宽
- T4 定位器：条目与表内变量一致、只定位不添加、MatchContains
- T0 右键/复制守卫：tab 右键分析自带 df/model、复制透传 tab df（旧版直接崩溃）
- 审查后续项：R7 表头右键菜单（删除列/置灰反馈）、R8 tab 不钉住 loader 时间缓存、
  R10 嵌套程序化滚动的锚点抑制、R11 短缺数据补空值而非伪造平线、
  locate_time/jump_to_data 按真正入表的变量定位并选中目标列

替身策略（同 test_table_dialog_blink.py）：直接构造 DataTableDialog 并
实例级替换 _resolve_loader，不走 popup/add_variables，避免污染类级单例。
"""

import time

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import Qt, QItemSelectionModel, QCoreApplication, QPoint
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QMenu,
    QMessageBox,
    QTableView,
)

from tests.fixtures.data_factory import write_mdf
from src.data.mdf_lazy_loader import MDFLazyLoader
from src.ui.file_loader_manager import FileLoaderManager
from src.ui.table_dialog import (
    DataTableDialog,
    PandasTableModel,
    _nearest_time_row,
    _state_time_array,
)
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


class SwappedGroupLoader:
    """代理 loader，只把 get_var_group_index 的 group 归属按 mapping 改写。

    用于模拟“换了一个 MDF，同名通道落在另一个 channel group”——此时旧
    快照里的 group 索引在新文件里语义已变，重建必须按名反查。
    """

    def __init__(self, inner, mapping):
        self.__dict__["_inner"] = inner
        self.__dict__["_mapping"] = mapping

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def get_var_group_index(self, display_name):
        return self._mapping.get(
            self._inner.get_var_group_index(display_name),
            self._inner.get_var_group_index(display_name),
        )


# ---------- 审查回归 R1（P0）：重建后自关闭不得让调用方解引用 None ----------

def test_refresh_table_dialog_survives_self_close(qapp, app_settings, monkeypatch):
    """变量数值表开着 → 重载一个变量名完全不重叠的文件。

    update_data 会因表空自行 close()，closeEvent 经类名把 _instance 置
    None；旧版调用方继续解引用 DataTableDialog._instance → AttributeError
    中断 _post_load_actions 后续步骤。走真实调用链 FileLoaderManager。
    """
    monkeypatch.setattr(
        "src.ui.table_dialog.QMessageBox.information", staticmethod(lambda *a, **k: None)
    )
    dlg = DataTableDialog()
    monkeypatch.setattr(DataTableDialog, "_instance", None)
    DataTableDialog._instance = dlg
    dlg._add_variable_to_table("a", pd.Series([1.0, 2.0, 3.0], name="a"))
    dlg.show()
    pump(60)
    assert DataTableDialog._instance is dlg

    # file_loader_manager 的真实顺序：_release_old_data 先快照再清空
    dlg.clear_all_columns()
    mgr = FileLoaderManager.__new__(FileLoaderManager)  # 不跑 __init__
    mgr._refresh_table_dialog(FakeCsvLoader(pd.DataFrame({"other": [1, 2, 3]})))

    assert DataTableDialog._instance is None, "表空应已自关"
    assert dlg.isVisible() is False
    assert not dlg.has_table_content()
    dlg.deleteLater()
    pump(20)


# ---------- 审查回归 R2（P1）：group 归属以变量名为准 ----------

def test_add_variable_to_tab_regroups_by_name(tab_dialog, mdf_loader, monkeypatch):
    """传错的 group_index 必须按名归位，不得塞进别人组的时间轴。"""
    warns = []
    monkeypatch.setattr(
        "src.ui.table_dialog.logger.warning",
        lambda msg, *a, **k: warns.append(msg % a if a else msg),
    )

    # Press_G1 实属 G1（6 点 @0.2s），却请求加入 G0（12 点 @0.1s）
    state = tab_dialog._add_variable_to_tab("Press_G1", 0)

    assert state is not None
    assert set(tab_dialog._group_tabs) == {1}, "tab 必须按名归位到 G1"
    assert len(state.df) == len(mdf_loader.get_series("Press_G1")), "不得被 G0 时轴裁剪/填充"
    assert any("按名归位" in w for w in warns), "应留下告警"


def test_update_data_follows_new_file_group(tab_dialog, mdf_loader, monkeypatch):
    """换文件后重建不得沿用旧 group 索引（快照键）。"""
    tab_dialog._add_variable_to_tab("Press_G1", 1)
    tab_dialog.clear_all_columns()
    assert tab_dialog._tab_vars_snapshot == {1: ["Press_G1"]}, "快照基于旧文件"

    swapped = SwappedGroupLoader(mdf_loader, {1: 0})
    tab_dialog._resolve_loader = lambda: swapped
    tab_dialog.update_data(swapped)

    assert set(tab_dialog._group_tabs) == {0}, "新文件里 Press_G1 属 G0"
    state = tab_dialog._group_tabs[0]
    assert len(state.df) == len(mdf_loader.get_group_time_array(0)) == 12
    assert "Press_G1" in state.df.columns


# ---------- 审查回归 R5（P2）：空 group 的 0 行 tab 不算内容 ----------

def test_empty_group_tab_is_not_content(qapp, app_settings, tmp_path, monkeypatch):
    monkeypatch.setattr(DataTableDialog, "_instance", None)
    path = write_mdf(
        tmp_path / "empty_grp.mf4",
        version="4.10",
        n=12,
        with_single_shot_group=True,
        with_empty_group=True,
    )
    loader = MDFLazyLoader(str(path))
    dlg = DataTableDialog()
    dlg._resolve_loader = lambda: loader

    gi = loader.get_var_group_index("EmptyCh")
    state = dlg._add_variable_to_tab("EmptyCh", gi)

    assert state is not None and len(state.df) == 0, "空组 tab 0 行"
    assert dlg.has_table_content() is False, "0 行空 tab 不得误判为“有内容”"
    loader.close()
    dlg.deleteLater()
    pump(20)


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
    if sb0.maximum() <= 0:
        pytest.skip("窗口字体/行高导致无滚动范围，本用例环境下无效")

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
    if sb0.maximum() <= 0:
        pytest.skip("窗口字体/行高导致无滚动范围，本用例环境下无效")
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
    if sb0.maximum() <= 0:
        pytest.skip("窗口字体/行高导致无滚动范围，本用例环境下无效")

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


# ---------- 审查回归 R3（P1）：新建 tab 就要对齐时间锚点 ----------

def test_new_tab_aligns_anchor_on_creation(shown_tab_dialog):
    """当前 tab 滚到某时刻 → 首次添加另一个 group 的变量→新建 tab 必须
    直接落在同一时刻，而非停在表头（旧版 state 注册晚于 setCurrentIndex，
    _on_tab_switched 反查为 None 直接 return）。
    """
    dlg = shown_tab_dialog
    state0 = dlg._add_variable_to_tab("Press_G0", 0)  # 0.1s 步长
    pump(100)
    sb0 = state0.view.verticalScrollBar()
    if sb0.maximum() <= 0:
        pytest.skip("窗口字体/行高导致无滚动范围，本用例环境下无效")

    sb0.setValue(4)  # 锚点 0.4s（G1 时轴 [0,1.0] 覆盖得到）
    pump(100)
    assert dlg._current_time_anchor == pytest.approx(0.4)

    state1 = dlg._add_variable_to_tab("Press_G1", 1)  # 新建 tab（自动切页）
    pump(100)

    assert dlg._tab_widget.currentIndex() == state1.widget_index
    first = state1.view.indexAt(QPoint(0, 0))
    assert first.isValid()
    assert float(state1.df["time"].iloc[first.row()]) == pytest.approx(0.4), (
        "新 tab 首可见行时刻应等于当前锚点"
    )


# ---------- R7：表头右键菜单（删除列 / 置灰反馈 / 清空） ----------

class _NoExecMenu(QMenu):
    """替身菜单：记录构造出的实例，按 pick 序号直接返回某个动作。

    QMenu.exec 是阻塞模态调用，offscreen 下会挂住测试；而 PySide6 里在
    基类上 setattr("exec") 不生效（实例查找绕过类字典），只能把
    src.ui.table_dialog.QMenu 换成这个重写了 exec 的子类。
    """

    menus: list[QMenu] = []
    pick: int | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        _NoExecMenu.menus.append(self)

    def exec(self, *args, **kwargs):
        acts = self.actions()
        return None if _NoExecMenu.pick is None else acts[_NoExecMenu.pick]


@pytest.fixture()
def menu_stub(monkeypatch):
    _NoExecMenu.menus.clear()
    _NoExecMenu.pick = None
    monkeypatch.setattr("src.ui.table_dialog.QMenu", _NoExecMenu)
    yield _NoExecMenu


def _header_hit(view, logical_col: int) -> QPoint:
    """取该逻辑列表头在 header 坐标系里的一个命中点（供菜单入口直接调用）。"""
    header = view.horizontalHeader()
    return QPoint(header.sectionViewportPosition(logical_col) + 2, header.height() // 2)


def test_tab_header_context_menu_is_wired(tab_dialog):
    """tab 视图表头必须挂上右键菜单：删除列/复制变量名/清空列表以只单表可达。"""
    state = tab_dialog._add_variable_to_tab("Press_G0", 0)
    assert (
        state.view.horizontalHeader().contextMenuPolicy()
        == Qt.ContextMenuPolicy.CustomContextMenu
    )


def test_tab_header_menu_items_and_enabled(tab_dialog, menu_stub):
    """变量列：删除可用、冻结置灰（不再静默无效）；time 列：删除置灰。"""
    state = tab_dialog._add_variable_to_tab("Press_G0", 0)

    tab_dialog._on_header_right_click(_header_hit(state.view, 1), state.view)
    acts = menu_stub.menus[-1].actions()
    assert [a.text() for a in acts] == [
        '删除列 "Press_G0"',
        "冻结列（tab 模式不支持）",
        "复制变量名",
        "清空列表",
    ]
    assert acts[0].isEnabled() and not acts[1].isEnabled()

    tab_dialog._on_header_right_click(_header_hit(state.view, 0), state.view)
    acts = menu_stub.menus[-1].actions()
    assert acts[0].text() == '删除列 "time"' and not acts[0].isEnabled()


def test_tab_header_menu_deletes_column(shown_tab_dialog, menu_stub):
    """菜单项选中删除后：只影响本 tab 的列与定位框条目，时间列仍在。"""
    dlg = shown_tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("State", 0)
    assert list(state.df.columns) == ["time", "Press_G0", "State"]

    menu_stub.pick = 0  # 选中第一项：删除列
    dlg._on_header_right_click(_header_hit(state.view, 2), state.view)

    assert list(state.df.columns) == ["time", "Press_G0"]
    assert state.model.columnCount() == 2
    assert dlg.get_column_names() == ["Press_G0"]
    assert dlg._var_locator.count() == 1


def test_removing_last_var_closes_tab_and_remaps_index(tab_dialog):
    """删到该组无变量列 → 整页移除；后续页的 widget_index 必须重映射。"""
    dlg = tab_dialog
    state0 = dlg._add_variable_to_tab("Press_G0", 0)
    state1 = dlg._add_variable_to_tab("Press_G1", 1)
    assert (state0.widget_index, state1.widget_index) == (0, 1)

    dlg._remove_tab_column(state0, "Press_G0")

    assert dlg._tab_widget.count() == 1
    assert list(dlg._group_tabs) == [1]
    assert state1.widget_index == 0, "页序前移后未重映射 → 反查会拿到别的 state"
    assert dlg._group_state_by_widget_index(0) is state1
    assert dlg._tab_widget.widget(0) is state1.view
    assert dlg.get_column_names() == ["Press_G1"]


def test_removing_final_tab_falls_back_to_single_table_ui(tab_dialog):
    """最后一个 tab 被删空：退回单表 UI，不留一个空标签栏。"""
    dlg = tab_dialog
    state0 = dlg._add_variable_to_tab("Press_G0", 0)

    dlg._remove_tab_column(state0, "Press_G0")

    assert dlg._tab_mode is False and dlg._group_tabs == {}
    assert dlg.has_table_content() is False
    assert not dlg.splitter.isHidden()  # 单表视图重新可见（对话框本身未 show）


def test_clear_all_columns_in_tab_mode(tab_dialog, monkeypatch):
    """tab 模式“清空列表”：旧写法用 self.model（单表模为 None）直接 AttributeError。"""
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    dlg = tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)

    dlg._clear_all_columns()

    assert dlg._group_tabs == {} and dlg._tab_mode is False
    assert dlg._table_vars_snapshot == [] and dlg._tab_vars_snapshot == {}
    assert dlg.has_table_content() is False


# ---------- R19：删空退回单表时的空态观感 ----------

def test_reset_to_single_table_shows_empty_table_not_blank_panels(tab_dialog):
    """删空最后一个 tab 退回单表：必须是一个空表，而不是左右两块白板。

    进 tab 模式时单表模型被置 None，退回时若不补回空模型，_update_views 会直接
    return → 两个视图都没内容、frozen_view 也得不到隐藏（线上截图里就是两块白板）。
    """
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)

    dlg._remove_tab_column(state, "Press_G0")

    assert dlg.model is not None, "退回单表后必须补回空模型"
    assert dlg.model.columnCount() == 0 and dlg.model.rowCount() == 0
    assert dlg.main_view.model() is dlg.model
    assert dlg.frozen_view.model() is dlg.model
    assert dlg.frozen_view.isHidden(), "无冻结列时 frozen_view 必须隐藏"
    assert not dlg.main_view.isHidden()


def test_reset_to_single_table_keeps_existing_model(tab_dialog):
    """单表原本有内容时退回（_switch_to_group_mode 路径）不得被空模型覆盖。"""
    dlg = tab_dialog
    df = pd.DataFrame({"Press_G0": np.arange(12.0)})
    dlg._df = df
    dlg.model = PandasTableModel(df, dlg.units)
    dlg.main_view.setModel(dlg.model)
    dlg.frozen_view.setModel(dlg.model)

    dlg._reset_tab_mode()

    assert dlg.model is not None
    assert dlg.model.rowCount() == 12, "退回不得抹掉已有单表内容"
    assert dlg.main_view.model() is dlg.model


# ---------- R8：tab 不得钉住 loader 的时间轴缓存 ----------

def test_tab_does_not_pin_loader_time_cache(tab_dialog, mdf_loader):
    """时间轴必须只存于 tab 自己的 df：另存一份缓存数组本体会让 LRU 形同虚设。"""
    dlg = tab_dialog
    t_cached = mdf_loader.get_group_time_array(0)
    state = dlg._add_variable_to_tab("Press_G0", 0)

    assert not hasattr(state, "time_array"), "state 不得另存时间轴副本"
    assert not np.shares_memory(state.df["time"].to_numpy(), t_cached)

    mdf_loader._time_cache.clear()  # 模拟 LRU 逐出该 group

    assert np.array_equal(_state_time_array(state), t_cached)


# ---------- R10：嵌套程序化滚动的锚点抑制 ----------

def test_nested_programmatic_scroll_keeps_anchor_suppression(tab_dialog):
    """守卫必须是深度计数：内层 finally 把 bool 复位会提前解除外层抑制。"""
    dlg = tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    state1 = dlg._add_variable_to_tab("Press_G1", 1)  # 0.2s 步长，当前页

    dlg._tab_sync_depth = 1  # 模拟外层正在程序化滚动
    try:
        dlg._on_tab_switched(1)
        assert dlg._tab_sync_depth == 1, "内层不得把外层抑制清零"
        dlg._update_time_anchor(state1, 3)
        assert dlg._current_time_anchor is None, "抑制期内不得回写锚点"
    finally:
        dlg._tab_sync_depth = 0

    dlg._update_time_anchor(state1, 3)
    assert dlg._current_time_anchor == pytest.approx(0.6)


# ---------- R11：短缺数据补空值，不伪造平线 ----------

def test_align_len_pads_missing_tail_with_empty_values():
    """旧写法用尾值填充，会在数据缺失区间伪造一条平线且长度看起来正常。"""
    f = DataTableDialog._align_len

    out = f(np.array([1.0, 2.0]), 5)
    assert out[0] == 1.0 and out[1] == 2.0
    assert np.isnan(out[2:]).all()

    ints = f(np.array([1, 2], dtype=np.int64), 4)
    assert list(ints[:2]) == [1, 2]
    assert ints[2] is None and ints[3] is None  # 整型无法承载 NaN

    assert list(f(np.array([1.0, 2.0, 3.0]), 2)) == [1.0, 2.0]  # 过长仍截断


def test_short_series_shows_empty_cells(tab_dialog, mdf_loader, monkeypatch):
    """整列变短时表里必须是空格子（模型渲染为 ""），不是重复的末值。"""
    dlg = tab_dialog
    real = mdf_loader.get_series("Press_G0")
    short = real.iloc[: len(real) // 2]
    monkeypatch.setattr(mdf_loader, "get_series", lambda name: short)

    state = dlg._add_variable_to_tab("Press_G0", 0)
    vals = state.df["Press_G0"].to_numpy()

    assert len(vals) == len(real) == 12
    assert not pd.isna(vals[: len(short)]).any()
    assert pd.isna(vals[len(short):]).all()
    assert (
        state.model.data(state.model.index(11, 1), Qt.ItemDataRole.DisplayRole) == ""
    )


# ---------- 定位目标列：jump_to_data / locate_time ----------

def test_jump_to_data_targets_first_var_present_in_table(tab_dialog, mdf_loader, monkeypatch):
    """曲线名全集里被跳过的那个会让 var_names[0] 在表里查不到（旧版 KeyError）。"""
    dlg = tab_dialog
    dlg._add_variable_to_tab("Press_G1", 1)
    calls = []
    monkeypatch.setattr(
        dlg,
        "locate_time",
        lambda gi, t, var_name=None: calls.append((gi, t, var_name)),
    )

    class FakePlot:
        factor = 1.0
        offset = 0.0

    DraggableGraphicsLayoutWidget._jump_to_data_mdf_tab(
        FakePlot(), dlg, ["NoSuchCurve", "Press_G1"], 0.55, mdf_loader
    )
    assert calls == [(1, 0.55, "Press_G1")]

    calls.clear()
    DraggableGraphicsLayoutWidget._jump_to_data_mdf_tab(
        FakePlot(), dlg, ["NoSuchCurve"], 0.55, mdf_loader
    )
    assert calls == [], "没有变量入表时不得盲目跳转"


def test_locate_time_selects_target_variable_column(tab_dialog):
    """locate_time 传入 var_name 时选中该变量列，而非总落在第 0 列 time。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("State", 0)

    assert dlg.locate_time(0, 0.5, var_name="State") is True
    pump(80)
    sel = {
        (i.row(), i.column()) for i in state.view.selectionModel().selectedIndexes()
    }
    assert sel == {(5, 2)}

    assert dlg.locate_time(0, 0.5) is True
    pump(80)
    sel = {
        (i.row(), i.column()) for i in state.view.selectionModel().selectedIndexes()
    }
    assert sel == {(5, 0)}, "未指定变量时回退 time 列"

    assert dlg.locate_time(0, 0.5, var_name="NotInThisTab") is True
    pump(80)
    sel = {
        (i.row(), i.column()) for i in state.view.selectionModel().selectedIndexes()
    }
    assert sel == {(5, 0)}, "变量不在本 tab 时回退 time 列"


# ---------- R18：行号列宽度跟着行位数走（线上截图发现） ----------

def test_row_header_width_grows_with_row_count(qapp):
    """行号表头右对齐，宽度不足被裁的是高位数字（固定 48px 只容 5 位）。"""
    font = QTableView().font()
    fm = QFontMetrics(font)

    narrow = DataTableDialog._row_header_width(999, font)
    wide = DataTableDialog._row_header_width(1_000_000, font)

    assert narrow == 48, "小表维持原有 48px 观感，不得变宽"
    assert wide > narrow
    # 断言相对关系而非绝对像素：offscreen 与真机字体不同
    assert wide >= fm.horizontalAdvance("999999") + 2, "6 位行号不得被裁切"
    assert DataTableDialog._row_header_width(0, font) == 48, "空表不得算出 0 位"


def test_tab_row_header_width_matches_row_count(tab_dialog):
    """建 tab 时必须按该 group 的行数定宽，而不是写死 48px。"""
    state = tab_dialog._add_variable_to_tab("Press_G0", 0)
    vh = state.view.verticalHeader()
    expect = DataTableDialog._row_header_width(len(state.df), state.view.font())

    assert vh.minimumWidth() == vh.maximumWidth() == expect
    assert vh.width() >= QFontMetrics(state.view.font()).horizontalAdvance(
        str(len(state.df))
    ), "行号文本宽度必须小于列宽"


def test_tab_row_header_widens_for_six_digit_rows(tab_dialog, mdf_loader, monkeypatch):
    """6 位行号的长时程文件：行号列必须自动变宽（旧写法锁 48px 会裁掉高位）。"""
    n = 120_000
    monkeypatch.setattr(
        mdf_loader, "get_group_time_array", lambda gi: np.arange(n) * 0.01
    )
    monkeypatch.setattr(mdf_loader, "get_series", lambda name: pd.Series(np.zeros(n)))

    state = tab_dialog._add_variable_to_tab("Press_G0", 0)
    vh = state.view.verticalHeader()

    assert len(state.df) == n
    assert vh.maximumWidth() > 48, "仍被锁在 48px → 高位数字会被裁"
    assert vh.maximumWidth() >= QFontMetrics(state.view.font()).horizontalAdvance(
        str(n)
    ) + 2


# ---------- tab 标签右键菜单：批量添加本组变量 / 关闭标签页 ----------

def _tab_bar_hit(dlg, widget_index: int) -> QPoint:
    """取第 widget_index 个标签在 tabBar 坐标系里的中心（需 show 后才有有效 rect）。"""
    return dlg._tab_widget.tabBar().tabRect(widget_index).center()


def _tab_menu_texts(menu_stub) -> list[str]:
    """最近一次弹出的标签菜单里的非分隔项文本。"""
    return [a.text() for a in menu_stub.menus[-1].actions() if not a.isSeparator()]


def _tab_menu_pick(menu_stub, prefix: str) -> int:
    """按文本前缀在菜单里找动作序号（有 separator 占位，硬编序号太脆）。"""
    acts = menu_stub.menus[-1].actions()
    for i, a in enumerate(acts):
        if a.text().startswith(prefix):
            return i
    raise AssertionError(f"菜单缺少以 {prefix} 开头的项：{[a.text() for a in acts]}")


def _stub_answer(monkeypatch, answer) -> list:
    """把确认框固定答成 answer，并记下 (标题, 正文) 便于断言文案。"""
    asked: list = []

    def fake(*args, **kwargs):
        asked.append((args[1], args[2]))
        return answer

    monkeypatch.setattr(QMessageBox, "question", fake)
    # 避免取数失败/超上限分支弹出模态信息框挂住 offscreen 测试
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    return asked


def test_tab_bar_context_menu_is_wired(shown_tab_dialog, menu_stub):
    """标签栏必须真挂上 CustomContextMenu。

    其余菜单用例都是直接调 handler，只改接线不改 handler 时它们全绿；
    而“右键没菜单”正是用户会看见的唯一故障现象，必须单独钉住。
    """
    dlg = shown_tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    bar = dlg._tab_widget.tabBar()
    pump(50)

    assert bar.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    assert bar.tabAt(_tab_bar_hit(dlg, 0)) == state.widget_index

    bar.customContextMenuRequested.emit(_tab_bar_hit(dlg, 0))
    pump(50)
    assert menu_stub.menus, "信号未接通 handler → 右键不会弹菜单"


def test_tab_bar_context_menu_items(shown_tab_dialog, menu_stub):
    """菜单四项的文本必须带准确数量（构建菜单时就算，不靠事后“猜”）。"""
    dlg = shown_tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("Press_G1", 1)
    pump(50)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert _tab_menu_texts(menu_stub) == [
        "添加本组其余变量（+2 列）",  # G0 共 3 个变量，已加 1 个
        "关闭此标签页（移除 1 个变量）",
        "关闭其他标签页（1 页 / 1 个变量）",
        "关闭所有标签页（2 页 / 2 个变量）",
    ]
    acts = menu_stub.menus[-1].actions()
    assert acts[2].isSeparator(), "“添加”与“关闭”两组之间必须有分隔线"
    assert all(a.isEnabled() for a in acts if not a.isSeparator())
    # tooltip 必须同时给出页号与组名：标签上只写 G0，组名只存在于 tab tooltip
    tip = acts[0].toolTip()
    assert "G0" in tip and "尚未添加的 2 个变量加入本页" in tip, tip


def test_add_group_item_greys_out_when_loader_lacks_group_api(
    shown_tab_dialog, menu_stub
):
    """CSV/Excel 的 loader 没有 get_group_variables → 置灰并说明原因。

    这走的是 _pending_group_var_count 的 -1 分支：不能错报成“+0 列”，也不能“点了
    没反应”：静默无效是这类菜单最糟糕的失败模式。
    """
    dlg = shown_tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    dlg._resolve_loader = lambda: FakeCsvLoader(state.df)
    pump(50)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    add_act = next(a for a in menu_stub.menus[-1].actions() if "本组其余变量" in a.text())
    assert add_act.text() == "添加本组其余变量（数据源不可用）"
    assert not add_act.isEnabled()
    assert dlg._pending_group_var_count(state) == -1
    # 关闭系项与数据源无关，必须照常可用
    assert "关闭此标签页（移除 1 个变量）" in _tab_menu_texts(menu_stub)


def test_add_group_remaining_from_tab_menu_adds_all_missing_columns(
    shown_tab_dialog, mdf_loader, menu_stub, monkeypatch
):
    """从菜单项进批量添加的完整派发路径（其余用例都是直调 method）。

    handler 里“选中 act_add_group → _add_group_remaining_variables”这一行分发
    写错（归错动作、或 DRY 下用 is 比较包装对象失配）时，只有从菜单入口才能发现。
    """
    dlg = shown_tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    pump(50)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    menu_stub.pick = _tab_menu_pick(menu_stub, "添加本组其余变量")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert set(state.df.columns) == {"time"} | set(mdf_loader.get_group_variables(0))
    assert dlg._var_locator.count() == 3


def test_single_tab_menu_hides_close_others(shown_tab_dialog, menu_stub):
    """只有一个 tab 时，“关闭其他/关闭所有”与“关闭此页”重复 → 不列出。"""
    dlg = shown_tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    pump(50)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert _tab_menu_texts(menu_stub) == [
        "添加本组其余变量（+2 列）",
        "关闭此标签页（移除 1 个变量）",
    ]


def test_tab_bar_blank_area_right_click_shows_no_menu(shown_tab_dialog, menu_stub):
    """右键落在标签右侧空白：不得弹菜单（宁可不响应，也不能对错误的 tab 动手）。"""
    dlg = shown_tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    pump(50)

    dlg._on_tab_bar_right_click(QPoint(10000, 2))

    assert menu_stub.menus == []


def test_no_tab_menu_while_single_table(tab_dialog, menu_stub):
    """单表期：容器被 setVisible(False) 收起、一个 tab 也没有 → 右键不弹菜单。

    “单表天然无此菜单”的判据不是 _tab_widget 为 None（它已在 __init__ 里建好），
    而是根本没有页可命中：tabAt 返回 -1 → 反查不到 state → 直接 return。
    """
    dlg = tab_dialog
    assert dlg._tab_container_widget.isHidden()
    assert dlg._tab_widget.count() == 0 and dlg._group_tabs == {}

    dlg._on_tab_bar_right_click(QPoint(5, 5))

    assert menu_stub.menus == []


def test_add_group_remaining_adds_all_missing_columns(tab_dialog, mdf_loader):
    """一次加完整组：列集合对齐 loader 的组内变量，行数不变。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    rows = len(state.df)

    dlg._add_group_remaining_variables(state)

    assert set(state.df.columns) == {"time"} | set(mdf_loader.get_group_variables(0))
    assert len(state.df) == rows, "行数必须仍等于原时间轴长度，不得因列长度不一被拉伸"
    assert state.model.columnCount() == len(state.df.columns)
    assert dlg._var_locator.count() == 3


def test_add_group_remaining_is_idempotent_and_greys_out(shown_tab_dialog, menu_stub):
    """连点两次列数不变；已全加时菜单项变“已全部添加”并置灰。"""
    dlg = shown_tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_group_remaining_variables(state)
    cols = list(state.df.columns)

    dlg._add_group_remaining_variables(state)
    assert list(state.df.columns) == cols

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    first = menu_stub.menus[-1].actions()[0]
    assert first.text() == "本组变量已全部添加" and not first.isEnabled()


def test_add_group_remaining_rebuilds_model_once(tab_dialog, monkeypatch):
    """性能护栏：整批只重建一次 model。退化回逐列 setModel 时此条先红。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    real = PandasTableModel
    built: list = []

    class CountingModel(real):
        def __init__(self, *args, **kwargs):
            built.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("src.ui.table_dialog.PandasTableModel", CountingModel)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    dlg._add_group_remaining_variables(state)  # 本次要补 2 列

    assert len(built) == 1, f"整批只应重建一次模型，实际 {len(built)} 次"
    assert state.view.model() is state.model


def test_add_group_remaining_skips_unreadable_channel(tab_dialog, mdf_loader, monkeypatch):
    """单通道取数失败只跳过它，不能毁掉整批；失败数合并成一条提示。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    real_get_series = mdf_loader.get_series

    def boom(name):
        if name == "State":
            raise KeyError(name)
        return real_get_series(name)

    info: list = []
    monkeypatch.setattr(mdf_loader, "get_series", boom)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info.append(a[2]))

    dlg._add_group_remaining_variables(state)

    assert "State" not in state.df.columns
    assert "Label" in state.df.columns, "一个通道失败不得影响其余"
    assert len(info) == 1 and "1 个取数失败" in info[0]
    assert state.model.columnCount() == len(state.df.columns)


def test_add_group_remaining_asks_confirmation_above_threshold(tab_dialog, monkeypatch):
    """超静默阈值 → 先确认；答 No 则一列不加。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    asked = _stub_answer(monkeypatch, QMessageBox.StandardButton.No)
    monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_SILENT_COLS", 1)

    dlg._add_group_remaining_variables(state)

    assert len(asked) == 1 and asked[0][0] == "添加本组其余变量"
    assert "尚未添加" in asked[0][1]
    assert list(state.df.columns) == ["time", "Press_G0"], "取消后不得有副作用"


def test_add_group_remaining_refuses_above_hard_cap(tab_dialog, monkeypatch):
    """超硬上限：一个都不加、也不走确认框，只提示改用逐个拖拽。

    这是分级保护的最外层（计划设计决定 6）：误点一个 5000 列的组不应该
    变成“确认一下然后卡死 3 分钟”，而是直接拒绝并给出替代路径。
    """
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    msgs: list = []
    monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_COLS", 1)

    def no_question(*args, **kwargs):
        raise AssertionError("超上限应直接拒绝，不该再走确认框")

    monkeypatch.setattr(QMessageBox, "question", no_question)
    monkeypatch.setattr(
        QMessageBox, "information", lambda *a, **k: msgs.append((a[1], a[2]))
    )

    dlg._add_group_remaining_variables(state)

    assert list(state.df.columns) == ["time", "Press_G0"], "超上限不得有任何副作用"
    assert msgs and msgs[0][0] == "本组变量过多"
    assert "拖拽" in msgs[0][1], "必须告诉用户改用逐个添加"


def test_bulk_add_time_estimate_comes_from_measured_probe(tab_dialog, monkeypatch):
    """确认框里的耗时预估必须先实测单列取数，不能用固定常数。

    实测同一文件里单列成本相差 15 倍（G310 32 ms/列 vs G158 2.2 ms/列），
    拿常数去乘会把 0.4 秒的操作说成 7 秒，不准的读数会让人忽略所有提醒。
    """
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    probed: list = []
    monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_SILENT_COLS", 1)
    monkeypatch.setattr(
        DataTableDialog,
        "_probe_series_seconds",
        staticmethod(lambda loader, name: probed.append(name) or 0.0005),
    )
    asked = _stub_answer(monkeypatch, QMessageBox.StandardButton.Yes)

    dlg._add_group_remaining_variables(state)

    assert probed == ["State"], "预估必须先试取第一列"
    assert "不到 1 秒" in asked[0][1], "不足 1 秒不得写成“约 0 秒”"


def test_add_group_remaining_confirmation_yes_adds_all(tab_dialog, mdf_loader, monkeypatch):
    """同一阈值下答 Yes → 全组入列。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    _stub_answer(monkeypatch, QMessageBox.StandardButton.Yes)
    monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_SILENT_COLS", 1)

    dlg._add_group_remaining_variables(state)

    assert set(state.df.columns) == {"time"} | set(mdf_loader.get_group_variables(0))


def test_add_group_remaining_cancel_keeps_added_columns(tab_dialog, monkeypatch):
    """进度框取消：保留已插入的列、照常重建一次模型，不整批回滚。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_SILENT_COLS", 1)
    monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_PROGRESS_COLS", 0)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    class _FakeProgress:
        """够用的 QProgressDialog 替身：第一次 setValue 之后就算被取消。"""

        instances: list = []

        def __init__(self, *args, **kwargs):
            self.value = 0
            self.closed = False
            _FakeProgress.instances.append(self)

        def setWindowTitle(self, *a): ...
        def setWindowModality(self, *a): ...
        def setAutoClose(self, *a): ...
        def setMinimumDuration(self, *a): ...
        def show(self): ...
        def setValue(self, v): self.value = v
        def wasCanceled(self): return self.value >= 1
        def close(self): self.closed = True

    _FakeProgress.instances.clear()
    monkeypatch.setattr("src.ui.table_dialog.QProgressDialog", _FakeProgress)

    dlg._add_group_remaining_variables(state)

    assert list(state.df.columns) == ["time", "Press_G0", "State"], "取消后已插入的列必须保留"
    progress = _FakeProgress.instances[0]
    assert progress.closed, "退出批量必须关掉进度框"
    assert state.model.columnCount() == 3, "取消路径也要补上一次性重建的模型"


def test_add_group_remaining_below_threshold_adds_silently(tab_dialog, mdf_loader, monkeypatch):
    """默认阈值下（3 列 x 12 行）绝不弹确认框：“不打扰绝大多数组”的核心验收点。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)

    def forbidden(*args, **kwargs):
        raise AssertionError(f"小批量不应弹确认框：{args[2] if len(args) > 2 else args}")

    monkeypatch.setattr(QMessageBox, "question", forbidden)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    dlg._add_group_remaining_variables(state)

    assert set(state.df.columns) == {"time"} | set(mdf_loader.get_group_variables(0))


def test_close_tab_removes_only_target_tab(shown_tab_dialog, menu_stub, monkeypatch):
    """“关闭此标签页”只动被右键那一页，其余页的 widget_index 重映射仍正确。"""
    dlg = shown_tab_dialog
    s0 = dlg._add_variable_to_tab("Press_G0", 0)
    s1 = dlg._add_variable_to_tab("Press_G1", 1)
    pump(50)
    asked = _stub_answer(monkeypatch, QMessageBox.StandardButton.Yes)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭此标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert 0 not in dlg._group_tabs and 1 in dlg._group_tabs
    assert dlg._group_state_by_view(s0.view) is None
    assert s1.widget_index == 0, "前移后未重映射 → 切页/锚点会查到错的 state"
    assert dlg._tab_widget.widget(0) is s1.view
    assert asked[0][0].startswith("关闭标签页 ")
    assert "Press_G0" in asked[0][1], "确认框必须列出将被移除的变量名"


def test_close_tab_cancel_keeps_everything(shown_tab_dialog, menu_stub, monkeypatch):
    """确认框答 No → 页、列、定位框全部保持原样。"""
    dlg = shown_tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("Press_G1", 1)
    pump(50)
    asked = _stub_answer(monkeypatch, QMessageBox.StandardButton.No)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭此标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert list(dlg._group_tabs) == [0, 1]
    assert len(asked) == 1
    assert "已绘制的曲线不受影响" in asked[0][1], "必须写清关闭的后果"


def test_close_others_keeps_target(shown_tab_dialog, menu_stub, monkeypatch):
    """“关闭其他标签页”：只剩目标页且它就在前台。"""
    dlg = shown_tab_dialog
    s0 = dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("Press_G1", 1)
    pump(50)
    _stub_answer(monkeypatch, QMessageBox.StandardButton.Yes)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭其他标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert list(dlg._group_tabs) == [0]
    assert dlg._tab_widget.currentWidget() is s0.view
    assert dlg.get_column_names() == ["Press_G0"]


def test_close_all_tabs_returns_to_single_table_ui(shown_tab_dialog, menu_stub, monkeypatch):
    """“关闭所有标签页”退回单表 UI，必须是空表而不是两块白板（R19 回归）。"""
    dlg = shown_tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("Press_G1", 1)
    pump(50)
    asked = _stub_answer(monkeypatch, QMessageBox.StandardButton.Yes)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 1))
    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭所有标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 1))
    pump(50)

    assert "将关闭 2 个标签页，共移除 2 个变量列" in asked[0][1]
    assert dlg._group_tabs == {} and dlg._tab_mode is False
    assert dlg.model is not None, "退回单表后必须补回空模型"
    assert dlg.main_view.model() is dlg.model
    assert dlg.frozen_view.isHidden(), "无冻结列时 frozen_view 必须隐藏"
    assert dlg.has_table_content() is False


def test_close_tab_refreshes_locator_and_column_names(shown_tab_dialog, monkeypatch):
    """关闭页后对外口径同步：列名快照与定位框条目一起减少。"""
    dlg = shown_tab_dialog
    s0 = dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("State", 0)
    dlg._add_variable_to_tab("Press_G1", 1)
    pump(50)
    assert dlg.get_column_names() == ["Press_G0", "State", "Press_G1"]
    assert dlg._var_locator.count() == 3
    _stub_answer(monkeypatch, QMessageBox.StandardButton.Yes)

    dlg._close_tab(s0)

    assert dlg.get_column_names() == ["Press_G1"]
    assert dlg._var_locator.count() == 1
    assert dlg._group_state_by_view(s0.view) is None

