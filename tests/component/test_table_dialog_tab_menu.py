"""变量数值表：表头/标签栏右键菜单、分组批量添加与关闭。

由 test_table_dialog_tab_mode.py 拆分而来（只搬运，未改任何断言）：
表头菜单删除列、标签栏菜单的分组补全/关闭分支、批量添加的确认与硬上限、
以及执行期间标签页失效的兜底。
"""

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import (
    QMenu,
    QMessageBox,
)

from tests.fixtures.waits import settle
from src.ui.table_dialog import (
    DataTableDialog,
    PandasTableModel,
)

from tests.component._tab_mode_shared import FakeCsvLoader

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


def _forbid_confirmation(monkeypatch) -> None:
    """把确认框打成“调用即失败”，信息框静默：用于钉住“这条路径不该弹框”。"""

    def forbidden(*args, **kwargs):
        raise AssertionError(f"不该弹确认框：{args[1] if len(args) > 1 else args}")

    monkeypatch.setattr(QMessageBox, "question", forbidden)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)


class _NoCancelProgress:
    """不取消的 QProgressDialog 替身（测试里不想真弹一个窗口）。"""

    instances: list = []

    def __init__(self, *args, **kwargs):
        _NoCancelProgress.instances.append(self)

    def setWindowTitle(self, *a): ...
    def setWindowModality(self, *a): ...
    def setAutoClose(self, *a): ...
    def setMinimumDuration(self, *a): ...
    def show(self): ...
    def setValue(self, v): ...
    def wasCanceled(self): return False
    def close(self): ...


def test_tab_bar_context_menu_is_wired(shown_tab_dialog, menu_stub):
    """标签栏必须真挂上 CustomContextMenu。

    其余菜单用例都是直接调 handler，只改接线不改 handler 时它们全绿；
    而“右键没菜单”正是用户会看见的唯一故障现象，必须单独钉住。
    """
    dlg = shown_tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    bar = dlg._tab_widget.tabBar()
    settle()

    assert bar.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    assert bar.tabAt(_tab_bar_hit(dlg, 0)) == state.widget_index

    bar.customContextMenuRequested.emit(_tab_bar_hit(dlg, 0))
    settle()
    assert menu_stub.menus, "信号未接通 handler → 右键不会弹菜单"


def test_tab_bar_context_menu_items(shown_tab_dialog, menu_stub):
    """菜单四项的文本必须带准确数量（构建菜单时就算，不靠事后“猜”）。"""
    dlg = shown_tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("Press_G1", 1)
    settle()

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
    settle()

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
    settle()
    # 不只打桩 information：question 不打桩的话，一旦有人加回确认框，本用例会
    # 在 offscreen 下挂死而不是失败（改前的变异验证就是这么卡满 15 分钟的）
    _forbid_confirmation(monkeypatch)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    menu_stub.pick = _tab_menu_pick(menu_stub, "添加本组其余变量")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert set(state.df.columns) == {"time"} | set(mdf_loader.get_group_variables(0))
    assert dlg._var_locator.count() == 3


def test_single_tab_menu_hides_close_others(shown_tab_dialog, menu_stub):
    """只有一个 tab 时，“关闭其他/关闭所有”与“关闭此页”重复 → 不列出。"""
    dlg = shown_tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    settle()

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert _tab_menu_texts(menu_stub) == [
        "添加本组其余变量（+2 列）",
        "关闭此标签页（移除 1 个变量）",
    ]


def test_tab_bar_blank_area_right_click_shows_no_menu(shown_tab_dialog, menu_stub):
    """右键落在标签右侧空白：不得弹菜单（宁可不响应，也不能对错误的 tab 动手）。"""
    dlg = shown_tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    settle()

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


def test_add_group_remaining_adds_all_missing_columns(tab_dialog, mdf_loader, monkeypatch):
    """一次加完整组：列集合对齐 loader 的组内变量，行数不变。

    本组所有进入批量添加的用例都要 _forbid_confirmation：不只是断言“不该弹框”，
    更是 offscreen 下的自保——真弹一个模态框会挂死整个测试进程而不是失败。
    """
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    rows = len(state.df)
    _forbid_confirmation(monkeypatch)

    dlg._add_group_remaining_variables(state)

    assert set(state.df.columns) == {"time"} | set(mdf_loader.get_group_variables(0))
    assert len(state.df) == rows, "行数必须仍等于原时间轴长度，不得因列长度不一被拉伸"
    assert state.model.columnCount() == len(state.df.columns)
    assert dlg._var_locator.count() == 3


def test_add_group_remaining_is_idempotent_and_greys_out(
    shown_tab_dialog, menu_stub, monkeypatch
):
    """连点两次列数不变；已全加时菜单项变“已全部添加”并置灰。"""
    dlg = shown_tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    _forbid_confirmation(monkeypatch)
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
    _forbid_confirmation(monkeypatch)

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
    # 顺序不能倒：先 _forbid_confirmation 再盖 information，否则本用例拿不到正文
    _forbid_confirmation(monkeypatch)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info.append(a[2]))

    dlg._add_group_remaining_variables(state)

    assert "State" not in state.df.columns
    assert "Label" in state.df.columns, "一个通道失败不得影响其余"
    assert len(info) == 1 and "1 个取数失败" in info[0]
    assert state.model.columnCount() == len(state.df.columns)


def test_add_group_remaining_never_asks_confirmation(tab_dialog, mdf_loader, monkeypatch):
    """批量添加不弹确认框（用户要求）：进度框路径上也必须直接开工。

    右键标签本身就是明确的页级意图，多点一次“确定”只会挡住结果，还会把
    激活交给主窗口。这里把 question 打成抛错，任何退回确认框的写法会立即变红。
    """
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    _forbid_confirmation(monkeypatch)
    monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_PROGRESS_COLS", 0)
    _NoCancelProgress.instances.clear()
    monkeypatch.setattr("src.ui.table_dialog.QProgressDialog", _NoCancelProgress)

    dlg._add_group_remaining_variables(state)

    assert len(_NoCancelProgress.instances) == 1, "进度框仍是唯一的进度反馈"
    assert set(state.df.columns) == {"time"} | set(mdf_loader.get_group_variables(0))


def test_add_group_remaining_refuses_above_hard_cap(tab_dialog, monkeypatch):
    """超硬上限：一个都不加、也不走确认框，只提示改用逐个拖拽。

    上限是取消确认框后唯一保留的护栏：误点一个几千列的组不应该变成
    “先卡死三分钟”，而是直接拒绝并给出替代路径。
    """
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    msgs: list = []
    restored: list = []
    monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_COLS", 1)
    _forbid_confirmation(monkeypatch)
    monkeypatch.setattr(
        QMessageBox, "information", lambda *a, **k: msgs.append((a[1], a[2]))
    )
    monkeypatch.setattr(DataTableDialog, "_restore_foreground", lambda self: restored.append(1))

    dlg._add_group_remaining_variables(state)

    assert list(state.df.columns) == ["time", "Press_G0"], "超上限不得有任何副作用"
    # P2-5 后标题改为「本组数据量过大」：护栏不止列数一条轴，列数不够格时也会因内存被拦
    assert msgs and msgs[0][0] == "本组数据量过大"
    assert "拖拽" in msgs[0][1], "必须告诉用户改用逐个添加"
    assert restored == [1], "弹过信息框也必须把窗口拉回前台"


def test_restore_foreground_raises_and_activates(tab_dialog, monkeypatch):
    """两个调用缺一不可：raise_ 管窗口叠序，activateWindow 管键盘焦点。"""
    calls: list = []
    monkeypatch.setattr(DataTableDialog, "raise_", lambda self: calls.append("raise"))
    monkeypatch.setattr(
        DataTableDialog, "activateWindow", lambda self: calls.append("activate")
    )

    tab_dialog._restore_foreground()

    assert calls == ["raise", "activate"]


def test_add_group_remaining_cancel_keeps_added_columns(tab_dialog, monkeypatch):
    """进度框取消：保留已插入的列、照常重建一次模型，不整批回滚。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_PROGRESS_COLS", 0)
    _forbid_confirmation(monkeypatch)
    restored: list = []
    monkeypatch.setattr(DataTableDialog, "_restore_foreground", lambda self: restored.append(1))

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
    assert restored == [1], "进度框关掉后必须把表格拉回前台（实测 Qt 会交给主窗口）"


def test_small_bulk_add_touches_no_modal_or_foreground(tab_dialog, mdf_loader, monkeypatch):
    """默认阈值下（2 列待加）全程零模态框：既不弹框也不动窗口叠序。"""
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    _forbid_confirmation(monkeypatch)
    restored: list = []
    monkeypatch.setattr(DataTableDialog, "_restore_foreground", lambda self: restored.append(1))

    dlg._add_group_remaining_variables(state)

    assert set(state.df.columns) == {"time"} | set(mdf_loader.get_group_variables(0))
    assert restored == [], "没弹模态框就不该 raise/activate，避免无谓改窗口叠序"


def test_bulk_add_250_columns_touches_no_modal(tab_dialog, mdf_loader, monkeypatch):
    """用户那个量级（135~250 列）在默认阈值下必须零模态框。

    真实文件实测 200 列 x 107,001 行只要 0.25 s，旧阈值 30 列却会为它弹一个
    进度框；而模态框关闭时 Qt 会把窗口叠序交给主窗口，就是“点完确定窗口不
    见了”的根源。阈值抬到 300 后这条路径上一个框都不弹，从根本上不会丢前台。
    """
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    names = [f"Sig{i}" for i in range(250)]
    _forbid_confirmation(monkeypatch)
    monkeypatch.setattr(mdf_loader, "get_group_variables", lambda gi: names)
    monkeypatch.setattr(mdf_loader, "get_series", lambda name: pd.Series(np.zeros(12)))

    def no_progress(*a, **k):
        raise AssertionError("默认阈值下 250 列的快批量不该弹进度框")

    monkeypatch.setattr("src.ui.table_dialog.QProgressDialog", no_progress)
    restored: list = []
    monkeypatch.setattr(DataTableDialog, "_restore_foreground", lambda self: restored.append(1))

    dlg._add_group_remaining_variables(state)

    assert len(state.df.columns) == 2 + 250
    assert restored == [], "全程没弹框，不需要也不应该去动窗口叠序"


def test_close_tab_removes_only_target_tab(shown_tab_dialog, menu_stub, monkeypatch):
    """“关闭此标签页”只动被右键那一页，其余页的 widget_index 重映射仍正确。"""
    dlg = shown_tab_dialog
    s0 = dlg._add_variable_to_tab("Press_G0", 0)
    s1 = dlg._add_variable_to_tab("Press_G1", 1)
    settle()
    _forbid_confirmation(monkeypatch)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭此标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert 0 not in dlg._group_tabs and 1 in dlg._group_tabs
    assert dlg._group_state_by_view(s0.view) is None
    assert s1.widget_index == 0, "前移后未重映射 → 切页/锚点会查到错的 state"
    assert dlg._tab_widget.widget(0) is s1.view


def test_closing_tabs_never_asks_confirmation(shown_tab_dialog, menu_stub, monkeypatch):
    """三个关闭项全部直连：确认框原本要拦三道，用户要求一律去掉。

    去掉确认框后，“重载数据不会自动恢复”这条后果必须仍活在菜单 tooltip 里 ——
    信息可以从弹框挪走，不能凭空消失；这也是本用例存在的另一半理由。
    """
    dlg = shown_tab_dialog
    dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("Press_G1", 1)
    settle()
    _forbid_confirmation(monkeypatch)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 1))
    tips = {
        a.text(): a.toolTip() for a in menu_stub.menus[-1].actions() if not a.isSeparator()
    }
    for text, tip in tips.items():
        if text.startswith("关闭"):
            assert "曲线不受影响" in tip and "不会自动恢复" in tip, f"{text} 缺少后果说明：{tip}"

    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭其他标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 1))
    assert list(dlg._group_tabs) == [1]

    # 只剩一页时不再给“关闭其他/所有”，用“关闭此标签页”收尾（同样不弹框）
    # pick 不清零的话“查看菜单”的那一次右键会直接执行上一次的动作
    menu_stub.pick = None
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    assert not any(t.startswith("关闭其他") for t in _tab_menu_texts(menu_stub))
    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭此标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    assert dlg._group_tabs == {}


def test_close_others_keeps_target(shown_tab_dialog, menu_stub, monkeypatch):
    """“关闭其他标签页”：只剩目标页且它就在前台。"""
    dlg = shown_tab_dialog
    s0 = dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("Press_G1", 1)
    settle()
    _forbid_confirmation(monkeypatch)

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
    settle()
    _forbid_confirmation(monkeypatch)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 1))
    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭所有标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 1))
    settle()

    assert dlg._group_tabs == {} and dlg._tab_mode is False
    assert dlg.model is not None, "退回单表后必须补回空模型"
    assert dlg.main_view.model() is dlg.model
    assert dlg.frozen_view.isHidden(), "无冻结列时 frozen_view 必须隐藏"
    assert dlg.has_table_content() is False


def test_close_tab_refreshes_locator_and_column_names(shown_tab_dialog, menu_stub, monkeypatch):
    """关闭页后对外口径同步：列名快照与定位框条目一起减少。"""
    dlg = shown_tab_dialog
    s0 = dlg._add_variable_to_tab("Press_G0", 0)
    dlg._add_variable_to_tab("State", 0)
    dlg._add_variable_to_tab("Press_G1", 1)
    settle()
    assert dlg.get_column_names() == ["Press_G0", "State", "Press_G1"]
    assert dlg._var_locator.count() == 3
    _forbid_confirmation(monkeypatch)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))
    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭此标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert dlg.get_column_names() == ["Press_G1"]
    assert dlg._var_locator.count() == 1
    assert dlg._group_state_by_view(s0.view) is None


def test_add_group_remaining_aborts_when_tab_reset_midway(
    tab_dialog, mdf_loader, monkeypatch
):
    """批量途中后台重载完成（_reset_tab_mode）：立即中止，不写僵尸页、不弹误导框。

    P1-1 回归：旧实现里循环只判 wasCanceled，tab 被重置后仍把余下列写进已清空
    的 state.df，对已摘除的 view 重建模型；若 loader 同时被 close，余下列全部
    KeyError 计入 skipped，最后弹“N 个取数失败”——真实原因（页已不存在）被完全
    掩盖。现在每轮开头与收尾前都用代际令牌 + 在位双重判定。
    """
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)
    real_get_series = mdf_loader.get_series
    info: list = []

    def reload_then(name):
        # 取第一列（State）时模拟"重载完成回调抵达"：整个 tab 模式被重置
        if name == "State":
            dlg._reset_tab_mode()
        return real_get_series(name)

    monkeypatch.setattr(mdf_loader, "get_series", reload_then)
    _forbid_confirmation(monkeypatch)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info.append(a[2]))

    dlg._add_group_remaining_variables(state)  # 本组还差 State / Label 两列

    assert dlg._group_tabs == {} and dlg._tab_mode is False
    assert state.model is None, "失效后不得再给死页重建模型"
    assert "Label" not in state.df.columns, "失效后不得继续往僵尸 df 写列"
    assert info == [], "不能对已不存在的页弹误导性的取数失败提示"


def test_add_group_remaining_degrades_when_group_api_raises(
    tab_dialog, mdf_loader, monkeypatch
):
    """P2-2：数据源在菜单构建后死掉，批量入口必须与 _pending_group_var_count 同口径降级。

    loader 已 close 时 _ensure_open 抛 KeyError（"未知组返空列表"只对越界组
    成立），旧写法裸调会把异常以未捕获形式抛给 Qt 槽，用户视角是"点了菜单
    没反应"。现在入口 try/except：零副作用 + 一条提示 + 拉回前台。
    """
    dlg = tab_dialog
    state = dlg._add_variable_to_tab("Press_G0", 0)

    def boom(gi):
        raise KeyError("MDF 数据源已关闭")

    msgs: list = []
    restored: list = []
    monkeypatch.setattr(mdf_loader, "get_group_variables", boom)
    _forbid_confirmation(monkeypatch)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: msgs.append(a[2]))
    monkeypatch.setattr(DataTableDialog, "_restore_foreground", lambda self: restored.append(1))

    dlg._add_group_remaining_variables(state)

    assert list(state.df.columns) == ["time", "Press_G0"], "中止入口不得有任何副作用"
    assert msgs and "数据源不可用" in msgs[0], "必须告知失败原因而非静默无效"
    assert restored == [1], "弹过提示框也要把窗口拉回前台"
    assert dlg._pending_group_var_count(state) == -1, "菜单侧同口径降级仍在位"


def test_tab_menu_action_skipped_when_tab_died_during_exec(
    shown_tab_dialog, menu_stub, monkeypatch
):
    """菜单 exec 的嵌套事件循环里发生重载重建：exec 返回后动作必须作废。

    不加闸时 _remove_tab(state) 按 group_index pop，会把重建后的同键新页
    （用户根本没右键过它）连坐关掉；正确行为是什么都不做。
    """
    dlg = shown_tab_dialog
    s0 = dlg._add_variable_to_tab("Press_G0", 0)
    settle()
    _forbid_confirmation(monkeypatch)

    real_exec = menu_stub.exec

    def exec_then_reload(self, *args, **kwargs):
        picked = real_exec(self, *args, **kwargs)
        if picked is not None:
            # 模拟"菜单打开期间重载完成并重新建了同组的一页"
            dlg._reset_tab_mode()
            dlg._add_variable_to_tab("Press_G0", 0)
        return picked

    monkeypatch.setattr(menu_stub, "exec", exec_then_reload)

    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))  # 先看一眼菜单（pick=None）
    menu_stub.pick = _tab_menu_pick(menu_stub, "关闭此标签页")
    dlg._on_tab_bar_right_click(_tab_bar_hit(dlg, 0))

    assert list(dlg._group_tabs) == [0], "陈旧 state 放行会误摘重建后的同键新页"
    assert dlg._group_tabs[0] is not s0
    assert dlg._tab_widget.count() == 1


class TestBulkAddMemoryGuard:
    """P2-5：批量添加护栏要看内存估算，不能只数格子。

    阈值直接改 monkeypatch，而不是造 150 万行的表：本项要钉的是「哪条轴该触发」
    这个判定，不是估算公式的浮点行为。
    """

    @staticmethod
    def _info_sink(monkeypatch):
        info: list = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info.append(a[2]))
        return info

    def test_memory_axis_blocks_below_the_column_cap(self, tab_dialog, monkeypatch):
        dlg = tab_dialog
        state = dlg._add_variable_to_tab("Press_G0", 0)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_EST_MB", 0.0001)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_COLS", 2000)
        info = self._info_sink(monkeypatch)

        dlg._add_group_remaining_variables(state)

        assert len(info) == 1, f"该被内存轴拦下: {info}"
        assert "内存" in info[0]
        assert list(state.df.columns) == ["time", "Press_G0"], "拦下时一列都不该进表"

    def test_column_axis_still_blocks_a_tiny_table(self, tab_dialog, monkeypatch):
        """列数上限独立生效：行数极少、内存估算近乎 0 时也不许一次灌进来。"""
        dlg = tab_dialog
        state = dlg._add_variable_to_tab("Press_G0", 0)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_EST_MB", 1e9)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_COLS", 1)
        info = self._info_sink(monkeypatch)

        dlg._add_group_remaining_variables(state)

        assert len(info) == 1
        assert list(state.df.columns) == ["time", "Press_G0"]

    def test_batch_within_both_limits_still_adds_everything(self, tab_dialog, monkeypatch):
        """正向对照：两轴都宽松时必须照常补齐，否则前两条是在测空跑。"""
        dlg = tab_dialog
        state = dlg._add_variable_to_tab("Press_G0", 0)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_EST_MB", 1e9)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_COLS", 2000)
        info = self._info_sink(monkeypatch)

        dlg._add_group_remaining_variables(state)

        assert info == []
        assert {"State", "Label"} <= set(state.df.columns)

    def test_memory_axis_message_quotes_rows_and_estimate(self, tab_dialog, monkeypatch):
        dlg = tab_dialog
        state = dlg._add_variable_to_tab("Press_G0", 0)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_EST_MB", 0.0001)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_COLS", 2000)
        info = self._info_sink(monkeypatch)

        dlg._add_group_remaining_variables(state)

        text = info[0]
        assert "12 行" in text, f"要让用户知道按多少行估的: {text}"
        assert "内存" in text
        assert "列的批量上限" not in text, "报的应是真正触发的那条轴"

    def test_column_axis_message_does_not_quote_a_zero_memory_estimate(
        self, tab_dialog, monkeypatch
    ):
        """列数触发时不提内存：2 列 × 12 行的估算只有 0.0002 MB，写出来是噪声。"""
        dlg = tab_dialog
        state = dlg._add_variable_to_tab("Press_G0", 0)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_EST_MB", 1e9)
        monkeypatch.setattr("src.ui.table_dialog._BULK_ADD_MAX_COLS", 1)
        info = self._info_sink(monkeypatch)

        dlg._add_group_remaining_variables(state)

        text = info[0]
        assert "超过 1 列的批量上限" in text, text
        assert "内存" not in text
