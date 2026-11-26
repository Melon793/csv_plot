"""CustomViewBox 右键菜单结构测试（offscreen）。

菜单完全自建（不再继承 pyqtgraph 的 ViewBoxMenu）：
- 顶层顺序固定：按 X 调节 Y / 游标 → 数据三项 → 布局两项 → 导出 → 清除；
- pyqtgraph 的 View All / X/Y axis / Plot Options / Mouse Mode 不再出现；
- 导出改为复用 Scene 原生动作（scene.showExportDialog 能力保留）；
- 缓存同一个 QMenu、每次右键清空重排，反复右键不累积对象。

游标选项的显示文案改为中文后，`request_set_cursor_mode` 仍必须发英文
内部标识符（cursor_sync_manager / file_loader_manager 按它比对）。

「隐藏/显示游标数值」与四个模式正交，已收进「游标选项」子菜单：
它不加入互斥组、不显示勾选态，只随状态翻转文案，并与模式项用分隔线隔开。
"""

import pytest

from PySide6.QtCore import QPoint, QPointF, Qt

from tests.fixtures.waits import flush_deferred_deletes
from src.ui.widgets.custom_viewbox import (
    ZH_ADJUST_HEIGHT,
    ZH_ARRANGE_LAYOUT,
    ZH_AUTO_Y_IN_X,
    ZH_CLEAR_PLOT,
    ZH_COPY_NAME,
    ZH_CURSOR_MENU,
    ZH_CURSOR_MODE_LABELS,
    ZH_EXPORT,
    ZH_HIDE_CURSOR_VALUE,
    ZH_JUMP_TO_DATA,
    ZH_RESET_ALL_HEIGHT,
    ZH_SHOW_CURSOR_VALUE,
    ZH_VAR_EDITOR,
)

# 改造前的英文字面量：任何一条重新出现在菜单里都说明有比对点漏改
LEGACY_ENGLISH_TEXTS = [
    "Jump to Data",
    "Autoscale in x-Range",
    "Cursor Mode",
    "1 free cursor",
    "1 anchored cursor",
    "2 anchored cursor",
    "off",
    "Show Cursor Value",
    "Hide Cursor Value",
    "Copy Name",
    "Plot Variable Editor",
    "Adjust Height",
    "100% to all",
    "Clear Plot",
]


class _FakeMenuEvent:
    """只提供 getMenu / addParentContextMenus 用到的字段"""

    def __init__(self, x=0.0, y=0.0, accepted_item=None):
        self._pos = QPointF(x, y)
        self.acceptedItem = accepted_item

    def scenePos(self):
        return self._pos


@pytest.fixture()
def view_box(plot_factory):
    """带最小 plot_context 替身的 CustomViewBox。

    FakePlotContext 只覆盖绘图主路径，菜单状态查询用到的成员在此补齐
    （CustomViewBox 用 hasattr(plot_widget, "plot_context") 判定可用）。
    """
    pw = plot_factory()
    vb = pw.view_box
    ctx = pw.plot_context
    ctx.is_cursor_enabled = lambda: False
    ctx.cursor_mode = "1 free cursor"
    ctx.cursor_values_hidden = False
    ctx.get_row_height = lambda row: 100
    ctx._plot_col_max_default = 1
    ctx.plot_widgets = []
    return vb


def _top_texts(menu):
    return [a.text() for a in menu.actions()]


def _submenu(menu, title):
    """按标题取子菜单（返回 QMenu 列表，用于断言数量）"""
    return [
        a.menu()
        for a in menu.actions()
        if a.menu() is not None and a.menu().title() == title
    ]


def test_native_viewbox_menu_is_never_created(view_box):
    """原生菜单不应成为无宿主顶层窗口，也不应成为 plot 的可见子控件。"""
    assert view_box.menu is None
    assert view_box.menuEnabled() is False
    ctrl_menu = view_box.plot_widget.plot_item.ctrlMenu
    assert ctrl_menu.parent() is view_box.plot_widget
    assert ctrl_menu.isWindow()
    assert ctrl_menu.windowType() == Qt.WindowType.Popup
    assert not ctrl_menu.isVisible()
    assert view_box.plot_widget.childAt(QPoint(10, 10)) is not ctrl_menu


def test_menu_uses_chinese_labels(view_box):
    """自定义项全部中文，且无改造前的英文字面量残留"""
    menu = view_box.getMenu(_FakeMenuEvent())
    texts = _top_texts(menu)

    for zh in (
        ZH_JUMP_TO_DATA,
        ZH_AUTO_Y_IN_X,
        ZH_CURSOR_MENU,
        ZH_COPY_NAME,
        ZH_VAR_EDITOR,
        ZH_ADJUST_HEIGHT,
        ZH_CLEAR_PLOT,
    ):
        assert zh in texts, f"缺少中文菜单项 {zh!r}，实际：{texts}"

    # 数值开关已收进子菜单，顶层不得残留任何状态下的文案
    assert ZH_HIDE_CURSOR_VALUE not in texts
    assert ZH_SHOW_CURSOR_VALUE not in texts

    cursor_actions = _submenu(menu, ZH_CURSOR_MENU)[0].actions()
    cursor_texts = [a.text() for a in cursor_actions]
    for mode in ("1 free cursor", "1 anchored cursor", "2 anchored cursor", "off"):
        assert ZH_CURSOR_MODE_LABELS[mode] in cursor_texts

    # 动作项文案是当前状态的反面：cursor_values_hidden=False（数值可见）→ 提供「隐藏游标数值」
    assert ZH_HIDE_CURSOR_VALUE in cursor_texts
    assert ZH_SHOW_CURSOR_VALUE not in cursor_texts

    # 结构：四个模式项 → 分隔线 → 数值开关（开关不进互斥组的视觉证据）
    assert [a.isSeparator() for a in cursor_actions] == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert cursor_actions[-1].text() == ZH_HIDE_CURSOR_VALUE

    # 互斥组只含四个模式项；数值开关不属于任何互斥组
    mode_groups = {a.actionGroup() for a in cursor_actions[:4]}
    assert len(mode_groups) == 1
    assert cursor_actions[0].actionGroup().isExclusive() is True
    assert cursor_actions[-1].actionGroup() is None, "数值开关不得加入互斥组"

    all_texts = texts + cursor_texts
    for legacy in LEGACY_ENGLISH_TEXTS:
        assert legacy not in all_texts, f"英文残留文案 {legacy!r} 未替换干净"


def test_pyqtgraph_default_items_removed(view_box):
    """原生项全部退出菜单；导出保留为 Scene 动作并汉化"""
    menu = view_box.getMenu(_FakeMenuEvent())
    actions = {a.text(): a for a in menu.actions() if not a.isSeparator()}

    for native in ("View All", "X axis", "Y axis", "Mouse Mode", "Plot Options"):
        assert native not in actions, f"原生菜单项 {native!r} 未移除"
    assert "Export..." not in actions
    assert ZH_EXPORT in actions

    scene = view_box.scene()
    assert scene.contextMenu == [], "原生导出项仍会被 Scene 自动追加"
    assert scene._csv_plot_export_action is actions[ZH_EXPORT]

    # 停用的只是 PlotItem 菜单入口，ctrlMenu 本身原样保留
    plot_item = view_box.plot_widget.plot_item
    assert plot_item.menuEnabled() is False
    assert plot_item.ctrlMenu is not None


def test_export_action_keeps_native_chain(view_box):
    """导出项仍走 Scene.showExportDialog；真实追加链路不会重复塞原生项"""
    vb = view_box
    ev = _FakeMenuEvent(accepted_item=vb)
    menu = vb.getMenu(ev)
    scene = vb.scene()

    class _FakeDialog:
        def __init__(self):
            self.shown = []

        def show(self, item):
            self.shown.append(item)

    fake = _FakeDialog()
    scene.exportDialog = fake
    try:
        # 复现 ViewBox.mouseClickEvent → addParentContextMenus 的真实顺序
        scene.addParentContextMenus(vb, menu, ev)
        texts = [a.text() for a in menu.actions()]
        assert texts.count(ZH_EXPORT) == 1, "导出项被重复追加"
        assert "Plot Options" not in texts

        next(a for a in menu.actions() if a.text() == ZH_EXPORT).trigger()
        assert fake.shown == [vb]
    finally:
        scene.exportDialog = None


def test_top_level_order_is_fixed(view_box):
    """顶层顺序固定：视图 → 数据 → 布局 → 导出 → 清除"""
    menu = view_box.getMenu(_FakeMenuEvent())
    texts = [a.text() for a in menu.actions()]

    assert texts == [
        ZH_AUTO_Y_IN_X,
        ZH_CURSOR_MENU,
        "",
        ZH_JUMP_TO_DATA,
        ZH_COPY_NAME,
        ZH_VAR_EDITOR,
        "",
        ZH_ADJUST_HEIGHT,
        ZH_ARRANGE_LAYOUT,
        "",
        ZH_EXPORT,
        "",
        ZH_CLEAR_PLOT,
    ]


def test_get_menu_is_idempotent(view_box):
    """连续两次 getMenu 不重复插入（比对文案与创建文案同源的回归）"""
    view_box.getMenu(_FakeMenuEvent())
    menu = view_box.getMenu(_FakeMenuEvent())
    texts = _top_texts(menu)

    for zh in (
        ZH_JUMP_TO_DATA,
        ZH_AUTO_Y_IN_X,
        ZH_COPY_NAME,
        ZH_VAR_EDITOR,
        ZH_CLEAR_PLOT,
    ):
        assert texts.count(zh) == 1, f"{zh!r} 出现 {texts.count(zh)} 次：菜单项重复插入"

    assert texts.count(ZH_CURSOR_MENU) == 1
    assert ZH_HIDE_CURSOR_VALUE not in texts
    assert len(_submenu(menu, ZH_CURSOR_MENU)) == 1
    assert len(_submenu(menu, ZH_ADJUST_HEIGHT)) == 1

    cursor = _submenu(menu, ZH_CURSOR_MENU)[0]
    value_texts = [
        a.text()
        for a in cursor.actions()
        if a.text() in (ZH_SHOW_CURSOR_VALUE, ZH_HIDE_CURSOR_VALUE)
    ]
    assert value_texts == [ZH_HIDE_CURSOR_VALUE], f"数值开关重复插入：{value_texts}"

    adjust = _submenu(menu, ZH_ADJUST_HEIGHT)[0]
    reset_texts = [
        a.text() for a in adjust.actions() if a.text() == ZH_RESET_ALL_HEIGHT
    ]
    assert len(reset_texts) == 1


def test_cursor_mode_signal_still_emits_internal_identifier(view_box):
    """点选中文化后的模式项，信号仍发英文标识符"""
    menu = view_box.getMenu(_FakeMenuEvent())
    cursor_menu = _submenu(menu, ZH_CURSOR_MENU)[0]
    by_label = {a.text(): a for a in cursor_menu.actions()}

    received = []
    view_box.signals.request_set_cursor_mode.connect(
        lambda mode, pw, ctx_x: received.append(mode)
    )

    by_label["双固定游标"].trigger()
    by_label["关闭游标"].trigger()

    assert received == ["2 anchored cursor", "off"]


def test_cursor_value_action_in_submenu_state_and_signals(view_box):
    """数值开关：未开游标置灰；文案随隐藏态翻转；点击发对应信号"""
    ctx = view_box.plot_widget.plot_context

    # 数值信号的真实落点（EventHandler）会读 cursor_btn；替身补一个"未勾选"桩，
    # 让信号穿过真实 EventHandler 落点（未勾选分支），不越界访问主窗口控件
    class _CursorBtnStub:
        def isChecked(self):
            return False

    ctx.cursor_btn = _CursorBtnStub()

    # 光标关闭（fixture 默认态）：开关仍在子菜单内，但置灰
    cursor_menu = _submenu(view_box.getMenu(_FakeMenuEvent()), ZH_CURSOR_MENU)[0]
    hide_act = {a.text(): a for a in cursor_menu.actions()}[ZH_HIDE_CURSOR_VALUE]
    assert hide_act.isEnabled() is False
    assert hide_act.isCheckable() is False, "开关不得带勾选态，否则会被当成第 5 种模式"

    shown, hidden = [], []
    view_box.signals.request_show_cursor_value.connect(lambda pw: shown.append(pw))
    view_box.signals.request_hide_cursor_value.connect(lambda pw: hidden.append(pw))

    # 光标开启 + 数值已隐藏：文案翻转为「显示游标数值」，点击发 show 信号
    ctx.is_cursor_enabled = lambda: True
    ctx.cursor_values_hidden = True
    cursor_menu = _submenu(view_box.getMenu(_FakeMenuEvent()), ZH_CURSOR_MENU)[0]
    by_text = {a.text(): a for a in cursor_menu.actions() if not a.isSeparator()}
    assert ZH_HIDE_CURSOR_VALUE not in by_text
    show_act = by_text[ZH_SHOW_CURSOR_VALUE]
    assert show_act.isEnabled() is True
    show_act.trigger()
    assert shown == [view_box.plot_widget]
    assert hidden == []

    # 回到数值可见态：同一位置换成「隐藏游标数值」，点击发 hide 信号
    ctx.cursor_values_hidden = False
    cursor_menu = _submenu(view_box.getMenu(_FakeMenuEvent()), ZH_CURSOR_MENU)[0]
    by_text = {a.text(): a for a in cursor_menu.actions() if not a.isSeparator()}
    by_text[ZH_HIDE_CURSOR_VALUE].trigger()
    assert hidden == [view_box.plot_widget]
    assert shown == [view_box.plot_widget]


def _live_object_counts(menu):
    from PySide6.QtGui import QAction, QActionGroup
    from PySide6.QtWidgets import QMenu

    return {
        "QAction": len(menu.findChildren(QAction)),
        "QMenu": len(menu.findChildren(QMenu)),
        "QActionGroup": len(menu.findChildren(QActionGroup)),
    }


def test_repeated_get_menu_does_not_accumulate_objects(view_box, qapp):
    """连续右键不累积 QObject：removeAction 之后必须真正销毁

    基线取第一轮 getMenu（此时 pyqtgraph 自身项与我们的自定义项都已建好），
    再连点 5 次；游标选项子菜单与「调整高度」子菜单每轮都会重建，
    旧对象若只 remove 不 delete 就会一直挂在缓存菜单的 children 上。
    """
    menu = view_box.getMenu(_FakeMenuEvent())
    flush_deferred_deletes()
    baseline = _live_object_counts(menu)
    visible_baseline = len(menu.actions())

    for _ in range(5):
        view_box.getMenu(_FakeMenuEvent())
        flush_deferred_deletes()

    assert _live_object_counts(menu) == baseline
    assert len(menu.actions()) == visible_baseline
