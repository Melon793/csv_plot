"""CustomViewBox 右键菜单中文文案 component 测试（offscreen）。

覆盖两件事：
1. 我们自己添加的菜单项全部为中文（pyqtgraph 默认项按设计保持英文）；
2. 幂等性——`getMenu` 返回的是 pyqtgraph 缓存的同一个 QMenu，创建文案与
   「判重/移除」比对文案必须同源，否则每次右键都会重复插入一份。

游标模式的显示文案改为中文后，`request_set_cursor_mode` 仍必须发英文
内部标识符（cursor_sync_manager / file_loader_manager 按它比对）。
"""

import pytest

from PySide6.QtCore import QPointF

from tests.fixtures.waits import flush_deferred_deletes
from src.ui.widgets.custom_viewbox import (
    ZH_ADJUST_HEIGHT,
    ZH_AUTO_Y_IN_X,
    ZH_CLEAR_PLOT,
    ZH_COPY_NAME,
    ZH_CURSOR_MODE,
    ZH_CURSOR_MODE_LABELS,
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
    """只提供 getMenu 用到的 scenePos() 的伪事件"""

    def __init__(self, x=0.0, y=0.0):
        self._pos = QPointF(x, y)

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
    return [a.menu() for a in menu.actions() if a.menu() is not None
            and a.menu().title() == title]


def test_menu_uses_chinese_labels(view_box):
    """自定义项全部中文，且无改造前的英文字面量残留"""
    menu = view_box.getMenu(_FakeMenuEvent())
    texts = _top_texts(menu)

    for zh in (
        ZH_JUMP_TO_DATA,
        ZH_AUTO_Y_IN_X,
        ZH_CURSOR_MODE,
        ZH_COPY_NAME,
        ZH_VAR_EDITOR,
        ZH_ADJUST_HEIGHT,
        ZH_CLEAR_PLOT,
    ):
        assert zh in texts, f"缺少中文菜单项 {zh!r}，实际：{texts}"

    # 动作项文案是当前状态的反面：cursor_values_hidden=False（数值可见）→ 提供「隐藏游标数值」
    assert ZH_HIDE_CURSOR_VALUE in texts
    assert ZH_SHOW_CURSOR_VALUE not in texts

    cursor_texts = [a.text() for a in _submenu(menu, ZH_CURSOR_MODE)[0].actions()]
    for mode in ("1 free cursor", "1 anchored cursor", "2 anchored cursor", "off"):
        assert ZH_CURSOR_MODE_LABELS[mode] in cursor_texts

    all_texts = texts + cursor_texts
    for legacy in LEGACY_ENGLISH_TEXTS:
        assert legacy not in all_texts, f"英文残留文案 {legacy!r} 未替换干净"


def test_pyqtgraph_default_items_untouched(view_box):
    """pyqtgraph 默认项按设计保持英文，且原有的按英文隐藏逻辑仍生效"""
    menu = view_box.getMenu(_FakeMenuEvent())
    actions = {a.text(): a for a in menu.actions()}

    assert "View All" in actions
    assert "Mouse Mode" in actions
    assert actions["Mouse Mode"].isVisible() is False


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
        ZH_HIDE_CURSOR_VALUE,
    ):
        assert texts.count(zh) == 1, f"{zh!r} 出现 {texts.count(zh)} 次：菜单项重复插入"

    assert len(_submenu(menu, ZH_CURSOR_MODE)) == 1
    assert len(_submenu(menu, ZH_ADJUST_HEIGHT)) == 1

    adjust = _submenu(menu, ZH_ADJUST_HEIGHT)[0]
    reset_texts = [a.text() for a in adjust.actions() if a.text() == ZH_RESET_ALL_HEIGHT]
    assert len(reset_texts) == 1


def test_cursor_mode_signal_still_emits_internal_identifier(view_box):
    """点选中文化后的模式项，信号仍发英文标识符"""
    menu = view_box.getMenu(_FakeMenuEvent())
    cursor_menu = _submenu(menu, ZH_CURSOR_MODE)[0]
    by_label = {a.text(): a for a in cursor_menu.actions()}

    received = []
    view_box.signals.request_set_cursor_mode.connect(
        lambda mode, pw, ctx_x: received.append(mode)
    )

    by_label["双固定游标"].trigger()
    by_label["关闭游标"].trigger()

    assert received == ["2 anchored cursor", "off"]


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
    再连点 5 次；游标模式子菜单与「调整高度」子菜单每轮都会重建，
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

