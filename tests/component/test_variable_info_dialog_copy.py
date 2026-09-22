"""变量信息窗口：值列复制按钮与几何（行高 / 列宽 / 文件路径复制）。

由 test_variable_info_dialog.py 拆分而来（只搬运，未改任何断言）：
复制按钮的几何、命中区、与选区互不干扰，以及行高列宽的落地守卫。
"""

import sys

import pytest

from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QTreeWidgetItem,
)
from shiboken6 import isValid

from tests.fixtures.waits import (
    flush_deferred_deletes,
    settle,
)
from src.core.config import (
    VAR_INFO_COL0_MIN_WIDTH,
    VAR_INFO_COPY_BTN_MARGIN,
    VAR_INFO_COPY_BTN_SIZE,
    VAR_INFO_ROW_HEIGHT,
)
from src.data import var_info
from src.ui.dialogs import variable_info_dialog as vid_mod
from src.ui.dialogs.variable_info_dialog import (
    CopyFieldDelegate,
    copy_button_rect,
)

# ---------------------------------------------------------------------------
# 值列行尾悬停复制按钮
# ---------------------------------------------------------------------------

_SENTINEL = "SENTINEL-MUST-NOT-BE-CLOBBERED"


def _find_row(tree, key: str):
    """按「属性」列文本定位行（递归遍历，分组标题同样可命中）。"""

    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            if child.text(0) == key:
                return child
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(tree.invisibleRootItem())


def _value_cell_rect(tree, item) -> QRect:
    """某个 item 的「值」列单元格矩形（视口坐标），并卡住“行可见”前提。

    visualItemRect 返回横跨整行的矩形，拿不到分列边界；视图自己算单元格
    用的是 columnViewportPosition + columnWidth，此处必须同口径，否则
    测的是另一块地方。

    末尾的断言是全部调用点的公共前置：行在视口外时 visualItemRect 仍会
    给出一套落在其它行上的坐标，拿它去点击就是静默误测（行高调大后可视
    行数变少，此风险从“测不出”变成“很容易撞上”）。需要时先 scrollToItem。
    """
    row_rect = tree.visualItemRect(item)
    assert 0 <= row_rect.top() < tree.viewport().height(), (
        f"目标行不在视口内，点击会落到别的行：{item.text(0)} row={row_rect} "
        f"视口高={tree.viewport().height()}"
    )
    return QRect(
        tree.columnViewportPosition(1),
        row_rect.top(),
        tree.columnWidth(1),
        row_rect.height(),
    )


def _viewport_image(tree) -> QImage:
    img = QImage(tree.viewport().size(), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.white)
    tree.viewport().render(img)
    return img


def _region_diff(a: QImage, b: QImage, rect: QRect) -> int:
    """rect 内颜色不同的像素数：用于区分“图标只落在热区”与“整行重绘”。"""
    count = 0
    for y in range(max(rect.top(), 0), min(rect.bottom(), a.height() - 1) + 1):
        for x in range(max(rect.left(), 0), min(rect.right(), a.width() - 1) + 1):
            if a.pixelColor(x, y) != b.pixelColor(x, y):
                count += 1
    return count


@pytest.fixture()
def qt_exceptions():
    """捕获 Qt 回调（委托 paint / 事件出口）里抛出的 Python 异常。

    PySide6 对虚函数重写里的异常只走 sys.excepthook、不向调用方上抛，
    offscreen 下更是看不见 stderr —— 不挂钩就会把崩溃类缺陷测成“通过”。
    """
    caught: list = []
    original = sys.excepthook

    def hook(exc_type, exc, tb):
        caught.append(exc)

    sys.excepthook = hook
    yield caught
    sys.excepthook = original


class TestFieldCopyButton:
    def test_tree_uses_copy_delegate(self, page):
        """接线前提：页面用的必须是带委托的子类，不是裸 QTreeWidget。"""
        _, pg = page
        assert isinstance(pg.tree, vid_mod.VarInfoTree)
        assert isinstance(pg.tree.itemDelegate(), CopyFieldDelegate)

    def test_copy_button_rect_geometry(self):
        """纯几何：正方形、右端对齐留 margin、垂直居中、不超出单元格。"""
        cell = QRect(120, 40, 600, 24)
        btn = copy_button_rect(cell)

        assert btn.width() == btn.height() == VAR_INFO_COPY_BTN_SIZE
        assert cell.right() - btn.right() == VAR_INFO_COPY_BTN_MARGIN
        assert btn.top() >= cell.top() and btn.bottom() <= cell.bottom()
        # 与文本区的分界：整条预留宽度内不允许被文本占用
        assert btn.left() == cell.right() - VAR_INFO_COPY_BTN_MARGIN - VAR_INFO_COPY_BTN_SIZE + 1

    def test_copy_button_rect_follows_runtime_constant(self, monkeypatch):
        """默认尺寸必须在调用时读全局，不得焊死在 def 上。

        踩过的坑：默认参数直接写 VAR_INFO_COPY_BTN_SIZE 时，它在 def 执行
        时就绑定了 config 的值，而 paint 里的文本区收缩读的是运行时全局
        —— 运行时改常量会让两处错位，“不重启预览尺寸”直接失效。
        本用例在旧写法下必然失败（拿到的还是 config 里的旧边长）。
        """
        monkeypatch.setattr(vid_mod, "VAR_INFO_COPY_BTN_SIZE", 30)
        monkeypatch.setattr(vid_mod, "VAR_INFO_COPY_BTN_MARGIN", 7)
        cell = QRect(0, 0, 600, 40)

        btn = copy_button_rect(cell)

        assert btn.width() == btn.height() == 30, "边长没跟运行时常量走"
        assert cell.right() - btn.right() == 7
        assert btn.top() == cell.center().y() - 15

    def test_is_copyable_rules(self, page):
        """值列有内容才给按钮；标题行、占位行、属性列一律不给。"""
        _, pg = page
        tree = pg.tree

        normal = _find_row(tree, "变量名")
        assert CopyFieldDelegate.is_copyable(tree.indexFromItem(normal, 1)) is True
        # 同一行的属性列不画按钮（否则满屏两个把手）
        assert CopyFieldDelegate.is_copyable(tree.indexFromItem(normal, 0)) is False

        header = _find_row(tree, "基本信息")
        assert header.parent() is None, "分组标题必须是顶层项"
        assert CopyFieldDelegate.is_copyable(tree.indexFromItem(header, 1)) is False

        dash = QTreeWidgetItem(tree, ["占位", "-"])
        assert CopyFieldDelegate.is_copyable(tree.indexFromItem(dash, 1)) is False

    def test_click_on_button_copies_value(self, page, qt_exceptions):
        """端到端：点击热区→剪贴板得到值列原文，状态栏回显键名。

        反向验证：未接线时（删掉 copy_requested 的 connect）本用例
        必然失败 —— 剪贴板停在哨兵值上。
        """
        dlg, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QApplication.clipboard().setText(_SENTINEL)

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        assert QApplication.clipboard().text() == item.text(1) == "speed"
        assert "变量名" in dlg.status_label.text()
        # 提示只报键名，不得把值也回显出去
        assert "speed" not in dlg.status_label.text()
        assert qt_exceptions == []

    def test_release_outside_button_does_not_copy(self, page, qt_exceptions):
        """按下在热区内、释放拖到热区外 → 取消（按按钮的通用语义）。"""
        _, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QApplication.clipboard().setText(_SENTINEL)

        QTest.mousePress(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        QTest.mouseRelease(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(btn.left() - VAR_INFO_COPY_BTN_MARGIN, btn.center().y()),
        )
        settle()

        assert QApplication.clipboard().text() == _SENTINEL
        assert qt_exceptions == []

    def test_press_on_button_leaves_selection_untouched(self, page):
        """复制是“读”操作，不得顺带把用户的选区弄乱。"""
        _, pg = page
        tree = pg.tree
        tree.clearSelection()
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        assert tree.selectionModel().selectedRows() == []

    def test_press_outside_button_still_selects_row(self, page):
        """热区外必须完全保持原有行为（补位验证上一条不是“整个事件被吞”）。"""
        _, pg = page
        tree = pg.tree
        tree.clearSelection()
        item = _find_row(tree, "变量名")
        cell = _value_cell_rect(tree, item)
        btn = copy_button_rect(cell)

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(cell.left() + 6, cell.center().y()),
        )
        settle()

        assert item in tree.selectedItems()
        assert btn.contains(QPoint(cell.left() + 6, cell.center().y())) is False

    def test_icon_only_drawn_on_hover_and_never_on_text(self, page, qt_exceptions):
        """悬停才出现图标，且只出现在热区 —— 文本区一个像素都不能变。"""
        _, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        cell = _value_cell_rect(tree, item)
        btn = copy_button_rect(cell)
        text_area = QRect(cell.left(), cell.top(), btn.left() - cell.left(), cell.height())

        before = _viewport_image(tree)
        QTest.mouseMove(tree.viewport(), btn.center())
        settle()
        hovered = _viewport_image(tree)

        assert _region_diff(before, hovered, btn) > 0, "悬停后热区内应画出图标"
        assert _region_diff(before, hovered, text_area) == 0, "预留宽度恒定，文本区不得因悬停而变"

        # 直接递一个 Leave 事件：offscreen 下合成“移出控件”的鼠标位不可靠，
        # 而被测的是我们自己的 leaveEvent 处理，不必依赖平台光标跟踪
        tree.leaveEvent(QEvent(QEvent.Type.Leave))
        settle()
        left = _viewport_image(tree)

        assert _region_diff(left, before, btn) == 0, "离开后图标必须消失"
        assert qt_exceptions == []

    def test_hover_state_survives_tree_rebuild(self, page, env, qt_exceptions):
        """悬停中重建整棵树 → 不崩，且新行仍可悬停复制。

        这就是悬停态用 QPersistentModelIndex 的全部理由：render() 里的
        tree.clear() 会重置模型，旧索引必须自动失效而不是被继续取用。
        """
        _, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QTest.mouseMove(tree.viewport(), btn.center())
        settle()

        snapshot = var_info.build_snapshot(env.loader_a, "speed", 0)
        old_stats = pg.stats
        pg.render(snapshot, old_stats)
        settle()
        assert qt_exceptions == [], "重建后旧悬停索引被取用会在此报异常"

        new_item = _find_row(tree, "变量名")
        new_btn = copy_button_rect(_value_cell_rect(tree, new_item))
        QApplication.clipboard().setText(_SENTINEL)
        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, new_btn.center(),
        )
        settle()
        assert QApplication.clipboard().text() == "speed"
        assert qt_exceptions == []

    def test_stats_backfilled_rows_are_copyable(self, page):
        """异步回填走 takeChildren 重建子行，新行同样要点得着。"""
        _, pg = page
        tree = pg.tree
        item = _find_row(tree, "最大值")
        assert item is not None, "前置：统计已回填"
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QApplication.clipboard().setText(_SENTINEL)

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        assert QApplication.clipboard().text() == item.text(1) == "30"

    def test_long_value_shows_key_only_and_never_widens_window(self, page):
        """长文件路径：剪贴板给全量，提示只报键名，且窗口不得被提示顶宽。

        后半段是回归闸：QLabel 的 sizeHint 会经布局抬高整个窗口的
        minimumWidth，实测复制一条长路径后窗口 700 → 769 且拖不回
        去；靠 status_label 的 Ignored 宽度策略消除。只断言提示文本
        变短是拦不住这个的 —— 真正的因是宽度策略。
        """
        dlg, pg = page
        tree = pg.tree
        long_value = "/Volumes/data/" + "measurement_" * 30 + ".mf4"
        item = QTreeWidgetItem(tree, ["文件路径", long_value])
        # 必须先滚入视口：行高变大后可视行数变少，追加到树末尾的行很
        # 可能已经在视口外，而 visualItemRect 仍会给出一套落在其它行上的
        # 坐标，不滚上来就是点错行的误测
        tree.scrollToItem(item)
        settle()
        btn = copy_button_rect(_value_cell_rect(tree, item))
        width_before = dlg.width()

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        assert QApplication.clipboard().text() == long_value
        notice = dlg.status_label.text()
        assert notice == "已复制「文件路径」"
        assert long_value not in notice
        assert dlg.width() == width_before, "提示不得把窗口顶宽"
        assert dlg.minimumWidth() <= width_before, "minimumWidth 被抬高就缩不回去"

    def test_copy_then_tab_close_leaves_no_crash(self, page, qt_exceptions):
        """复制过的页被关闭：委托与持久索引随页销毁，不得有残留回调。

        必须真把 DeferredDelete 投递完（理由见 force_delete 的注释），否则
        “没崩”只是因为页还没死，测不出任何东西。
        """
        dlg, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QTest.mouseMove(tree.viewport(), btn.center())
        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        dlg._on_tab_close(0)
        flush_deferred_deletes()

        assert dlg._pages == {}
        assert isValid(tree) is False, "页面必须真被销毁，否则本用例是空转的"
        assert qt_exceptions == []


class TestRowHeightAndColumnWidth:
    """行高与列宽配置项的落地守卫（断言值全部取自 config，不写死）。

    这两个尺寸都不是独立可调的：行高受复制按钮边长托底，列宽下限
    实际作用于所有列。调优时改坏任一约束，靠本类兜住。
    """

    @pytest.mark.skipif(VAR_INFO_ROW_HEIGHT <= 0, reason="=0 表示沿用系统默认，无固定值可断言")
    def test_row_height_applies_to_every_row(self, page):
        """委托给的高度要真成为行高，且分组行与数据行同值。

        uniformRowHeights 开着时逐行给不同高度会被拉平成最大值，本用例
        同时把这个前提测出来：一旦有人关掉该开关去追求不等高，这里先失败。
        """
        _, pg = page
        tree = pg.tree
        group = tree.topLevelItem(0)
        assert tree.visualItemRect(group).height() == VAR_INFO_ROW_HEIGHT
        assert tree.visualItemRect(group.child(0)).height() == VAR_INFO_ROW_HEIGHT

    def test_row_height_fits_copy_button(self, page):
        """行高不得小于复制按钮，否则图标会溢出骑到相邻行上。

        绘制不裁剪到单元格，所以后果不是“被裁掉”而是“遮挡别的内容”
        （实测 40 px 按钮配 18 px 行高：上下共溢出 22 px、横跨约 3 行）。

        这条是常量间的耦合关系：把 VAR_INFO_ROW_HEIGHT 调小到低于
        VAR_INFO_COPY_BTN_SIZE 时，必须在这里报错而不是默默花屏。约
        束方向是“行高→按钮”：改大按钮不会把行高顶高。
        """
        _, pg = page
        tree = pg.tree
        cell = _value_cell_rect(tree, _find_row(tree, "变量名"))
        btn = copy_button_rect(cell)
        assert btn.top() >= cell.top() and btn.bottom() <= cell.bottom(), (
            f"行高 {cell.height()} 容不下 {VAR_INFO_COPY_BTN_SIZE} px 按钮"
        )

    def test_col0_min_width_clamps_narrow_drag(self, page):
        """往窄拖的下限就是该常量；名字里的 col0 不等于只管第 0 列。"""
        _, pg = page
        tree = pg.tree
        assert tree.header().minimumSectionSize() == VAR_INFO_COL0_MIN_WIDTH
        tree.setColumnWidth(0, 1)
        assert tree.columnWidth(0) == VAR_INFO_COL0_MIN_WIDTH

    def test_col0_default_width_not_clamped_by_minimum(self, page):
        """调小下限不会把默认列宽一起拖窄：默认值由内容算出，高于下限。"""
        _, pg = page
        tree = pg.tree
        assert tree.columnWidth(0) > VAR_INFO_COL0_MIN_WIDTH


class _FakeAction:
    """QAction 替身：只记录 triggered 的接线，可手动 fire。"""

    def __init__(self, text: str):
        self.text = text
        self._slots = []

    @property
    def triggered(self):
        return self

    def connect(self, slot):
        self._slots.append(slot)

    def fire(self):
        for slot in self._slots:
            slot()


class _FakeMenu:
    """QMenu 替身：记录每次弹出的菜单与其中的项，不真画出来。

    offscreen 下 ``exec()`` 会阻塞等用户，而菜单内容本身（有哪几项、每项
    复制出什么）才是被测对象，因此直接拦掉渲染。``created`` 同时用于
    断言“不该弹的时候不弹”。
    """

    def __init__(self, parent=None):
        self._items = []
        _FakeMenu.created.append(self)

    def addAction(self, text):  # noqa: N802 - 跟 Qt 同名
        action = _FakeAction(text)
        self._items.append(action)
        return action

    def exec(self, pos):  # noqa: N802 - 跟 Qt 同名
        return None

    @property
    def items(self):
        return list(self._items)


@pytest.fixture()
def fake_menu(monkeypatch):
    _FakeMenu.created = []
    monkeypatch.setattr(vid_mod, "QMenu", _FakeMenu)
    yield _FakeMenu


class TestFilePathCopy:
    r"""「文件路径」行的写法风格 / 引号 / 右键菜单（跨平台复制）。

    背景（用户实测报障）：Windows 上拖拽进主窗口的网盘文件，路径形如
    ``//fileserver/team-share/… TC01_ EU7 Calibration/…1am=0.8_Map.csv``
    —— 正斜杠 UNC + 空格 + ``=``，粘回 Windows 被 Shell 当 URL 交给浏览器
    （实测跳 Edge），只有 ``\\host\share`` 加引号才能直接用。本组用例盯的就是
    “复制出去能不能用”，以及风格开关是否真的在调用时生效。
    """

    #: 报障串的结构复刻（保留空格、``=`` 与两层前导斜杠；主机名/项目号/客户名
    #: 均为虚构标识符，请勿改回真实内网路径）
    RAW = (
        "//fileserver/team-share/PRJ-0000-00_Demo_ENG01_TC01_ EU7 Calibration"
        "/20260101_DEMO_ENG01_1am=0.8_Map.csv"
    )
    WIN = RAW.replace("/", "\\")

    def _add_path_row(self, pg, value=RAW):
        """手工注入一行「文件路径」并滚入视口（同长路径用例的做法）。"""
        tree = pg.tree
        item = QTreeWidgetItem(tree, [var_info.ROW_KEY_FILE_PATH, value])
        tree.scrollToItem(item)
        settle()
        return item

    def _click_copy_button(self, pg, item):
        tree = pg.tree
        QApplication.clipboard().setText(_SENTINEL)
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()
        return QApplication.clipboard().text()

    def test_row_key_is_the_literal_ui_and_data_agree_on(self):
        """键名是 UI 查行的依据，被改字面就会静默失效（菜单与引号全不生效）。"""
        assert var_info.ROW_KEY_FILE_PATH == "文件路径"

    def test_button_follows_windows_style_from_config(self, page, monkeypatch):
        r"""PATH_COPY_STYLE=windows → 反斜杠 + UNC 两层前导保留 + 自动加引号。

        反向验证两处：旧写法（直接 ``setText(item.text(1))``）会原样给出
        正斜杠串 —— 就是报障里跳浏览器的那一条；而把风格常量绑在 def
        默认参数上时，monkeypatch 不生效，拿到的仍是 native。
        """
        dlg, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_STYLE", "windows")
        item = self._add_path_row(pg)

        copied = self._click_copy_button(pg, item)

        assert copied == '"' + self.WIN + '"'
        # 拆写避免转义看错：前导必为两个反斜杠（UNC），其后各段单反斜杠
        unc_prefix = '"' + "\\\\" + "fileserver" + "\\"
        assert copied.startswith(unc_prefix)
        assert dlg.status_label.text() == "已复制「文件路径」"

    def test_default_native_style_adds_quotes_but_keeps_slashes(self, page):
        """默认 native：macOS 上保持正斜杠，但含空格/``=`` 必须已被引号保护。"""
        _, pg = page
        item = self._add_path_row(pg)

        assert self._click_copy_button(pg, item) == '"' + self.RAW + '"'

    def test_quote_never_config_disables_wrapping(self, page, monkeypatch):
        """PATH_COPY_QUOTE=never → 粘进 Excel 单元格时不带引号。"""
        _, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_QUOTE", "never")
        item = self._add_path_row(pg)

        assert self._click_copy_button(pg, item) == self.RAW

    def test_quote_always_wraps_a_clean_path(self, page, monkeypatch):
        """PATH_COPY_QUOTE=always → 无特殊字符也包引号（对齐 Explorer）。"""
        _, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_STYLE", "windows")
        monkeypatch.setattr(vid_mod, "PATH_COPY_QUOTE", "always")
        clean = r"D:\Messung\x.dat"
        item = self._add_path_row(pg, clean)

        assert self._click_copy_button(pg, item) == '"' + clean + '"'

    def test_other_rows_are_copied_verbatim(self, page):
        """只有路径行参与转换：其余行必须原样取走（多一层处理反而不可预期）。"""
        _, pg = page
        item = _find_row(pg.tree, "变量名")

        assert self._click_copy_button(pg, item) == "speed"

    def test_context_menu_offers_three_styles(self, page, fake_menu, monkeypatch):
        """右键路径行 → 三项菜单，逐项得到各自写法（默认项跟配置走）。"""
        _, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_QUOTE", "never")
        item = self._add_path_row(pg)
        pos = _value_cell_rect(pg.tree, item).center()

        pg._on_tree_context_menu(pos)

        assert len(fake_menu.created) == 1
        actions = fake_menu.created[0].items
        assert len(actions) == 3
        assert [a.text for a in actions] == [
            "复制为 当前平台",
            "复制为 Windows \\host\\share",
            "复制为 POSIX //host/share",
        ]
        QApplication.clipboard().setText(_SENTINEL)
        actions[2].fire()
        assert QApplication.clipboard().text() == self.RAW
        actions[1].fire()
        assert QApplication.clipboard().text() == self.WIN

    def test_context_menu_skipped_for_other_rows_and_placeholder(
        self, page, fake_menu
    ):
        """非路径行不弹菜单；路径行的值为占位符 ``-`` 时也不弹。"""
        _, pg = page
        other = _find_row(pg.tree, "变量名")
        pg._on_tree_context_menu(_value_cell_rect(pg.tree, other).center())
        assert fake_menu.created == [], "非路径行不得弹菜单"

        dash = self._add_path_row(pg, "-")
        pg._on_tree_context_menu(_value_cell_rect(pg.tree, dash).center())
        assert fake_menu.created == [], "空路径无内容可复制，不得弹菜单"

    def test_ctrl_c_uses_the_same_formatting(self, page, monkeypatch):
        """Ctrl+C 与行尾按钮口径一致，否则会得到两种写法。"""
        _, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_STYLE", "windows")
        item = self._add_path_row(pg)
        pg.tree.clearSelection()
        item.setSelected(True)

        pg._on_copy_selection()

        text = QApplication.clipboard().text()
        assert text.startswith(var_info.ROW_KEY_FILE_PATH + "\t")
        assert text.split("\t")[1] == '"' + self.WIN + '"'

    def test_placeholder_path_is_not_promoted_to_a_fake_path(self, page):
        r"""值为占位符 ``-`` 时，三条复制出口一律原样给出 ``-``。

        反向验证：缺守卫时 ``display_path("-")`` 会把它当相对路径按当前工作
        目录绝对化，复制出 ``<cwd>/-`` —— 一条看着能用、实际打不开的假路径，
        而状态栏照样提示「已复制」，用户无从察觉。右键菜单此前已在调用点
        守了这条，本用例盯的是守卫进唯一出口后三条路径口径一致。
        """
        _, pg = page
        dash = self._add_path_row(pg, "-")

        # 唯一出口本身：三种风格一律不得改写占位符，空串同样原样返回
        for style in (None, "windows", "posix"):
            assert pg._format_path("-", style) == "-"
        assert pg._format_path("", None) == ""

        # Ctrl+C：此前唯一可达的漏口（行尾按钮被 is_copyable 拦在绘制层）
        pg.tree.clearSelection()
        dash.setSelected(True)
        QApplication.clipboard().setText(_SENTINEL)

        pg._on_copy_selection()

        assert QApplication.clipboard().text() == var_info.ROW_KEY_FILE_PATH + "\t-"

        # 行尾按钮的回调直调：is_copyable 只挡绘制，挡不住未来新增的 emit 方
        QApplication.clipboard().setText(_SENTINEL)
        pg._on_copy_field(dash)
        assert QApplication.clipboard().text() == "-"

    def test_tree_has_custom_context_menu_policy(self, page):
        """接线闸：菜单策略没改成 CustomContextMenu 时槽函数永不被调。"""
        _, pg = page
        assert pg.tree.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
