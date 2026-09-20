"""e2e：状态栏左段 → 文件信息上翻抽屉（P0-2 Step B）。

覆盖三件事：抽屉真能弹在状态栏上方（几何量测，不是"看起来对"）、内容就是
当前文件的真实信息（换文件后不许残留）、以及三个动作出口（复制入剪贴板、
打开走系统文件管理器且不拼 shell、Esc/加载/换文件都能收起）。

字段口径本身在 tests/unit/data/test_file_info.py 里钉；这里只验 UI 层。
"""

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QToolButton

from src.core.config import STATUS_DRAWER_EDGE_MARGIN, STATUS_DRAWER_WIDTH
from src.data.file_info import KEY_FILE_NAME, KEY_FOLDER
from src.data.var_info import ROW_KEY_FILE_PATH


@pytest.fixture
def qapp_clipboard():
    """剪贴板先插一枚哨兵：断言"真的被写进去了"，而不是残留上一用例的值。"""
    clipboard = QApplication.clipboard()
    clipboard.setText("__sentinel__")
    return clipboard


def _click_segment(widget, qapp):
    """按真实路径点一下状态栏段：mousePressEvent → clicked 信号 → 开抽屉。"""
    local = QPointF(widget.rect().center())
    widget.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            local,
            QPointF(widget.mapToGlobal(local.toPoint())),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    qapp.processEvents()


def _drag_select(widget, start_x, end_x):
    """按住左键从 start_x 拖到 end_x（可以拖出控件外）：走 QLineEdit 自己的
    mousePressEvent/mouseMoveEvent，和真实拖选的代码路径一致。"""
    for kind, x, buttons in (
        (QEvent.Type.MouseButtonPress, start_x, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseMove, end_x, Qt.MouseButton.LeftButton),
    ):
        local = QPointF(x, widget.height() // 2)
        ev = QMouseEvent(
            kind,
            local,
            QPointF(widget.mapToGlobal(local.toPoint())),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        if kind is QEvent.Type.MouseButtonPress:
            widget.mousePressEvent(ev)
        else:
            widget.mouseMoveEvent(ev)


def _button_for(drawer, key):
    """按行取行尾按钮：同一次网格排布里，键标签与按钮的垂直中心一致。"""
    labels = [lb for lb in drawer.findChildren(QLabel) if lb.text() == key]
    assert labels, f"抽屉里没有「{key}」这一行"
    center_y = labels[0].geometry().center().y()
    buttons = [
        b
        for b in drawer.findChildren(QPushButton)
        if abs(b.geometry().center().y() - center_y) <= 2
    ]
    assert len(buttons) == 1, f"「{key}」行尾按钮数异常: {[b.text() for b in buttons]}"
    return buttons[0]


def _open(loaded_window, qapp):
    _click_segment(loaded_window._file_segment, qapp)
    drawer = loaded_window._file_info_drawer
    assert drawer is not None and drawer.isVisible(), "点左段应弹出文件信息抽屉"
    return drawer


def test_click_opens_drawer_above_status_bar(loaded_window, qapp):
    """抽屉贴在状态栏正上方、左右不越界，且比窗口矮得多。"""
    mw = loaded_window
    drawer = _open(mw, qapp)

    bar = mw.statusBar()
    bar_top = bar.mapToGlobal(QPoint(0, 0)).y()
    window_left = mw.mapToGlobal(QPoint(0, 0)).x()

    assert drawer.y() + drawer.height() <= bar_top, "抽屉压住了状态栏"
    gap = bar_top - (drawer.y() + drawer.height())
    assert 0 <= gap <= STATUS_DRAWER_EDGE_MARGIN, f"抽屉与状态栏脱开或贴死: gap={gap}"
    assert drawer.x() >= window_left, "抽屉左沿越出窗口"
    assert drawer.width() <= min(STATUS_DRAWER_WIDTH, bar.width())
    assert drawer.height() < mw.height() / 2, "抽屉太高会盖掉绘图区"


def test_drawer_shows_current_file(loaded_window, qapp):
    """内容取自当前 loader：文件名/路径/大小/格式/行数/耗时都在。"""
    mw = loaded_window
    drawer = _open(mw, qapp)
    values = dict(drawer._rows)

    assert values[KEY_FILE_NAME] == "e2e_demo.csv"
    assert values[ROW_KEY_FILE_PATH].endswith("e2e_demo.csv")
    assert values[KEY_FOLDER] == values[ROW_KEY_FILE_PATH][: -len("/e2e_demo.csv")]
    assert values["文件大小"].endswith(("B", "KB", "MB"))
    assert values["格式"].startswith("CSV 文本")
    assert values["数据行数"] == "50 行"
    assert values["加载耗时"].endswith("s")


def test_no_copy_all_button_and_no_validity_rows(loaded_window, qapp):
    """本轮定稿：撤掉「复制全部」按钮与有效性/变量数行，用例钉住防回潮。"""
    drawer = _open(loaded_window, qapp)

    assert {b.text() for b in drawer.findChildren(QPushButton)} == {"打开", "复制"}
    keys = {key for key, _ in drawer._rows}
    for banned in ("有效变量", "常量变量", "无效变量", "变量数", "有效性", "数据质量"):
        assert banned not in keys


def test_copy_button_writes_clipboard_and_broadcasts(loaded_window, qapp, qapp_clipboard):
    """行尾「复制」：路径行按配置的写法风格入剪贴板，反馈进状态栏消息区。"""
    mw = loaded_window
    drawer = _open(mw, qapp)
    expected = dict(drawer._rows)[ROW_KEY_FILE_PATH]

    _button_for(drawer, ROW_KEY_FILE_PATH).click()
    qapp.processEvents()

    assert qapp_clipboard.text() == expected
    assert mw._message_label.text() == "已复制「文件路径」"

    _button_for(drawer, KEY_FILE_NAME).click()
    qapp.processEvents()
    assert qapp_clipboard.text() == "e2e_demo.csv", "文件名不许被绝对化成别的东西"


def test_open_button_asks_system_file_manager(loaded_window, qapp, monkeypatch):
    """「打开」用 argv 列表调用系统命令，不拼 shell 字符串（路径里有空格也安全）。"""
    calls = []
    monkeypatch.setattr(
        "src.ui.widgets.status_drawer.subprocess.Popen",
        lambda args, **kwargs: calls.append(args),
    )
    drawer = _open(loaded_window, qapp)

    _button_for(drawer, KEY_FOLDER).click()
    qapp.processEvents()

    assert len(calls) == 1
    args = calls[0]
    assert isinstance(args, list) and all(isinstance(a, str) for a in args)
    assert args[0] in {"open", "explorer", "xdg-open"}, args
    assert any("e2e_demo.csv" in a for a in args[1:]), f"没把目标文件交给系统: {args}"


def test_open_button_when_file_moved_says_so(loaded_window, qapp, monkeypatch):
    """文件已不在原位置：不调用系统命令，给一句能读懂的原因。"""
    calls = []
    monkeypatch.setattr(
        "src.ui.widgets.status_drawer.subprocess.Popen",
        lambda args, **kwargs: calls.append(args),
    )
    mw = loaded_window
    drawer = _open(mw, qapp)
    drawer._file_path = "/nowhere/e2e_demo.csv"

    _button_for(drawer, KEY_FOLDER).click()
    qapp.processEvents()

    assert calls == []
    assert "文件已不在原位置" in mw._message_label.text()


def test_escape_closes_drawer(loaded_window, qapp):
    """Qt.Popup 自带 Esc / 点击外部关闭：抽屉不该需要用户找关闭按钮。"""
    drawer = _open(loaded_window, qapp)

    QApplication.sendEvent(
        drawer,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier),
    )
    qapp.processEvents()
    assert not drawer.isVisible()


def test_new_load_closes_drawer_and_reopen_shows_new_file(loaded_window, qapp, tmp_path):
    """加载一开始就收起抽屉；再次点开是**重新取数**，不残留旧快照。

    这里直接换 ``mw.loader`` 而不走"再点一次导入"：整条重载链（互斥锁、防抖
    刷新、模板恢复）有自己的用例，本用例只管抽屉的取数时机。
    """
    from src.data.loader import FastDataLoader
    from tests.fixtures.data_factory import make_simple_rows, write_csv

    mw = loaded_window
    drawer = _open(mw, qapp)

    # 列数与第一个文件一致：make_simple_rows 固定产 4 列，改列数会写成坏文件
    other = write_csv(
        tmp_path / "e2e_second.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(12),
    )
    mw.begin_load_feedback(str(other))
    assert not drawer.isVisible(), "加载期间抽屉必须收起（内容即将失效）"

    mw.loader = FastDataLoader(str(other))
    drawer = _open(mw, qapp)
    values = dict(drawer._rows)
    assert values[KEY_FILE_NAME] == "e2e_second.csv"
    assert values["数据行数"] == "12 行"


def test_click_without_data_only_broadcasts(main_window, qapp):
    """未加载文件：左段还在（占位），点开只给一句提示，不弹空抽屉。"""
    mw = main_window
    _click_segment(mw._file_segment, qapp)

    assert mw._file_info_drawer is None
    assert mw._message_label.text() == "尚未加载数据文件"


def test_click_while_loading_is_gated(main_window, qapp):
    """后台线程在跑时不开抽屉：旧 loader 随时会被释放。"""

    class _FakeThread:
        def isRunning(self):
            return True

    mw = main_window
    mw._thread = _FakeThread()

    _click_segment(mw._file_segment, qapp)

    assert mw._file_info_drawer is None
    assert "正在加载数据，请稍候再查看文件信息" in mw._message_label.text()


def test_drawer_does_not_widen_main_window(loaded_window, qapp):
    """回归护栏：长绝对路径不许抬高窗口最小宽度（状态栏上踩过同一坑）。"""
    mw = loaded_window
    before = mw.minimumWidth()

    _open(mw, qapp)
    qapp.processEvents()

    assert mw.minimumWidth() == before


def test_key_column_is_muted_and_right_aligned(loaded_window, qapp):
    """排印层级钉桩：键列压灰右对齐、值列沿用应用字号。

    对齐只能按 alignment 断言 —— 标签的几何位置由网格列宽决定，改对齐标志
    不会动矩形，量几何看不出来。值列"不写字号"是刻意的密度决定（写 13px 后
    长路径可见字符从 59 掉到 53），所以这里断言样式串里没有 font-size。
    """
    from src.ui import theme

    drawer = _open(loaded_window, qapp)
    labels = {lb.text(): lb for lb in drawer.findChildren(QLabel)}

    for key, _value in drawer._rows:
        key_label = labels.get(key)
        if key_label is None:
            continue  # 值与键同文的行（本抽屉没有）不参与断言
        assert key_label.alignment() & Qt.AlignmentFlag.AlignRight
        assert key_label.styleSheet() == theme.muted_text()

    value = drawer._labels[ROW_KEY_FILE_PATH]
    assert theme.TEXT_PRIMARY in value.styleSheet()
    assert "font-size" not in value.styleSheet()


def test_path_rows_are_read_only_scrollable_fields(loaded_window, qapp):
    """三行路径字段是只读 QLineEdit：有视口，框外的字才选得到。

    QLabel 没有视口 —— 文本按控件宽度排一次版，超出部分根本没参与排版，拖选
    永远走不到尾部。ClickFocus 是"别看起来能编辑"的补偿（不点就没有光标）。
    """
    from PySide6.QtWidgets import QLineEdit

    from src.ui.widgets.status_drawer import _CLIP_KEYS

    drawer = _open(loaded_window, qapp)

    for key, widget in drawer._labels.items():
        path_row = key in _CLIP_KEYS
        assert isinstance(widget, QLineEdit) is path_row, key
        if path_row:
            assert widget.isReadOnly(), key
            assert widget.focusPolicy() == Qt.FocusPolicy.ClickFocus, key
        # 两列共用同一档内边距，否则值列出现两条文字左沿
        assert "padding: 0 " in widget.styleSheet(), key


def test_hidden_part_of_long_path_is_laid_out_and_selectable(loaded_window, qapp):
    """放不下的那半截路径必须还在：视口滚得到、拖选/全选拿到的都是真路径。

    作者报的真问题，前两版都不行 —— elidedText 把 … 写进 text()，粘出来不是真
    路径；硬切的 QLabel 没有视口，框外的字符压根没参与排版，拖多远都只到可见的
    末字符。这里量的是"光标从文末走回头，位移比整框还宽"（实测 675 px > 框宽
    363 px）：看不见的字符也排了版，视口走得完全程。

    全程不 pump 事件：offscreen 弹窗拿不到键盘（"This plugin does not support
    grabbing the keyboard"），一次 processEvents 就发回 FocusOut，把选区清成
    -1/-1（实测），cocoa 上不会。所以本用例只钉数据口径，像素表现见 tmp 截图。
    """
    drawer = _open(loaded_window, qapp)
    field = drawer._labels[ROW_KEY_FILE_PATH]
    full = drawer._values[ROW_KEY_FILE_PATH]

    assert field.text() == full, "显示串不许被改写"
    if field.fontMetrics().horizontalAdvance(full) <= field.width():
        pytest.skip("这条测试路径本来就放得下，测不到视口")

    # 露尾＝文末光标的矩形落在框内。不写 `cursorPosition() == len(full)`：
    # 新建 QLineEdit 的光标本来就在末尾（实测 109），那是条空断言
    tail_x = field.cursorRect().x()
    assert 0 <= tail_x <= field.width(), f"默认没露出尾部：cursorX={tail_x}"
    field.setCursorPosition(0)
    assert tail_x - field.cursorRect().x() > field.width(), (
        "光标位移不足一屏：说明根本没有可滚的视口"
    )

    field.selectAll()
    assert field.selectedText() == full, "全选必须拿到完整真路径"
    assert "…" not in field.selectedText()

    field.end(False)  # 前面 setCursorPosition(0) 把视口滚到开头了，拨回用户看到的尾部姿态
    _drag_select(field, field.width() - 30, field.width() + 200)
    assert field.selectionEnd() == len(full), "拖过右缘选不到真正的结尾"
    assert field.selectedText() == full[field.selectionStart():]


def test_folder_row_keeps_only_open(loaded_window, qapp):
    """「所在文件夹」行只留「打开」：多一枚方片，值列就少 53 px（实测 368→315），
    而值列才是这扇抽屉要读的东西。作者定的取舍，钉住防回潮。"""
    drawer = _open(loaded_window, qapp)

    assert _button_for(drawer, KEY_FOLDER).text() == "打开"
    row_buttons = [
        b
        for b in drawer.findChildren(QPushButton)
        if abs(b.geometry().center().y()
               - drawer._labels[KEY_FOLDER].geometry().center().y()) <= 2
    ]
    assert [b.text() for b in row_buttons] == ["打开"]


def test_footer_states_how_to_close_and_get_full_values(loaded_window, qapp):
    """抽屉没有关闭按钮，页脚必须把"怎么收起"和"被截断怎么办"写在脸上。"""
    drawer = _open(loaded_window, qapp)
    footer = [
        lb.text()
        for lb in drawer.findChildren(QLabel)
        if "Esc" in lb.text() or "复制" in lb.text()
    ]

    assert footer, "页脚提示不见了"
    assert any("Esc" in text for text in footer)
    assert any("复制" in text for text in footer)


def test_body_scroll_does_not_paint_its_own_background(loaded_window, qapp):
    """主体滚动区必须透底：cocoa 真机上它自己铺一层窗口灰，实测同一扇抽屉
    "页脚 #FFFFFF、主体 #ECECEC"。offscreen/Fusion 复现不出来（所以这条在
    CI 上只是防拆钉），但样式串就是那台机器上唯一压得住灰的写法。
    """
    from PySide6.QtWidgets import QScrollArea

    drawer = _open(loaded_window, qapp)
    scroll = next(s for s in drawer.findChildren(QScrollArea) if "transparent" in s.styleSheet())

    sheet = scroll.styleSheet()
    assert "QScrollArea { background: transparent" in sheet
    # 视口是滚动区的直接子 QWidget，漏掉这条链就只透边框、内容照旧铺灰
    assert "QScrollArea > QWidget > QWidget { background: transparent; }" in sheet


def test_action_buttons_are_platform_native(loaded_window, qapp):
    """行尾动作按钮（打开 / 复制）一律不带 QSS，交给平台绘制。

    一带样式表就被 QStyleSheetStyle 接管、退出平台绘制，与主窗口顶栏按钮成了
    两套灰阶（作者定：普通按钮回原生，见 src/ui/theme.py 的模块 docstring）。
    类也得对：QPushButton 而非 QToolButton —— 两个类的"原生"长得不一样。
    """
    drawer = _open(loaded_window, qapp)
    buttons = drawer.findChildren(QPushButton)

    assert buttons, "行尾动作按钮不见了"
    assert all(b.styleSheet() == "" for b in buttons), "有按钮自带样式表"
    assert not drawer.findChildren(QToolButton), "抽屉按钮退回 QToolButton 了"
