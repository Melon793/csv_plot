"""e2e：状态栏左段 → 文件信息上翻抽屉（P0-2 Step B）。

覆盖三件事：抽屉真能弹在状态栏上方（几何量测，不是"看起来对"）、内容就是
当前文件的真实信息（换文件后不许残留）、以及三个动作出口（复制入剪贴板、
打开走系统文件管理器且不拼 shell、Esc/加载/换文件都能收起）。

字段口径本身在 tests/unit/data/test_file_info.py 里钉；这里只验 UI 层。
"""

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel, QToolButton

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


def _button_for(drawer, key, text=None):
    """按行取行尾按钮：同一次网格排布里，键标签与按钮的垂直中心一致。

    ``text`` 用来在一行有多个动作时点名（所在文件夹行是 复制 + 打开）。
    """
    labels = [lb for lb in drawer.findChildren(QLabel) if lb.text() == key]
    assert labels, f"抽屉里没有「{key}」这一行"
    center_y = labels[0].geometry().center().y()
    buttons = [
        b
        for b in drawer.findChildren(QToolButton)
        if abs(b.geometry().center().y() - center_y) <= 2
        and (text is None or b.text() == text)
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

    assert {b.text() for b in drawer.findChildren(QToolButton)} == {"打开", "复制"}
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

    _button_for(drawer, KEY_FOLDER, "打开").click()
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

    _button_for(drawer, KEY_FOLDER, "打开").click()
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


def test_only_path_rows_get_the_hover_field(loaded_window, qapp):
    """三行路径字段挂 hover 框（暗示"这块能拉选"），数值行不挂。

    框是 FieldLabel 在 paintEvent 里画的（样式表 :hover 合成事件下实测涂不出
    像素，见该类文档串），所以既断言类型也断言"投一个 Enter 之后真的多出墨"。
    """
    from src.ui.widgets.status_drawer import _COPY_KEYS, FieldLabel

    drawer = _open(loaded_window, qapp)

    for key, label in drawer._labels.items():
        hoverable = key in _COPY_KEYS
        assert isinstance(label, FieldLabel) is hoverable, key
        # 透明描边 + 内边距对所有行常驻，否则值列会出现两条文字左沿
        assert "padding: 0 4px" in label.styleSheet(), key

    target = drawer._labels[ROW_KEY_FILE_PATH]
    before = target.grab().toImage()
    # 必须投真的 QEnterEvent：投一个类型是 Enter 的裸 QEvent，QWidget::event 会
    # 把它 static_cast 成 QEnterEvent，实测直接 SIGSEGV（Leave 不转，裸事件安全）
    from PySide6.QtGui import QEnterEvent

    QApplication.sendEvent(
        target, QEnterEvent(QPointF(4, 4), QPointF(4, 4), QPointF(4, 4))
    )
    qapp.processEvents()
    after = target.grab().toImage()
    painted = sum(
        1
        for x in range(after.width())
        for y in range(after.height())
        if before.pixelColor(x, y) != after.pixelColor(x, y)
    )
    # 框铺满整块（值列宽 × 行高，数千像素），只画一条边不可能到这个数
    assert painted >= 2000, f"hover 框没画出来：只有 {painted} 个像素变了"

    QApplication.sendEvent(target, QEvent(QEvent.Type.Leave))
    qapp.processEvents()
    back = target.grab().toImage()
    assert sum(
        1
        for x in range(back.width())
        for y in range(back.height())
        if back.pixelColor(x, y) != before.pixelColor(x, y)
    ) == 0, "离开后框没收回"


def test_long_path_is_clipped_not_elided(loaded_window, qapp, qapp_clipboard):
    """放不下的路径只许硬切：拉选拿到的必须是真路径里的连续片段。

    回归的是上一版：用 elidedText 裁中间后，省略号是写进 text() 的真字符，会被
    一起拉选复制 —— 作者实测（Win11）粘出来 "Users/demo/Data…/demo_50pts"，
    不是任何真实路径。现在改成硬切 + FieldLabel 右缘渐隐。
    """
    from PySide6.QtGui import QMouseEvent

    drawer = _open(loaded_window, qapp)
    label = drawer._labels[ROW_KEY_FILE_PATH]
    full = drawer._values[ROW_KEY_FILE_PATH]

    assert label.text() == full, "显示串不许被改写"
    if label.fontMetrics().horizontalAdvance(full) <= label.width():
        pytest.skip("这条测试路径本来就放得下，测不到截断")
    assert label._is_clipped(), "该出现渐隐时没判定为裁切"
    assert not drawer._labels[KEY_FILE_NAME]._is_clipped(), "短值不该被判定为裁切"

    def drag(kind, x):
        pos = QPointF(x, label.height() / 2)
        return QMouseEvent(
            kind,
            pos,
            label.mapToGlobal(pos.toPoint()),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    QApplication.sendEvent(label, drag(QEvent.Type.MouseButtonPress, 6))
    QApplication.sendEvent(label, drag(QEvent.Type.MouseMove, label.width() - 6))
    QApplication.sendEvent(
        label, drag(QEvent.Type.MouseButtonRelease, label.width() - 6)
    )
    qapp.processEvents()

    selected = label.selectedText()
    assert selected, "拉选没选中任何字符"
    assert "…" not in selected
    assert selected in full, f"选中片段不是真路径的一部分: {selected!r}"

    _button_for(drawer, ROW_KEY_FILE_PATH).click()
    qapp.processEvents()
    assert qapp_clipboard.text() == full or "…" not in qapp_clipboard.text()


def test_folder_row_copies_as_well_as_opens(loaded_window, qapp, qapp_clipboard):
    """所在文件夹行必须两个动作都有：只有「打开」时，被裁掉的尾部没有出口。"""
    from src.core.config import PATH_COPY_QUOTE, PATH_COPY_STYLE
    from src.utils.paths import format_for_copy

    drawer = _open(loaded_window, qapp)
    expected = format_for_copy(
        drawer._values[KEY_FOLDER], PATH_COPY_STYLE, PATH_COPY_QUOTE
    )

    _button_for(drawer, KEY_FOLDER, "复制").click()
    qapp.processEvents()

    assert qapp_clipboard.text() == expected
    assert "…" not in qapp_clipboard.text()
    assert loaded_window._message_label.text() == "已复制「所在文件夹」"


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
