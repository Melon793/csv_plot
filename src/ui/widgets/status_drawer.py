"""状态栏上翻抽屉（P0-2）：贴在状态栏上方的轻量弹层。

外壳 :class:`StatusDrawer` 只管"贴在状态栏正上方 + 生命周期"，内容通过
``set_body`` 注入：文件信息抽屉（Step B）放一张键值表，x 轴时间基准抽屉
（Step C）放一组输入控件。

为什么用 ``Qt.WindowType.Popup`` 而不是自绘遮罩或普通子控件：

- Esc 关闭、点击外部关闭、随主窗口失焦收起 —— 全部是 Qt popup 的既有行为，
  自己实现要挂全局事件过滤器，那是本仓踩过的重入坑位。
- 不占用 ``QMainWindow`` 的中央布局：抽屉若作为中央区控件插入，会把绘图区
  往上顶，等于把 P0-2 已经论证过的"常驻行占高"换个形式复现。

代价是抽屉打开期间主窗口被它独占鼠标。这符合"看一眼信息 / 改一个参数"的
短交互定位，因此接受。
"""

from __future__ import annotations

import os
import subprocess
import sys

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    PATH_COPY_QUOTE,
    PATH_COPY_STYLE,
    STATUS_DRAWER_EDGE_MARGIN,
    STATUS_DRAWER_MIN_HEIGHT,
    STATUS_DRAWER_WIDTH,
    STATUS_DRAWER_WIDTH_AXIS,
    X_AXIS_FACTOR_MAX,
    X_AXIS_FACTOR_MIN,
    X_AXIS_FREQUENCY_PRESETS,
)
from src.core.logger import get_logger
from src.data.file_info import KEY_FILE_NAME, KEY_FOLDER, build_file_rows
from src.data.var_info import ROW_KEY_FILE_PATH
from src.ui import theme
from src.utils.paths import format_for_copy

logger = get_logger(__name__)

#: 值会被列宽裁掉、需要一键取全文的两行
_COPY_KEYS = {KEY_FILE_NAME, ROW_KEY_FILE_PATH}
#: 值是目录、交给系统文件管理器打开的那行
_REVEAL_KEYS = {KEY_FOLDER}
#: 值可能被列宽裁掉的三行：只有它们用只读 QLineEdit 当值控件（有视口，框外的字
#: 才拉选得到）。与 _COPY_KEYS 刻意不同：作者定过「所在文件夹」只留「打开」，
#: 所以那一行要字段框但不要复制
_CLIP_KEYS = {KEY_FILE_NAME, ROW_KEY_FILE_PATH, KEY_FOLDER}
#: 路径缺失时 file_info 给的占位符，复制它等于复制一个无意义的横杠
_PLACEHOLDER = "-"


def reveal_in_file_manager(file_path: str, folder: str) -> str:
    """在系统文件管理器里定位文件（选中它，而不是只打开目录）。

    返回空串表示已把请求交给系统，非空是失败原因。三平台各一支：macOS
    ``open -R``、Windows ``explorer /select,`` 支持"打开目录并选中该文件"，
    Linux 的 ``xdg-open`` 没有选中语义，只能开目录。

    参数一律走 argv 列表、不拼 shell 字符串：文件路径里的空格、``&``、``$``
    都是常事（本仓 paths 模块为此专门做过引号策略），拼进一条命令串就是命令
    注入与断词两个坑。
    """
    target = file_path or folder
    if not target:
        return "没有可打开的路径"
    if not os.path.exists(target):
        return "文件已不在原位置"

    if sys.platform == "darwin":
        args = ["open", "-R", file_path] if file_path else ["open", folder]
    elif sys.platform == "win32":
        # explorer 的 /select 必须是**一个**参数且逗号后无空格；它成功时也常
        # 返回非零退出码，因此不看返回码
        args = ["explorer", f"/select,{file_path}"] if file_path else ["explorer", folder]
    else:
        args = ["xdg-open", folder or file_path]

    try:
        # 不 wait：launch 型子进程自己会退，等它等于把 UI 线程挂在 Finder 上
        subprocess.Popen(args, close_fds=True)
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("打开所在文件夹失败: %s", args, exc_info=True)
        return f"无法调用系统文件管理器：{e}"
    return ""


class StatusDrawer(QFrame):
    """抽屉外壳：贴在状态栏正上方的 Popup 面板。"""

    #: 宽度上限。两扇抽屉的内容宽度差得远，各自定一档（见 config 的实测说明）
    width_cap = STATUS_DRAWER_WIDTH

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        # 白底 + 发丝描边：与「修改布局」网格选择器共用一份色板（src/ui/theme.py）。
        # 选择器必须带 id —— 裸写 ``QFrame {...}`` 会连带命中内部的 QScrollArea，
        # 给它也画出一层边框。
        self.setObjectName("statusDrawer")
        self.setStyleSheet(theme.panel_style("QFrame#statusDrawer"))
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(12, 10, 12, 10)
        self._root.setSpacing(6)
        self._anchor_bar = None

    def _footer(self, hint: str, actions: list | None = None) -> QWidget:
        """分隔线 + 一行次级提示（右侧可挂动作按钮），照网格选择器的页脚。

        抽屉没有标题栏也没有关闭按钮，"怎么关掉它"必须写在脸上；值被省略号
        截断时，取全文的出口在哪一句也要说清。

        ``actions`` 放页脚而不是内容区：网格选择器的「取消」就在这一行，两处
        同一位置才不用重新学；更实际的原因是 x 轴抽屉的确认键原先与预览行同
        排，把大号预览挤到裁字（cocoa 实测可用 228 px / 需要 258 px）。
        """
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setLineWidth(0)
        line.setFixedHeight(1)
        line.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        line.setStyleSheet(theme.separator_style())
        layout.addWidget(line)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        label = QLabel(hint)
        label.setStyleSheet(theme.muted_text())
        # Ignored：页脚文案是整行里最长的一句，不放开就会把抽屉顶得比窗口还宽
        label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        row.addWidget(label)
        # 刻意不加 addStretch：label 已是 Ignored（可扩展、最小 0），它会吃掉
        # 全部余量把按钮推到右端。再放一个 stretch 项就和 label 抢余量，长提示
        # 反被挤到裁字
        for button in actions or []:
            row.addWidget(button)
        layout.addLayout(row)
        return box

    def _chip(self, text: str, tip: str = "") -> QPushButton:
        """抽屉里的小按钮。两个刻意的选择：

        - **不带样式表**（作者定，见 ``theme`` 模块 docstring）：一带 QSS 就被
          ``QStyleSheetStyle`` 接管、退出平台绘制。
        - **用 ``QPushButton`` 而不是 ``QToolButton``**：两者都是"原生"，但各自
          的原生长得不一样 —— 实测同一份文字在 macOS 下 QPushButton 是 56x32 的
          aqua 白胶囊、QToolButton 是 36x22 的灰色斜面方块（Windows 75x26 vs
          33x18，Fusion 80x24 vs 34x19）。主窗口顶栏全是 QPushButton，抽屉跟着
          用同一类才谈得上"和系统一致"。

        尺寸策略钉成 Fixed：QPushButton 的水平策略是 Minimum（可长大），而页脚
        那行没有 addStretch（label 是 Ignored，见 ``_footer``），不钉死就会被
        拉去吃掉整行余量 —— 实测「恢复默认」「应用」会被撑到一屏宽。
        """
        button = QPushButton()
        button.setText(text)
        button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        if tip:
            button.setToolTip(tip)
        return button

    # -- 内容 ---------------------------------------------------------------

    def set_body(self, body: QWidget) -> None:
        """替换主体内容。外壳不感知内容是什么。

        调用方负责在**隐藏状态**下换 body：可见时换子控件后布局不会立刻重算，
        紧接着量的 ``sizeHint()`` 会拿到塌陷值（实测 409 → 18），抽屉就按错误
        高度弹出来。
        """
        while self._root.count():
            item = self._root.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._root.addWidget(body)
        self._root.activate()

    # -- 定位 ---------------------------------------------------------------

    def open_above(self, status_bar) -> bool:
        """顶到状态栏正上方并显示。返回 False 表示上方放不下。"""
        self._anchor_bar = status_bar
        window = status_bar.window()
        if window is None:
            return False

        gap = STATUS_DRAWER_EDGE_MARGIN
        bar_top_left = status_bar.mapToGlobal(QPoint(0, 0))
        window_top = window.mapToGlobal(QPoint(0, 0)).y()
        room = bar_top_left.y() - window_top - 2 * gap
        if room < STATUS_DRAWER_MIN_HEIGHT:
            self.hide()
            return False

        # 先定宽再量高：值列的省略与行数都按最终宽度算
        width = min(self.width_cap, max(200, status_bar.width() - 2 * gap))
        self.resize(width, min(self.sizeHint().height(), room))
        self.move(bar_top_left.x() + gap, bar_top_left.y() - self.height() - gap)
        self.show()
        self.raise_()
        return True

    def reposition(self) -> None:
        """跟随主窗口移动/缩放重新贴一次（浮在原地比回到状态栏上方更奇怪）。"""
        if self._anchor_bar is not None and self.isVisible():
            self.open_above(self._anchor_bar)

    # -- 反馈 ---------------------------------------------------------------

    def _notify(self, text: str, level: str = "info") -> None:
        """把抽屉里的反馈送进主窗口状态栏消息区（抽屉自己不放状态栏）。

        用 getattr 而不是直接 ``parentWidget()._broadcast``：单元测试里抽屉的
        parent 可能是任意 QWidget，缺方法时这条辅助反馈该静默失效。
        """
        broadcast = getattr(self.parentWidget(), "_broadcast", None)
        if broadcast is not None:
            broadcast(text, level=level)

    def _mw(self):
        """抽屉的主人（主窗口）：parent 就是它，不另存引用免得失效。"""
        owner = self.parentWidget()
        return owner if hasattr(owner, "loader") else None


class _BodyScroll(QScrollArea):
    """按内容报 sizeHint 的滚动区。

    QAbstractScrollArea 的 sizeHint 与内容无关（它只给一个固定值），抽屉照它
    排版就不知道该长多高 —— 实测要么裁掉最后几行，要么在明明放得下时也滚出
    滚动条。这里把高度改成内容需要的高度，宽度仍走默认（值列允许被省略，
    不该反过来决定抽屉宽度）。

    只改 sizeHint 不改 minimumSizeHint：矮窗口放不下时仍要能压缩，压缩的代价
    由滚动条承担。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # 不写这一句，cocoa 上视口会自己铺一层窗口灰，抽屉就成了"页脚白、主体灰"
        self.setStyleSheet(theme.transparent_scroll_style())

    def sizeHint(self):
        hint = super().sizeHint()
        widget = self.widget()
        if widget is not None:
            hint.setHeight(widget.sizeHint().height())
        return hint


class FileInfoDrawer(StatusDrawer):
    """左段抽屉：文件级信息，外加复制与「打开所在文件夹」。

    内容每次打开都从 loader 重建 —— 快照成本可以忽略（只读元信息），换来的
    是一定不会看到上一个文件的信息。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list = []
        self._values: dict = {}
        self._labels: dict = {}  # key → 值列 QLabel，重算省略号与测试都要用它
        self._file_path = ""
        self._folder = ""

    # -- 打开 ---------------------------------------------------------------

    def open_for(self, mw, elapsed_s: float | None = None) -> bool:
        """从主窗口取 loader 元信息并打开抽屉。"""
        loader = getattr(mw, "loader", None)
        rows = build_file_rows(loader, elapsed_s=elapsed_s)
        if not rows:
            return False

        self._rows = rows
        self._values = dict(rows)
        # 「打开」动作复用表里那两行的**原值**（不是屏上显示串）：按钮打开的东西
        # 必须与用户看到的那条路径同源，另算一遍就会出现"显示 A 打开 B"；显示串
        # 可能被裁成 …，原值与它同源，只是少了中间一段
        self._file_path = self._plain(ROW_KEY_FILE_PATH)
        self._folder = self._plain(KEY_FOLDER)

        self.hide()  # 见 set_body：可见状态下换 body 会把 sizeHint 量塌
        self.set_body(self._build_body())
        return self.open_above(mw.statusBar())

    def _plain(self, key: str) -> str:
        """取该行的真实路径，占位符与空值一律当"没有"。"""
        value = self._values.get(key, "")
        return "" if value == _PLACEHOLDER else value

    # -- 内容构建 -----------------------------------------------------------

    def _build_body(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._build_table(), 1)
        layout.addWidget(
            self._footer(
                "点抽屉外任意处或按 Esc 收起 · 路径行可拉选 · 行尾「复制」取全文"
            )
        )
        return body

    def _build_table(self) -> QScrollArea:
        """键值表：键列右对齐 / 值列吃满剩余宽度 / 行尾动作列。

        键列右对齐是这套排印里收益最大的一处：左对齐时各键右边缘参差，值列
        跟着参差（网格选择器那套量法测得视觉间隙极差 39 px），右对齐后归零。

        套 QScrollArea 是给矮窗口兜底：行数固定而可用高度不够时，宁可出
        滚动条，也不能把最后几行裁掉。
        """
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(1, 1)

        self._labels = {}
        for row, (key, value) in enumerate(self._rows):
            grid.addWidget(self._key_label(key), row, 0)
            label = (
                self._path_field(value)
                if key in _CLIP_KEYS
                else self._value_label(value)
            )
            self._labels[key] = label
            grid.addWidget(label, row, 1)
            button = self._action_button(key, value)
            if button is not None:
                grid.addWidget(button, row, 2)

        content = QWidget()
        content.setLayout(grid)

        scroll = _BodyScroll()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _key_label(key: str) -> QLabel:
        label = QLabel(key)
        label.setStyleSheet(theme.muted_text())
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        return label

    @staticmethod
    def _value_label(value: str) -> QLabel:
        """普通值：一行裸文本（没有拉选需求，全文出口是 tooltip）。"""
        label = QLabel(value)
        # 「格式」行的值没有硬上界（工作表名 + 不定长 notes，实测一条 40 个 CJK
        # 字符 ≈ 560 px，超过值列净宽 441 px），而它既不在 _CLIP_KEYS（没有视口）
        # 也没有行尾按钮 —— tooltip 是它唯一的全文出口
        label.setToolTip(value)
        # 字段框占位与路径字段同一套内边距，否则整列出现两条文字左沿
        label.setStyleSheet(theme.field_style())
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        # Ignored：QLabel 的 minimumSizeHint 等于整串文本宽度，长值会把抽屉顶得
        # 比窗口还宽（状态栏上同一件事已修过一次，见 _ElideStatusBar）
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        return label

    @staticmethod
    def _path_field(value: str) -> QLineEdit:
        """路径字段：只读 QLineEdit，带视口，所以**框外的字也选得到**。

        QLabel 没有视口（文本按控件宽度排一次版，超出部分根本没参与排版），
        拖选永远拿不到尾部；实测同一条 103 字符路径，只读 QLineEdit 的 home/end
        两态差 12408 个像素，QLabel 是 0。

        - ``ClickFocus``：不点就没有光标，尽量不像"可编辑的输入框"；点进去之后
          Ctrl+A / Ctrl+C / 方向键都能走到尾部。
        - 露出尾部：路径里有信息的是文件名那一截，所以把视口滚到末尾。新建的
          QLineEdit 光标本身就在末尾（实测 cursorPosition==len 且 cursorRect 已在
          框内），显式 ``end()`` 只是不依赖 Qt 未承诺的默认可见位置；停在 0 那侧
          露出来的是 /Users/... 那截噪音。
        - 刻意不改写显示串：省略号是写进 ``text()`` 的真字符，会被一起拉选复制，
          作者实测粘出来 "Users/demo/Data…/demo_50pts" 不是任何真实路径。
        """
        field = QLineEdit(value)
        field.setReadOnly(True)
        field.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        field.setStyleSheet(theme.path_field_style())
        field.setToolTip(value)
        field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        field.end(False)  # 视口滚到末尾，露文件名那一截
        return field

    def _action_button(self, key: str, value: str) -> QPushButton | None:
        """行尾动作：文件名与路径行给「复制」，所在文件夹行给「打开」。

        一行只放一个动作：动作列每多一枚方片，值列就少 53 px（实测 368 → 315），
        而值列正是这扇抽屉要读的东西。作者据此把「所在文件夹」的「复制」撤了，
        那一行的全文出口是 tooltip（以及上一行「文件路径」的复制）。
        """
        if key in _REVEAL_KEYS:
            button = self._chip("打开", "在系统文件管理器里显示该文件")
            button.clicked.connect(self._on_open_folder)
            return button
        if key in _COPY_KEYS:
            button = self._chip("复制", f"复制{key}（完整值）")
            button.clicked.connect(
                lambda _checked=False, k=key, v=value: self._on_copy(k, v)
            )
            return button
        return None

    # -- 动作 ---------------------------------------------------------------

    def _on_copy(self, key: str, value: str) -> None:
        """单行复制：值原样入剪贴板，路径行按配置的写法风格与引号策略转写。

        与变量信息窗口同一口径（见 ``VarInfoPage._on_copy_field``）：复制出来
        的东西要能直接粘进报告单元格或命令行，所以不拼属性名。
        """
        if not value or value == _PLACEHOLDER:
            self._notify(f"「{key}」没有可复制的内容", level="warn")
            return
        if key == ROW_KEY_FILE_PATH:
            # 只有整条路径才转写法/包引号：format_for_copy 内部会 display_path，
            # 喂「文件名」会被绝对化成"<当前目录>/a.csv"这种看着能用实则错的东西
            value = format_for_copy(value, PATH_COPY_STYLE, PATH_COPY_QUOTE)
        QApplication.clipboard().setText(value)
        self._notify(f"已复制「{key}」")

    def _on_open_folder(self) -> None:
        error = reveal_in_file_manager(self._file_path, self._folder)
        if error:
            self._notify(f"打开所在文件夹失败：{error}", level="warn")
        else:
            self._notify("已在文件管理器中显示该文件")


class XAxisDrawer(StatusDrawer):
    """中段抽屉：x 轴时间基准（采样频率 / 系数 / 偏移）。

    采样频率是主输入：现场只知道"这台设备 100 Hz 采的"，不会去算 1/100，
    而顶部「时间修正」对话框只给系数与偏移，正是这一步要补的缺口。系数由频率
    在全精度下反推后写入全局 factor；要手写系数必须先勾"手写"，勾上之后频率
    框退成只读回显 —— 两个框同时可编辑就会出现"改一个另一个跟不跟"的歧义，
    以及 3 Hz ↔ 0.333333 这类往返舍入漂移（实测 3M 点会漂出约 1 s）。
    """

    #: 这扇抽屉最紧的一行是页脚（提示 + 恢复默认 + 应用），实测 440 就够；
    #: 沿用文件抽屉的 600 会在右侧白留 ~160 px 空
    width_cap = STATUS_DRAWER_WIDTH_AXIS

    #: 当前档的标记：只用圆点，不加粗（作者定）。选 U+25CF 而不是 U+2022 ——
    #: 实测前者在本机两套 style 下字宽 12 px、墨迹足，后者只有 6 px，
    #: 孤零零一个小点摆在按钮里太弱
    _MARK = "●"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._syncing = False
        self._by_factor = False
        self._preset_buttons: list = []
        self._marked_index = -1  # 当前被标记的档位下标，-1 = 哪档都不对
        self._build()

    # -- 打开 ---------------------------------------------------------------

    def open_for(self, mw) -> bool:
        """用当前的全局 factor/offset 起算并打开抽屉。"""
        self._load_from(mw.factor, mw.offset)
        return self.open_above(mw.statusBar())

    # -- 构建 ---------------------------------------------------------------

    def _build(self) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(2, 1)

        self.freq_spin = QDoubleSpinBox()
        # 频率与系数互为倒数，两框的范围也就互为端点（见 config 的同源说明）
        self.freq_spin.setRange(1.0 / X_AXIS_FACTOR_MAX, 1.0 / X_AXIS_FACTOR_MIN)
        self.freq_spin.setDecimals(6)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(self._on_freq_edited)

        self.factor_spin = QDoubleSpinBox()
        self.factor_spin.setRange(X_AXIS_FACTOR_MIN, X_AXIS_FACTOR_MAX)
        self.factor_spin.setDecimals(6)
        self.factor_spin.valueChanged.connect(self._on_factor_edited)

        self.manual_check = QCheckBox("手写系数")
        self.manual_check.setStyleSheet(theme.muted_text())
        self.manual_check.setToolTip(
            "勾上后直接填每样本间隔（秒），频率框退成回显。\n"
            "默认按采样频率输入 —— 频率是 10 的幂，系数是小数，手打更容易错一位。"
        )
        self.manual_check.toggled.connect(self._on_authority_changed)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1e9, 1e9)
        self.offset_spin.setDecimals(6)
        self.offset_spin.valueChanged.connect(self._refresh_preview)

        grid.addWidget(self._field_label("采样频率"), 0, 0)
        grid.addWidget(self.freq_spin, 0, 1)
        # 档位单独一行，且从**第 1 列**起（与频率框左沿对齐）：它们就是"给上面
        # 那个框填值"的快捷入口，对齐到框下面才读得出这层从属关系。
        # 不再挤在频率框右侧 —— QPushButton 在 Fusion 下有 ~80 px 最小宽度，四枚
        # 329 px 放右侧会把抽屉自然宽从 450 顶到 600，远超这扇自己的上限
        # STATUS_DRAWER_WIDTH_AXIS=440（mac 下 233 px 刚好卡住，但布局不能只按
        # 一台机器定）。
        grid.addLayout(self._build_presets(), 1, 1, 1, 2)
        grid.addWidget(self._field_label("系数"), 2, 0)
        grid.addWidget(self.factor_spin, 2, 1)
        grid.addWidget(self.manual_check, 2, 2, Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(self._field_label("偏移"), 3, 0)
        grid.addWidget(self.offset_spin, 3, 1)

        # 结果区照网格选择器的双层写法：大号"会变成什么" + 小号"改动的范围"。
        # 整行通栏：原先只占前两列（输入区宽度），18px 的预览被挤到裁字
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(2)
        self.preview = QLabel()
        self.preview.setStyleSheet(theme.result_text())
        self.preview.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        texts.addWidget(self.preview)
        scope = QLabel("应用后所有已开子图的横轴同步重算")
        scope.setStyleSheet(theme.muted_text())
        scope.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        texts.addWidget(scope)
        grid.addLayout(texts, 4, 0, 1, 3)

        self.reset_btn = self._chip("恢复默认", "填回默认系数并立即应用（撤销本次修正）")
        self.reset_btn.clicked.connect(self._on_reset)
        self.apply_btn = self._chip(
            "应用",
            "按上面的输入重算所有子图的横轴（焦点不在输入框里时，按 Enter 同效）",
        )
        self.apply_btn.clicked.connect(self._on_apply)

        body = QWidget()
        outer = QVBoxLayout(body)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addLayout(grid)
        outer.addWidget(
            self._footer(
                "点抽屉外任意处或按 Esc 收起（不落地）",
                actions=[self.reset_btn, self.apply_btn],
            )
        )
        self.set_body(body)
        self._apply_authority()

    def keyPressEvent(self, event) -> None:
        """Enter = 应用，与网格选择器同一套键盘模型（预览 → 确认，Esc 取消）。

        刻意用 ``keyPressEvent`` 而不是 ``QShortcut``：popup 的键盘 grab 在
        offscreen 下根本不存在（实测 ``hasFocus()`` 恒为 False），shortcut 的
        focus 作用域永不命中 —— 写了既验证不了也维护不了。焦点停在输入框里时
        回车由 spinbox 自己吃掉，那时点「应用」仍然在原地；不做"任何位置的回车
        都算应用"这种承诺。
        """
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_apply()
            return
        super().keyPressEvent(event)

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(theme.muted_text())
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        return label

    def _build_presets(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        for hz, label in X_AXIS_FREQUENCY_PRESETS:
            # 单位取 hz 而不是 label：label 已自带 "Hz"，再拼一次就成了 "1Hz Hz"
            button = self._chip(label, f"采样频率 {hz:g} Hz（系数 {1.0 / hz:g}）")
            button.clicked.connect(
                lambda _checked=False, value=hz: self._pick_preset(value)
            )
            row.addWidget(button)
            self._preset_buttons.append(button)

        # 四枚钉成等宽：标记是往文字前面加一个圆点，不钉的话每换一档整行都抖
        # 一次（实测 macOS 下 55 → 71 px）。宽度按"被标记那一枚"取最大值，
        # 所以抖动一次都不会有
        probes = [
            self._chip(self._marked_text(label))
            for _hz, label in X_AXIS_FREQUENCY_PRESETS
        ]
        pin = max(probe.sizeHint().width() for probe in probes)
        for probe in probes:
            probe.deleteLater()
        for button in self._preset_buttons:
            button.setFixedWidth(pin)

        row.addStretch(1)
        return row

    @staticmethod
    def _marked_text(label: str) -> str:
        return f"{XAxisDrawer._MARK} {label}"

    def _refresh_preset_states(self) -> None:
        """把与当前基准相等的那一档前缀一个圆点，其余回到普通文字。

        哪一档都不等（手写出来的 0.0015 s 之类）就没有任何一档被标记，这本身
        也是信息。

        刻意**不借 checked 态**：实测 macOS style 把选中按钮的文字画成白字压在
        白底上（深色像素 266 → 0），palette 改 ButtonText / Button 四种设法全都
        救不回来，字直接看不见；Fusion 与 Windows 虽可读，但同一处三平台两种
        结果，不如统一用文字标。也不加粗 —— 作者定：单一个圆点就够认。
        """
        factor, _offset = self.candidate()
        hz = 1.0 / factor if factor else 0.0
        marked = -1
        for index, (preset_hz, _label) in enumerate(X_AXIS_FREQUENCY_PRESETS):
            if abs(preset_hz - hz) <= 1e-9 * max(preset_hz, hz):
                marked = index
                break
        if marked == self._marked_index:
            # 每敲一个字符都会刷一次，绝大多数时候落在同一档，重设文字是白折腾
            return
        self._marked_index = marked
        for index, button in enumerate(self._preset_buttons):
            label = X_AXIS_FREQUENCY_PRESETS[index][1]
            button.setText(self._marked_text(label) if index == marked else label)

    # -- 输入同步 -----------------------------------------------------------

    def _pick_preset(self, hz: float) -> None:
        self.freq_spin.setValue(hz)

    def _on_freq_edited(self, value: float) -> None:
        if self._syncing or self._by_factor:
            return
        self._syncing = True
        try:
            self.factor_spin.setValue(1.0 / value if value else 1.0)
        finally:
            self._syncing = False
        self._refresh_preview()

    def _on_factor_edited(self, value: float) -> None:
        if self._syncing or not self._by_factor:
            return
        self._syncing = True
        try:
            self.freq_spin.setValue(1.0 / value if value else 1.0)
        finally:
            self._syncing = False
        self._refresh_preview()

    def _on_authority_changed(self, checked: bool) -> None:
        self._by_factor = checked
        self._apply_authority()
        self._refresh_preview()

    def _apply_authority(self) -> None:
        """哪一框可编辑，哪一框就是权威值，另一框只回显。"""
        self.freq_spin.setEnabled(not self._by_factor)
        for button in self._preset_buttons:
            button.setEnabled(not self._by_factor)
        self.factor_spin.setEnabled(self._by_factor)

    def _load_from(self, factor: float, offset: float) -> None:
        self._syncing = True
        try:
            self.factor_spin.setValue(factor)
            self.freq_spin.setValue(1.0 / factor if factor else 1.0)
            self.offset_spin.setValue(offset)
        finally:
            self._syncing = False
        self._refresh_preview()

    def candidate(self) -> tuple:
        """当前输入对应的 (factor, offset)。

        由频率反推时**不取**频率框的显示值再四舍五入：spinbox 只有 6 位小数，
        3 Hz 会变成 0.333333，3M 点横轴就漂出约 1 秒。1/value 在 double 下
        即全精度，落进全局 factor 的仍是精确倒数。
        """
        if self._by_factor:
            factor = self.factor_spin.value()
        else:
            hz = self.freq_spin.value()
            factor = 1.0 / hz if hz else 1.0
        return factor, self.offset_spin.value()

    def _refresh_preview(self) -> None:
        """预览行 = 落地后状态栏中段会显示的那句原文（同一函数、同一套格式）。

        唯一差别是 ``always_show_correction=True``（作者定）：抽屉里即使还在
        默认基准上，也要把"比例系数 1 / 偏移量 0"写出来 —— 这一行是"你现在要
        落地成什么"的回执，空着像没数据；状态栏是常驻段，默认值写出来才是噪音。
        """
        mw = self._mw()
        if mw is None:
            return
        factor, offset = self.candidate()
        text = mw._axis_segment_text(factor, offset, always_show_correction=True)
        self.preview.setText(text)
        self.preview.setToolTip(f"应用后状态栏显示：{text}")
        self._refresh_preset_states()

    # -- 动作 ---------------------------------------------------------------

    def _on_apply(self) -> None:
        mw = self._mw()
        if mw is None:
            return
        factor, offset = self.candidate()
        if not mw.layout_manager.apply_time_correction(factor, offset):
            self._notify("系数必须是正数", level="warn")
            return
        self.hide()

    def _on_reset(self) -> None:
        """填回默认基准并立即应用。

        刻意不只清输入框：「恢复默认」的意图就是撤销修正，只改框不落地会让人
        以为已经恢复，而图上的横轴没变。
        """
        mw = self._mw()
        if mw is None:
            return
        self._load_from(
            getattr(mw, "_factor_default", 1.0), getattr(mw, "_offset_default", 0.0)
        )
        self._on_apply()
