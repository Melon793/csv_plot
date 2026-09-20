"""界面色板与文字层级：「修改布局」网格选择器与状态栏抽屉共用。

值全部取自 ``src/ui/dialogs/layout_grid_selector.py`` 的既有实现 —— 那是本仓
唯一一处成体系的视觉语言，作者认可它并要求其他界面与之对齐。抽到这里的目的是
让抽屉不再自带一份颜色：将来做深浅主题 token 化时只改这一个文件。

已知代价（选它而不是 ``palette()`` 自动跟随的理由）：这是一份**浅色主题**的
写死色板，网格选择器今天也是这样。同一屏上出现两套灰阶比深色主题下不一致更
刺眼，所以两处必须继续用同一份值。

**这里刻意不放按钮样式**（作者定）：抽屉那 9 枚方片与网格选择器的「取消」原先
各带一份 QSS，实测一带 ``border``/``background-color`` 就被 ``QStyleSheetStyle``
接管 —— 同一枚按钮在 macOS / Windows / Fusion 下尺寸一律 106x31、深浅 palette
两态像素不变，而顶栏原生按钮三档各不相同（78x33 / 100x30 / 80x27）且跟着变，
于是全应用只剩这两枚不跟随系统，成了第三套灰阶。现在它们都退回平台绘制。
唯一保留自定义绘制的是「修改布局」那 12 枚网格方块：三态色是那扇对话框唯一的
信息通道，原生布尔态表达不了"当前布局"与"将要切换"同框。
"""

from __future__ import annotations

# === 色板 ===
BG = "#FFFFFF"
TEXT_PRIMARY = "#1F2329"
TEXT_MUTED = "#8A8F98"
SEP = "#EAECEF"

# 状态栏段间竖线。不能复用 SEP：抽屉坐在白底面板上，SEP 才有 1 px 发丝感；
# 状态栏坐在系统窗口条上（实测底色 #efefef），SEP 与它差 5/255，量出来
# "可见线列数 = 0"，等于没有线。这一档深到 #C9CDD4（对比度 13）。
SEP_ON_BAR = "#C9CDD4"

# 网格方片三态（只有「修改布局」那 12 格用）：未选浅灰、当前淡蓝、新值实心蓝
CHIP_OFF_BG = "#F2F4F7"
CHIP_OFF_BD = "#DFE3E8"
CHIP_CUR_BG = "#E8F1FD"
CHIP_CUR_BD = "#8FBBEA"
CHIP_SEL_BG_TOP = "#5AA2E6"
CHIP_SEL_BG_BOT = "#3B82D0"
CHIP_SEL_BD = "#2F6FB4"

# === 度量 ===
R_FIELD = 4  # 值列字段框的圆角：框比按钮大一号，用更小一档才不显肿
F_MUTED_PX = 12  # 次级文字（标签、页脚提示、副文案）
F_RESULT_PX = 18  # 大号结果行：网格选择器用 22px，抽屉要矮一档

# === 文字层级 ===


def muted_text(px: int = F_MUTED_PX) -> str:
    """次级文字：标签、页脚提示、副文案。"""
    return f"color: {TEXT_MUTED}; font-size: {px}px;"


def value_text() -> str:
    """正文：键值表的值列。

    刻意**不写字号**：正文沿用应用字体（与变量列表、对话框一致），钉一个 px
    值会让同一屏出现两套正文尺寸；实测写 13px 之后长路径的可见字符数从 59 掉
    到 53 —— 字大了，一屏放下的信息反而更少。
    """
    return f"color: {TEXT_PRIMARY};"


def result_text(px: int = F_RESULT_PX) -> str:
    """大号结果行（预览）：与网格选择器的 "4 × 1" 同一层级。"""
    return f"color: {TEXT_PRIMARY}; font-size: {px}px; font-weight: 600;"


def field_style() -> str:
    """值列普通行（不可滚的短值）的字段框占位：透明描边 + 4 px 内边距。

    存在的唯一理由是对齐：三行路径字段用的是只读 QLineEdit（能拉选到框外，见
    ``path_field_style``），它的文本左沿天然不在 0；这一列要是没有同一套内边距，
    表格就会出现两条文字左沿。

    声明值 1px 描边 + 4px padding，实测每侧吃掉 5 px（``contentsRect`` 报
    ``(5,1,w-10,h-2)``），与声明一致。
    """
    return (
        f"QLabel {{ {value_text()}"
        f"border: 1px solid transparent;"
        f"border-radius: {R_FIELD}px;"
        f"padding: 0 4px; }}"
    )


def path_field_style() -> str:
    """三行路径字段：只读 QLineEdit 的底色、描边与内边距。

    用 QLineEdit 而不是 QLabel，是因为 QLabel **没有视口**：文本按控件宽度排一次
    版，超出部分根本没参与排版，所以框外那几个字符既看不见也选不到（实测同一条
    103 字符路径，只读 QLineEdit 的 home/end 两态差 12408 个像素 = 有视口能滚，
    QLabel 是 0）。

    ``padding`` 取 7 不是随手写的：QLabel 那列的首个墨列实测在第 9 px，QLineEdit
    原生在第 2 px，补 7 才能让整列文字左沿对齐。
    """
    return (
        f"QLineEdit {{ {value_text()}"
        f"background-color: {BG};"
        f"border: 1px solid transparent;"
        f"border-radius: {R_FIELD}px;"
        f"padding: 0 7px; }}"
        f"QLineEdit:hover {{ background-color: {CHIP_OFF_BG};"
        f"border: 1px solid {CHIP_CUR_BD}; }}"
        f"QLineEdit:focus {{ background-color: {BG};"
        f"border: 1px solid {CHIP_CUR_BD}; }}"
    )


def separator_style() -> str:
    """发丝分隔线：1px，比描边更浅（配合 WA_StyledBackground 使用）。"""
    return f"background-color: {SEP}; border: none;"


def panel_style(selector: str) -> str:
    """面板底色 + 发丝描边。

    必须带 id 或类型限定：在 ``QFrame`` 上直接写 ``QFrame {{ ... }}`` 会连带
    命中内部的 QScrollArea（它也是 QFrame），把滚动区的框一起画出来。
    """
    return (
        f"{selector} {{"
        f"background-color: {BG};"
        f"border: 1px solid {CHIP_OFF_BD};"
        f"}}"
    )


def transparent_scroll_style() -> str:
    """让滚动区把面板底色透出来。

    cocoa 真机上 QScrollArea 的视口会自己填 #ECECEC（窗口灰），而它父级的白底
    只在没被盖住的地方露出来 —— 实测同一扇抽屉里"页脚白、主体灰"（主体中
    #ECECEC / 页脚 #FFFFFF）。给视口刷白（``background-color`` 或 palette）都
    压不住，只有把整条链设成 transparent 让父级白底透上来才干净。

    必须写成 ``QScrollArea > QWidget > QWidget``：Qt 样式表里视口是滚动区的
    直接子 QWidget，只写前者只会命中滚动区自己的边框。
    """
    return (
        "QScrollArea { background: transparent; border: none; }"
        "QScrollArea > QWidget > QWidget { background: transparent; }"
    )
