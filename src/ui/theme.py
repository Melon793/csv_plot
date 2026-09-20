"""界面色板与文字层级：「修改布局」网格选择器与状态栏抽屉共用。

值全部取自 ``src/ui/dialogs/layout_grid_selector.py`` 的既有实现 —— 那是本仓
唯一一处成体系的视觉语言，作者认可它并要求其他界面与之对齐。抽到这里的目的是
让抽屉不再自带一份颜色：将来做深浅主题 token 化时只改这一个文件。

已知代价（选它而不是 ``palette()`` 自动跟随的理由）：这是一份**浅色主题**的
写死色板，网格选择器今天也是这样。同一屏上出现两套灰阶比深色主题下不一致更
刺眼，所以两处必须继续用同一份值。
"""

from __future__ import annotations

# === 色板 ===
BG = "#FFFFFF"
TEXT_PRIMARY = "#1F2329"
TEXT_MUTED = "#8A8F98"
SEP = "#EAECEF"

# 方片（单元格 / 小按钮）三态：未选浅灰、当前淡蓝、新值实心蓝
CHIP_OFF_BG = "#F2F4F7"
CHIP_OFF_BD = "#DFE3E8"
CHIP_CUR_BG = "#E8F1FD"
CHIP_CUR_BD = "#8FBBEA"
CHIP_SEL_BG_TOP = "#5AA2E6"
CHIP_SEL_BG_BOT = "#3B82D0"
CHIP_SEL_BD = "#2F6FB4"
BTN_PRESS_BG = "#E6E9ED"

# === 度量 ===
R_CHIP = 6  # 方片圆角半径：与网格单元格同一档，两处一起看才像一家人
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
    """值列的"字段框"占位：透明描边 + 4 px 内边距，所有行一律带上。

    为什么连不做 hover 的行也要留描边：只给部分行加 padding 会让它们的文字左沿
    比别的行缩进 5 px，一张表两条基线。

    hover 那一档**不在这里**：样式表的 ``:hover`` 在合成事件下实测涂不出任何像素
    （变化=0），无法证明真机会亮；改由 ``status_drawer.FieldLabel`` 在 paintEvent
    里自己画，与状态栏可点击段同一套做法，可测。
    """
    return (
        f"QLabel {{ {value_text()}"
        f"border: 1px solid transparent;"
        f"border-radius: {R_FIELD}px;"
        f"padding: 0 4px; }}"
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


def chip_style(selector: str, bg: str = BG, border: str = CHIP_OFF_BD) -> str:
    """小方片按钮：细边 + 圆角 + hover/pressed 两级反馈。

    ``selector`` 传控件类型（``QToolButton`` / ``QPushButton``）：Qt 样式表里
    一个 widget 只能有一份样式串，后设的整份覆盖前者，所以选中态必须由调用方
    重设整串（见 status_drawer 里的 _refresh_preset_states）。
    """
    return (
        f"{selector} {{"
        f"padding: 2px 8px;"
        f"border: 1px solid {border};"
        f"border-radius: {R_CHIP}px;"
        f"background-color: {bg};"
        f"color: {TEXT_PRIMARY};"
        f"}}"
        f"{selector}:hover {{ background-color: {CHIP_OFF_BG}; }}"
        f"{selector}:pressed {{ background-color: {BTN_PRESS_BG}; }}"
        # 禁用态必须写：一旦给了 background-color，Qt 就不再自动压灰，
        # 不写会出现"看着可点、点了没反应"的假可用按钮
        f"{selector}:disabled {{"
        f"background-color: {CHIP_OFF_BG};"
        f"color: {TEXT_MUTED};"
        f"}}"
    )


def chip_style_selected(selector: str) -> str:
    """方片的"当前生效"态：淡蓝底 + 蓝描边，与网格选择器的当前布局同色。"""
    return chip_style(selector, CHIP_CUR_BG, CHIP_CUR_BD)


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
