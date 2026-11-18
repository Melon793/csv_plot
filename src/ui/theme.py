"""界面色板与文字层级：「修改布局」网格选择器与状态栏抽屉共用。

值全部取自 ``src/ui/dialogs/layout_grid_selector.py`` 的既有实现 —— 那是本仓
唯一一处成体系的视觉语言，作者认可它并要求其他界面与之对齐。抽到这里的目的是
让抽屉不再自带一份颜色：全 app 的蓝色统一由 :data:`ACCENT` 派生（Windows 跟
系统强调色，其它平台品牌蓝），散落字面量逐步收敛到这里的 token。

已知代价（选它而不是 ``palette()`` 自动跟随的理由）：这是一份**浅色主题**的
写死色板，网格选择器今天也是这样。同一屏上出现两套灰阶比深色主题下不一致更
刺眼，所以两处必须继续用同一份值。

**这里刻意不放按钮样式**（作者定）：抽屉那 9 枚方片与网格选择器的「取消」原先
各带一份 QSS，实测一带 ``border``/``background-color`` 就被 ``QStyleSheetStyle``
接管 —— 同一枚按钮在 macOS / Windows / Fusion 下尺寸一律 106x31、深浅 palette
两态像素不变，而顶栏原生按钮三档各不相同（78x33 / 100x30 / 80x27）且跟着变，
于是全应用只剩这两枚不跟随系统，成了第三套灰阶。现在它们都退回平台绘制。
唯一保留自定义绘制的是「修改布局」那 15 枚网格方块：三态色是那扇对话框唯一的
信息通道，原生布尔态表达不了"当前布局"与"将要切换"同框。
"""

from __future__ import annotations

import colorsys
import sys

from PySide6.QtGui import QColor

# === 强调色（accent） ===
# 全 app 只有一个蓝色源头：Windows 跟随系统强调色（DWM AccentColor），
# 其余平台固定品牌蓝。启动时解析一次——改系统强调色后重启生效（不做
# 运行中重刷，避免为 WM_SETTINGCHANGE 引入一整套重绘链路）。
BRAND_BLUE = "#2F6FB4"        # mac / Linux：本仓既有视觉锚点（原 CHIP_SEL_BD）
WINDOWS_FALLBACK = "#0078D4"  # Windows 读不到 AccentColor 时的系统默认蓝
_MAX_ACCENT_LUM = 0.45        # 亮度上限：浅黄/浅粉等强调色压暗到线框可读


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c))):02X}" for c in rgb)


def mix_hex(base: str, other: str, t: float) -> str:
    """线性混合两色：t=0 全 base，t=1 全 other。"""
    a, b = _hex_to_rgb(base), _hex_to_rgb(other)
    return _rgb_to_hex(x + (y - x) * t for x, y in zip(a, b))


def lighten_hex(value: str, factor: float) -> str:
    """按 HSV 明度缩放（保留色相/饱和度）：factor>1 提亮，<1 压暗。"""
    rgb = [c / 255 for c in _hex_to_rgb(value)]
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return _rgb_to_hex(
        c * 255 for c in colorsys.hsv_to_rgb(h, s, min(1.0, max(0.0, v * factor)))
    )


def relative_luminance(value: str) -> float:
    """WCAG 相对亮度（0=黑，1=白）。"""

    def channel(c: int) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = _hex_to_rgb(value)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def clamp_luminance(value: str, max_lum: float = _MAX_ACCENT_LUM) -> str:
    """把过亮的颜色逐档压暗到亮度上限内（Windows 强调色可以是浅黄）。"""
    rgb = tuple(float(c) for c in _hex_to_rgb(value))
    while relative_luminance(_rgb_to_hex(rgb)) > max_lum and max(rgb) > 0:
        rgb = tuple(c * 0.9 for c in rgb)
    return _rgb_to_hex(rgb)


def _abgr_to_hex(dword: int) -> str:
    """DWM AccentColor 是 0xAABBGGRR：低位是 R，alpha 字节忽略。"""
    return _rgb_to_hex((dword & 0xFF, (dword >> 8) & 0xFF, (dword >> 16) & 0xFF))


def _read_windows_accent() -> str | None:
    """读 HKCU 下的 DWM AccentColor（Software/Microsoft/Windows/DWM），失败返回 None。"""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\DWM"
        ) as key:
            dword, _ = winreg.QueryValueEx(key, "AccentColor")
    except OSError:
        return None
    if not isinstance(dword, int):
        return None
    return _abgr_to_hex(dword)


def resolve_accent() -> str:
    """当前平台强调色：Windows 读系统强调色，其余平台品牌蓝。"""
    if sys.platform == "win32":
        return clamp_luminance(_read_windows_accent() or WINDOWS_FALLBACK)
    return BRAND_BLUE


def on_accent_for(value: str) -> str:
    """实色上的文字：在白字/深字中取对比度更高的一档（延迟取 TEXT_PRIMARY）。"""
    lum = relative_luminance(value)
    white_ratio = 1.05 / (lum + 0.05)
    dark_ratio = (lum + 0.05) / (relative_luminance(TEXT_PRIMARY) + 0.05)
    return "#FFFFFF" if white_ratio >= dark_ratio else TEXT_PRIMARY


ACCENT = resolve_accent()
BORDER_SOFT = mix_hex(ACCENT, "#FFFFFF", 0.5)   # 弱边框：hover/focus/分类格
TINT_SOFT = mix_hex(ACCENT, "#FFFFFF", 0.9)     # 白底上的 10% 浅底（不透明版）
TINT_ALPHA = 64        # 25%：半透明覆盖（标记区域/闪烁/按压/选中 tint）
TINT_SOFT_ALPHA = 26   # 10%：更弱的半透明覆盖（数值列表十字引导）


def accent_qcolor(alpha: int = 255) -> QColor:
    color = QColor(ACCENT)
    color.setAlpha(alpha)
    return color


def tint_qcolor(alpha: int = TINT_ALPHA) -> QColor:
    """accent 的半透明版本：盖在内容上必须透出底层。"""
    return accent_qcolor(alpha)


# === 色板 ===
BG = "#FFFFFF"
TEXT_PRIMARY = "#1F2329"
TEXT_MUTED = "#8A8F98"
SEP = "#EAECEF"

# 状态栏段间竖线。不能复用 SEP：抽屉坐在白底面板上，SEP 才有 1 px 发丝感；
# 状态栏坐在系统窗口条上（实测底色 #efefef），SEP 与它差 5/255，量出来
# "可见线列数 = 0"，等于没有线。这一档深到 #C9CDD4（对比度 13）。
SEP_ON_BAR = "#C9CDD4"

# 网格方片三态（只有「修改布局」那 15 格用）：未选浅灰、当前淡蓝、新值实心蓝
CHIP_OFF_BG = "#F2F4F7"
CHIP_OFF_BD = "#DFE3E8"
CHIP_CUR_BG = TINT_SOFT
CHIP_CUR_BD = BORDER_SOFT
CHIP_SEL_BG_TOP = lighten_hex(ACCENT, 1.35)
CHIP_SEL_BG_BOT = lighten_hex(ACCENT, 0.87)
CHIP_SEL_BD = ACCENT

# 实色（accent）上的文字：白字对比度不足 4.5:1 时改用深字
ON_ACCENT = on_accent_for(ACCENT)

# 变量名悬停提示：Qt 对富文本提示不套默认内边距，字几乎贴框，必须显式 padding。
# QSS padding **不是 1:1**，各平台叠的底座不同（实测"文字墨迹→可见边框"）：
#   macOS   1px 2px → 6.5 / 6.5 / 6.5 / 6.5 pt
#   Windows 1px 2px → 6 / 7 / 6 / 8 px @96dpi（windowsvista，12px 字体）
# Windows 侧更紧（win10_env 实测），改数值前两个平台都要量，别按声明值想象。
#
# 刻意**不画边框和圆角**：QToolTip 的浮层是不透明方窗（实测
# WA_TranslucentBackground=False、windowOpacity=0.949），自绘圆角会在方窗里露出
# 四个直角，成为"直角套圆角"。外形交给平台：Windows 自带 1px 深色边框，
# macOS 是浅色边缘。
#
# 这是全应用唯一的应用级样式表，不违反上面"不放按钮 QSS"的决策：判据是规则有没有
# 命中原生控件，本规则只有 QToolTip 选择器。实测（tmp 里 check_qss_metric_leak.py）
# 加上它以后 QPushButton / QCheckBox / QComboBox / QLineEdit / QTabWidget 的
# sizeHint、表格行高、滚动条宽度全部不变。
TOOLTIP_QSS = f"""QToolTip {{
    padding: 1px 2px;
    background: {BG};
    color: {TEXT_PRIMARY};
}}"""

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


# 表格选中项：显式给出背景/文字色，让 QStyleSheetStyle 接管 item 绘制，
# 跳过 Windows 11 原生样式在选中项左缘画的 1px 强调色竖条（Qt 的
# CE_ItemViewItem vertical marker，会压在首字符上）。
# 颜色走 palette(highlight)/palette(highlighted-text)，跟随系统强调色与
# 深浅模式；只命中 item，表头/滚动条/未选中项仍走平台绘制。
TABLE_SELECTION_QSS = (
    "QTableWidget::item:selected, QTableView::item:selected {"
    " background-color: palette(highlight);"
    " color: palette(highlighted-text);"
    "}"
)
