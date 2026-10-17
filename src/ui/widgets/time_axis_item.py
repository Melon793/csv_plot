"""时间戳 X 轴图元 —— 索引空间的底轴把刻度渲染成时间戳（竖排文字）

数据空间仍是索引（1..N，含 offset/factor 时间校正），本轴只改**渲染**：
``tickStrings`` 把每个刻度的显示 X 经 mapper 翻译成 ``HH:MM:SS``，
``drawPicture`` 把文字旋转 -90° 竖排（读向自下而上，与主流采集软件一致）。

pyqtgraph 0.14 的 ``AxisItem`` 不支持旋转刻度文字，``drawPicture`` 的
实现足够薄（轴线 + 刻度线 + 逐条 drawText），这里整体复刻并把 drawText
换成"平移到刻度正下方 → rotate(-90) → 定向绘制"。

高度自适应必须接管：父类按"刻度文字的**高**"预留轴高（``textHeight``），
而旋转后占位的是文字的**宽**（"13:03:45" ≈ 55px ≫ 字高 14px），不接管
就会把竖排文字整段裁掉。

时序约束：**不能在绘制中改几何**。``tickStrings`` 在 ``generateDrawSpecs``
里被调用（正处于 paint 途中），此刻若 ``setMaximumHeight`` 触发重新布局，
本帧作废、画面停回旧高度。因此：

- 挂 mapper 时（``set_index_to_text``，事件循环中）用 mapper 的样本文本
  预量宽度并就地撑高 —— 首绘前预算就位，这是常态路径；
- ``tickStrings`` 里只**记录**量得的更宽值，高度更新经 ``QTimer``
  推迟到事件循环（带日期格式等罕见变宽场景的兜底）。
"""

from __future__ import annotations

from typing import Any, Callable

import pyqtgraph as pg
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer
from PySide6.QtGui import QFontMetricsF
from PySide6.QtWidgets import QApplication

# 时间轴刻度密度（相对 pyqtgraph 默认 1.0 的倍数）。
# pyqtgraph 的 tickSpacing：spacing = dif / (2.25 × density × √(size/ref))，
# 即 density 越大间距越小、标签越密 —— 2.0 = 密度加倍（用户实测默认太稀疏）。
# 竖排文字横向只占字高（~14px），加倍后依然远宽于文字，不会被密度判定跳过。
TICK_DENSITY = 2.0

# mapper 契约：显示 X（float）→ 时间文本；映射不了返回 None/""（该刻度留空）。
# 可选属性 sample_text：一条代表性文本（如 "13:03:45"），供首绘前预量宽度。
IndexToText = Callable[[float], "str | None"]

_VERTICAL_FLAGS = (
    pg.QtCore.Qt.AlignmentFlag.AlignLeft
    | pg.QtCore.Qt.AlignmentFlag.AlignVCenter
    | pg.QtCore.Qt.TextFlag.TextDontClip
)


class IndexToTimeTextAdapter:
    """显示 X → 行号 → 时间文本 的组合适配器（mw 的 mapping + pw 的 factor/offset）

    显示 X = offset + factor × 行号（1-based，时间校正系统的既有口径），
    反解行号后交给 ``TimeAxisMapping`` 取文本。作为独立小类而非闭包：
    ``TimestampAxisItem`` 靠 ``sample_text`` 属性在首绘前预量宽度。
    """

    def __init__(self, mapping: Any, pw: Any):
        self._mapping = mapping
        self._pw = pw
        size = max(int(getattr(mapping, "size", 0)), 1)
        self.sample_text = (
            mapping.text_for_index(1.0)
            or mapping.text_for_index(size / 2.0)
            or "00:00:00"
        )

    def __call__(self, display_x: float) -> str | None:
        pw = self._pw
        factor, offset = pw.factor, pw.offset
        if not factor or not np.isfinite(factor):
            return None
        return self._mapping.text_for_index((display_x - offset) / factor)


class TimestampAxisItem(pg.AxisItem):
    """把索引刻度渲染成时间戳的底轴

    mapper 由 MainWindow 侧绑定（要读 pw.factor / pw.offset 与时间列），
    轴本身只认"显示 X → 文本"这一个回调，不反向依赖任何管理者。
    """

    def __init__(self, *args: Any, index_to_text: IndexToText | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.setTickDensity(TICK_DENSITY)
        self._index_to_text = index_to_text
        self._rotated_text_width = 0.0
        self._height_refresh_pending = False
        self._pending_widest_text = ""
        if index_to_text is not None:
            self._grow_rotated_width(self._sample_text(index_to_text))

    # ---- 宽度 / 高度 -------------------------------------------------------

    @staticmethod
    def _sample_text(mapper: IndexToText) -> str:
        """mapper 的代表性文本：优先显式 sample_text，退化探测 index=1"""
        sample = getattr(mapper, "sample_text", None)
        if sample:
            return str(sample)
        try:
            return str(mapper(1.0) or "")
        except Exception:
            return ""

    def _measure_texts(self, texts: list[str]) -> str:
        """返回这批文本里最宽的一条（无有效文本返回空串）"""
        font = self.style["tickFont"] or self.font() or QApplication.font()
        metrics = QFontMetricsF(font)
        widest, text = 0.0, ""
        for candidate in texts:
            if candidate:
                width = metrics.horizontalAdvance(candidate)
                if width > widest:
                    widest, text = width, candidate
        return text

    def _grow_rotated_width(self, text: str) -> None:
        """按样本文本撑高轴预算；只在事件循环里调用（不在绘制中）"""
        if not text:
            return
        font = self.style["tickFont"] or self.font() or QApplication.font()
        width = QFontMetricsF(font).horizontalAdvance(text)
        if width > self._rotated_text_width:
            self._rotated_text_width = width
            AxisItem_updateMaxTextSize = pg.AxisItem._updateMaxTextSize
            AxisItem_updateMaxTextSize(self, width)

    def _apply_pending_height(self) -> None:
        """延迟撑高（事件循环里执行）；widget 可能已销毁，必须守卫

        与 plot_widget._interaction_timer 同一个陷阱：QTimer 回调会在
        deleteLater 与 C++ 析构之间的任意时刻触发，裸调 self.font() 会抛
        RuntimeError 并污染解释器状态。
        """
        self._height_refresh_pending = False
        try:
            text = self._pending_widest_text
            self._pending_widest_text = ""
            if text:
                self._grow_rotated_width(text)
        except RuntimeError:
            pass  # C++ 对象已析构，无事可做

    def _updateMaxTextSize(self, x: float) -> None:  # noqa: N802 (pyqtgraph 命名)
        # 父类在 generateDrawSpecs 末尾会传"刻度文字的高"；旋转后预算按
        # "文字宽"算。父类逻辑只增不减，这里取 max 即可保住已撑大的预算。
        pg.AxisItem._updateMaxTextSize(self, max(float(x), self._rotated_text_width))

    # ---- 刻度文本 ---------------------------------------------------------

    def set_index_to_text(self, mapper: IndexToText | None) -> None:
        self._index_to_text = mapper
        if mapper is not None:
            self._grow_rotated_width(self._measure_texts([self._sample_text(mapper)]))
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph 命名)
        mapper = self._index_to_text
        if mapper is None:
            return super().tickStrings(values, scale, spacing)
        texts = []
        for value in values:
            try:
                text = mapper(float(value))
            except Exception:
                text = None
            texts.append(text or "")
        widest = self._measure_texts(texts)
        # 绘制中不改几何：量到更宽文本（如带日期格式）时推迟到事件循环撑高
        if widest and not self._height_refresh_pending:
            self._pending_widest_text = widest
            self._height_refresh_pending = True
            QTimer.singleShot(0, self._apply_pending_height)
        return texts

    # ---- 绘制 -------------------------------------------------------------

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):  # noqa: N802
        """复刻 AxisItem.drawPicture，文字块改为竖排

        几何（底轴）：锚点 A = (刻度水平居中点, 文字矩形顶边)。rotate(-90) 后
        局部 +x 指向屏幕上方、+y 指向屏幕右方，因此局部矩形
        ``(-L, -W/2, L, W)``（L = 文本自然宽、W = 文本自然高）正好覆盖
        "自 A 向下延伸 L 像素、沿刻度左右各 W/2"的区域；
        ``AlignLeft`` 使文字从局部 x=-L（屏幕底部）起笔、向屏幕上方读，
        ``AlignVCenter`` 沿刻度居中 —— 与水平排版"贴轴 + 居中"同款。
        """
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)

        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)

        for tpen, t1, t2 in tickSpecs:
            p.setPen(tpen)
            p.drawLine(t1, t2)

        if self.style["tickFont"] is not None:
            p.setFont(self.style["tickFont"])
        p.setPen(self.textPen())
        p.setClipRect(self.boundingRect())
        for rect, _flags, text in textSpecs:
            if not text:
                continue
            length = rect.width()   # 文本自然宽 → 旋转后的竖向占用
            thick = rect.height()   # 文本自然高 → 旋转后的横向占用
            p.save()
            p.translate(QPointF(rect.center().x(), rect.top()))
            p.rotate(-90)
            p.drawText(
                QRectF(-length, -thick / 2.0, length, thick),
                int(_VERTICAL_FLAGS),
                text,
            )
            p.restore()
