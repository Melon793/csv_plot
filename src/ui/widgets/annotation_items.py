"""标注图元 —— 各类 AnnotationItem 对应的 pyqtgraph 图元与交互门控

七条实测结论（改动前请先读这段，都是踩空过的坑）：

1. 自动范围隔离必须靠「覆写 dataBounds() 返回 (None, None)」
   ViewBox.childrenBounds 收到 (None, None) 会把该轴标记 useX/useY=False 后 continue，
   图元就彻底退出 autorange。pyqtgraph 0.14.0 已无 setIgnoreBounds（老资料里到处
   在写的那个 API 不存在，照抄必踩空）。
   ViewBox.addItem(item, ignoreBounds=True) 也能隔离，但它的实现只是「不把 item
   放进 ViewBox.addedItems」，语义是调用方的责任 —— 换成覆写，所有图元走同一条
   通路，不依赖每个调用点是否记得传参。
   特别提醒：ArrowItem 自带 dataBounds，在 pxMode 下返回 [0, 0]（一个退化点），
   不覆写的话把箭头拖到数据范围外照样会污染自动范围；TextItem 同理，它的
   dataBounds 返回 [anchor, anchor]。

1b. 字号恒定**不要**用 ItemIgnoresTransformations
   这条与第 1 条同一量级，单独列是因为它坑得很隐蔽 —— 它不会报错、不会画错，
   只会让图元**点不中**。
   pyqtgraph 的 TextItem 自带 updateTransform，源码注释原话是 "similar to setting
   ItemIgnoresTransformations = True, but does not break mouse interaction and
   collision detection"，也就是说字号恒定本来就成立。再叠加这个标志，等于用白拿
   的性质换掉了鼠标交互：实测带标志时 scene.items(文字中心) 里根本没有文字图元
   （编辑态拖不动），去掉后立刻正常，而 sceneBoundingRect 在 1× 与 10× 下逐位一致
   （都是 92×22）。根因是 Qt 的 BSP 空间索引按 sceneTransform 存包围盒，对这种
   图元取值不一致。TargetLabel 同为 TextItem，一并适用。

2. 非编辑态必须同时拔掉**四条**交互通路，缺一条都会抢走 ViewBox 的鼠标事件
   - ROI 本体：translatable / resizable / rotatable 全 False。
     ROI.hoverEvent 只在 translatable 为真时才 acceptDrags，而这是 GraphicsScene
     挑选拖拽目标的唯一门槛；全 False 后它连 acceptClicks 都不会调（实测
     acceptDrags 调用次数 0），平移/缩放/框选全部落回 ViewBox。
   - ROI 手柄：显式 setVisible(False) + setAcceptedMouseButtons(NoButton)。
     Handle.hoverEvent 无条件 acceptDrags，根本不看 ROI 的门控 —— 只看 ROI
     标志位会漏掉这一条。
   - 折线的线段（``_PolyLineSegment``）：独立图元，容器关不住它。pyqtgraph 在
     ``addSegment`` 里给它设了 ``LeftButton``，点一下就在段上插一个折点，必须
     单独拔按键。连带效应：``addSegment`` 还会给两端手柄 ``setDeletable(True)``，
     把手柄的按键**或上** ``RightButton``（右键菜单有"删掉这个折点"）——
     订阅式地 OR 会越关越多，所以本文件的门控一律用**赋值**而不是 ``|=``。
     线段抢不抢拖拽看的是**父容器的 translatable**（``_PolyLineSegment.hoverEvent``
     里那句 acceptDrags 的条件就是它），容器一关线段连 hover 阶段都不会登记拖拽，
     视图平移自然落回 ViewBox。这层因果绕，改 ``translatable`` 前务必回读第 5 条。
   - TextItem：setAcceptedMouseButtons(NoButton)。Qt 只向命中该位的 item 投递 press。

3. ArrowItem 的 pos 就是箭尖，且 pxMode 下渲染尺寸像素恒定
   makeArrowPath 把顶点放在 (0,0)、箭身朝 -x；实测 headLen=20/headWidth=14 的箭头
   在 4.5 与 22 px/unit 两种视图缩放下都渲染成 21x28 设备像素。所以箭尖直接用
   数据坐标 setPos 即可，方向按「屏幕像素方向」反算。
   不要用 sceneBoundingRect 去推算它 —— 该值对 ItemIgnoresTransformations 图元
   不可信（实测同一箭头返回 83.6x122.6 的虚高值）。

4. ROI 的几何读写必须换算到父坐标（数据坐标）
   自由手柄（addFreeHandle）的 pos() 是 ROI 的局部坐标，ROI 整体平移只改自己的
   位置、手柄一个都不动；而 RectROI 这类"pos + size"型 ROI 的 getState()['pos']
   本身就是数据坐标。两者混用过的代价：直线/箭头被整体拖走后模型里还是老坐标，
   存模板再套回来就跳回原位（实测踩过）。统一走 mapToParent / mapFromParent。

5. PolyLineROI 的子图元要单独管门控**和 z 序**
   线段是独立图元，z 只在建段那一刻取一次 self.zValue() + 1，之后容器改 z 它不跟
   （实测容器 z 从 50 改到 90，线段还停在 11）—— 不等同步，描边会被别的标注盖住、
   选中也压不住别人。

   实测记录（挑落点精确落在段上的做法，见 tests/component/test_annotation_polyline.py）：
   容器可平移时线段在 hover 阶段就登记了 LeftButton，之后 GraphicsScene 的
   sendDragEvent 走 init 分支会**直接用登记到的线段当 dragItem，不再回退问 ViewBox**
   —— 视图就平不动了。容器不可平移时 dragItems 为空，拖拽照旧归 ViewBox。
   也就是说"浏览态不平移视图"这条保证，靠的是容器 translatable=False，而不只是
   线段 acceptedMouseButtons=NoButton（后者挡的是"点段插折点"那条路）。
   编辑态下线段抢走拖拽是**设计目标**（拖折线就该改折线而不是平移视图）。

6. TargetItem 的标签是「format 字符串」陷阱
   标靶标签走 ``pg.TargetItem.setLabel``，而它把「含 ``{}`` 的字符串」当 format
   模板：``"目标 {24℃}"`` 会被 ``str.format(x, y)`` 抛 ``KeyError: '24℃'``
   （实测）。用户文案里出现花括号是常事（"区间 {0,1}"、"{主驾}"），所以本文件
   一律把文案包成 callable 再交出去 —— callable 分支永不解析花括号。
   空文案用 ``setLabel(None)``，那才是"不显示标签"。

7. **每种图元都必须在构造期把自己的坐标落下去**
   模型（数据坐标）与图元（屏幕）是两套状态，任何一类图元漏了这一步都会变成
   "模型说在这、屏幕画在那"的静默分裂 —— 不报错、不崩溃，只是标注跑偏，而
   模板往返测试照样全绿（往返读的是模型）。
   P1 的文字标注就是这么错的：``AnnotationTextItem`` 的构造函数没有位置参数、
   ``create_annotation_item`` 也没补一步写入，于是**每条文字标注都停在 ViewBox
   原点（子图左上角）**，模型里却记着用户点的坐标（实测：模型 [(20,20)]、图元
   pos (0,0)、sceneBoundingRect 落在子图角上）。
   守卫见 ``tests/component/test_annotation_placement.py`` —— 它逐类比对
   "模型坐标 → 期望像素位置" 与图元的实际位置，新加 kind 忘了落坐标会直接红。
"""

from __future__ import annotations

import math
from typing import Any, Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsItem
import pyqtgraph as pg

from src.core.annotation_models import (
    DEFAULT_FILL_ALPHA,
    DEFAULT_FONT_SIZE,
    DEFAULT_STROKE,
    DEFAULT_STROKE_WIDTH,
    MAX_POLYLINE_POINTS,
    Z_ANNOTATION,
    Z_ANNOTATION_SELECTED,
)

# 箭头头部尺寸（设备像素）：pxMode 下与视图缩放无关，截图上是稳定大小
ARROW_HEAD_LEN = 14
ARROW_HEAD_WIDTH = 11

# 标靶直径（设备像素）：同为截图友好的固定值，不随视图缩放变化。
# 需要"按数据尺寸圈一片允差带"的场合请用椭圆标注，标靶只负责"指着这个点"。
TARGET_SIZE = 14

# 标靶标签相对标靶的屏幕偏移（像素）：TargetItem 自带 TargetLabel 的默认值，
# 这里只是把它写出来当作契约，改这里等于改观感，别改语义。
TARGET_LABEL_OFFSET_PX = 20

# 新建矩形/椭圆的最小边长（数据坐标），避免出现零面积图元
MIN_SHAPE_SIZE = 1e-9

_FONT_CACHE: dict[tuple[str, int], QFont] = {}
_FONT_FAMILY: str | None = None


# 模块级私有属性名不受 dataclass/异常保护，统一用 getattr + 默认值访问
def _flag(item: Any, name: str, default: Any) -> Any:
    return getattr(item, name, default)


def _annotation_font(pixel_size: int) -> QFont:
    """按像素字号取字体（带缓存）

    用 setPixelSize 而不是 setPointSize：点值受屏幕 DPI 影响，同一份标注在
    不同机器上渲染出的物理大小会变，像素值才是截图可复现的那一个。
    """
    global _FONT_FAMILY
    if _FONT_FAMILY is None:
        app = QApplication.instance()
        _FONT_FAMILY = app.font().family() if app is not None else ""

    key = (_FONT_FAMILY, int(pixel_size))
    font = _FONT_CACHE.get(key)
    if font is None:
        font = QFont(_FONT_FAMILY)
        font.setPixelSize(int(pixel_size))
        _FONT_CACHE[key] = font
    return font


class _NoBoundsMixin:
    """让标注图元退出 ViewBox 自动范围计算（见模块 docstring 第 1 条）"""

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        return (None, None)


# ======================================================================
# 图元定义
# ======================================================================

class AnnotationTextItem(_NoBoundsMixin, pg.TextItem):
    """文字标注

    **不要开 ItemIgnoresTransformations**（这是 P1 的一个真实缺陷，已修）：
    pyqtgraph 的 ``TextItem.updateTransform`` 自带一套等价机制，其源码注释原话是
    "similar to setting ItemIgnoresTransformations = True, **but does not break
    mouse interaction and collision detection**"。我们再叠一个标志，等于是用
    "字号恒定"这个**本来就成立**的性质，换掉了鼠标交互：
    实测（800×600、数据 0~100）带标志时 ``scene.items(文字中心)`` 里根本没有
    文字图元 —— 编辑态点不中、拖不动；去掉标志立刻正常，而 ``sceneBoundingRect``
    在 1× 与 10× 下逐位不变（都是 92×22）。
    根因是 Qt 的 BSP 空间索引按 sceneTransform 存包围盒，对这种图元取值不一致。

    拖动自己实现、不用 ItemIsMovable：位置挂在数据坐标上，交给 Qt 换算容易抖动；
    改成按 scenePos 反算数据坐标，位移结果可精确断言。
    """

    def __init__(self, text: str = "", *, annotation_id: str = "",
                 editable: bool = False, point=None):
        super().__init__(text=text, color=DEFAULT_STROKE, anchor=(0.0, 0.5))
        init_annotation_item(self, annotation_id, editable)
        # 位置必须在构造期就落下去：本图元不像 ROI / 标靶那样把坐标写进构造函数的
        # 第一个参数，漏了这一步它会停在 ViewBox 原点（子图左上角），而模型里记着
        # 用户点的那个坐标 —— 屏幕与模型静默分裂（P1 实测踩过，见模块 docstring）。
        if point is not None:
            self.setPos(float(point[0]), float(point[1]))
        self._dragging = False
        # 由 AnnotationManager 注入：几何变化时回写模型
        self._on_geometry_changed: Callable[[str], None] | None = None
        # 由 AnnotationManager 注入：手势起点 / 终点（文字图元没有 ROI 那套信号，
        # 只能靠自己按下/松开这一对鼠标事件，见 AnnotationManager 的手势合并）
        self._on_edit_started: Callable[[str], None] | None = None
        self._on_edit_finished: Callable[[str], None] | None = None

    # -- 注入钩子的调用 ----------------------------------------------
    def _notify(self, hook: str) -> None:
        callback = _flag(self, hook, None)
        if callback is not None:
            callback(self.annotation_id)

    # -- 拖拽 --------------------------------------------------------
    def mousePressEvent(self, ev):
        if _flag(self, "_editable", False) and ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._notify("_on_edit_started")
            ev.accept()
            return
        ev.ignore()

    def mouseMoveEvent(self, ev):
        if not (_flag(self, "_editable", False) and self._dragging):
            ev.ignore()
            return
        view_box = self.getViewBox()
        if view_box is None:
            ev.ignore()
            return
        point = view_box.mapSceneToView(ev.scenePos())
        self.setPos(point.x(), point.y())
        ev.accept()

    def mouseReleaseEvent(self, ev):
        was_dragging = self._dragging
        self._dragging = False
        if was_dragging:
            self._notify("_on_edit_finished")
        super().mouseReleaseEvent(ev)

    def itemChange(self, change, value):
        # 程序化 setPos 与用户拖拽都会走到这里；回写是幂等的（只写模型，不反写图元），
        # 不存在「模型→图元→模型」的回环
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            callback = _flag(self, "_on_geometry_changed", None)
            if callback is not None and not _flag(self, "_suppress_sync", False):
                callback(self.annotation_id)
        return super().itemChange(change, value)


class _AnnotationFillMixin:
    """半透明填充的公共部分：brush 记账 + 画家状态纪律

    pyqtgraph 的 RectROI / EllipseROI / PolyLineROI 的 paint 都不吃 brush，填充要
    自己画。save/restore 必须成对：漏了 restore，紧接着的 super().paint() 会继承
    这份变换，描边被二次缩放成一大片色块（实测过，不是理论担忧）。
    """

    _fill_brush: QBrush | None = None

    def apply_fill(self, color: str | None, alpha: int = DEFAULT_FILL_ALPHA) -> None:
        if not color:
            self._fill_brush = None
        else:
            qcolor = QColor(color)
            qcolor.setAlpha(int(alpha))
            self._fill_brush = QBrush(qcolor)
        self.update()

    def _begin_fill(self, painter: QPainter) -> bool:
        """给填充铺好画家状态；返回 False 表示没填充可画（调用方直接返回即可）"""
        if self._fill_brush is None:
            return False
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(pg.mkPen(None))
        painter.setBrush(self._fill_brush)
        return True


class _AnnotationFilledROI(_AnnotationFillMixin):
    """矩形 / 椭圆的填充：形状是单位方块，靠画家变换缩到实际尺寸"""

    def _local_rect(self) -> QRectF:
        size = self.state["size"]
        return QRectF(0.0, 0.0, float(size[0]), float(size[1])).normalized()

    def _paint_fill(self, painter: QPainter, draw: Callable[[QPainter], None]) -> None:
        rect = self._local_rect()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        if not self._begin_fill(painter):
            return
        painter.translate(rect.left(), rect.top())
        painter.scale(rect.width(), rect.height())
        draw(painter)
        painter.restore()


class AnnotationRectROI(_AnnotationFilledROI, _NoBoundsMixin, pg.RectROI):
    """矩形标注：圈出关注区间"""

    def __init__(self, pos, size, *, annotation_id: str = "", editable: bool = False):
        # sideScalers=True 多给两条边手柄：本 ROI 没有平移手柄，靠手柄缩放即可
        # 覆盖移动需求，编辑起来不会出现"拖不动"的死角
        super().__init__(pos, size, rotatable=False, removable=False, sideScalers=True)
        init_annotation_item(self, annotation_id, editable)

    def paint(self, painter, opt, widget):
        self._paint_fill(painter, lambda p: p.drawRect(0, 0, 1, 1))
        super().paint(painter, opt, widget)


class AnnotationEllipseROI(_AnnotationFilledROI, _NoBoundsMixin, pg.EllipseROI):
    """椭圆标注：圈出局部特征"""

    def __init__(self, pos, size, *, annotation_id: str = "", editable: bool = False):
        super().__init__(pos, size, rotatable=False, removable=False)
        init_annotation_item(self, annotation_id, editable)

    def paint(self, painter, opt, widget):
        self._paint_fill(painter, lambda p: p.drawEllipse(QRectF(0, 0, 1, 1)))
        super().paint(painter, opt, widget)


class AnnotationLineROI(_NoBoundsMixin, pg.LineSegmentROI):
    """直线 / 箭头共用的线段载体（两端各一个手柄，箭头另有头部图元）"""

    def __init__(self, points, *, annotation_id: str = "", editable: bool = False):
        super().__init__(points, removable=False)
        init_annotation_item(self, annotation_id, editable)


class AnnotationPolyLineROI(_AnnotationFillMixin, _NoBoundsMixin, pg.PolyLineROI):
    """折线标注：手工勾勒包络

    三条实测约束（踩空一次就会**静默**出错，不是崩溃）：

    1. 折点手柄的 ``pos()`` 是 ROI 的**局部**坐标。ROI 整体平移只改自己的位置，
       手柄一个都不动 —— 几何读写必须走 ``mapToParent`` / 反向 ``mapFromParent``。
       照抄"直接读手柄坐标"会把整体平移丢掉：屏幕上线条移了、模型里还是老坐标，
       存模板再套回来就跳回原位（P1 的直线/箭头就踩在这上面，已一并修）。
    2. 线段（``_PolyLineSegment``）是**独立图元**，pyqtgraph 在 ``addSegment`` 里给
       它设了 ``LeftButton``，点一下就在段上插一个折点。只关容器的
       ``acceptedMouseButtons`` 不够，浏览态下照样会多出折点。
    3. 线段的 z 只在建段那一刻取一次 ``self.zValue() + 1``，之后容器改 z 它不跟。
       不同步的话描边停在 11：被任何一条 z=50 的标注盖住，选中抬到 90 也盖不住
       别人。覆写 ``setZValue`` 一处解决建/选中/任何改 z 的路径。
    """

    def __init__(self, points, *, annotation_id: str = "", editable: bool = False):
        # closed=False：描边不闭合（勾勒包络要的就是开口折线），
        # 但**填充**仍按闭合多边形画，见 paint
        super().__init__(points, closed=False, removable=False, rotatable=False)
        init_annotation_item(self, annotation_id, editable)

    def setZValue(self, z: float) -> None:
        super().setZValue(z)
        for segment in getattr(self, "segments", None) or []:
            segment.setZValue(z + 1)

    def segmentClicked(self, segment, ev=None, pos=None):
        """段上点击加点（pyqtgraph 自带交互）

        在**加点之前**挡上限，而不是事后由管理器拒绝回写：后者会让画面多出一个
        模型里不存在的折点，两边分裂且再也对不上。上限与模型层同一个常量
        （模板里的 points 超限由 from_dict 截断）。
        """
        if len(self.handles) >= MAX_POLYLINE_POINTS:
            return
        super().segmentClicked(segment, ev=ev, pos=pos)

    def paint(self, painter, opt, widget):
        """只画填充；描边由线段子项负责

        pyqtgraph 0.14.0 里 ``PolyLineROI.paint`` 是 ``pass``，所以这里不调
        ``super().paint()`` —— 调了也只是空转。
        """
        path = self.shape()
        # 两个共线折点时包围盒高度为 0，填充无面积可言，跳过
        if path.boundingRect().isEmpty():
            return
        if not self._begin_fill(painter):
            return
        painter.drawPath(path)
        painter.restore()


class AnnotationTargetItem(_NoBoundsMixin, pg.TargetItem):
    """标靶：标定工况点 / 目标值（十字 + 圆圈，屏幕上尺寸恒定）

    直接用 pyqtgraph 自带的 TargetItem —— 它自己就是按 deviceTransform 反算形状的，
    10× 缩放下仍是 14 像素，正好满足「截图可读」。

    两条实测约束（都会**静默**出错或直接抛异常，不是理论担忧）：

    1. 标签文案必须包成 callable 再交给 ``setLabel``。pyqtgraph 会把「含 ``{}``
       的字符串」判成 format 模板，``"目标 {24℃}"`` 会走到 ``str.format(x, y)``
       上抛 ``KeyError: '24℃'``（实测）。callable 分支永不解析花括号。
       空文案 = 不显示标签（``setLabel(None)``）。
    2. 浏览态必须连 ``acceptedMouseButtons`` 一起拔掉，与 ROI 子图元同一条理由。
       本图元 ``acceptHoverEvents()`` 默认 False（实测），``hoverEvent`` 那条通路
       本来就不成立，但 ``mousePress`` 照常投递 —— 光靠 ``movable=False`` 不够稳。
       ``movable`` 仍要跟着关：它同时挡 ``mouseDragEvent`` 与本图元对标签的转发。

    另注：``dataBounds()`` 本来就返回 ``None``（实测），autorange 天然隔离；
    这里仍旧混入 ``_NoBoundsMixin``，是为了七类图元走同一条通路、不留特例。
    """

    def __init__(self, point, *, annotation_id: str = "", editable: bool = False, text: str = ""):
        x, y = float(point[0]), float(point[1])
        super().__init__(pos=(x, y), size=TARGET_SIZE, symbol="crosshair")
        init_annotation_item(self, annotation_id, editable)
        # 由 AnnotationManager 注入：拖动过程中回写模型（TargetItem 的
        # sigPositionChanged 在 setPos 里发出，程序化与拖拽都会走到）
        self._on_geometry_changed: Callable[[str], None] | None = None
        # 由 AnnotationManager 注入：手势起点/终点，供撤销栈把一次拖动合并成一条
        self._on_edit_started: Callable[[str], None] | None = None
        self._on_edit_finished: Callable[[str], None] | None = None
        self.set_label_text(text)

    # -- 标签 --------------------------------------------------------
    def set_label_text(self, text: str) -> None:
        """设置标签文案；空文案表示不显示标签

        改文案必然重建标签（pyqtgraph 只提供"换掉整个 TargetLabel"这一条路），
        所以先比对、没变就早退 —— 否则每次调个颜色都会重建一遍标签。
        重建出来的新标签是默认样式，调用方必须紧接着 ``apply_style_to_item``。
        """
        body = str(text or "").strip()
        if body == getattr(self, "_label_text", ""):
            return
        self._label_text = body
        if not body:
            self.setLabel(None)
            return
        self.setLabel(lambda x, y, t=body: t)
        # 也不给标签开 ItemIgnoresTransformations：TargetLabel 自己会在
        # viewTransformChanged 里按 viewPixelSize 重算偏移，实测 1×/10× 下
        # sceneBoundingRect 都是 80×22；而这个标志会让图元在场景 BSP 索引里
        # 查不到，落点击中判定失效（同 AnnotationTextItem 的类 docstring）
        # 标签重建 = 图元状态变了，门控要重上：新手柄/新标签默认什么都吃
        _apply_gating(self)

    # -- 手势钩子 ----------------------------------------------------
    def mouseDragEvent(self, ev):
        if ev.isStart():
            callback = _flag(self, "_on_edit_started", None)
            if callback is not None:
                callback(self.annotation_id)
        super().mouseDragEvent(ev)


class AnnotationArrowHead(_NoBoundsMixin, pg.ArrowItem):
    """箭头头部：纯装饰，永远不吃鼠标事件"""

    def __init__(self, *, annotation_id: str = ""):
        super().__init__(angle=0, headLen=ARROW_HEAD_LEN, headWidth=ARROW_HEAD_WIDTH, tailLen=None)
        self.annotation_id = annotation_id
        self._editable = False
        self._handles_visible = True
        init_annotation_item(self, annotation_id, editable=False)


# ======================================================================
# 门控与样式
# ======================================================================

def init_annotation_item(item: Any, annotation_id: str, editable: bool = False) -> Any:
    """标注图元统一初始化：挂 id、定 Z 序、按当前模式上门控"""
    item.annotation_id = annotation_id
    item._editable = False
    item._handles_visible = True
    item.setZValue(Z_ANNOTATION)
    set_item_editable(item, editable)
    if isinstance(item, pg.TextItem):
        item.setFont(_annotation_font(DEFAULT_FONT_SIZE))
    return item


def _gate_roi_children(item: Any, editable: bool, handles_on: bool) -> None:
    """ROI 的**子图元**必须单独关，光关容器盖不住

    - 手柄：``Handle.hoverEvent`` 无条件 acceptDrags，根本不看 ROI 的门控
      （只看容器标志位会漏掉这一条），所以显式 ``setVisible(False)``
      —— Qt 不向隐藏图元投递鼠标事件。顺带把按键也拔掉，双保险。
      按键用**赋值**不用 ``|=``：``PolyLineROI.addSegment`` 会把手柄按键
      OR 上 ``RightButton``（右键能删折点），订阅式地 OR 会越关越多。
    - 折线的线段：是独立图元，自带 LeftButton，浏览态下点线段会插折点。
    """
    for handle in getattr(item, "handles", []) or []:
        handle_item = handle.get("item") if isinstance(handle, dict) else None
        if handle_item is None:
            continue
        handle_item.setVisible(handles_on)
        handle_item.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton if handles_on else Qt.MouseButton.NoButton
        )
    for segment in getattr(item, "segments", []) or []:
        segment.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton if editable else Qt.MouseButton.NoButton
        )


def _apply_gating(item: Any) -> None:
    """按「编辑态 × 手柄请求态」重算图元的交互开关（唯一权威实现）"""
    editable = bool(_flag(item, "_editable", False))
    handles_on = editable and bool(_flag(item, "_handles_visible", True))

    if isinstance(item, pg.ROI):
        item.translatable = editable
        item.resizable = editable
        # 旋转恒关：模型用「两个对角点」表达矩形，转起来会丢角度信息
        item.rotatable = False
        item.removable = False
        item.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton if editable else Qt.MouseButton.NoButton
        )
        _gate_roi_children(item, editable, handles_on)
    elif isinstance(item, pg.TextItem):
        item.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton if editable else Qt.MouseButton.NoButton
        )
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, editable)
        # 位移由 mouseMoveEvent 自己算，见 AnnotationTextItem 的类 docstring
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
    elif isinstance(item, pg.TargetItem):
        # movable 挡的是鼠标拖拽与本图元对标签的转发；按键是更外层的闸门。
        # 两个都要关：只关一个都留一条能抢走 ViewBox 拖拽的路（见模块 docstring 第 6 条）
        item.movable = editable
        item.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton if editable else Qt.MouseButton.NoButton
        )
        # 标签是**独立子图元**，容器的按键状态管不到它。它自己实现 mouseClickEvent /
        # mouseDragEvent（都转发给父标靶），浏览态照吃按键 —— 从标签上起手拖拽会
        # 被它接走，视图就平不动了。同 ROI 手柄那条，得单独拔。
        label = item.label()
        if label is not None:
            label.setAcceptedMouseButtons(
                Qt.MouseButton.LeftButton if editable else Qt.MouseButton.NoButton
            )
    else:
        # 纯装饰图元（箭头头部）：永远不参与交互
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)


def set_item_editable(item: Any, editable: bool) -> None:
    item._editable = bool(editable)
    _apply_gating(item)


def set_item_handles_visible(item: Any, visible: bool) -> None:
    """演示视图用：只藏手柄，不动可编辑性"""
    item._handles_visible = bool(visible)
    _apply_gating(item)


def set_item_selected(item: Any, selected: bool) -> None:
    """选中态：抬 Z 序置顶

    P1 只做置顶（保证被选中的标注不被别的标注盖住），虚线选中框留 P3 ——
    描边换个颜色会覆盖用户自己选的颜色，代价比收益大。
    """
    item.setZValue(Z_ANNOTATION_SELECTED if selected else Z_ANNOTATION)


def _apply_handle_pens(item: Any, pen) -> None:
    """让 ROI 手柄跟着标注颜色走（默认手柄是青色的，和红色标注放一起很跳）"""
    for handle in getattr(item, "handles", []) or []:
        handle_item = handle.get("item") if isinstance(handle, dict) else None
        if handle_item is None or not hasattr(handle_item, "pen"):
            continue
        try:
            handle_item.pen = pg.mkPen(pen)
            handle_item.hoverPen = pg.mkPen(pen)
            handle_item.update()
        except Exception:
            # 手柄样式纯属观感，失败不该影响标注本身能不能用
            pass


def apply_style_to_item(item: Any, model: Any) -> None:
    """把模型的样式字段落到图元上（颜色 / 线宽 / 字号 / 填充）"""
    raw_width = int(getattr(model, "stroke_width", DEFAULT_STROKE_WIDTH))
    stroke = getattr(model, "stroke", DEFAULT_STROKE) or DEFAULT_STROKE
    fill = getattr(model, "fill", None)
    alpha = int(getattr(model, "fill_alpha", DEFAULT_FILL_ALPHA))

    if isinstance(item, pg.TextItem):
        item.setColor(stroke)
        item.setFont(_annotation_font(int(getattr(model, "font_size", DEFAULT_FONT_SIZE))))
        item.fill = pg.mkBrush(fill) if fill else pg.mkBrush(None)
        # 线宽 0 = 无外边框：必须显式 NoPen —— Qt 里 width=0 的笔是
        # cosmetic 1px，视觉上仍是 1 像素的框；TextItem.paint 只认
        # border.style() == NoPen 才跳过绘制
        item.border = pg.mkPen(None) if raw_width <= 0 else pg.mkPen(stroke, width=raw_width)
        item.update()
        return

    # 非文字类型不允许 0：直线/箭头宽 0 会隐形，标靶会只剩标签 —— 图元层钳回 1
    width = max(1, raw_width)
    pen = pg.mkPen(stroke, width=width)

    if isinstance(item, pg.ROI):
        item.setPen(pen)
        # hoverPen 只加粗、不换色：hover 变色会和「选中」的视觉语义打架
        item.hoverPen = pg.mkPen(stroke, width=width + 1)
        _apply_handle_pens(item, pen)
        apply_fill = getattr(item, "apply_fill", None)
        if callable(apply_fill):
            apply_fill(fill, alpha)
        return

    if isinstance(item, pg.TargetItem):
        # TargetItem 自带四个画笔（pen/hoverPen/brush/hoverBrush），默认是黄-紫、
        # 蓝-青的撞色组合，不一起改写的话标靶永远不跟标注颜色走
        item.setPen(pen)
        item.setHoverPen(pg.mkPen(stroke, width=width + 1))
        if fill:
            qcolor = QColor(fill)
            qcolor.setAlpha(alpha)
            item.setBrush(QBrush(qcolor))
            item.setHoverBrush(QBrush(qcolor))
        else:
            # 不填充：只留十字与圆环（给个全透明笔刷，别退回 pyqtgraph 的默认蓝）
            item.setBrush(pg.mkBrush(None))
            item.setHoverBrush(pg.mkBrush(None))
        label = item.label()
        if label is not None:
            label.setColor(stroke)
            label.setFont(_annotation_font(int(getattr(model, "font_size", DEFAULT_FONT_SIZE))))
        return

    if isinstance(item, pg.ArrowItem):
        item.setStyle(pen=pg.mkPen(stroke), brush=pg.mkBrush(stroke))


def apply_visible_to_item(item: Any, visible: bool) -> None:
    """显隐是图元自己的标志位；手柄的显隐由 _apply_gating 单独管"""
    item.setVisible(bool(visible))
    if isinstance(item, pg.ROI):
        # ROI 的手柄是子项，isVisible() 会跟随父项，但用户直接拖过手柄后
        # Qt 内部状态可能残留，显式同步一次更稳
        _apply_gating(item)


# ======================================================================
# 几何读写
# ======================================================================

def _free_handle_points(item: Any) -> list[tuple[float, float]]:
    """把「自由手柄」的坐标换算到父坐标系（= 数据坐标）

    手柄的 ``pos()`` 是 ROI 的**局部**坐标，ROI 整体平移只改自己的位置、手柄不动；
    直接读局部坐标会把整体平移丢掉（存模板再套回来就跳回原位，实测踩过）。
    用 ``mapToParent`` 而不是手工加偏移：ROI 哪天带上缩放/旋转也不会算错。
    """
    points = []
    for handle in item.getHandles():
        mapped = item.mapToParent(handle.pos())
        points.append((float(mapped.x()), float(mapped.y())))
    return points


def _to_local_points(item: Any, points) -> list[tuple[float, float]]:
    """数据坐标 → ROI 局部坐标（``_free_handle_points`` 的逆运算）"""
    local = []
    for x, y in points:
        mapped = item.mapFromParent(QPointF(float(x), float(y)))
        local.append((float(mapped.x()), float(mapped.y())))
    return local


def read_geometry(item: Any) -> list[tuple[float, float]]:
    """从图元回读几何（数据坐标），返回与模型 points 同构的点列表"""
    # PolyLineROI 不是 LineSegmentROI 的子类，两个分支互不覆盖，但都必须排在
    # 通用的 pg.ROI 分支**之前**（后者按 pos/size 解释，折线身上没有这两个语义）
    if isinstance(item, pg.PolyLineROI):
        return _free_handle_points(item)
    if isinstance(item, pg.LineSegmentROI):
        return _free_handle_points(item)[:2]
    if isinstance(item, pg.ROI):
        state = item.getState()
        pos, size = state["pos"], state["size"]
        rect = QRectF(float(pos[0]), float(pos[1]), float(size[0]), float(size[1])).normalized()
        return [(rect.left(), rect.top()), (rect.right(), rect.bottom())]
    if isinstance(item, pg.TargetItem):
        # 单点图元：TargetItem 的 pos 就是数据坐标（形状在像素里另算，与 pos 无关）
        point = item.pos()
        return [(float(point.x()), float(point.y()))]
    if isinstance(item, pg.TextItem):
        point = item.pos()
        return [(float(point.x()), float(point.y()))]
    return []


def write_geometry(item: Any, points: list[tuple[float, float]]) -> None:
    """把模型几何写到图元上（与 read_geometry 严格互逆）"""
    if isinstance(item, pg.PolyLineROI):
        item.setPoints(_to_local_points(item, points))
        # setPoints 会把手柄与线段**全部重建**（clearPoints + addFreeHandle +
        # addSegment），新手柄默认可见、新线段默认吃 LeftButton —— 门控必须重上，
        # 否则浏览态下凭空多出一批能拖的手柄
        _apply_gating(item)
    elif isinstance(item, pg.LineSegmentROI):
        # LineSegmentROI.setState 会直接读 state['pos']，缺键就 KeyError；
        # 先取全量 state 再改 points，避免自己拼一个残缺字典
        state = dict(item.getState())
        local = _to_local_points(item, points[:2])
        state["points"] = [local[0], local[1]]
        state.setdefault("pos", (0.0, 0.0))
        state.setdefault("size", (1.0, 1.0))
        state.setdefault("angle", 0)
        item.setState(state)
    elif isinstance(item, pg.ROI):
        (x0, y0), (x1, y1) = points[0], points[1]
        item.setState({
            "pos": (min(x0, x1), min(y0, y1)),
            "size": (abs(x1 - x0), abs(y1 - y0)),
            "angle": 0,
        })
    elif isinstance(item, pg.TargetItem):
        item.setPos(float(points[0][0]), float(points[0][1]))
    elif isinstance(item, pg.TextItem):
        item.setPos(float(points[0][0]), float(points[0][1]))


def update_arrow_head(head_item: AnnotationArrowHead, view_box: Any,
                      tail: tuple[float, float], head: tuple[float, float]) -> None:
    """把箭头头部摆到线段的头部端点，并按屏幕像素方向定向

    方向必须在屏幕像素里算：data 空间的角度经过非等比缩放后和视觉方向不一致。
    ArrowItem 的 pos 即箭尖（makeArrowPath 顶点在 0,0），所以直接 setPos 到
    head 点即可，不需要按头长回退。
    """
    if view_box is None:
        return
    try:
        tail_scene = view_box.mapViewToScene(QPointF(float(tail[0]), float(tail[1])))
        head_scene = view_box.mapViewToScene(QPointF(float(head[0]), float(head[1])))
    except Exception:
        return

    dx = head_scene.x() - tail_scene.x()
    dy = head_scene.y() - tail_scene.y()
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        # 两端重合：方向无定义，保持原样而不是给一个随机方向
        return

    unit_x, unit_y = dx / norm, dy / norm
    # angle=0 时箭身朝屏幕 +x（箭尖朝 -x）；箭身需朝 tail 侧 = (-ux, -uy)。
    # QTransform.rotate 在屏幕坐标系（y 向下）里正角为顺时针，所以 unit_y 取反：
    # 直接 atan2(unit_y, -unit_x) 会把头部沿水平轴镜像 —— 朝上的箭头三角朝下、
    # 斜箭头三角反折（实测 8 方向只有水平两支是对的）。
    angle = math.degrees(math.atan2(-unit_y, -unit_x))
    head_item.setStyle(
        angle=angle,
        headLen=ARROW_HEAD_LEN,
        headWidth=ARROW_HEAD_WIDTH,
        tailLen=None,
    )
    head_item.setPos(float(head[0]), float(head[1]))


# ======================================================================
# 工厂与拆解
# ======================================================================

def create_annotation_item(model: Any, *, editable: bool = False) -> Any:
    """按模型创建图元（箭头的头部由调用方另建，见 create_arrow_head）"""
    if model.kind == "text":
        item = AnnotationTextItem(
            model.text,
            annotation_id=model.id,
            editable=editable,
            point=model.points[0],
        )
    elif model.kind == "target":
        item = AnnotationTargetItem(
            model.points[0],
            annotation_id=model.id,
            editable=editable,
            text=model.text,
        )
    elif model.kind in ("rect", "ellipse"):
        (x0, y0), (x1, y1) = model.points[0], model.points[1]
        pos = (min(x0, x1), min(y0, y1))
        size = (max(abs(x1 - x0), MIN_SHAPE_SIZE), max(abs(y1 - y0), MIN_SHAPE_SIZE))
        cls = AnnotationRectROI if model.kind == "rect" else AnnotationEllipseROI
        item = cls(pos, size, annotation_id=model.id, editable=editable)
    elif model.kind in ("line", "arrow"):
        item = AnnotationLineROI(model.points[:2], annotation_id=model.id, editable=editable)
    elif model.kind == "polyline":
        item = AnnotationPolyLineROI(model.points, annotation_id=model.id, editable=editable)
    else:
        # 未知 kind：模型层白名单之外，不渲染（from_dict 已挡掉，这里是兜底）
        return None

    apply_style_to_item(item, model)
    apply_visible_to_item(item, bool(getattr(model, "visible", True)))
    set_item_editable(item, editable)
    return item


def create_arrow_head(model: Any) -> AnnotationArrowHead:
    head = AnnotationArrowHead(annotation_id=model.id)
    apply_style_to_item(head, model)
    head.setZValue(Z_ANNOTATION)
    return head


def teardown_annotation_item(view_box: Any, item: Any, decor: Any = None) -> None:
    """从视图与场景中拆除标注图元

    必须走 view_box.removeItem：它同时清 ViewBox.addedItems 与场景，
    只调 scene().removeItem 会把 item 永久留在 addedItems 里。
    先断开子图元的交互，避免 deleteLater 到真正析构之间那一轮事件循环里
    手柄/线段仍能吃到拖拽（项目里 mark_region 踩过同类问题）。
    """
    if isinstance(item, pg.ROI):
        for handle in getattr(item, "handles", []) or []:
            handle_item = handle.get("item") if isinstance(handle, dict) else None
            if handle_item is not None:
                try:
                    handle_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    handle_item.setVisible(False)
                except RuntimeError:
                    pass
        for segment in getattr(item, "segments", []) or []:
            try:
                segment.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            except RuntimeError:
                pass

    for target in (decor, item):
        if target is None:
            continue
        try:
            if view_box is not None:
                view_box.removeItem(target)
            elif target.scene() is not None:
                target.scene().removeItem(target)
        except RuntimeError:
            # C++ 侧已析构：拆解本就是收尾动作，不该再抛出去打断调用方
            continue
        # 注意：ArrowItem 继承 QGraphicsPathItem，是纯 QGraphicsItem 而不是
        # QObject，根本没有 deleteLater。摘出场景后由 Python 引用计数负责回收
        # （调用方随之丢弃对它的引用），这里不能无条件调用。
        if hasattr(target, "deleteLater"):
            try:
                target.deleteLater()
            except RuntimeError:
                continue
