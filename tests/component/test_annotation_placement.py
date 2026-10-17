"""标注落点测试：**模型坐标 → 屏幕位置** 的逐类核对。

为什么值得单独一个文件
----------------------
标注的状态有两套：模型里的数据坐标（会被存进模板、被换布局回放）和图元在
屏幕上的位置。任何一类图元漏了"构造期把坐标写下去"这一步，都会变成
"模型说在这、屏幕画在那"的**静默分裂** —— 不报错、不崩溃，只是画偏了；而
模板往返、布局回放、序列化幂等这些测试读的都是模型，照样全绿。

P1 的文字标注就是这么错的：``AnnotationTextItem`` 的构造函数没有位置参数，
``create_annotation_item`` 也没补一步写入，于是每条文字标注都停在 ViewBox
原点（子图左上角），模型里却记着用户点的坐标。这个文件就是那类错误的守卫 ——
新增 kind 忘了落坐标，第 2 条断言会直接红。

两条断言（逐 kind）
-------------------
1. ``read_geometry(item)`` 与 ``model.points`` 一致（回读闭环）
2. 图元锚点在屏幕上的位置，与"把 model.points[0] 经 ViewBox 投影出来的像素"
   一致（2 像素内）—— 这条才是真正守住落点的

非空转守卫
----------
两条断言都可能被"两边都退化成同一个点"骗过（例如双方都取到了子图原点）。
所以每个用例先断言**期望像素离子图原点足够远**：真把某类图元丢在原点，
位置对不上的同时这条守卫也会亮。
"""

from __future__ import annotations

import pandas as pd
import pytest

from PySide6.QtCore import QPointF
import pyqtgraph as pg

from src.ui.widgets.annotation_items import read_geometry

#: 参与核对的全部图元类型（与零干扰对照集合保持同一份口径）
ALL_KINDS = ("text", "rect", "ellipse", "line", "arrow", "polyline", "target")

#: 允许的像素误差：图元坐标往返里有 float→Qt 的取整，1 px 内是常态
TOLERANCE_PX = 2.0

#: 非空转守卫用的最小偏移：期望落点离子图原点至少这么远才算"测到了东西"
MIN_OFFSET_FROM_ORIGIN_PX = 80.0


def _annotation_geometry(kind: str, anchor: tuple[float, float]) -> list[tuple[float, float]]:
    """给 kind 造一组"几何上合法、且与 anchor 相关"的点"""
    x, y = anchor
    if kind in ("text", "target"):
        return [(x, y)]
    if kind == "polyline":
        return [(x, y), (x + 12.0, y + 18.0), (x + 24.0, y + 4.0)]
    # rect / ellipse / line / arrow：两个对角点，首个点就是左上/起点
    return [(x, y), (x + 14.0, y + 14.0)]


def _expected_pixel(pw, point: tuple[float, float]):
    """数据坐标 → 控件像素"""
    return pw.mapFromScene(pw.view_box.mapViewToScene(QPointF(float(point[0]), float(point[1]))))


def _item_origin_pixel(pw, item):
    """图元在自己身上"对应 model.points[0]"的那个点，换算成控件像素

    各类图元表达"第一个点"的位置不一样，这里是唯一需要分类的地方：
    - 折线 / 线段：第一个自由手柄（``read_geometry`` 也是这么读的）
    - 矩形 / 椭圆：ROI 的局部原点就是归一化后的左上角
    - 文字 / 标靶：局部原点就是锚点
    """
    if isinstance(item, (pg.LineSegmentROI, pg.PolyLineROI)):
        scene_point = item.mapToScene(item.getHandles()[0].pos())
    else:
        scene_point = item.mapToScene(QPointF(0.0, 0.0))
    return pw.mapFromScene(scene_point)


@pytest.fixture()
def placed(plot_factory, qapp):
    """已绘图、视图范围固定成 1:1（100×100 数据 → 800×600 控件）的 widget

    1:1 不是必须的，但让"像素误差"这句话有个稳定含义，排障时不用先换算。
    """
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    pw = plot_factory(df)
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a")
    pw.view_box.setXRange(0, 100, padding=0)
    pw.view_box.setYRange(0, 100, padding=0)
    qapp.processEvents()
    return pw


class TestPlacementMatchesModel:
    """逐类核对：图元画在哪，模型说在哪"""

    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_item_lands_on_the_model_point(self, placed, qapp, kind):
        pw = placed
        anchors: list[tuple[float, float]] = [(20.0, 20.0), (55.0, 70.0), (85.0, 35.0)]
        for anchor in anchors:
            points = _annotation_geometry(kind, anchor)
            style = {"text": f"{kind} 标注"} if kind in ("text", "target") else {}
            aid = pw.annotation_manager.create(kind, points, **style)
            assert aid is not None, f"{kind} 创建失败"
            qapp.processEvents()
            item = pw.annotation_manager._graphics[aid].item
            model = pw.annotation_manager.get(aid)

            # 1) 回读闭环
            read_back = read_geometry(item)
            assert read_back == pytest.approx(model.points, abs=1e-6), (
                f"{kind} 图元回读的几何与模型不一致"
            )

            # 2) 落点：锚点的屏幕位置 == 该数据点投影出来的像素
            origin_px = _item_origin_pixel(pw, item)
            expected_px = _expected_pixel(pw, points[0])
            distance = (
                (origin_px.x() - expected_px.x()) ** 2
                + (origin_px.y() - expected_px.y()) ** 2
            ) ** 0.5

            # 非空转守卫见 test_expected_pixel_is_not_the_corner
            assert distance <= TOLERANCE_PX, (
                f"{kind} 落点偏离：模型 {points[0]} 应在像素 {expected_px.toTuple()}，"
                f"实际在 {origin_px.toTuple()}（差 {distance:.1f}px）"
            )

            pw.annotation_manager.remove(aid)
            qapp.processEvents()

    def test_expected_pixel_is_not_the_corner(self, placed):
        """非空转守卫：期望落点必须离子图原点足够远

        没有这一条，上面那组断言可能在"两边都退化成子图原点"时依然通过。
        """
        pw = placed
        origin_px = pw.mapFromScene(pw.view_box.mapViewToScene(QPointF(0.0, 0.0)))
        for anchor in ((20.0, 20.0), (55.0, 70.0), (85.0, 35.0)):
            far = _expected_pixel(pw, anchor)
            distance = (
                (far.x() - origin_px.x()) ** 2 + (far.y() - origin_px.y()) ** 2
            ) ** 0.5
            assert distance > MIN_OFFSET_FROM_ORIGIN_PX, (
                f"数据点 {anchor} 的期望像素离原点只有 {distance:.1f}px，"
                "落点断言会退化成空转"
            )

    def test_text_is_draggable_in_edit_mode(self, placed, qapp):
        """文字标注在编辑态下必须点得中

        它曾经因为多开了一个 ItemIgnoresTransformations 而彻底点不中：
        Qt 的 BSP 空间索引按 sceneTransform 存包围盒，对这种图元取值不一致，
        ``scene.items(文字中心)`` 里根本查不到它。这是"能看见但摸不着"，
        比画错更隐蔽。
        """
        pw = placed
        pw.annotation_manager.set_edit_mode(True)
        aid = pw.annotation_manager.create("text", [(50.0, 50.0)], text="峰值 5.2℃")
        qapp.processEvents()
        item = pw.annotation_manager._graphics[aid].item

        in_widget = [
            entry for entry in pw.plot_item.scene().items(item.sceneBoundingRect().center())
        ]
        assert any(entry is item for entry in in_widget), (
            "文字图元不在落点的命中列表里 —— 场景索引查不到它，编辑态拖不动"
        )

        # 文字仍须保持固定像素尺寸（pyqtgraph 的 updateTransform 负责，不需要那个标志）
        before = item.sceneBoundingRect().size()
        pw.view_box.setXRange(0, 10, padding=0)
        pw.view_box.setYRange(0, 10, padding=0)
        qapp.processEvents()
        after = item.sceneBoundingRect().size()
        assert (round(after.width()), round(after.height())) == (
            round(before.width()),
            round(before.height()),
        ), "10× 缩放后文字尺寸变了，截图不可用"
