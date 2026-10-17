"""标靶图元 component 测试（offscreen）。

标靶（``target``）在七类图元里有两处独一份，本文件专门守它们：

1. **带附属标签子图元**。``TargetLabel`` 是独立的 ``TextItem``，自己实现
   ``mouseClickEvent`` / ``mouseDragEvent`` 并转发给父标靶 —— 容器的按键门控
   管不到它（同 ROI 手柄那条）。零干扰对照里的 ``_decorate_widget`` 用
   ``create_at_context("target")`` 造出来的是**无文案**标靶，等于没有标签，
   那条通路至今没被覆盖 —— 本文件补上，并且是从**标签身上起手拖拽**。
2. **标签文案必须包成 callable**。pyqtgraph 把含 ``{}`` 的字符串判成 format
   模板，``"目标 {24℃}"`` 会走到 ``str.format(x, y)`` 上抛 ``KeyError``。
   这是实测踩出来的，不是理论担忧。

另外两条与其他类共用、但在这里再钉一次（挡回归）：

3. **改文案 = 重建标签**，而重建出来的是**默认门控**（吃按键）。所以
   ``set_label_text`` 末尾必须重上门控 —— 少了那一句，浏览态下把标靶文案改一改
   就会凭空长出一个能吃鼠标的图元。
4. 拖动必须回写模型（``sigPositionChanged`` → ``_on_item_geometry_changed``）。

门控状态本身（容器按键 = NoButton）在 ``test_annotation_no_interference.py``
的 ``TestGating`` 里已按类覆盖；本文件只补**标签**那一层。
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPoint, Qt

from src.core.annotation_models import Z_ANNOTATION, Z_ANNOTATION_SELECTED
from src.ui.widgets.annotation_items import TARGET_SIZE, read_geometry
from tests.component._viewbox_events import mouse_event

#: 屏幕上"落在标靶符号上"的落点（与其它标注测试的中央区域一致）
_LANDING = QPoint(400, 300)
_DRAG_TO = QPoint(340, 260)


@pytest.fixture()
def annotated(plot_factory, qapp):
    """已绘图、视图范围固定成 0..100 的 widget（1:1 换算，便于算落点）"""
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    pw = plot_factory(df)
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a"), "曲线未绘出，后续断言失去意义"
    pw.view_box.setXRange(0, 100, padding=0)
    pw.view_box.setYRange(0, 100, padding=0)
    qapp.processEvents()
    return pw


def _manager(pw):
    return pw.annotation_manager


def _item(pw, aid):
    return _manager(pw)._graphics[aid].item


def _label(pw, aid):
    return _item(pw, aid).label()


def _make_target(pw, point, text: str = "") -> str:
    aid = _manager(pw).create("target", [point], text=text)
    assert aid is not None, f"标靶未创建: {point}"
    return aid


def _view_range(pw) -> tuple[float, float, float, float]:
    x, y = pw.view_box.viewRange()
    return (float(x[0]), float(x[1]), float(y[0]), float(y[1]))


def _reset_view(pw, qapp) -> tuple[float, float, float, float]:
    pw.view_box.setXRange(0, 100, padding=0)
    pw.view_box.setYRange(0, 100, padding=0)
    qapp.processEvents()
    return _view_range(pw)


def _rate_limit_gap() -> None:
    """GraphicsScene 有 ``mouseRateLimit``（默认 200/s = 5ms）：两次 move 挨得太近，
    后一次会被整个丢掉。真实事件循环里鼠标移动天然有间隔，这里补上。
    """
    time.sleep(0.02)


def _settle_autorange(pw, qapp, rounds: int = 10) -> tuple[float, float, float, float]:
    """把 ``autoRange()`` 推到不动点后返回视图范围

    ``ViewBox.autoRange`` 不是纯函数：它读当前 viewRect/targetRect 再写回，Y 又开着
    ``autoVisibleOnly``（Y 跟着可见 X 段走），X↔Y 互相影响，要几轮才稳定。
    因此**不能跨 widget 比较**两次 autoRange 的结果 —— 两个实例的历史不同，
    收敛值也不同，断言会变成噪声。能成立的比较是**同一 widget 的前后对照**。
    """
    for _ in range(rounds):
        pw.view_box.autoRange()
        qapp.processEvents()
        qapp.processEvents()
    return _view_range(pw)


def _hover_then_drag_at(pw, qapp, landing: QPoint, drag_to: QPoint):
    """先 hover 再 press+move，返回这一串手势最终落到谁身上

    必须补 hover：GraphicsScene 在 hover 阶段收集 ``dragItems``，一旦有图元登记了
    左键，``sendDragEvent`` 走 init 分支会**直接用登记的图元当 dragItem，不再回退
    去问 ViewBox** —— 视图就平不动了。只发 press+move 测不到这条路径。
    """
    scene = pw.plot_item.scene()
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    plain = Qt.KeyboardModifier.NoModifier

    _rate_limit_gap()
    pw.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, pw, landing, none, none, plain))
    qapp.processEvents()

    scene.dragItem = None
    scene.dragButtons = []
    _rate_limit_gap()
    pw.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, pw, landing, left, left, plain))
    qapp.processEvents()
    pw.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, pw, drag_to, none, left, plain))
    qapp.processEvents()

    dragged = scene.dragItem
    pw.mouseReleaseEvent(
        mouse_event(QEvent.Type.MouseButtonRelease, pw, drag_to, left, none, plain)
    )
    qapp.processEvents()
    return dragged


def _label_landing(pw, aid) -> QPoint:
    """标签包围盒中心在控件坐标里的位置（"从标签身上起手"用的落点）"""
    label = _label(pw, aid)
    assert label is not None, "标靶没有标签，本用例的前置条件不成立"
    center = label.sceneBoundingRect().center()
    return pw.mapFromScene(center)


# ---------------------------------------------------------------------------
# 标签：文案
# ---------------------------------------------------------------------------

class TestLabelText:
    def test_empty_text_means_no_label(self, annotated):
        aid = _make_target(annotated, (50.0, 50.0))
        assert _label(annotated, aid) is None, "空文案不该长出标签"

    def test_plain_text_shows_up(self, annotated, qapp):
        aid = _make_target(annotated, (50.0, 50.0), text="目标 24℃")
        qapp.processEvents()
        label = _label(annotated, aid)
        assert label is not None
        assert label.toPlainText() == "目标 24℃"

    def test_braces_in_text_do_not_crash(self, annotated, qapp):
        """回归：花括号文案曾直接把 setLabel 打成 KeyError

        pyqtgraph 对**非 callable** 的文案会先 ``string.Formatter().parse()``，
        只要含 ``{}`` 就判成 format 模板，随后 ``str.format(x, y)`` —— 花括号里是
        中文或非字段名就抛 ``KeyError``。包成 callable 后这条分支永不解析花括号。
        """
        aid = _make_target(annotated, (30.0, 40.0), text="目标 {24℃}")
        qapp.processEvents()
        label = _label(annotated, aid)
        assert label is not None
        assert label.toPlainText() == "目标 {24℃}"

    def test_long_text_is_truncated(self, annotated, qapp):
        """文案长度上限在模型层收口（与文字标注同一条）"""
        from src.core.annotation_models import MAX_TEXT_LENGTH

        aid = _make_target(annotated, (10.0, 10.0), text="甲" * (MAX_TEXT_LENGTH + 50))
        qapp.processEvents()
        assert len(_manager(annotated).get(aid).text) == MAX_TEXT_LENGTH
        assert _label(annotated, aid).toPlainText() == "甲" * MAX_TEXT_LENGTH

    def test_setting_the_same_text_does_not_rebuild(self, annotated, qapp):
        """同文案早退：否则改个颜色都会重建一遍标签"""
        aid = _make_target(annotated, (20.0, 20.0), text="A")
        qapp.processEvents()
        before = _label(annotated, aid)

        _item(annotated, aid).set_label_text("A")
        assert _label(annotated, aid) is before, "文案没变却重建了标签"

    def test_clearing_text_removes_the_label(self, annotated, qapp):
        aid = _make_target(annotated, (20.0, 20.0), text="有字")
        qapp.processEvents()
        assert _label(annotated, aid) is not None

        _manager(annotated).apply_style(aid, text="")
        qapp.processEvents()
        assert _label(annotated, aid) is None


# ---------------------------------------------------------------------------
# 标签：门控（容器的门控管不到它）
# ---------------------------------------------------------------------------

class TestLabelGating:
    def test_browse_mode_takes_the_label_buttons_away(self, annotated, qapp):
        aid = _make_target(annotated, (50.0, 50.0), text="目标")
        _manager(annotated).set_edit_mode(False)
        qapp.processEvents()

        label = _label(annotated, aid)
        assert label is not None
        assert label.acceptedMouseButtons() == Qt.MouseButton.NoButton, (
            "浏览态标签仍吃左键 —— 从标签上起手拖拽会被它接走，视图平不动"
        )

    def test_edit_mode_opens_the_label(self, annotated, qapp):
        """对照组：证明上一条不是恒真"""
        aid = _make_target(annotated, (50.0, 50.0), text="目标")
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()

        assert _label(annotated, aid).acceptedMouseButtons() == Qt.MouseButton.LeftButton

    def test_rebuilding_the_label_does_not_reopen_it(self, annotated, qapp):
        """**关键回归**：改文案重建标签后，门控必须重上

        ``setLabel`` 会造一个全新的 ``TargetLabel``，它的 ``acceptedMouseButtons``
        是默认值（吃键）。``set_label_text`` 末尾那句 ``_apply_gating`` 就是为
        这条存在的 —— 删掉它，浏览态下改个文案就凭空多出一个吃鼠标的图元，
        而画面上没有任何变化。
        """
        aid = _make_target(annotated, (50.0, 50.0), text="甲")
        _manager(annotated).set_edit_mode(False)
        qapp.processEvents()
        assert _label(annotated, aid).acceptedMouseButtons() == Qt.MouseButton.NoButton

        # 浏览态下改文案（模板套用 / 属性对话框都会走到这条路）
        _manager(annotated).apply_style(aid, text="乙")
        qapp.processEvents()

        label = _label(annotated, aid)
        assert label.toPlainText() == "乙", "标签没重建，本用例失去意义"
        assert label.acceptedMouseButtons() == Qt.MouseButton.NoButton, (
            "重建出来的标签带着默认按键，门控没有重上"
        )

    def test_style_change_keeps_the_label_gated(self, annotated, qapp):
        """只改颜色也走同一条重建路径（apply_style 内部会先 set_label_text）"""
        aid = _make_target(annotated, (50.0, 50.0), text="目标")
        _manager(annotated).set_edit_mode(False)
        qapp.processEvents()

        _manager(annotated).apply_style(aid, stroke="#00A0FF")
        qapp.processEvents()
        assert _label(annotated, aid).acceptedMouseButtons() == Qt.MouseButton.NoButton


# ---------------------------------------------------------------------------
# 标签：样式跟随
# ---------------------------------------------------------------------------

class TestLabelStyle:
    """注意 ``TextItem.color`` 是 **QColor 属性**（``setColor`` 里赋的值），
    不是方法 —— 写 ``.color()`` 会抛 ``TypeError: not callable``。
    """

    def test_label_follows_the_annotation_colour(self, annotated, qapp):
        """标签默认是浅灰字，不跟着标注颜色走的话在任何浅色主题上都看不清"""
        aid = _make_target(annotated, (50.0, 50.0), text="目标")
        _manager(annotated).apply_style(aid, stroke="#00A0FF")
        qapp.processEvents()

        assert _label(annotated, aid).color.name() == "#00a0ff"

    def test_rebuilt_label_keeps_the_colour(self, annotated, qapp):
        """重建顺序陷阱：``set_label_text`` 必须排在 ``apply_style_to_item`` **之前**

        反过来的话新标签是默认样式，紧接着那次染色就落在**旧标签**身上了 ——
        颜色看起来"随机丢掉一半"。
        """
        aid = _make_target(annotated, (50.0, 50.0), text="甲")
        _manager(annotated).apply_style(aid, stroke="#00A0FF")

        _manager(annotated).apply_style(aid, text="乙")
        qapp.processEvents()

        label = _label(annotated, aid)
        assert label.toPlainText() == "乙"
        assert label.color.name() == "#00a0ff", "重建后的标签丢了标注色"

    def test_symbol_pen_follows_the_annotation_colour(self, annotated, qapp):
        """TargetItem 自带四个画笔（pen/hoverPen/brush/hoverBrush），都要改写"""
        aid = _make_target(annotated, (50.0, 50.0), text="目标")
        _manager(annotated).apply_style(aid, stroke="#00A0FF", fill="#00FF00", fill_alpha=60)
        qapp.processEvents()

        item = _item(annotated, aid)
        assert item.pen.color().name() == "#00a0ff"
        assert item.brush.color().name() == "#00ff00"
        assert item.brush.color().alpha() == 60

    def test_no_fill_does_not_fall_back_to_the_default_blue(self, annotated, qapp):
        """不填充时笔刷必须是 ``NoBrush``

        直接读 ``brush.color().alpha()`` 会得到 255（``NoBrush`` 的 color 是无效
        默认值），看上去像"被填成不透明黑"。真正的判据是画笔风格。
        """
        aid = _make_target(annotated, (50.0, 50.0), text="目标")
        qapp.processEvents()
        assert _item(annotated, aid).brush.style() == Qt.BrushStyle.NoBrush, (
            "不填充时笔刷应退回无填充，不能是 pyqtgraph 的默认蓝"
        )


# ---------------------------------------------------------------------------
# 几何：单点读写 + 拖动回写
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_read_write_roundtrip(self, annotated, qapp):
        from src.ui.widgets.annotation_items import write_geometry

        aid = _make_target(annotated, (12.0, 34.0))
        item = _item(annotated, aid)
        assert read_geometry(item) == pytest.approx([(12.0, 34.0)])

        write_geometry(item, [(77.0, 8.0)])
        assert read_geometry(item) == pytest.approx([(77.0, 8.0)])

    def test_moving_the_symbol_writes_back_to_the_model(self, annotated, qapp):
        """真实拖拽由 ``setPos`` → ``sigPositionChanged`` 驱动（TargetItem 自己发）"""
        aid = _make_target(annotated, (20.0, 20.0))
        _item(annotated, aid).setPos(35.0, 42.0)
        qapp.processEvents()

        assert _manager(annotated).get(aid).points == pytest.approx([(35.0, 42.0)]), (
            "标靶移动没回写模型 —— 存模板再套回来会跳回原位"
        )

    def test_single_point_figures_are_rejected_when_empty(self, annotated):
        """几何下限在模型层收口：0 个点画不出标靶"""
        assert _manager(annotated).create("target", []) is None
        assert _manager(annotated).count() == 0

    def test_extra_points_are_truncated_to_one(self, annotated):
        """标靶是单点图元：多给的点由模型层截断（外部模板是自由输入）"""
        aid = _manager(annotated).create("target", [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
        assert aid is not None
        assert _manager(annotated).get(aid).points == pytest.approx([(1.0, 2.0)])

    def test_symbol_size_is_constant_on_screen(self, annotated, qapp):
        """标靶按 deviceTransform 反算形状：10× 缩放下屏幕尺寸逐位不变

        这与"截图可读"直接相关 —— 缩放后标靶跟着变大变小就废了。

        实测口径（``tmp/probe_target_size.py``）：本工程的场景坐标**就是控件
        像素**（1 scene 单位 = 1 屏幕像素），所以 ``sceneBoundingRect().size()``
        可以直接当屏幕尺寸用，不必再经 ``mapFromScene`` 换算。
        """
        aid = _make_target(annotated, (50.0, 50.0))
        item = _item(annotated, aid)
        qapp.processEvents()
        before = item.sceneBoundingRect().size()

        annotated.view_box.setXRange(0, 10, padding=0)
        annotated.view_box.setYRange(0, 10, padding=0)
        qapp.processEvents()
        after = item.sceneBoundingRect().size()

        assert (after.width(), after.height()) == pytest.approx(
            (before.width(), before.height())
        ), f"10× 缩放后标靶屏幕尺寸变了：{before.toTuple()} → {after.toTuple()}"
        # 且尺寸确实来自 TARGET_SIZE（不是退化成 0 或某个偶然值）
        assert before.width() >= TARGET_SIZE, (
            f"标靶屏幕尺寸只有 {before.width():.1f}px，小于设定的 {TARGET_SIZE}px"
        )

    def test_autorange_is_isolated(self, annotated, qapp):
        """远处标靶不该把自动范围撑开（``dataBounds`` 覆写隔离）

        比对必须都在**同一只 widget** 上做、且各自先把 ``autoRange`` 推到不动点：
        它读当前 viewRect/targetRect 再写回，两个实例的历史不同、收敛值也不同。
        唯一的变量只能是"有没有标靶"。
        """
        _manager(annotated).set_edit_mode(False)
        qapp.processEvents()

        baseline = _settle_autorange(annotated, qapp)
        # 不隔离的话自动范围会被撑到 1e6 量级
        _make_target(annotated, (1e6, 1e6), text="远处")
        with_target = _settle_autorange(annotated, qapp)

        assert with_target == baseline, (
            f"标靶参与了自动范围计算: 无标注 {baseline} / 有标靶 {with_target}"
        )


# ---------------------------------------------------------------------------
# 零干扰：从**标签**身上起手拖拽，仍须归 ViewBox
# ---------------------------------------------------------------------------

class TestPanParityFromTheLabel:
    """零干扰对照（``test_annotation_no_interference.py``）用的是无文案标靶，
    等于没测标签那条通路。这里补上：落点取在标签包围盒中心。
    """

    def test_landing_point_really_is_on_the_label(self, annotated, qapp):
        """前置条件：落点确实落在标签上（否则下面的对照是空转）"""
        aid = _make_target(annotated, (50.0, 50.0), text="目标值 24℃")
        qapp.processEvents()
        label = _label(annotated, aid)
        landing = _label_landing(annotated, aid)

        assert label.sceneBoundingRect().contains(annotated.mapToScene(landing)), (
            f"标签落点 {landing.toTuple()} 不在标签包围盒里"
        )
        # 且标签确实在符号右侧（offset=(20,0)），不是压在符号上
        assert landing.x() > annotated.mapFromScene(
            annotated.view_box.mapViewToScene(pg.Point(50.0, 50.0))
        ).x()

    def test_browse_mode_pan_is_identical_when_starting_on_the_label(self, annotated, qapp):
        """核心断言：从标签上起手拖，拖拽仍归 ViewBox，位移与无标注时逐位相同"""
        aid = _make_target(annotated, (50.0, 50.0), text="目标值 24℃")
        _manager(annotated).set_edit_mode(False)
        qapp.processEvents()
        landing = _label_landing(annotated, aid)

        decorated_before = _reset_view(annotated, qapp)
        decorated_drag = _hover_then_drag_at(annotated, qapp, landing, QPoint(landing.x() - 60, landing.y() - 40))
        decorated_after = _view_range(annotated)

        assert decorated_drag is annotated.view_box, (
            f"浏览态下从标签起手拖拽被 {decorated_drag} 接走了（视图平不动）"
        )

        # 对照：同一只 widget、同一串手势，只是把标注清掉
        _manager(annotated).clear()
        qapp.processEvents()
        clean_before = _reset_view(annotated, qapp)
        clean_drag = _hover_then_drag_at(annotated, qapp, landing, QPoint(landing.x() - 60, landing.y() - 40))
        clean_after = _view_range(annotated)

        assert clean_before == decorated_before
        assert clean_after != clean_before, "拖拽本身没生效，对照失去意义"
        assert clean_drag is annotated.view_box
        assert decorated_after == clean_after, (
            f"标靶标签改变了视图平移结果: 有标注 {decorated_after} / 无标注 {clean_after}"
        )

    def test_edit_mode_label_drag_moves_the_symbol_instead(self, annotated, qapp):
        """对照组：编辑态从标签上拖，本该移动标靶而不是平移视图"""
        aid = _make_target(annotated, (50.0, 50.0), text="目标值 24℃")
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()
        landing = _label_landing(annotated, aid)
        before_points = list(_manager(annotated).get(aid).points)
        before_view = _reset_view(annotated, qapp)

        dragged = _hover_then_drag_at(annotated, qapp, landing, QPoint(landing.x() - 60, landing.y() - 40))

        assert dragged is not annotated.view_box, "编辑态标签没接走拖拽，上一条断言可能恒真"
        assert _view_range(annotated)[:2] == before_view[:2], "编辑态拖标靶不该平移视图"
        assert _manager(annotated).get(aid).points != pytest.approx(before_points), (
            "编辑态从标签上拖没有移动标靶"
        )


# ---------------------------------------------------------------------------
# 图层序 / 模板往返 / 资源回收
# ---------------------------------------------------------------------------

class TestLifecycleAndRoundTrip:
    def test_selected_symbol_is_lifted(self, annotated):
        aid = _make_target(annotated, (50.0, 50.0), text="目标")
        assert _item(annotated, aid).zValue() == Z_ANNOTATION
        _manager(annotated).select(aid)
        assert _item(annotated, aid).zValue() == Z_ANNOTATION_SELECTED
        _manager(annotated).select(None)
        assert _item(annotated, aid).zValue() == Z_ANNOTATION

    def test_dump_load_keeps_text_and_position(self, annotated, qapp):
        manager = _manager(annotated)
        aid = _make_target(annotated, (18.0, 62.0), text="目标 {24℃}")
        payload = manager.dump()

        manager.clear()
        assert manager.load(payload) == 1
        assert manager.dump() == payload, "往返不收敛，模板会越存越偏"

        restored = manager.get(aid)
        assert restored is not None and restored.kind == "target"
        assert restored.points == pytest.approx([(18.0, 62.0)])
        assert restored.text == "目标 {24℃}"
        # 文案必须真的落到重建出来的标签上（不是只存在模型里）
        qapp.processEvents()
        assert _label(annotated, aid).toPlainText() == "目标 {24℃}"

    def test_create_remove_cycles_leave_no_scene_items(self, annotated, qapp):
        """标签是附属子图元：容器没回收干净的话场景里会留下幽灵"""
        manager = _manager(annotated)
        scene = annotated.plot_item.scene()
        baseline = len(scene.items())

        for _ in range(5):
            aid = _make_target(annotated, (25.0, 25.0), text="目标值 24℃")
            assert manager.remove(aid) is True
            qapp.processEvents()

        assert len(scene.items()) == baseline, "标靶或其标签没被回收"

    def test_clear_removes_the_label_too(self, annotated, qapp):
        manager = _manager(annotated)
        scene = annotated.plot_item.scene()
        baseline = len(scene.items())

        _make_target(annotated, (25.0, 25.0), text="目标值 24℃")
        qapp.processEvents()
        assert len(scene.items()) > baseline

        manager.clear()
        qapp.processEvents()
        assert len(scene.items()) == baseline, "clear 之后标签留在场景里"

    def test_presentation_view_keeps_the_symbol_visible(self, annotated, qapp):
        """演示视图只藏拖拽手柄；标靶没有手柄，必须原样可见"""
        aid = _make_target(annotated, (25.0, 25.0), text="目标")
        _manager(annotated).set_presentation(True)
        qapp.processEvents()
        assert _item(annotated, aid).isVisible()

        _manager(annotated).set_visible(aid, False)
        qapp.processEvents()
        assert not _item(annotated, aid).isVisible()
