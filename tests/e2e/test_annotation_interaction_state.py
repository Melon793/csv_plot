"""P0 出口条件收尾：真实 MainWindow 下，标注不得扰动 `_is_interacting` 状态机。

方案文档 P0 剩下的那一条出口条件写的是：

    真实主窗口下，加 10 个标注前后 sigRangeChanged 触发次数与 _is_interacting
    时序一致。

component 层已经守住「图元不抢鼠标」（``test_annotation_no_interference.py``），
但那边用的是**裸** ``PlotWidget`` —— 恰好绕过了本项目最容易出事的一条链路：

    ``EventHandler._sibling_is_interaction_source()`` 要经
    ``pw.window().plot_widgets`` 才看得见兄弟子图

裸 widget 没有 window、没有兄弟，级联抑制分支永远走不到；标注若在这条链路上留了
副作用（多一次范围变化、让防抖定时器早/晚一拍、把兄弟子图误判成交互源），只有
「真实窗口 + 多子图 + XLink 在链」才照得出来。

断言方式是**同一窗口前后对照**：先跑一遍脚本化交互序列录下时间线，再铺 10 条标注
（含拉到数据范围外的），从同一初始状态重跑同一序列，逐项比对。

为什么不做跨实例对照（baseline 窗口 vs decorated 窗口）：交互防抖、样式刷新、
autoRange 全都依赖各自的 repaint 历史，两个实例之间的差异本身就是噪声 —— 与
``tests/README.md`` 陷阱 #14 同源。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent, QPoint, Qt

from tests.component._viewbox_events import mouse_event, wheel_event

_FLAG = "flag"
_RANGE = "range"
_ANNOTATION_COUNT = 10

#: 每次跑序列前把窗口钉回这个尺寸。序列里那步 resize 会改几何，不钉的话两次运行
#: 的绘图区大小不同，像素→数据换算随之不同（实测拖拽 Y 平移量差 0.1），对照失效。
_WINDOW_W = 1100
_WINDOW_H = 720


class _Trace:
    """把 ``sigRangeChanged`` 与 ``_is_interacting`` 写进同一条时间线。

    两类事件必须**交错**记录：只比次数会漏掉「标志位翻转发生在错误的那一次范围
    变化之后」这类错位。

    记录点在 EventHandler 的槽**之后**（Qt 按连接顺序派发，EventHandler 的连接在
    widget 初始化时就建立，本记录器最后接上），所以每条范围事件记的是「这次范围
    变化处理完之后」的状态。这正好是要看的形态：同一次手势里，持有交互态的那个
    子图记成 ``flag=True``，被级联抑制的那个记成 ``flag=False``。

    顺带记下谁持有交互态：**普通手势（改主图范围）时持有者是从图**而不是被操作的主图。
    pyqtgraph 的 ``ViewBox.updateViewRange`` 先 ``link.linkedViewChanged()`` 通知从图、
    最后才 emit 自己的 ``sigRangeChanged``（见 ViewBox.py 该函数末尾），从图的
    ``_on_range_changed`` 因此先跑并抢到交互态，主图随后被级联抑制。这是既有行为
    （无标注时逐字相同），故下方断言只守住「同时只有一个持有者」这一结构性不变量，
    不写死是哪个索引 —— 直接写从图的范围时，持有者反倒是主图（反向传播）。
    """

    def __init__(self) -> None:
        self.entries: list[tuple] = []

    def range_changed(self, index: int, pw, view_box) -> None:
        (x0, x1), (y0, y1) = view_box.viewRange()
        self.entries.append(
            (
                _RANGE,
                index,
                round(float(x0), 9),
                round(float(x1), 9),
                round(float(y0), 9),
                round(float(y1), 9),
                bool(getattr(pw, "_is_interacting", False)),
                bool(getattr(pw, "_is_syncing_range", False)),
            )
        )

    def flag_changed(self, index: int, value) -> None:
        self.entries.append((_FLAG, index, bool(value)))

    def ranges(self) -> list[tuple]:
        return [e for e in self.entries if e[0] == _RANGE]

    def flags(self) -> list[tuple]:
        return [e for e in self.entries if e[0] == _FLAG]


def _format(entries: list[tuple]) -> str:
    return "\n".join(f"  [{i:>3}] {e}" for i, e in enumerate(entries))


def _range_slot(trace: _Trace, index: int, pw):
    """造一个 ``sigRangeChanged`` 槽。

    用 ``*args`` 而不是固定形参：本项目的 ``sigRangeChanged`` 实际带三个实参
    （第三个是 ``[x_changed, y_changed]``）。写成 ``(vb, rng, i=index)`` 这种
    带默认值的固定形参，第三个实参会**顶掉**那个默认值 —— 实测记录下来的
    "子图索引"直接变成了 ``[True, False]``，而时间线本身看着还挺像样。
    """

    def slot(*_args):
        trace.range_changed(index, pw, _args[0])

    return slot


def _install_flag_recorder(monkeypatch, widget_cls, index_of, trace):
    """把 ``_is_interacting`` 换成带记录的可写属性，返回还原函数。

    ``_is_interacting`` 既没有信号、也不是类属性 —— 它是 ``_setup_plot_area`` 里
    第一次赋值时才落到实例 ``__dict__`` 的普通 Python 属性。要记录「写入的那一刻」
    只能上属性，而且必须**上到类上**：写入方有三处 —— ``EventHandler`` 与
    ``CursorManager`` 各自的 property setter、以及 ``PlotUIManager``
    ``_setup_plot_area`` 里的直接赋值 —— 它们只有类这一层是公共的。

    真值搬去 ``_is_interacting_recorded``，对外仍叫 ``_is_interacting``：数据属性
    描述符的优先级高于实例 ``__dict__``，所以补丁期间同名旧条目自动失效；
    补丁在 widget 创建**之前**安装，新 widget 的首次赋值就直接落进新存储，不留旧条目。
    只记**翻转**而不是每次写入：重复写同一个值语义上是空操作，记进来只会让
    时间线对机械性差异过敏。
    """
    storage = "_is_interacting_recorded"

    def getter(self):
        return self.__dict__.get(storage, False)

    def setter(self, value):
        value = bool(value)
        changed = self.__dict__.get(storage, False) != value
        self.__dict__[storage] = value
        if not changed:
            return
        index = index_of.get(id(self))
        if index is None:
            return
        trace.flag_changed(index, value)

    # raising=False：这是**新增**类属性（原实现只有实例属性），不是改已有值
    monkeypatch.setattr(
        widget_cls, "_is_interacting", property(getter, setter), raising=False
    )

    def restore(widgets) -> None:
        """把真值搬回原名，别让窗口带着一个只在测试里存在的属性名收尾。

        还原时补丁还没撤，直接写 ``__dict__`` 绕开 property。
        """
        for pw in widgets:
            pw.__dict__["_is_interacting"] = pw.__dict__.pop(storage, False)

    return restore


def _spy_xlink_sync(mw) -> list:
    """统计 XLink 同步**跑完**的次数。

    不能只看 ``mw._pending_xlink_sync``：它在 ``_sync_linked_x_ranges`` 开头就落回
    False，等它等于 False 时同步可能还没跑完 —— 用它当完成条件会在同步中途放行，
    后面的步骤就跟一次正在进行的范围写入交错，时间线长度随机漂移。
    """
    done: list = []
    manager = mw.layout_manager
    original = manager._sync_linked_x_ranges

    def wrapper():
        original()
        done.append(True)

    manager._sync_linked_x_ranges = wrapper
    return done


class _Harness:
    """把「重置 → 跑序列 → 收时间线」这套动作收在一起。"""

    def __init__(self, mw, pws, trace, sync_done, qapp, qtbot) -> None:
        self.mw = mw
        self.pws = pws
        self.trace = trace
        self.sync_done = sync_done
        self.qapp = qapp
        self.qtbot = qtbot

    # ------------------------------------------------------------------
    # 状态控制
    # ------------------------------------------------------------------
    def quiesce(self) -> None:
        """排空交互防抖与 XLink 同步，回到静止态。

        每个手势之后都必须等：``_on_range_changed`` 起的是 50ms 防抖定时器，
        ``_end_interaction`` 会写一次 ``_is_interacting = False``。不等它，这次翻转
        就会漂到下一步里，两次运行的时间线长度对不上 —— 那是模拟的锅，不是功能的
        问题。
        """
        for pw in self.pws:
            self.qtbot.waitUntil(
                lambda pw=pw: not getattr(pw, "_is_interacting", False), timeout=3000
            )
        self.qtbot.waitUntil(
            lambda: not getattr(self.mw, "_pending_xlink_sync", False), timeout=3000
        )
        self.qapp.processEvents()

    def reset_view(self) -> None:
        """把窗口几何与视图范围钉回固定起点。

        **几何必须一起钉**：序列最后一步会 ``resize`` 撑大窗口，不收回来的话第二次
        运行就从一个更大的起点开始，拖拽的像素→数据换算随之改变（实测 Y 平移量差
        0.1），两次时间线根本不可比。这不是功能问题，是夹具没控制住变量。

        第 2 个子图只设 Y，X 由 XLink 跟随主图。
        """
        self.mw.resize(_WINDOW_W, _WINDOW_H)
        master = self.pws[0]
        master.view_box.setXRange(0.5, 4.2, padding=0)
        for pw in self.pws:
            pw.view_box.setYRange(5.0, 40.0, padding=0)
        self.quiesce()

    def plot_center(self, pw) -> QPoint:
        """绘图区中心（控件坐标）：用 ViewBox 的 scene 矩形映射回来。

        不写死像素坐标：真实窗口里 widget 的尺寸由布局（顶栏 + 侧栏 + splitter）
        决定，写死会随着窗口宽度变化落到绘图区外面，手势就静默失效了 —— 而静默
        失效在前后对照里看不出来。

        PySide6 的 ``QWidget.mapFromScene`` 传 QPointF 进来直接返回 QPoint，
        没有 ``.toPoint()`` 可用。
        """
        center = pw.view_box.sceneBoundingRect().center()
        return QPoint(pw.mapFromScene(center))

    # ------------------------------------------------------------------
    # 脚本化交互序列
    # ------------------------------------------------------------------
    def run_sequence(self) -> None:
        """刻意覆盖三条会写 ``_is_interacting`` 的路径。"""
        master = self.pws[0]
        center = self.plot_center(master)

        # 1) 程序化改范围：主图入交互态 → XLink 把范围传给兄弟 → 兄弟应被级联抑制
        master.view_box.setXRange(1.0, 4.0, padding=0)
        self.quiesce()

        # 2) 滚轮缩放（落点 = 视图中心，标注正压在下面）
        master.wheelEvent(wheel_event(master, center, 120))
        self.quiesce()

        # 3) 左键拖拽平移（同一落点起拖）
        self._drag(master, center, center - QPoint(40, 25))
        self.quiesce()

        # 4) 窗口级 XLink 同步：真实入口是 MainWindow.resizeEvent → _handle_resize
        #    → _schedule_xlink_sync。这里走真实 resize，并确认它真的排上了同步 ——
        #    否则本步会退化成空跑，而空跑在前后对照里是看不出来的。
        before = len(self.sync_done)
        self.mw.resize(self.mw.width() + 40, self.mw.height() + 24)
        self.qapp.processEvents()
        assert getattr(self.mw, "_pending_xlink_sync", False), (
            "resize 没触发 XLink 同步调度，第 4 步退化成空跑（前提失效）"
        )
        self.qtbot.waitUntil(
            lambda: len(self.sync_done) > before
            and not getattr(self.mw, "_pending_xlink_sync", False),
            timeout=3000,
        )
        self.quiesce()

        # 5) 断开从图的 XLink（"链接丢了"），再让窗口自己去同步。
        #    健康检查分支会在 _sync_guard 里重新 setXLink，这条路径才会真的写范围、
        #    也才走得到那条守卫 —— 这正是 P1-19 关心的形态，标注不得把它破坏掉。
        #
        #    试过更直白的"只把从图范围挪开"：没用。写从图会经
        #    ``link.linkedViewChanged`` 反向传播到主图，两边立刻又一致了，同步跑完
        #    什么都不用写（实测时间线里一条带守卫的回报都没有）。
        follower = self.pws[1]
        follower.view_box.setXLink(None)
        follower.view_box.setXRange(1.0, 3.0, padding=0)  # 掉队
        self.quiesce()
        before = len(self.sync_done)
        self.mw.layout_manager._schedule_xlink_sync()
        self.qtbot.waitUntil(
            lambda: len(self.sync_done) > before
            and not getattr(self.mw, "_pending_xlink_sync", False),
            timeout=3000,
        )
        self.quiesce()
        assert follower.view_box.linkedView(0) is self.pws[0].view_box, (
            "健康检查没把断掉的 XLink 接回来，第 5 步没走到守卫路径"
        )

    def _drag(self, pw, start: QPoint, end: QPoint) -> None:
        left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        plain = Qt.KeyboardModifier.NoModifier
        pw.mousePressEvent(
            mouse_event(QEvent.Type.MouseButtonPress, pw, start, left, left, plain)
        )
        self.qapp.processEvents()
        pw.mouseMoveEvent(
            mouse_event(QEvent.Type.MouseMove, pw, end, none, left, plain)
        )
        self.qapp.processEvents()
        pw.mouseReleaseEvent(
            mouse_event(QEvent.Type.MouseButtonRelease, pw, end, left, none, plain)
        )
        self.qapp.processEvents()

    # ------------------------------------------------------------------
    # 装饰
    # ------------------------------------------------------------------
    def decorate(self, count: int = _ANNOTATION_COUNT) -> None:
        """铺 count 条标注：多数压在当前视图内（含视图中心），一条远在数据范围外。

        几何按**当前视图跨度**算而不是写死数值：这样「矩形盖住手势落点」这个前提
        不会随数据或窗口尺寸漂移。远处那条顺带踩 autorange 隔离。
        """
        made = 0
        for pw in self.pws:
            (x0, x1), (y0, y1) = pw.view_box.viewRange()
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            w, h = (x1 - x0) / 8.0, (y1 - y0) / 8.0
            far = max(abs(x1), abs(y1)) * 50.0
            specs = (
                ("rect", [(cx - w, cy - h), (cx + w, cy + h)]),  # 盖住手势落点
                ("ellipse", [(x0 + w, y0 + h), (x0 + 3 * w, y0 + 3 * h)]),
                ("text", [(cx, y0 + h)]),
                ("arrow", [(cx - 2 * w, cy), (cx, cy)]),
                ("rect", [(far, far), (far * 1.1, far * 1.1)]),  # 数据范围之外
            )
            for kind, points in specs:
                assert pw.annotation_manager.create(kind, points) is not None
                made += 1
        assert made == count, f"应铺 {count} 条标注，实际 {made}"


@pytest.fixture
def harness(loaded_window, qapp, qtbot, monkeypatch):
    """真实 MainWindow + 2×1 XLink 子图 + 时间线记录器。"""
    from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

    mw = loaded_window

    # 顺序要紧：三件都在「建矩阵」之前装上 —— 建矩阵会把旧的 1×1 widget 换成新的
    # 一批，补丁若晚一步，新 widget 的 `_is_interacting` 就落回旧存储、记录不到；
    # XLink 同步的定时器也会拿着旧的 bound method，绕开计数器。
    trace = _Trace()
    index_of: dict[int, int] = {}
    restore = _install_flag_recorder(
        monkeypatch, DraggableGraphicsLayoutWidget, index_of, trace
    )
    sync_done = _spy_xlink_sync(mw)

    mw.layout_manager.create_subplots_matrix(2, 1)
    qapp.processEvents()

    pws = [c.plot_widget for c in mw.plot_widgets]
    assert len(pws) == 2, f"应有 2 个子图，实际 {len(pws)}"
    for index, pw in enumerate(pws):
        index_of[id(pw)] = index
        assert pw.plot_variable("speed"), f"子图 {index} 画图失败"
    qapp.processEvents()

    # 前提：子图间 X 轴 XLink 是活的，否则级联抑制分支根本走不到，整个文件变成空跑
    assert pws[1].view_box.linkedView(0) is pws[0].view_box, (
        "子图未建立 XLink，级联链路是死的"
    )

    for index, pw in enumerate(pws):
        # 槽必须收 *args：本项目的 sigRangeChanged 实际带第三个实参（changed
        # 三元组），写成固定形参又用默认值兜底的话，第三个实参会把默认值顶掉，
        # 记下来的"索引"就变成了 changed。
        pw.view_box.sigRangeChanged.connect(_range_slot(trace, index, pw))

    h = _Harness(mw, pws, trace, sync_done, qapp, qtbot)
    h.reset_view()  # 排掉建矩阵/画图留下的残留（可能还有一次待跑的 XLink 同步）
    yield h
    restore(pws)


# ---------------------------------------------------------------------------
# 核心：10 条标注前后，交互时间线逐项一致
# ---------------------------------------------------------------------------

class TestInteractionTimeline:
    def test_timeline_is_not_vacuous(self, harness):
        """先证明时间线本身有内容。

        没有这一条，下面「前后相等」的断言在序列静默失效时同样会通过 —— 两边都
        空着当然相等。
        """
        harness.reset_view()
        harness.trace.entries.clear()
        harness.run_sequence()

        assert len(harness.trace.ranges()) >= 4, (
            f"范围事件太少，序列没跑起来:\n{_format(harness.trace.entries)}"
        )
        values = [e[2] for e in harness.trace.flags()]
        assert True in values, "没有任何一次进入交互态"
        assert False in values, "没有任何一次退出交互态"
        assert any(e[7] is True for e in harness.trace.ranges()), (
            "整条时间线里没有一条带 _is_syncing_range 保护的回报 —— "
            "第 5 步（XLink 同步）退化成空跑，那条守卫路径没被走到\n"
            f"{_format(harness.trace.entries)}"
        )

    def test_timeline_identical_before_and_after_ten_annotations(self, harness):
        harness.reset_view()
        harness.trace.entries.clear()
        harness.run_sequence()
        baseline = list(harness.trace.entries)

        harness.decorate()
        harness.reset_view()
        harness.trace.entries.clear()
        harness.run_sequence()
        decorated = list(harness.trace.entries)

        assert decorated == baseline, (
            "标注存在时交互时间线发生偏移\n"
            f"--- 无标注（{len(baseline)} 项）---\n{_format(baseline)}\n"
            f"--- 有标注（{len(decorated)} 项）---\n{_format(decorated)}"
        )

    def test_decorating_does_not_move_the_view(self, harness):
        """铺标注本身不得改动视图范围（顺带证明上一条的起点是同一处）。"""
        harness.reset_view()
        harness.quiesce()
        before = {i: pw.view_box.viewRange() for i, pw in enumerate(harness.pws)}

        harness.decorate()
        harness.quiesce()

        for index, pw in enumerate(harness.pws):
            got = pw.view_box.viewRange()
            want = before[index]
            assert [[float(v) for v in axis] for axis in got] == [
                [float(v) for v in axis] for axis in want
            ], f"子图 {index} 铺标注后视图范围被改动: {want} -> {got}"

    def test_single_interaction_owner_per_gesture(self, harness):
        """每次手势只能有一个子图持有交互态，且进入/退出严格配对。

        这是 component 层照不到的：裸 widget 没有兄弟，级联抑制分支永远不执行，
        「同时只有一个持有者」这条不变量无从检验。标注若在 XLink 链路上多引发一次
        范围变化，第二个子图就会同时进入交互态 —— 表现就是这里出现连续的两次
        ``True``（没有配对的退出）。

        不写死持有者是哪个索引：实测是**从图**持有（见 ``_Trace`` 的说明），这是
        pyqtgraph 的信号派发顺序决定的既有行为，不是本特性的一部分。
        """
        harness.decorate()
        harness.reset_view()
        harness.trace.entries.clear()
        harness.run_sequence()

        transitions = [(e[1], e[2]) for e in harness.trace.flags()]
        assert transitions, "一次交互态翻转都没有，本用例是空跑"
        for position, (_index, value) in enumerate(transitions):
            entering = position % 2 == 0
            assert value is entering, (
                f"第 {position} 次翻转应为{'进入' if entering else '退出'}态，"
                f"实际 {value} —— 出现了同时持有者或未配对的翻转: {transitions}\n"
                f"{_format(harness.trace.entries)}"
            )
        entered = {transitions[i][0] for i in range(0, len(transitions) - 1, 2)}
        exited = {transitions[i][0] for i in range(1, len(transitions), 2)}
        assert entered == exited, (
            f"进入与退出的不是同一个子图: {transitions}\n"
            f"{_format(harness.trace.entries)}"
        )

    def test_timeline_returns_to_baseline_after_annotations_are_cleared(self, harness):
        """删掉标注后再跑一遍，时间线回到「从未加过标注」那一份（回收彻底）。

        与 ``test_autorange_after_removal_returns_to_baseline`` 同构：前者管自动范围，
        这条管交互状态机。标注若在场景里留下悬挂的图元/信号连接，这里会露出来。
        """
        harness.reset_view()
        harness.trace.entries.clear()
        harness.run_sequence()
        baseline = list(harness.trace.entries)

        harness.decorate()
        assert harness.pws[0].annotation_manager.count() > 0
        harness.pws[0].annotation_manager.clear()
        harness.pws[1].annotation_manager.clear()
        harness.reset_view()
        harness.trace.entries.clear()
        harness.run_sequence()

        assert harness.trace.entries == baseline, (
            "清掉标注后交互时间线没回到基线\n"
            f"--- 基线（{len(baseline)} 项）---\n{_format(baseline)}\n"
            f"--- 清理后（{len(harness.trace.entries)} 项）---\n"
            f"{_format(harness.trace.entries)}"
        )

