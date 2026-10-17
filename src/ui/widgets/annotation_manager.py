"""AnnotationManager —— 曲线标注管理器（责任链第 8 级，挂在链尾）

职责：标注的新建 / 删除 / 显隐 / 编辑模式 / 选中 / 序列化 / 箭头头部维护。

挂在链尾的理由：标注不参与任何既有数据流（不碰曲线、坐标轴、游标、标记区域），
只需要一个"能拿到 pw"的位置，因此对前七级是纯观察者，内部实现零改动。

状态存放：全部挂在 plot_widget 上（`_annotation_graphics`），不放在 Manager 里
—— Manager 随 widget 一起销毁重建（换布局时 create_subplots_matrix 会
deleteLater 掉全部 widget），状态跟着 widget 走才是同一生命周期。
跨布局保留由 MainWindow 级注册表负责（见 layout_manager.replay_annotations）。

零干扰的两条实现约束见 annotation_items 的模块 docstring。
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QRectF

from src.core.annotation_models import (
    ANNOTATION_KINDS,
    MAX_ANNOTATIONS_PER_PLOT,
    MAX_TEXT_LENGTH,
    MAX_UNDO_STEPS,
    AnnotationItem,
    minimum_points,
    new_annotation_id,
)
from src.core.logger import get_logger
from src.ui.widgets.annotation_items import (
    apply_style_to_item,
    apply_visible_to_item,
    create_annotation_item,
    create_arrow_head,
    read_geometry,
    set_item_editable,
    set_item_handles_visible,
    set_item_selected,
    teardown_annotation_item,
    update_arrow_head,
)

logger = get_logger("widget.annotation")

# apply_style 不可改的字段：改了不会同步图元，只会造成模型与画面分裂。
# kind 改了还会让已建好的图元类型对不上（矩形换椭圆不会变形状）。
_IMMUTABLE_FIELDS = frozenset({"id", "kind", "points", "anchor"})

# 粘贴位移占视图跨度的比例（4%）：够看出偏移、又不至于跑到视图外
PASTE_OFFSET_RATIO = 0.04


@dataclass
class _AnnotationGraphics:
    """一条标注的模型 + 图元（+ 箭头头部这类附属图元）"""

    model: AnnotationItem
    item: Any = None
    decor: Any = None
    slots: list = field(default_factory=list)   # [(signal, slot)]，拆解时逐条断
    kind: str = ""


class AnnotationManager:
    """标注管理器（不继承 QObject：与既有各 Manager 的形态保持一致）"""

    def __init__(self, event_handler: Any):
        if event_handler is None:
            raise ValueError("AnnotationManager requires a valid EventHandler instance")
        self._event_handler = event_handler
        self._edit_mode = False
        self._selected_id: str | None = None
        # 批量装载期间抑制逐条 emit：一次 load 只该有一次"变了"的播报，
        # 否则列表对话框会被重绘上百次
        self._syncing = False
        self._range_hooked = False
        self._arrow_count = 0
        # 撤销栈：快照式。_history 顶元素是**当前**状态，所以初始就是"空"这一份。
        # 存整份 dump 而不是命令对象，是为了让"新建/删除/拖动/改样式/粘贴"
        # 共用同一套机制 —— 每加一种操作就多写一个命令类，迟早漏一处。
        self._history: list[list[dict]] = [[]]
        self._future: list[list[dict]] = []
        self._restoring = False
        # 手势标记：一次拖动会连续触发几十次几何变更，逐次入栈会让撤销栈
        # 被一个手势冲垮（退一次只退一个像素），所以手势期间不入栈、收尾才入
        self._in_gesture = False

    # ------------------------------------------------------------------
    # 依赖链
    # ------------------------------------------------------------------
    @property
    def pw(self) -> Any:
        handler = self._event_handler
        if handler is None:
            raise RuntimeError(
                "AnnotationManager: dependency chain broken (_event_handler is None)"
            )
        return handler.pw

    def _view_box(self) -> Any:
        """取 ViewBox；widget 已进入销毁期时返回 None 而不是抛异常"""
        try:
            pw = self.pw
        except RuntimeError:
            return None
        if getattr(pw, "_is_being_destroyed", False):
            return None
        try:
            return getattr(pw, "view_box", None)
        except RuntimeError:
            return None

    @property
    def _graphics(self) -> dict[str, _AnnotationGraphics]:
        try:
            pw = self.pw
        except RuntimeError:
            return {}
        store = getattr(pw, "_annotation_graphics", None)
        if store is None:
            store = {}
            pw._annotation_graphics = store
        return store

    def _emit_changed(self) -> None:
        if self._syncing:
            return
        try:
            self.pw.annotations_changed.emit()
        except (RuntimeError, AttributeError):
            # widget 正在销毁：没人再听这条播报
            pass

    def _announce(self, text: str) -> None:
        """往主窗口消息区播报（独立 widget / 测试环境下静默跳过）"""
        try:
            window = self.pw.window()
        except RuntimeError:
            return
        if window is not None and hasattr(window, "_broadcast"):
            window._broadcast(text)

    # ------------------------------------------------------------------
    # 历史（撤销 / 重做）—— 快照式
    # ------------------------------------------------------------------
    def _snapshot(self) -> list[dict]:
        return self.dump()

    def _should_record(self) -> bool:
        """这次变更该不该进历史

        两类情况不记：
        - ``_syncing``：批量装载/回放期间，一次 load 只该是一步；
        - ``_restoring``：撤销自己触发的还原。不挡的话一次撤销会把"撤销"
          本身也记成一步，用户按 Ctrl+Z 会来回弹。
        """
        return not (self._syncing or self._restoring)

    def _record_history(self) -> None:
        """把"当前状态"记成新的一步（栈顶永远代表当前状态）"""
        # 显式操作一律视为手势结束：手势标记靠鼠标事件配对，图元若在手势中途
        # 被删掉就配不上对了，留一个 True 会让后续所有变更都不入栈
        self._in_gesture = False
        if not self._should_record():
            return
        state = self._snapshot()
        if self._history and self._history[-1] == state:
            # 状态没变（删了个不存在的 id、按了一下没拖动、重复设同一个颜色）
            # 不入栈，否则撤销栈会被一堆空步占满
            return
        self._history.append(state)
        del self._history[:-MAX_UNDO_STEPS]
        self._future.clear()

    def reset_history(self) -> None:
        """把当前状态当作新的历史起点

        换布局回放 / 模板装载之后要调：只清空栈会留下一份"空"的历史，
        下一次 Ctrl+Z 就会把刚回放出来的标注全抹掉。
        """
        self._in_gesture = False
        self._history = [self._snapshot()]
        self._future.clear()

    @property
    def can_undo(self) -> bool:
        return len(self._history) >= 2

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    def undo(self) -> bool:
        """撤销一步；没有可撤销的内容返回 False（不抛异常）"""
        if not self.can_undo:
            return False
        self._future.append(self._history.pop())
        self._restore(self._history[-1])
        self._announce(f"已撤销（还可撤销 {len(self._history) - 1} 步）")
        return True

    def redo(self) -> bool:
        """重做一步；没有可重做的内容返回 False"""
        if not self._future:
            return False
        state = self._future.pop()
        self._history.append(state)
        self._restore(state)
        self._announce(f"已重做（还可重做 {len(self._future)} 步）")
        return True

    def _restore(self, state: list[dict]) -> None:
        """把整份状态写回画面

        走 ``load(replace=True)`` 而不是逐个图元打补丁：模型层已经把"非法输入
        一律丢弃"收拢好了，打补丁等于把那份校验再写一遍、还必然漏。代价是
        整图重建（≤200 条，实测无感），换来撤销路径与装载路径**完全同源** ——
        装载路径有测试守着，撤销路径就跟着被守住了。
        """
        self._restoring = True
        try:
            self.load(state, replace=True)
        finally:
            self._restoring = False
            self._in_gesture = False

    # ------------------------------------------------------------------
    # 复制 / 粘贴
    # ------------------------------------------------------------------
    def _clipboard(self) -> list[dict] | None:
        """剪贴板存放位置：**容器级**，不挂在本 widget 上

        「把一张图的标注原样贴到另外几个子图」是这个功能的主场景（多分区对比图
        每个分区一套同样的标注），所以剪贴板必须活得比单个子图久、且被所有子图
        共享。真实应用里 ``pw.window()`` 是 MainWindow，天然是容器级；独立 widget
        与组件测试里 ``window()`` 退化成宿主替身，等于"同一宿主内共享"。
        不引入模块级全局：那会让跨用例互相污染，且违背"状态跟着生命周期走"。
        """
        try:
            pw = self.pw
        except RuntimeError:
            return None
        try:
            target = pw.window() or pw
        except RuntimeError:
            return None
        if target is None:
            return None
        store = getattr(target, "annotation_clipboard", None)
        if not isinstance(store, list):
            store = []
            try:
                target.annotation_clipboard = store
            except (AttributeError, RuntimeError):
                # C++ 侧已析构 / 宿主不允许挂属性：退化成"没有剪贴板"
                return None
        return store

    @property
    def clipboard_count(self) -> int:
        store = self._clipboard()
        return len(store) if store else 0

    def copy(self, *, all_items: bool = False) -> int:
        """把选中标注（或本图全部）放进剪贴板，返回复制的条数

        ``all_items=True`` 整图复制：多分区对比图里"把主驾那套标注照搬到副驾/
        后排"是常态，逐条复制太笨。没选中也没标注时返回 0（不抛异常）。
        """
        store = self._clipboard()
        if store is None:
            return 0
        if all_items:
            payload = self.dump()
        else:
            record = self._graphics.get(self._selected_id) if self._selected_id else None
            payload = [record.model.to_dict()] if record is not None else []
        if not payload:
            return 0
        # 原地改而不是重新赋值：别的子图可能已经持着这个 list 对象
        store[:] = payload
        self._announce(
            f"已复制本图全部 {len(payload)} 条标注" if all_items
            else f"已复制 {len(payload)} 条标注"
        )
        return len(payload)

    def copy_selection_or_all(self) -> int:
        """快捷键路径的复制：有选中就复制选中，没选中就整图复制

        菜单里两条是分开的（用户能看清自己在选哪个），但 Ctrl+C 只有一键 ——
        多分区场景里"把这张图的标注照搬到别的图"是主要用法，而按下 Ctrl+C 时
        经常并没有先点选某一条。此时退化成整图复制并播报，比什么都不发生有用；
        两条路径都会播报，用户不会不知道复制了什么。
        """
        if self._selected_id and self._graphics.get(self._selected_id) is not None:
            return self.copy()
        return self.copy(all_items=True)

    def paste(self) -> int:
        """把剪贴板内容粘贴到本子图，返回粘贴条数

        粘贴出的每条都换新 id：跨子图粘贴本身不会撞 id（各子图各有一张图元表），
        但两个子图共用同一个 id 会让标注列表、模板 diff 变得难读，也挡住
        "先粘一个、再改它"的正常用法。
        """
        store = self._clipboard()
        if not store:
            return 0
        dx, dy = self._paste_offset()
        payload: list[dict] = []
        for entry in store:
            if not isinstance(entry, dict):
                continue
            try:
                points = [
                    [float(point[0]) + dx, float(point[1]) + dy]
                    for point in entry.get("points") or []
                ]
            except (TypeError, ValueError, IndexError, KeyError):
                # 剪贴板内容是外部可写的（模板/其它子图的 dump），坏条目跳过就行
                continue
            if not points:
                continue
            item = dict(entry)
            item["id"] = new_annotation_id()
            item["points"] = points
            payload.append(item)
        if not payload:
            return 0
        attached = self.load(payload, replace=False)
        if attached:
            self._announce(f"已粘贴 {attached} 条标注")
        return attached

    def _paste_offset(self) -> tuple[float, float]:
        """粘贴位移：视图跨度的 4%，往右下错开

        完全不偏移会与原图严丝合缝地重合 —— 用户看不见发生了什么，会以为没粘上。
        """
        view_box = self._view_box()
        if view_box is None:
            return (0.0, 0.0)
        try:
            (x0, x1), (y0, y1) = view_box.viewRange()
            dx = (float(x1) - float(x0)) * PASTE_OFFSET_RATIO
            # 屏幕上是"往右下"：数据 y 轴向上，所以要取负
            dy = (float(y0) - float(y1)) * PASTE_OFFSET_RATIO
        except (TypeError, ValueError, RuntimeError):
            return (0.0, 0.0)
        if not (_finite(dx) and _finite(dy)):
            return (0.0, 0.0)
        return (dx, dy)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def edit_mode(self) -> bool:
        return self._edit_mode

    @property
    def selected_id(self) -> str | None:
        return self._selected_id

    def count(self) -> int:
        return len(self._graphics)

    def items(self) -> list[AnnotationItem]:
        """当前子图的标注模型列表（按创建/装载顺序）"""
        return [record.model for record in self._graphics.values()]

    def get(self, annotation_id: str) -> AnnotationItem | None:
        record = self._graphics.get(annotation_id)
        return record.model if record is not None else None

    def dump(self) -> list[dict]:
        """导出为可序列化列表（模板 / MainWindow 级注册表共用）"""
        return [record.model.to_dict() for record in self._graphics.values()]

    # ------------------------------------------------------------------
    # 创建 / 删除
    # ------------------------------------------------------------------
    def create(self, kind: str, points: list[tuple[float, float]], **style) -> str | None:
        """按给定几何新建一条标注，返回标注 id（失败返回 None）

        任何非法输入都返回 None 而不是抛异常：调用点分布在右键菜单、快捷键、
        模板装载等多条路径上，让异常漏出去会把一次"忽略这条"升级成一次崩溃。
        """
        if kind not in ANNOTATION_KINDS:
            logger.warning("未知标注类型，已忽略: %r", kind)
            return None
        if len(self._graphics) >= MAX_ANNOTATIONS_PER_PLOT:
            logger.warning("标注数量已达上限 %d，忽略新建", MAX_ANNOTATIONS_PER_PLOT)
            self._announce(f"标注数量已达上限（{MAX_ANNOTATIONS_PER_PLOT} 条）")
            return None

        try:
            clean_points = [[float(x), float(y)] for x, y in points]
        except (TypeError, ValueError):
            # 坐标不是数、或点不是二元组：整条丢弃（与模型层的"不丢单个坏点"同理）
            logger.warning("标注几何非法，已忽略: kind=%s points=%r", kind, points)
            return None

        payload = {"id": new_annotation_id(), "kind": kind, "points": clean_points}
        payload.update({k: v for k, v in style.items() if k not in ("id", "kind", "points")})
        model = AnnotationItem.from_dict(payload)
        if model is None or not model.is_valid():
            logger.warning("标注几何非法，已忽略: kind=%s points=%r", kind, points)
            return None
        annotation_id = self._attach(model)
        if annotation_id is not None:
            self._record_history()
        return annotation_id

    def create_at_context(self, kind: str) -> str | None:
        """在右键点击处新建标注，尺寸取当前视图跨度的 1/4

        不做"按下拖拽画框"：那条路要在 ViewBox 上拦截鼠标，会和既有的框选缩放
        正面冲突。改成"先给一个合适大小、再用手柄调"，编辑动作全部限定在
        标注图元自己身上，既有交互一行都不动。
        """
        view_box = self._view_box()
        if view_box is None:
            return None

        x_range, y_range = view_box.viewRange()
        x0, x1 = x_range
        y0, y1 = y_range
        center_x = getattr(view_box, "context_x", None)
        center_y = getattr(view_box, "context_y", None)
        if center_x is None or not _finite(center_x):
            center_x = (x0 + x1) / 2.0
        if center_y is None or not _finite(center_y):
            center_y = (y0 + y1) / 2.0

        half_w = abs(x1 - x0) / 8.0 or 1.0
        half_h = abs(y1 - y0) / 8.0 or 1.0

        if kind == "text":
            points = [(float(center_x), float(center_y))]
        elif kind == "target":
            points = [(float(center_x), float(center_y))]
        elif kind == "polyline":
            # 4 个折点的"之"字形：横跨上下文点两侧各半格，纵向上下各半格。
            # 给一个能直接看出形状的初始几何，用户拖手柄改比从两点拉起来快
            points = [
                (float(center_x - half_w), float(center_y - half_h)),
                (float(center_x), float(center_y + half_h)),
                (float(center_x + half_w), float(center_y - half_h)),
                (float(center_x + 2.0 * half_w), float(center_y + half_h)),
            ]
        else:
            points = [
                (float(center_x - half_w), float(center_y - half_h)),
                (float(center_x + half_w), float(center_y + half_h)),
            ]
        return self.create(kind, points)

    def remove(self, annotation_id: str) -> bool:
        record = self._graphics.pop(annotation_id, None)
        if record is None:
            return False
        if self._selected_id == annotation_id:
            self._selected_id = None
        self._detach(record)
        self._emit_changed()
        self._record_history()
        return True

    def clear(self, *, announce: bool = False, record: bool = True) -> int:
        """清空当前子图全部标注，返回清掉的条数

        ``record=False`` 只给 ``load`` 用：整份替换是**一次**操作，
        中间那个"空"的瞬时状态不该单独占一步撤销（否则撤一次只得到空图）。

        注意下面那个循环变量**不能叫 ``record``**：它会覆盖同名关键字参数，
        循环结束后 ``record`` 变成最后一个图元（真值），``record=False`` 整条失效
        （踩过：套用模板会多出一步"空图"，撤销一次得到空图）。
        """
        records = list(self._graphics.values())
        if not records:
            return 0
        self._graphics.clear()
        self._selected_id = None
        for entry in records:
            self._detach(entry)
        self._emit_changed()
        if announce:
            self._announce(f"已清除 {len(records)} 条标注")
        if record:
            self._record_history()
        return len(records)

    # ------------------------------------------------------------------
    # 装载 / 导出
    # ------------------------------------------------------------------
    def load(self, raw: Any, *, replace: bool = True, record: bool = True) -> int:
        """从序列化列表装载，返回成功装载条数

        逐条独立容错：单条非法只丢它自己。模板是用户手工攒的资产，
        不能因为一条坏记录整份装载失败。

        ``record=False`` 表示"把装载结果当成新的历史起点"（换布局回放用）：
        既不进撤销栈，也不留着旧历史 —— 留着的话下一次 Ctrl+Z 会把刚回放
        出来的标注整份抹掉。模板套用走默认的 True（那是可撤销的一步）。
        """
        models: list[AnnotationItem] = []
        skipped = 0
        for entry in raw or []:
            model = AnnotationItem.from_dict(entry)
            if model is None:
                skipped += 1
                continue
            models.append(model)
        if skipped:
            logger.warning("标注装载：跳过 %d 条非法记录", skipped)

        if replace:
            # 这次 clear 只算"整份替换"的中间态，不单独占一步撤销
            self.clear(record=False)

        attached = 0
        if models:
            # 保存/恢复而不是直接置 False：load 可能被 _restore 套着调，
            # 直接置 False 会让外层的手势与历史守卫在中途失效
            previous = self._syncing
            self._syncing = True
            try:
                for model in models:
                    if len(self._graphics) >= MAX_ANNOTATIONS_PER_PLOT:
                        logger.warning("标注装载：已达上限 %d，余下丢弃", MAX_ANNOTATIONS_PER_PLOT)
                        break
                    # id 冲突（同一份配置贴到两个子图）时改派新 id，避免图形表互相覆盖
                    if model.id in self._graphics:
                        model.id = new_annotation_id()
                    if self._attach(model) is not None:
                        attached += 1
            finally:
                self._syncing = previous
        self._emit_changed()

        if record:
            self._record_history()
        else:
            self.reset_history()
        return attached

    # ------------------------------------------------------------------
    # 编辑模式 / 显隐 / 选中
    # ------------------------------------------------------------------
    def set_edit_mode(self, on: bool) -> None:
        """切换编辑态。全 False 时图元连鼠标事件都不产生（见 annotation_items 约束 2）"""
        on = bool(on)
        if self._edit_mode == on:
            return
        self._edit_mode = on
        for record in self._graphics.values():
            set_item_editable(record.item, on)
            if record.decor is not None:
                set_item_editable(record.decor, False)
        if not on:
            self.select(None)

    def set_presentation(self, on: bool) -> None:
        """演示视图：只藏编辑期装饰（拖拽手柄），标注本身照常显示"""
        for record in self._graphics.values():
            set_item_handles_visible(record.item, not on)

    def set_visible(self, annotation_id: str, visible: bool) -> bool:
        record = self._graphics.get(annotation_id)
        if record is None:
            return False
        record.model.visible = bool(visible)
        apply_visible_to_item(record.item, record.model.visible)
        if record.decor is not None:
            record.decor.setVisible(record.model.visible)
        self._emit_changed()
        self._record_history()
        return True

    def select(self, annotation_id: str | None) -> None:
        if annotation_id is not None and annotation_id not in self._graphics:
            annotation_id = None
        previous = self._selected_id
        if previous == annotation_id:
            return
        self._selected_id = annotation_id
        if previous is not None and previous in self._graphics:
            set_item_selected(self._graphics[previous].item, False)
        if annotation_id is not None:
            set_item_selected(self._graphics[annotation_id].item, True)

    def apply_style(self, annotation_id: str, **changes) -> bool:
        """改样式 / 改文本

        走一遍模型层的 from_dict 而不是直接 setattr：颜色格式、线宽/字号范围、
        透明度范围都在那里收拢，绕过去就等于把校验丢掉了。

        **几何与类型不在可改字段内**：``points`` 改模型而不动图元会留下
        "模型说在这、屏幕画在那"的分裂状态（本方法不写图元几何）。几何只有
        两条合法入口 —— 用户在图上拖（走 sigRegionChanged 回写）和 ``load()``。
        """
        record = self._graphics.get(annotation_id)
        if record is None:
            return False

        payload = record.model.to_dict()
        payload.update(
            {k: v for k, v in changes.items() if k in payload and k not in _IMMUTABLE_FIELDS}
        )
        cleaned = AnnotationItem.from_dict(payload)
        if cleaned is None:
            return False

        record.model = cleaned
        if cleaned.kind == "target":
            # 改标靶文案必然重建标签（pyqtgraph 只给"换掉整个 TargetLabel"这一条路），
            # 所以必须排在 apply_style_to_item 之前 —— 重建出来的新标签是默认样式，
            # 紧接着那次调用才把它染成标注色
            record.item.set_label_text(cleaned.text[:MAX_TEXT_LENGTH])
        elif cleaned.kind == "text" and hasattr(record.item, "setText"):
            record.item.setText(cleaned.text[:MAX_TEXT_LENGTH])
        apply_style_to_item(record.item, cleaned)
        if record.decor is not None:
            apply_style_to_item(record.decor, cleaned)
        apply_visible_to_item(record.item, cleaned.visible)
        if cleaned.kind == "arrow":
            self._update_arrow_for(record)
        self._emit_changed()
        self._record_history()
        return True

    # ------------------------------------------------------------------
    # 内部：挂载 / 拆解 / 几何同步
    # ------------------------------------------------------------------
    #: 注入到图元上的回调名 → 管理器上的实现方法
    _HOOK_TARGETS = {
        "_on_geometry_changed": "_on_item_geometry_changed",
        "_on_edit_started": "_on_item_edit_started",
        "_on_edit_finished": "_on_item_edit_finished",
    }

    def _make_hook(self, hook: str, annotation_id: str):
        """把注入到图元上的回调绑到本管理器对应方法（图元只认 id 这一个参数）"""
        return functools.partial(getattr(self, self._HOOK_TARGETS[hook]), annotation_id)

    @staticmethod
    def _connect(item: Any, signal_name: str, handler: Any, annotation_id: str):
        """连一条信号，返回 ``(signal, slot)`` 供拆解时精确断开

        不用 ``disconnect()`` 全断：那会把别处（如既有管理器）挂上去的同名
        信号的连接一起摘掉，排障时极难定位。精确断就要求把 slot 留着。
        """
        signal = getattr(item, signal_name)
        slot = functools.partial(handler, annotation_id)
        signal.connect(slot)
        return (signal, slot)

    def _attach(self, model: AnnotationItem) -> str | None:
        view_box = self._view_box()
        if view_box is None:
            return None

        item = create_annotation_item(model, editable=self._edit_mode)
        if item is None:
            # 白名单里的 kind 都该有对应图元；走到这里说明模型与图形两侧脱节
            # （新加了 kind 却忘了补 create_annotation_item 的分支就是这种情况）。
            # 丢弃而不是留一条看不见也删不掉的"幽灵标注"。
            logger.debug("标注类型无对应图元，已跳过: %s", model.kind)
            return None

        decor = create_arrow_head(model) if model.kind == "arrow" else None
        try:
            # 走 PlotItem.addItem（内部转发到 ViewBox.addItem）：图元进 addedItems、
            # 拿到 _viewBox 引用，自动范围隔离由 dataBounds 覆写负责
            self.pw.plot_item.addItem(item)
            if decor is not None:
                self.pw.plot_item.addItem(decor)
        except RuntimeError:
            teardown_annotation_item(view_box, item, decor)
            return None

        # 注入回写回调（文字图元靠它同步位置；ROI 类走 sigRegionChanged）
        slots: list[tuple[Any, Any]] = []
        for hook in ("_on_geometry_changed", "_on_edit_started", "_on_edit_finished"):
            if hasattr(item, hook):
                setattr(item, hook, self._make_hook(hook, model.id))

        if hasattr(item, "sigRegionChanged"):
            slots.append(self._connect(item, "sigRegionChanged", self._on_item_geometry_changed, model.id))
        if hasattr(item, "sigPositionChanged"):
            # 标靶：TargetItem 在 setPos 里发这条，拖动与程序化定位都会走到
            slots.append(self._connect(item, "sigPositionChanged", self._on_item_geometry_changed, model.id))
        if hasattr(item, "sigRegionChangeStarted"):
            slots.append(self._connect(item, "sigRegionChangeStarted", self._on_item_edit_started, model.id))
        # 三个"手势结束"信号同名不同源，谁存在就连谁：ROI 拖本体/拖手柄走
        # sigRegionChangeFinished，标靶拖拽走 sigPositionChangeFinished，
        # 文字图元没有信号、在自己 mouseReleaseEvent 里调注入的钩子
        for signal_name in ("sigRegionChangeFinished", "sigPositionChangeFinished"):
            if hasattr(item, signal_name):
                slots.append(self._connect(item, signal_name, self._on_item_edit_finished, model.id))

        record = _AnnotationGraphics(
            model=model, item=item, decor=decor, slots=slots, kind=model.kind
        )
        self._graphics[model.id] = record
        if model.kind == "arrow":
            self._arrow_count += 1
            self._update_arrow_for(record)
        # 懒挂载：没有箭头的子图永远不会多这一条连接
        self._ensure_range_hook()

        self._emit_changed()
        return model.id

    def _detach(self, record: _AnnotationGraphics) -> None:
        if record.kind == "arrow":
            self._arrow_count = max(0, self._arrow_count - 1)
        for signal, slot in record.slots:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                # C++ 侧已析构 / 连接本就不存在：都是收尾路径上的正常情况
                pass
        for hook in self._HOOK_TARGETS:
            # hasattr 只吞 AttributeError；C++ 侧已析构时抛的是 RuntimeError，
            # 得自己兜住（收尾路径抛异常会把调用方的清理逻辑整段打断）
            try:
                if hasattr(record.item, hook):
                    setattr(record.item, hook, None)
            except RuntimeError:
                break
        teardown_annotation_item(self._view_box(), record.item, record.decor)

    def _on_item_geometry_changed(self, annotation_id: str, *_args) -> None:
        """图元几何变化 → 回写模型（只写模型，不反写图元，不存在回环）"""
        if self._syncing:
            return
        record = self._graphics.get(annotation_id)
        if record is None:
            return
        points = read_geometry(record.item)
        minimum = minimum_points(record.kind)
        if len(points) < minimum:
            # 中间态（例如折线重建手柄的间隙）：此时写进模型会让整条标注变非法，
            # 丢弃这一次，等图元稳定下来会再来一次
            return
        if len(points) != len(record.model.points) and record.kind != "polyline":
            # 除折线外点数由 kind 决定，对不上说明图元处在过渡状态，同丢弃
            return
        record.model.points = points
        if record.kind == "arrow":
            self._update_arrow_for(record)
        self._emit_changed()
        if not self._in_gesture:
            # 没被手势信号标住（例如有人直接调 write_geometry）：当场记一步，
            # 宁可多一步也不要漏
            self._record_history()

    def _on_item_edit_started(self, annotation_id: str, *_args) -> None:
        """手势开始：进入"连续变更合并成一步"的状态

        一次拖动会触发几十上百次 sigRegionChanged，逐次入栈的后果是撤销栈
        被一个手势冲垮 —— 用户按一次 Ctrl+Z 只退回一个像素。
        """
        self._in_gesture = True

    def _on_item_edit_finished(self, annotation_id: str, *_args) -> None:
        """手势结束：整个手势只记一步"""
        self._in_gesture = False
        self._record_history()

    # ------------------------------------------------------------------
    # 内部：箭头头部
    # ------------------------------------------------------------------
    def _ensure_range_hook(self) -> None:
        """按需挂 sigRangeChanged，用来给箭头头部重新定向

        懒挂载：没有箭头的子图永远不会多这一条连接。ArrowItem 是 pxMode
        （像素恒定），但方向必须在屏幕像素里算，视图一变就得重算。
        """
        if self._range_hooked or not self._arrow_count:
            return
        view_box = self._view_box()
        if view_box is None:
            return
        view_box.sigRangeChanged.connect(self._on_view_range_changed)
        self._range_hooked = True

    def _on_view_range_changed(self, *_args) -> None:
        # 早退守卫：这条连接在每次视图变化时都会被调用，必须近乎零成本
        if not self._arrow_count or self._syncing:
            return
        for record in list(self._graphics.values()):
            if record.kind == "arrow":
                self._update_arrow_for(record)

    def _update_arrow_for(self, record: _AnnotationGraphics) -> None:
        if record.decor is None or len(record.model.points) < 2:
            return
        update_arrow_head(
            record.decor,
            self._view_box(),
            record.model.points[0],
            record.model.points[-1],
        )
        record.decor.setVisible(record.model.visible)

    # ------------------------------------------------------------------
    # 命中测试（双击入口用）
    # ------------------------------------------------------------------
    #: 容差兜底的膨胀半径（设备像素）：底边/角点这类压边点击的抓取余量
    HIT_TOLERANCE_PX = 4.0

    def hit_test(self, scene_point: Any, tolerance_px: float = HIT_TOLERANCE_PX) -> str | None:
        """返回场景坐标处的标注 id，不命中返回 ``None``

        双击标注直接打开属性对话框的入口（plot_widget.mouseDoubleClickEvent）：
        在打开绘图变量编辑器**之前**先问一次本方法。命中判定走
        ``QGraphicsScene.items``（BSP 空间索引），与"能不能点中"是同一权威来源；
        图元是否接受鼠标事件不影响判定，所以浏览态（图元被门控成 NoButton）
        双击标注同样能进属性编辑。

        两段式：
        1. BSP 快路径 —— ``scene.items(点)``，沿 parent 链找 ``annotation_id``
           （手柄 / 折线线段 / 标靶标签这类子图元自身不带 id）。
        2. 容差兜底 —— Qt 的命中是半开区间（``QRectF.contains`` 不含右/下边界），
           恰好压在矩形底边或角点上的点 BSP 与 shape 精筛**都会漏**（实测双击
           矩形底边中点就触发），所以线性扫一遍本子图的标注图元，用膨胀
           ``tolerance_px`` 的探针矩形与 shape 相交判定。只会在快路径落空时
           才走到，双击频率下无性能顾虑；多条标注重叠的压边场景兜底可能
           返回非最上层的那条，属可接受误差。
        """
        try:
            scene = self.pw.plot_item.scene()
        except RuntimeError:
            return None
        if scene is None:
            return None
        for top in scene.items(scene_point):
            node = top
            while node is not None:
                annotation_id = getattr(node, "annotation_id", "")
                if annotation_id and annotation_id in self._graphics:
                    return annotation_id
                node = node.parentItem()
        return self._hit_test_linear(scene_point, tolerance_px)

    def _hit_test_linear(self, scene_point: Any, tolerance_px: float) -> str | None:
        """快路径的容差兜底（见 hit_test 的 docstring）"""
        try:
            x, y = float(scene_point.x()), float(scene_point.y())
        except (AttributeError, TypeError, ValueError):
            return None
        probe = QRectF(
            x - tolerance_px, y - tolerance_px, 2.0 * tolerance_px, 2.0 * tolerance_px
        )
        for record in self._graphics.values():
            for candidate in (record.item, record.decor):
                if candidate is None:
                    continue
                try:
                    path = candidate.mapToScene(candidate.shape())
                except RuntimeError:
                    continue
                if path.intersects(probe):
                    return record.model.id
        return None

    # ------------------------------------------------------------------
    # 对话框入口（独立 widget / 测试环境下窗口缺失时静默返回）
    # ------------------------------------------------------------------
    def open_property_dialog(self, annotation_id: str) -> bool:
        record = self._graphics.get(annotation_id)
        if record is None:
            return False
        try:
            from src.ui.dialogs.annotation_dialog import AnnotationDialog
        except ImportError:
            logger.warning("标注属性对话框不可用")
            return False
        try:
            parent = self.pw.window()
        except RuntimeError:
            parent = None
        dialog = AnnotationDialog(record.model, parent=parent)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return False
        self.apply_style(annotation_id, **dialog.result_changes())
        return True

    def open_list_dialog(self) -> bool:
        try:
            from src.ui.dialogs.annotation_list_dialog import AnnotationListDialog
        except ImportError:
            logger.warning("标注列表对话框不可用")
            return False
        try:
            parent = self.pw.window()
        except RuntimeError:
            parent = None
        dialog = AnnotationListDialog(self, parent=parent)
        dialog.exec()
        return True


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
