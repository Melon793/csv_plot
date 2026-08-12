"""
EventHandler - 事件处理管理

负责 DraggableGraphicsLayoutWidget 的 ViewBox 信号处理和交互事件：
- ViewBox 范围变化回调
- ViewBox 菜单信号处理
- 交互开始/结束事件
- 光标几何更新调度

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
import time
from typing import Any, TYPE_CHECKING

from PySide6.QtCore import QTimer

from src.core.config import (
    safe_callback,
    UI_DEBOUNCE_DELAY_MS,
    PERF_LOG_ENABLED,
    PERF_RANGE_CB_WARN_MS,
    PERF_INTERACTION_WARN_MS,
    ASYNC_Y_RESTORE_BATCH_SIZE,
)
from src.core.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.ui.widgets.mark_region_manager import MarkRegionManager


class EventHandler:
    """负责 ViewBox 信号处理和交互事件"""

    # 全局 Y restore 会话代数：任何 EventHandler 开启新交互时递增，
    # 用于作废其他 EventHandler 残留的异步 restore 批次（避免 plot B
    # 交互期间 plot A 的残留批次提前解冻兄弟子图，导致 Y 抖动短暂复发）
    _y_restore_generation = 0

    def __init__(self, mark_region_manager: MarkRegionManager):
        """初始化事件处理器，绑定到 MarkRegionManager 以获取依赖链"""
        if mark_region_manager is None:
            raise ValueError("EventHandler requires a valid MarkRegionManager instance")
        self._mark_region_manager = mark_region_manager

    @property
    def pw(self) -> Any:
        """关联的 DraggableGraphicsLayoutWidget 实例"""
        mrm = self._mark_region_manager
        if mrm is None:
            raise RuntimeError(
                "EventHandler: dependency chain broken (_mark_region_manager is None)"
            )
        return mrm.pw

    @property
    def _cursor_manager(self):
        """通过依赖链获取 CursorManager"""
        return self._mark_region_manager._cursor_manager

    @property
    def _multi_curve_manager(self):
        """通过依赖链获取 MultiCurveManager"""
        return self._cursor_manager._data_manager

    @property
    def _plot_data_manager(self):
        """通过依赖链获取 PlotDataManager"""
        return self._multi_curve_manager._data_manager

    @property
    def _axis_manager(self):
        """通过依赖链获取 AxisManager"""
        return self._plot_data_manager._axis_manager

    @property
    def _ui_manager(self):
        """通过依赖链获取 PlotUIManager"""
        return self._axis_manager._ui_manager

    @property
    def _is_interacting(self) -> bool:
        """用户是否正在交互（拖拽/缩放中）

        状态所有者：widget（self.pw._is_interacting）。
        所有 Manager 通过 self.pw 读写，不独立存储。
        """
        return getattr(self.pw, "_is_interacting", False)

    @_is_interacting.setter
    def _is_interacting(self, value: bool):
        self.pw._is_interacting = value

    @safe_callback
    def _on_range_changed(self, view_box, range, changed=None):
        _t0 = time.perf_counter() if PERF_LOG_ENABLED else 0
        try:
            if getattr(self.pw, '_is_updating_data', False) or getattr(self.pw, '_is_being_destroyed', False):
                self._cancel_ui_refresh()
                return

            if getattr(self.pw, '_is_syncing_range', False):
                logger.debug(
                    "[RANGE_CHANGED] _on_range_changed: short-circuit by _is_syncing_range, "
                    "new_range=(%.4f, %.4f)", range[0][0], range[0][1],
                )
                return

            # 交互轴检测：只有 X-only 交互才需要禁用/恢复 Y autoRange。
            # changed=[x_changed, y_changed]，None 时保守处理为 y_touched（不动 Y）。
            # 依据：wheelEvent 硬编码 (factor, 1) → [True, False]；
            #       拖拽 Y 轴 → [False, True]；绘图区平移/框选 → [True, True]。
            x_only = bool(changed is not None and changed[0] and not changed[1])

            # XLink 级联抑制：如果兄弟子图正在交互，本次 range 变化是 XLink 传播结果，
            # 无需进入交互模式或启动 timer。交互结束后由源子图统一广播刷新。
            if not self._is_interacting and self._sibling_is_interaction_source():
                return

            if not self._is_interacting:
                self._is_interacting = True
                # 记录本次交互是否为 X-only（用于 _end_interaction 决策是否恢复 Y）
                self.pw._interaction_x_only = x_only
                logger.debug(
                    "[RANGE_CHANGED] _on_range_changed: entering interacting (x_only=%s), "
                    "new_range=(%.4f, %.4f)", x_only, range[0][0], range[0][1],
                )
                # 只有 X-only 交互才禁用 Y autoVisible（Y-touched 交互尊重用户手动 Y）
                if x_only:
                    self._start_interaction()
            elif x_only and not getattr(self.pw, '_interaction_x_only', False):
                # 交互中途从 Y-touched 切换到 X-only：补充禁用 Y autoVisible
                self.pw._interaction_x_only = True
                self._start_interaction()
            elif not x_only and getattr(self.pw, '_interaction_x_only', False):
                # 交互中途从 X-only 切换到 Y-touched（如 box zoom 先 setXRange 再 setYRange，
                # 或 wheel 后拖拽 Y 轴）：用户手动操作了 Y 轴，_end_interaction 时
                # 不应恢复 Y autoRange，否则会覆盖用户指定的 yRange。
                # 但 _start_interaction 已冻结了兄弟子图，必须立即补解冻，
                # 否则兄弟 plot 的 autoRange[1] 永久停留在 False（Y 自适应失效）
                self.pw._interaction_x_only = False
                self._unfreeze_frozen_plots()

            timer = getattr(self.pw, '_interaction_timer', None)
            if timer is None:
                # 防御分支：防抖定时器缺失时复用 _end_interaction 完整收尾
                # （复位 _is_interacting + immediate 刷新 + 广播兄弟子图 + cursor geometry 兜底）
                self._end_interaction()
                return

            timer.stop()
            timer.start(UI_DEBOUNCE_DELAY_MS)
            # 交互期间（无论刚进入还是持续中）取消挂起的样式/光标刷新，
            # 防抖定时器超时后由 _end_interaction 统一兜底刷新
            self._cancel_ui_refresh('style', 'cursor')

            if PERF_LOG_ENABLED:
                _dt_ms = (time.perf_counter() - _t0) * 1000
                if _dt_ms > PERF_RANGE_CB_WARN_MS:
                    logger.warning(
                        "[PERF][RANGE_CB] slow _on_range_changed: %.2fms (interacting=%s)",
                        _dt_ms, self._is_interacting,
                    )
        except Exception:
            logger.warning("范围变化处理出错", exc_info=True)

    def _sibling_is_interaction_source(self) -> bool:
        """检查是否有兄弟子图正在作为交互源（用户直接操作的子图）。

        用于 XLink 级联抑制：当用户操作子图 A 时，XLink 会将 range 变化传播到
        子图 B-L。这些传播触发的 sigRangeChanged 不需要进入交互模式。
        """
        pw = self.pw
        main_window = pw.window() if hasattr(pw, 'window') else None
        if main_window is None or not hasattr(main_window, 'plot_widgets'):
            return False
        for container in main_window.plot_widgets:
            sibling = container.plot_widget
            if sibling is not pw and getattr(sibling, '_is_interacting', False):
                return True
        return False

    def _start_interaction(self):
        """交互开始时禁用所有可见 plot 的 Y autoVisible（仅限 X-only 交互）。

        条件式禁用：只对 autoRangeEnabled()[1] == True 的 plot 禁用 Y autoVisible，
        尊重用户手动取消过 Y autoRange 的 plot（完全不动）。
        交互期间 Y 轴范围冻结，避免每次 X 范围变化触发全数据 Y 扫描。
        交互结束后由 _end_interaction 异步恢复 Y auto-range。

        遍历所有 plot（不仅限于交互源）的原因：XLink 级联会将 range 变化
        传播到所有 linked plot，每个都会触发 Y auto-range。
        """
        _t0 = time.perf_counter() if PERF_LOG_ENABLED else 0

        pw = self.pw
        main_window = pw.window() if hasattr(pw, 'window') else None
        if main_window is None or not hasattr(main_window, 'plot_widgets'):
            return

        count = 0
        # 取消挂起的 ASYNC_Y_RESTORE 队列：用户在 restore 期间又发起新交互时，
        # 旧的 restore 队列已失效（X 范围已变），继续执行会导致 Y 轴范围抖动
        # （setYRange 用旧 X 范围计算的 Y 值，与新的 X 范围不匹配）。
        if hasattr(self, '_y_restore_queue'):
            self._y_restore_queue = []
        # 递增全局代数，作废所有 EventHandler 残留的 restore 批次
        # （self._y_restore_queue 只能清自己的，其他 handler 的队列靠代数拦截）
        EventHandler._y_restore_generation += 1

        for container in main_window.plot_widgets:
            sibling = container.plot_widget
            if not container.isVisible():
                continue
            if not hasattr(sibling, 'view_box'):
                continue
            try:
                # 读取当前 autoRange 状态 [x_auto, y_auto]
                # 关键：_flush_wheel_zoom 用 setXRange（而非 scaleBy）只设
                # autoRange[0]=False，不碰 autoRange[1]（scaleBy((factor,1)) 走
                # setRange(rect) 会禁用双轴 autoRange，污染 Y 冻结检测，见
                # tmp/yrange_jitter_after_phase2_analysis.md 根因 #1）。
                # 所以此时 autoRange[1] 反映的是用户手动操作的结果：
                #   - True：用户未手动取消 Y autoRange（正常情况）
                #   - False：用户手动取消过 Y autoRange（如拖拽 Y 轴）
                try:
                    auto_state = sibling.view_box.autoRangeEnabled()
                except (AttributeError, TypeError):
                    auto_state = sibling.view_box.state.get('autoRange', [False, False])

                y_auto = auto_state[1] if isinstance(auto_state, (list, tuple)) else False

                # 记录原始状态（用于 _schedule_async_y_restore 决策是否恢复 Y）
                sibling._pre_interaction_y_auto = y_auto

                # 只对 y_auto=True 的 plot 冻结 Y 轴
                # （用户手动取消过 Y autoRange 的 plot 完全不动，尊重用户意图）
                if y_auto:
                    # 直接修改 state 而非调用 setAutoVisible(y=False)，
                    # 避免触发 _autoRangeNeedsUpdate → prepareForPaint → 延迟 sigRangeChanged
                    # 用 viewbox_state_compat 封装，支持 pyqtgraph 版本升级后的安全降级
                    from src.ui.widgets.viewbox_state_compat import (
                        disable_y_autorange,
                        disable_y_autovisible,
                    )
                    disable_y_autovisible(sibling.view_box)
                    # 同时禁用 autoRange[1]：仅禁用 autoVisibleOnly 只改变
                    # auto-range 数据源，不能阻止 updateAutoRange 重算 Y；
                    # 必须禁用 autoRange[1] 才能真正冻结 Y 轴（根因 #3）
                    disable_y_autorange(sibling.view_box)
                    count += 1
            except Exception:
                logger.debug(
                    "[INTERACT] setAutoVisible(y=False) failed for plot",
                    exc_info=True,
                )

        if PERF_LOG_ENABLED:
            _dt_ms = (time.perf_counter() - _t0) * 1000
            logger.info(
                "[PERF][START_INTERACT] disabled Y autoVisible on %d plots in %.2fms",
                count, _dt_ms,
            )

    def _unfreeze_frozen_plots(self):
        """X-only → Y-touched 切换时的补解冻。

        _start_interaction 已冻结了所有 _pre_interaction_y_auto=True 的子图，
        切换到 Y-touched 后 _end_interaction 不会再走 _schedule_async_y_restore，
        必须在此同步恢复，否则兄弟子图 autoRange[1] 永久停留在 False。

        交互源 plot（self.pw）只恢复 autoVisibleOnly，不恢复 autoRange[1]：
        用户刚手动指定了它的 Y 范围（setYRange 已将其 autoRange[1] 置 False，
        这是正确终态）。兄弟子图则完整恢复（它们的 Y 未被用户指定）。
        """
        pw = self.pw
        main_window = pw.window() if hasattr(pw, 'window') else None
        if main_window is None or not hasattr(main_window, 'plot_widgets'):
            return

        from src.ui.widgets.viewbox_state_compat import (
            restore_y_autorange,
            restore_y_autovisible,
        )
        count = 0
        for container in main_window.plot_widgets:
            sibling = container.plot_widget
            if not container.isVisible() or not hasattr(sibling, 'view_box'):
                continue
            # 只解冻本次交互中被冻结过的 plot（_pre_interaction_y_auto=True）
            if not getattr(sibling, '_pre_interaction_y_auto', False):
                continue
            try:
                sibling._is_syncing_range = True
                if sibling is not pw:
                    restore_y_autorange(sibling.view_box)
                restore_y_autovisible(sibling.view_box)
                sibling._pre_interaction_y_auto = False
                count += 1
            except Exception:
                logger.debug("[INTERACT] unfreeze failed for plot", exc_info=True)
            finally:
                sibling._is_syncing_range = False
        logger.debug("[INTERACT] unfroze %d plots on x_only->y_touched switch", count)

    def _end_interaction(self):
        """结束交互：异步恢复 Y autoRange → 广播刷新。

        只有 X-only 交互才需要恢复 Y autoRange（Y-touched 交互尊重用户手动 Y）。
        Y 恢复通过 QTimer.singleShot(0) 异步执行，避免阻塞当前 wheel flush
        的最终呈现（WHEEL_LATENCY 从 ~200ms 降到 ~50ms）。
        """
        _t0 = time.perf_counter() if PERF_LOG_ENABLED else 0
        try:
            self._is_interacting = False

            # 只有 X-only 交互才需要恢复 Y autoRange
            if getattr(self.pw, '_interaction_x_only', False):
                self._schedule_async_y_restore()

            # 重置交互轴标记
            self.pw._interaction_x_only = False

            # 标准交互结束刷新
            self._queue_ui_refresh(immediate=True)
            # 广播刷新到兄弟子图（它们在交互期间被级联抑制跳过了刷新）
            self._refresh_siblings_after_interaction()
            if getattr(self.pw, '_pending_cursor_geometry_update', False):
                self.pw._pending_cursor_geometry_update = False
                self._schedule_cursor_geometry_update()
        except Exception:
            logger.warning("结束交互出错", exc_info=True)
        finally:
            if PERF_LOG_ENABLED:
                _dt_ms = (time.perf_counter() - _t0) * 1000
                if _dt_ms > PERF_INTERACTION_WARN_MS:
                    logger.warning(
                        "[PERF][END_INTERACT] slow _end_interaction: %.2fms",
                        _dt_ms,
                    )

    def _schedule_async_y_restore(self):
        """调度异步 Y auto-range 恢复。

        用 QTimer.singleShot(0) 推迟到下一事件循环，
        避免阻塞当前 wheel flush 的最终呈现。
        分批处理（每帧 ASYNC_Y_RESTORE_BATCH_SIZE 个 plot）避免单次 100ms+ 阻塞。
        只恢复 _pre_interaction_y_auto==True 的 plot（尊重用户手动 Y）。
        """
        pw = self.pw
        main_window = pw.window() if hasattr(pw, 'window') else None
        if main_window is None or not hasattr(main_window, 'plot_widgets'):
            return

        # 收集需要恢复 Y autoRange 的 plot（仅 _pre_interaction_y_auto=True 的）
        # 尊重用户手动取消过 Y autoRange 的 plot（完全不动）
        # _pre_interaction_y_auto 在 _start_interaction 中记录，反映用户手动操作结果
        targets = []
        for container in main_window.plot_widgets:
            sibling = container.plot_widget
            if not container.isVisible():
                continue
            if not hasattr(sibling, 'view_box'):
                continue
            if getattr(sibling, '_pre_interaction_y_auto', False):
                targets.append(sibling)

        if not targets:
            return

        # 交互源 plot 优先（放到队列首位，用户关注的 plot 最快恢复）
        if pw in targets:
            targets.remove(pw)
            targets.insert(0, pw)

        # 启动分批处理（记录当前全局代数，批次执行时若代数已变则整批作废）
        self._y_restore_queue = targets
        self._y_restore_batch_size = ASYNC_Y_RESTORE_BATCH_SIZE
        self._y_restore_gen_at_schedule = EventHandler._y_restore_generation
        QTimer.singleShot(0, self._process_y_restore_batch)

    def _process_y_restore_batch(self):
        """分批处理 Y auto-range 恢复（每帧 ASYNC_Y_RESTORE_BATCH_SIZE 个 plot）。

        每个 plot 用 _compute_and_set_visible_y_range 计算可见 X 范围内的 Y 范围
        并精确设置（避免 pyqtgraph 全数据扫描），然后恢复 setAutoVisible(y=True)。
        跨 EventHandler 协调：若 plot 已被另一个交互接管（_is_interacting=True），
        跳过本次恢复，避免与正在进行的交互打架。
        """
        if not hasattr(self, '_y_restore_queue') or not self._y_restore_queue:
            return

        # 代数校验：调度后若有新交互开启（_start_interaction 递增了代数），
        # 本队列已过期，整批作废（避免在其他 plot 交互期间提前解冻兄弟子图）
        if getattr(self, '_y_restore_gen_at_schedule', None) != EventHandler._y_restore_generation:
            self._y_restore_queue = []
            return

        # 取本批次
        batch = self._y_restore_queue[:self._y_restore_batch_size]
        self._y_restore_queue = self._y_restore_queue[self._y_restore_batch_size:]

        _t0 = time.perf_counter() if PERF_LOG_ENABLED else 0

        processed = 0
        for sibling in batch:
            # 跨 widget 协调：若该 plot 已被另一个交互接管，跳过恢复
            # （场景：A 调度 restore 后用户又操作了 B，B 的 _start_interaction
            #  无法清空 A 的队列，A 的 restore 仍会执行——此处跳过正在交互的 plot）
            if getattr(sibling, '_is_interacting', False):
                continue

            # 用 _is_syncing_range 抑制 setYRange 同步触发的 sigRangeChanged
            sibling._is_syncing_range = True
            try:
                # 用已有方法计算可见 Y 范围 + 精确设置（避免全数据扫描）
                # setYRange 是同步的，sigRangeChanged 在 _is_syncing_range=True 时被短路
                self._compute_and_set_visible_y_range(sibling)
            except Exception:
                logger.debug("[INTERACT] async Y restore failed", exc_info=True)
            finally:
                # 无论 _compute_and_set_visible_y_range 成败，都恢复 autoRange/autoVisibleOnly
                # 标志：setYRange 已将 autoRange[1] 设为 False（_is_syncing_range=True
                # 跳过了 _set_safe_y_range 内部的恢复分支），此处必须显式恢复，
                # 否则 plot 永久 autoRange[1]=False，后续 wheel 报 "0 plots"。
                # 用 viewbox_state_compat 封装，支持 pyqtgraph 版本升级后的安全降级
                from src.ui.widgets.viewbox_state_compat import (
                    restore_y_autorange, restore_y_autovisible,
                )
                restore_y_autorange(sibling.view_box)
                restore_y_autovisible(sibling.view_box)
                # setYRange 是同步的，sigRangeChanged 已在 _is_syncing_range=True 时被处理，
                # 可以立即清除标志（无延迟发射需要防范）
                sibling._is_syncing_range = False
                # 清除 _pre_interaction_y_auto 标记，避免陈旧状态残留
                sibling._pre_interaction_y_auto = False
                processed += 1

        if PERF_LOG_ENABLED:
            _dt_ms = (time.perf_counter() - _t0) * 1000
            remaining = len(self._y_restore_queue) if hasattr(self, '_y_restore_queue') else 0
            logger.info(
                "[PERF][ASYNC_Y_RESTORE] processed %d plots in %.2fms, %d remaining",
                processed, _dt_ms, remaining,
            )

        # 如果还有剩余，调度下一批
        if hasattr(self, '_y_restore_queue') and self._y_restore_queue:
            QTimer.singleShot(0, self._process_y_restore_batch)

    def _compute_and_set_visible_y_range(self, pw):
        """计算可见 X 范围内的 Y 范围并精确设置（避免全数据扫描）。

        复用 PlotDataManager._compute_visible_y_range_union（与
        multi_curve_manager._update_axes_for_multi_curve 共用同一实现）和
        pw._set_safe_y_range 包装方法。使用 ci.x_data / ci.y_data（原始数据，
        非降采样）。全曲线失败时记录 warning，便于排查"Y 范围冻结"问题。
        """
        if not hasattr(pw, 'curves') or not pw.curves:
            return

        # 获取当前可见 X 范围
        try:
            (x_min, x_max), _ = pw.view_box.viewRange()
        except Exception:
            return

        if x_min is None or x_max is None or abs(x_max - x_min) < 1e-12:
            return

        # 复用 PlotDataManager 的共享实现（含 per-curve X 范围快速排除）
        y_range = pw._compute_visible_y_range_union(x_min, x_max)
        if y_range is None:
            logger.warning(
                "[ASYNC_Y] no valid Y range for plot (curves=%d, x_window=(%.4f, %.4f)), "
                "leaving frozen range",
                len(pw.curves), x_min, x_max,
            )
            return

        final_min_y, final_max_y = y_range
        # 用 pw._set_safe_y_range 包装方法（含 NaN/sentinel 保护），
        # 不直接访问 pw._axis_manager._set_safe_y_range，保持封装一致
        pw._set_safe_y_range(final_min_y, final_max_y, set_limits=False)

    def _refresh_siblings_after_interaction(self):
        """交互结束后触发兄弟子图的样式/光标刷新。

        XLink 级联抑制期间，兄弟子图跳过了 _on_range_changed 中的刷新调度。
        交互结束后需统一补刷，确保 symbol/line 切换和光标状态正确。
        """
        pw = self.pw
        main_window = pw.window() if hasattr(pw, 'window') else None
        if main_window is None or not hasattr(main_window, 'plot_widgets'):
            return
        for container in main_window.plot_widgets:
            sibling = container.plot_widget
            if sibling is pw:
                continue
            if hasattr(sibling, '_queue_ui_refresh'):
                sibling._queue_ui_refresh(immediate=True)

    def _schedule_cursor_geometry_update(self):
        """调度光标几何更新"""
        if not hasattr(self.pw, "vline") or not self.pw.vline.isVisible():
            return
        if getattr(self.pw, "_cursor_refresh_timer", None) is None:
            return
        if getattr(self.pw, "_is_interacting", False):
            self.pw._pending_cursor_geometry_update = True
            return
        self.pw._pending_cursor_geometry_update = False
        # 重启单次定时器，合并短时间内的多次请求
        self.pw._cursor_refresh_timer.start(max(15, UI_DEBOUNCE_DELAY_MS))

    def _refresh_cursor_geometry(self):
        """刷新光标几何"""
        if not hasattr(self.pw, "vline") or not self.pw.vline.isVisible():
            return
        if getattr(self.pw, "_is_interacting", False):
            self.pw._pending_cursor_geometry_update = True
            return
        if self._cursor_manager.show_values_only:
            self._cursor_manager._show_x_position_only()
        else:
            self._cursor_manager.update_cursor_label()

    def _on_vb_jump(self, pw, ctx_x):
        """ViewBox 信号：跳转到数据"""
        if pw:
            pw.jump_to_data_impl(ctx_x)

    def _on_vb_clear(self, pw):
        """ViewBox 信号：清除绘图"""
        if pw:
            pw.clear_plot_item()
            if pw.plot_context:
                pw.plot_context.request_mark_stats_refresh(immediate=True)

    def _on_vb_auto_y(self, pw):
        """ViewBox 信号：自动 Y 轴"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "auto_y_in_x_range"):
            pw.plot_context.auto_y_in_x_range()

    def _on_vb_set_cursor_mode(self, mode, pw, ctx_x):
        """ViewBox 信号：设置光标模式"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "set_cursor_mode"):
            pw.plot_context.set_cursor_mode(mode, source_plot=pw, context_x=ctx_x)

    def _on_vb_show_cursor(self, pw):
        """ViewBox 信号：显示光标"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "cursor_values_hidden"):
            pw.plot_context.cursor_values_hidden = False
            if pw.plot_context.cursor_btn.isChecked():
                for c in pw.plot_context.plot_widgets:
                    c.plot_widget.toggle_cursor(True)

    def _on_vb_hide_cursor(self, pw):
        """ViewBox 信号：隐藏光标"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "cursor_values_hidden"):
            pw.plot_context.cursor_values_hidden = True
            if pw.plot_context.cursor_btn.isChecked():
                for c in pw.plot_context.plot_widgets:
                    c.plot_widget.toggle_cursor(False, hide_values_only=True)

    def _on_vb_set_row_height(self, pct, pw):
        """ViewBox 信号：设置行高"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "plot_widgets"):
            for idx, c in enumerate(pw.plot_context.plot_widgets):
                if c.plot_widget is pw:
                    row, _ = divmod(idx, pw.plot_context._plot_col_max_default)
                    pw.plot_context.set_row_height(row, pct)
                    break

    def _on_vb_set_all_row_height(self, pct):
        """ViewBox 信号：设置所有行高"""
        if self.pw.plot_context and hasattr(self.pw.plot_context, "set_all_row_height"):
            self.pw.plot_context.set_all_row_height(pct)

    def _on_vb_copy_name(self, pw):
        """ViewBox 信号：复制变量名"""
        if not pw:
            return
        var_names = []
        if pw.curves:
            var_names = list(pw.curves.keys())
        if var_names:
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setText(" ".join(var_names))

    def _on_vb_var_editor(self, pw):
        """ViewBox 信号：打开变量编辑器"""
        if pw:
            from src.ui.plot_variable_editor import PlotVariableEditorDialog

            parent = pw.window() if pw.window() else None
            dialog = PlotVariableEditorDialog(pw, parent)
            dialog.show()
            dialog.raise_()

    def _connect_viewbox_signals(self):
        """连接 ViewBox 信号"""
        vb = self.pw.view_box
        vb.plot_widget = self.pw
        vb.signals.request_jump_to_data.connect(self._on_vb_jump)
        vb.signals.request_clear_plot.connect(self._on_vb_clear)
        vb.signals.request_auto_y.connect(self._on_vb_auto_y)
        vb.signals.request_set_cursor_mode.connect(self._on_vb_set_cursor_mode)
        vb.signals.request_show_cursor_value.connect(self._on_vb_show_cursor)
        vb.signals.request_hide_cursor_value.connect(self._on_vb_hide_cursor)
        vb.signals.request_set_row_height.connect(self._on_vb_set_row_height)
        vb.signals.request_set_all_row_height.connect(self._on_vb_set_all_row_height)
        vb.signals.request_copy_name.connect(self._on_vb_copy_name)
        vb.signals.request_variable_editor.connect(self._on_vb_var_editor)

    def _cancel_ui_refresh(self, *types):
        """取消 UI 刷新"""
        if hasattr(self.pw, "_cancel_ui_refresh"):
            self.pw._cancel_ui_refresh(*types)

    def _queue_ui_refresh(self, immediate=False):
        """队列 UI 刷新"""
        if hasattr(self.pw, "_queue_ui_refresh"):
            self.pw._queue_ui_refresh(immediate=immediate)
