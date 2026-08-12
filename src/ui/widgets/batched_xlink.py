"""
BatchedXLinkSync - 批量 XLink 同步管理器（Phase 3）

替代 pyqtgraph 原生 setXLink 级联机制，使用 16ms PreciseTimer 将多个
同步请求批处理为一次批量操作，消除 N-1 次独立回调链。

设计要点：
- 监听所有 plot 的 sigRangeChanged（不仅 master），以支持任意 plot 上的交互
  原生 setXLink 是单向的（master→slave），无法处理 slave 上的用户操作
- 使用 _is_syncing_range 标志防止反馈循环：
  批量同步期间设置此标志，slave 的 _on_range_changed 会短路返回
  本监听器也检查此标志，跳过由自身 setXRange 触发的 sigRangeChanged
- 通过 `changed` 参数判断是否 X 轴变化，避免 Y-only 变化触发不必要的 X 同步
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QTimer, Qt

from src.core.config import (
    PERF_LOG_ENABLED,
    PERF_BATCHED_XLINK_INTERVAL_MS,
)
from src.core.logger import get_logger

logger = get_logger(__name__)


class BatchedXLinkSync:
    """批量 XLink 同步管理器

    生命周期：
    - 在 layout_manager.create_subplots_matrix 中通过 setup() 创建
    - 监听所有 plot 的 view_box.sigRangeChanged
    - 在布局变化、可见性变化时通过 update_plots() 更新 plot 列表
    - 在窗口销毁或重建矩阵时通过 dispose() 释放资源
    """

    def __init__(self):
        self._plots: list[Any] = []  # 所有 plot_widget 列表
        self._timer = QTimer()
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(PERF_BATCHED_XLINK_INTERVAL_MS)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._flush_sync)
        self._pending = False
        self._source_pw: Any = None  # 最近一次发起变化的 plot
        self._disposed = True  # 初始未 setup

    def setup(self, plot_widgets: list[Any]) -> None:
        """初始化或重建：连接所有 plot 的 sigRangeChanged

        Args:
            plot_widgets: 所有 plot_widget 实例列表（包括 master）
        """
        self.dispose()
        self._plots = list(plot_widgets)
        for pw in self._plots:
            self._connect_plot(pw)
        self._disposed = False
        logger.debug(
            "[BATCHED_XLINK] setup: connected %d plots", len(self._plots),
        )

    def update_plots(self, plot_widgets: list[Any]) -> None:
        """更新 plot 列表（布局变化后调用）

        断开已移除的 plot，连接新增的 plot。
        """
        new_set = set(id(p) for p in plot_widgets)
        old_set = set(id(p) for p in self._plots)

        # 断开已移除的 plot
        for pw in self._plots:
            if id(pw) not in new_set:
                self._disconnect_plot(pw)

        # 连接新增的 plot
        for pw in plot_widgets:
            if id(pw) not in old_set:
                self._connect_plot(pw)

        self._plots = list(plot_widgets)
        logger.debug(
            "[BATCHED_XLINK] update_plots: %d plots", len(self._plots),
        )

    def _connect_plot(self, pw: Any) -> None:
        """连接单个 plot 的 sigRangeChanged

        不做防御性 disconnect：setup 时所有 plot 都是新建的；
        update_plots 时只对新增 plot 调用此方法，不会重复连接。
        """
        try:
            vb = pw.view_box
            vb.sigRangeChanged.connect(self._on_plot_range_changed)
        except (RuntimeError, AttributeError):
            logger.debug(
                "[BATCHED_XLINK] connect failed for plot (C++ object may be destroyed)",
                exc_info=True,
            )

    def _disconnect_plot(self, pw: Any) -> None:
        """断开单个 plot 的 sigRangeChanged

        使用 warnings.catch_warnings 抑制 PySide6 对未连接 slot 的
        RuntimeWarning（warning 不是 exception，try/except 无法捕获）。
        """
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pw.view_box.sigRangeChanged.disconnect(self._on_plot_range_changed)
        except (TypeError, RuntimeError, AttributeError):
            pass

    def _on_plot_range_changed(self, view_box, range, changed=None) -> None:
        """任意 plot 的 range 变化时调度批量同步

        Args:
            view_box: 发起变化的 ViewBox
            range: 新的 viewRange [(xmin, xmax), (ymin, ymax)]
            changed: [x_changed, y_changed]，None 时保守处理为 X 变化
        """
        if self._disposed:
            return

        # 仅在 X 轴变化时调度同步（changed[0]=True 或 None）
        # 跳过 Y-only 变化（如 _process_y_restore_batch 触发的 setYRange）
        if changed is not None and not changed[0]:
            return

        # 定位源 plot
        source_pw = None
        for pw in self._plots:
            try:
                if pw.view_box is view_box:
                    source_pw = pw
                    break
            except (RuntimeError, AttributeError):
                continue

        if source_pw is None:
            return

        # 反馈循环防护：跳过由本管理器 setXRange 触发的 sigRangeChanged
        # _is_syncing_range=True 表示该 plot 正在被批量同步，不应再次触发
        if getattr(source_pw, '_is_syncing_range', False):
            return

        self._source_pw = source_pw
        self._pending = True
        if not self._timer.isActive():
            self._timer.start()

    def _flush_sync(self) -> None:
        """Timer 触发：批量同步所有 plot 的 X 范围到 source plot 的 X 范围"""
        if not self._pending or self._disposed:
            return
        self._pending = False

        source = self._source_pw
        self._source_pw = None

        if not self._plots or source is None:
            return

        # 检查 source 仍然有效（C++ 对象可能已销毁）
        try:
            source_vb = source.view_box
        except (RuntimeError, AttributeError):
            return
        if source_vb is None:
            return

        try:
            x_range = source_vb.viewRange()[0]
        except (RuntimeError, AttributeError):
            return

        xmin, xmax = x_range
        if xmin is None or xmax is None or abs(xmax - xmin) < 1e-12:
            return

        _t0 = time.perf_counter() if PERF_LOG_ENABLED else 0
        synced = 0
        skipped_hidden = 0
        skipped_unchanged = 0
        # 相对容差：与 _sync_linked_x_ranges 一致，消除 float32 精度误报
        range_width = abs(xmax - xmin)
        tolerance = 1e-4 * range_width
        for pw in self._plots:
            if pw is source:
                continue
            # 跳过不可见 plot（节省 setXRange 开销）
            container = pw.parentWidget()
            if container is not None and not container.isVisible():
                skipped_hidden += 1
                continue
            try:
                vb = pw.view_box
                # 跳过 X range 已一致的 slave（避免不必要的 setXRange + paint）
                try:
                    cur_range = vb.viewRange()[0]
                    if (
                        abs(cur_range[0] - xmin) < tolerance
                        and abs(cur_range[1] - xmax) < tolerance
                    ):
                        skipped_unchanged += 1
                        continue
                except (RuntimeError, AttributeError):
                    pass  # viewRange 获取失败时保守执行 setXRange

                pw._is_syncing_range = True
                # disableAutoRange(x) 防止 X autoRange 覆盖手动 setXRange
                # 与原生 _sync_linked_x_ranges 实现一致
                vb.enableAutoRange(x=False)
                vb.setXRange(xmin, xmax, padding=0)
                synced += 1
            except (RuntimeError, AttributeError):
                logger.debug(
                    "[BATCHED_XLINK] sync failed for plot (C++ object may be destroyed)",
                    exc_info=True,
                )
            except Exception:
                logger.debug("[BATCHED_XLINK] sync failed", exc_info=True)
            finally:
                # setXRange 是同步的，sigRangeChanged 已在 _is_syncing_range=True
                # 时被短路，可立即清除标志
                try:
                    pw._is_syncing_range = False
                except (RuntimeError, AttributeError):
                    pass

        if PERF_LOG_ENABLED:
            _dt_ms = (time.perf_counter() - _t0) * 1000
            logger.info(
                "[PERF][BATCHED_XLINK] synced %d slaves in %.2fms "
                "(skipped %d hidden, %d unchanged, source_x=(%.4f, %.4f))",
                synced, _dt_ms, skipped_hidden, skipped_unchanged, xmin, xmax,
            )

    def sync_now(self, source_pw: Any = None) -> None:
        """立即执行同步（绕过 16ms 延迟）

        用于布局变更、可见性变更等需要立即同步的场景。

        Args:
            source_pw: 指定源 plot；None 则使用最近一次 _source_pw
        """
        if self._disposed:
            return
        if source_pw is not None:
            self._source_pw = source_pw
        # 停止挂起的 timer，立即 flush
        if self._timer.isActive():
            self._timer.stop()
        # 强制 flush（即使 _pending=False）
        self._pending = True
        self._flush_sync()

    def dispose(self) -> None:
        """释放资源，断开所有信号连接"""
        self._disposed = True
        self._timer.stop()
        self._pending = False
        for pw in self._plots:
            self._disconnect_plot(pw)
        self._plots.clear()
        self._source_pw = None
