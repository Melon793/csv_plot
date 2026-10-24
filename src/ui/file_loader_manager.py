"""FileLoaderManager - 文件加载管理

负责 MainWindow 的文件加载功能：
- 文件对话框与路径验证
- 同步/异步加载路由
- 数据加载进度与状态管理
- 加载后 UI 同步
"""

from __future__ import annotations
import gc
import os
import re
import sys
import time
import warnings
from collections import OrderedDict

from PySide6.QtCore import Qt, QStandardPaths, QTimer, QSignalBlocker
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
)

from src.core.config import (
    BLF_MAX_MESSAGES,
    BLF_MAX_PAYLOAD_BYTES,
    BLF_TEXT_CANCELLED,
    BLF_TEXT_DB_MISSING,
    BLF_TEXT_NO_DB_SELECTED,
    BLF_TEXT_NO_SIGNAL,
    BLF_TEXT_TRUNCATED,
    BLF_TEXT_TRUNCATED_PAYLOAD,
    FILE_SIZE_LIMIT_BACKGROUND_LOADING,
    safe_qt_op,
    widget_alive as _qt_alive,
)
from src.core.data_types import AutoDetectError
from src.data.loader import DataLoadThread, FastDataLoader
from src.data.mdf_lazy_loader import MDFLazyLoader
from src.utils.paths import normalize_input_path
from src.ui.main_window_base_manager import MainWindowBaseManager
from src.core.logger import get_logger

logger = get_logger("ui.file_loader")

# 可加载的后缀白名单：分卷文件（data.csv.0）由 _extract_file_extension 的正则单独识别
SUPPORTED_EXTENSIONS = (
    ".csv",
    ".mfile",
    ".t00",
    ".t01",
    ".t10",
    ".t11",
    ".txt",
    ".mf4",
    ".mdf",
    ".dat",
    ".blf",
    ".xlsx",
    ".xlsm",
)


class FileLoaderManager(MainWindowBaseManager):
    """负责文件加载相关功能"""

    @property
    def _has_valid_loader(self) -> bool:
        return hasattr(self.mw, "loader") and self.mw.loader is not None

    @property
    def _has_valid_data(self) -> bool:
        return (
            self._has_valid_loader
            and hasattr(self.mw.loader, "datalength")
            and self.mw.loader.datalength > 0
        )

    @property
    def _current_data_length(self) -> int:
        return self.mw.loader.datalength if self._has_valid_loader else 0

    # BLF / CAN 数据库状态（§5）。在 __init__ 之外声明类级默认值，避免
    # 未走过 BLF 路径时 getattr 一路传默认值、也让单能直接断言初始态。
    _blf_progress = None
    _blf_db_paths: list | None = None
    # 取消标记按**代次**记录而不是一个全局布尔：进度框的 canceled 是同步落地，
    # 而旧线程的 cancelled/finished 会在后来的事件循环里迟到（见 _start_blf_load
    # 的保活注释），单布尔会被旧线程把新一轮导入误杀。
    _blf_load_version = -1
    # ``None`` = 本轮没取消过。不能用 -1 当哨兵：-1 也是「直接调用不传代次」
    # 的默认形参，两者混淆会让取消恢复一次也不跑。
    _blf_cancelled_version = None
    _blf_thread_refs: list | None = None
    _blf_force_dialog = False
    _blf_preset_db_paths = None
    # F 路径（库与报文完全不通）暂存**已经扫完**的 loader：报文桶与数据库是解耦
    # 的，重开对话框换库只需要重绑（毫秒级），不必把整文件再扫一遍。
    _blf_pending_loader = None

    @staticmethod
    def load_dict(path: str, *, default=None) -> dict:
        import ujson as json

        if not os.path.exists(path):
            return {} if default is None else default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise

    @staticmethod
    def _resolve_config_path(filename: str) -> str | None:
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            candidate = os.path.join(exe_dir, filename)
            if os.path.isfile(candidate):
                return candidate
        cwd_candidate = os.path.join(os.getcwd(), filename)
        if os.path.isfile(cwd_candidate):
            return cwd_candidate
        return None

    def load_btn_click(self):
        if getattr(self.mw, "_is_loading_new_data", False):
            return

        self.mw.load_btn.setEnabled(False)

        try:
            initial_dir = self._get_dialog_initial_directory()
            file_filter = (
                "All Files (*.*);;"
                "CSV/TXT Files (*.csv *.txt *.mfile *.t00 *.t01 *.t10 *.t11);;"
                "MDF Files (*.mf4 *.mdf *.dat);;"
                "BLF Files (*.blf);;"
                "Excel Files (*.xlsx *.xlsm)"
            )

            file_path, _ = QFileDialog.getOpenFileName(
                self.mw, "选择数据文件", initial_dir, file_filter
            )

            if file_path:
                self.load_csv_file(file_path)
            else:
                logger.debug("用户取消文件选择")
                self.mw.load_btn.setEnabled(True)
        except Exception:
            self.mw.load_btn.setEnabled(True)
            raise

    def _validate_file_path(self, file_path: str) -> bool:
        if not file_path or not isinstance(file_path, str):
            QMessageBox.warning(self.mw, "文件错误", "请选择一个有效的文件")
            return False

        if not os.path.isfile(file_path):
            QMessageBox.warning(self.mw, "文件错误", "文件不存在")
            return False

        return True

    def _check_file_size(self, file_path: str) -> bool:
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                QMessageBox.warning(self.mw, "文件错误", "文件为空")
                return False

            if file_size > 1024 * 1024 * 1024:
                reply = QMessageBox.question(
                    self.mw,
                    "文件过大",
                    f"文件大小 {file_size/(1024*1024*1024):.1f}GB 较大，加载可能需要较长时间，是否继续？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                return reply == QMessageBox.StandardButton.Yes

            return True

        except OSError as e:
            QMessageBox.critical(self.mw, "文件访问错误", f"无法访问文件: {e}")
            return False

    def _detach_vlines_from_scene(self, widget):
        """将 vline/vline2 从 scene 中物理移除（v5.11）。

        reload 期间 macOS 异步 paint (sendSpontaneousEvent) 可能遍历 BSP 树，
        访问 InfiniteLine 内部已失效的 C++ 坐标缓存导致 NULL+0x90 SIGSEGV。
        将 vline 从 scene 移除后，C++ 层不可能遍历到它。
        
        注意：必须使用 PlotItem.removeItem() 而非 scene.removeItem()，
        因为 PlotItem 内部维护 items 列表，scene.removeItem() 不会清理它，
        导致后续 PlotItem.addItem() 因重复而跳过。
        """
        if not hasattr(widget, "plot_item") or widget.plot_item is None:
            return
        for attr_name in ("vline", "vline2"):
            vline = getattr(widget, attr_name, None)
            if vline is None:
                continue
            try:
                # 使用 PlotItem.removeItem() 同时清理 scene 和 PlotItem.items
                widget.plot_item.removeItem(vline)
            except RuntimeError:
                # C++ 对象已销毁（预期场景），记录 debug 便于排查
                logger.debug("移除 %s 时 C++ 对象已销毁，跳过", attr_name, exc_info=True)
            except Exception:
                logger.debug("移除 %s 时异常", attr_name, exc_info=True)

    def _reattach_vlines_to_scene(self, widget):
        """将 vline/vline2 重新添加到 scene 中（v5.11）。

        在 _restore_cursor_state_after_reload 中调用，与 _detach_vlines_from_scene 配对。
        注意：_detach 使用 PlotItem.removeItem() 已同时清理 scene 和 PlotItem.items，
        因此这里可以安全使用 PlotItem.addItem() 重新添加。
        """
        if not hasattr(widget, "plot_item") or widget.plot_item is None:
            return
        for attr_name in ("vline", "vline2"):
            vline = getattr(widget, attr_name, None)
            if vline is None:
                continue
            try:
                # 仅当 vline 不在 scene 中时才重新添加
                if vline.scene() is None:
                    widget.plot_item.addItem(vline, ignoreBounds=True)
                # else: vline 已在 scene 中，跳过
            except RuntimeError:
                # C++ 对象已销毁（预期场景），记录 debug 便于排查
                logger.debug("添加 %s 时 C++ 对象已销毁，跳过", attr_name, exc_info=True)
            except Exception:
                logger.debug("添加 %s 时异常", attr_name, exc_info=True)

    def _reconnect_vline_signals(self, widget):
        """幂等重连 vline/vline2 的 sigPositionChanged 信号。

        v5.x 修复：_begin_data_reload 断开了 vline 信号，但 _restore_cursor_state_after_reload
        在 free cursor 路径下提前 return 跳过重连，导致后续开启 anchored cursor 时
        on_vline_position_changed 永不触发、跨 plot cursor link 失效。
        本方法在所有路径下统一重连，先 disconnect 再 connect 保证幂等。
        """
        for line_attr in ("vline", "vline2"):
            line = getattr(widget, line_attr, None)
            if line is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    line.sigPositionChanged.disconnect(widget.on_vline_position_changed)
            except (TypeError, RuntimeError):
                pass  # 原本就未连接，忽略
            try:
                line.sigPositionChanged.connect(widget.on_vline_position_changed)
            except (TypeError, RuntimeError) as e:
                logger.debug("重连 %s 信号失败: %s", line_attr, e)

    def _clamp_pinned_to_global_range(self, pinned_values):
        """将 pinned 值钳制到全局数据 X 范围内。

        v5.x 修复问题 A：reload 后新数据范围可能变小，旧 pinned 值可能超出范围。
        这里显式钳制的目的不是「补 setPos 的漏」（InfiniteLine.setPos 本来就会按
        maxRange 钳制，见 pyqtgraph InfiniteLine.setPos），而是让 MainWindow 与
        各 widget 的 pinned_x_values 在恢复前就一致，避免把越界值写进状态后再被
        各图 bounds 各自钳成不同数值。
        """
        if not pinned_values:
            return list(pinned_values)
        try:
            x_min, x_max = self.mw.cursor_sync_manager.collect_global_x_range()
            if x_min is None or x_max is None:
                return list(pinned_values)
            return [max(float(x_min), min(float(x_max), v)) for v in pinned_values]
        except Exception:
            logger.debug("钳制 pinned 值失败，返回原值", exc_info=True)
            return list(pinned_values)

    def _begin_data_reload(self):
        if self.mw._is_loading_new_data:
            return
        self.mw._is_loading_new_data = True
        self.mw._data_version += 1

        # v5.8 修复：取消之前的 safety timer，防止跨 reload 的过期 timer 触发 _force_unlock_all。
        # 之前的实现仅靠版本号防污染，但 _safety_unlock_version 会被最新 _end_data_reload 覆盖，
        # 导致旧 timer 触发时版本检查通过，误执行 _force_unlock_all 清除新曲线。
        self._stop_safety_timer()
        self._safety_unlock_version = -1

        if hasattr(self.mw, "_crosshair_update_timer"):
            self.mw._crosshair_update_timer.stop()
        self.mw._pending_crosshair_x = None

        # 保存 pinned cursor 状态以便重载后恢复
        pinned_x_values = list(getattr(self.mw, "pinned_x_values", []))
        cursor_mode = getattr(self.mw, "cursor_mode", "1 free cursor")
        self.mw._saved_pinned_x_values = pinned_x_values
        self.mw._saved_cursor_mode = cursor_mode

        try:
            self.mw.cursor_sync_manager.reset_all_pin_states()
        except Exception:
            logger.debug("重置 pin 状态失败（可能数据已变更）")
        for container in getattr(self.mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if not widget:
                continue
            # v5.x: 断开 vline/vline2 信号，防止中间态触发回调导致 SIGSEGV
            # 重复 reload 时信号可能已断开，使用 catch_warnings 抑制 libpyside 的 RuntimeWarning
            if hasattr(widget, "vline"):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        widget.vline.sigPositionChanged.disconnect(
                            widget.on_vline_position_changed
                        )
                except (TypeError, RuntimeError) as e:
                    logger.debug("断开 vline 信号失败(可能已断开): %s", e)
            if hasattr(widget, "vline2"):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        widget.vline2.sigPositionChanged.disconnect(
                            widget.on_vline_position_changed
                        )
                except (TypeError, RuntimeError) as e:
                    logger.debug("断开 vline2 信号失败(可能已断开): %s", e)
            # v5.11: 将 vline/vline2 从 scene 中物理移除，防止 reload 期间
            # macOS 异步 paint (sendSpontaneousEvent) 遍历 BSP 树时访问 InfiniteLine
            # 内部已失效的 C++ 坐标缓存 → NULL+0x90 SIGSEGV。
            # reload 完成后在 _restore_cursor_state_after_reload 中重新添加。
            self._detach_vlines_from_scene(widget)
            if hasattr(widget, "_safe_clear_plot_items"):
                try:
                    widget._safe_clear_plot_items()
                except Exception:
                    logger.debug("_safe_clear_plot_items 失败（数据重载期间）")
            # 清除旧 X limits，防止跨文件切换时旧 limits 残留导致 X-link 钳制
            if hasattr(widget, "plot_item") and widget.plot_item is not None:
                try:
                    widget.plot_item.setLimits(xMin=None, xMax=None)
                except Exception:
                    logger.debug("清除 X limits 失败（数据重载期间）")
            if hasattr(widget, "_cursor_trash_bin"):
                widget._cursor_trash_bin.clear()
            if hasattr(widget, "_pending_delete_items"):
                # 【崩溃修复】pending item 必须先从 scene 物理移除再释放 Python 引用。
                # PySide6 中 QGraphicsItem 的 Python 引用归零即销毁 C++ 对象，
                # 若 item 仍注册在 scene 中，BSP 树残留悬空指针 → 后续 paint
                # 遍历到它时 EXC_BAD_ACCESS/SIGSEGV（topLevelItem 崩溃签名）。
                # 移除后移入 trash bin 保持引用，延迟到下一轮清理时释放。
                # 注：trash bin 中是上一轮已 removeItem 的 item，此处清空安全。
                pending = widget._pending_delete_items
                scene = None
                plot_item = getattr(widget, "plot_item", None)
                if plot_item is not None:
                    try:
                        scene = plot_item.scene()
                    except RuntimeError:
                        scene = None
                trash = getattr(widget, "_cursor_trash_bin", None)
                for item in pending:
                    if item is None:
                        continue
                    try:
                        in_scene = item.scene() is not None
                    except RuntimeError:
                        in_scene = False
                    if in_scene:
                        # PlotItem.removeItem 同时清理 scene 与 PlotItem.items
                        # （scene.removeItem 会残留 PlotItem.items 引用）
                        removed = False
                        if plot_item is not None:
                            try:
                                plot_item.removeItem(item)
                                removed = True
                            except Exception:
                                logger.debug(
                                    "plot_item.removeItem 失败", exc_info=True
                                )
                        if not removed and scene is not None:
                            safe_qt_op(scene.removeItem, item)
                    safe_qt_op(item.setVisible, False)
                    if trash is not None:
                        trash.append(item)
                widget._pending_delete_items.clear()
            widget._is_updating_data = True
            widget._cached_data_version = 0  # 设为0跳过版本比对，由 _is_updating_data 和 _is_loading_new_data 提供锁定
            # 暂停视图更新，防止 scene 中间状态触发 paint event 导致 SIGSEGV
            widget.setUpdatesEnabled(False)
            if hasattr(widget, "_cancel_ui_refresh"):
                widget._cancel_ui_refresh()
            if hasattr(widget, "_cursor_refresh_timer"):
                widget._cursor_refresh_timer.stop()
            if hasattr(widget, "_interaction_timer"):
                widget._interaction_timer.stop()

    def _end_data_reload(self):
        if not self.mw._is_loading_new_data:
            return

        # 不在此时清除任何锁（_is_updating_data / _is_loading_new_data）
        # v5.0：锁的清除最终在 _post_reload_ui_refresh() 中执行（经 _restore_cursor_state_after_reload 调度），
        # 确保 setUpdatesEnabled(True) 和场景刷新完成后才放行 paintEvent

        # 安全性兜底：如果 _restore_cursor_state_after_reload 因极端情况未执行，
        # 3 秒后强制解锁。使用版本号防止跨 reload 污染。
        self._safety_unlock_version = self.mw._data_version
        self._safety_timer = QTimer(self.mw)
        self._safety_timer.setSingleShot(True)
        self._safety_timer.timeout.connect(self._safety_force_unlock)
        self._safety_timer.start(3000)

        # 必须走 _schedule_delayed（QTimer 实例，parent 归 mw）：三参
        # QTimer.singleShot 在 Nuitka 打包形态下会抛 TypeError，这一句失败
        # 就等于整条收尾链路不再执行 → 3 秒后兜底解锁接管
        self._schedule_delayed(0, self._restore_cursor_state_after_reload)

    def _schedule_delayed(self, msec: int, slot):
        """在 ``msec`` 毫秒后把 ``slot`` 投递到 GUI 线程；窗口销毁即取消。

        不用 ``QTimer.singleShot``：三参重载在 Nuitka 打包形态下重载解析
        不稳定（PySide6 post-load 补丁只保活两参形态的槽）会抛 TypeError；
        两参重载又没有 context，窗口销毁后回调还会打到已删控件。QTimer 实例
        的 parent 归 ``mw``，窗口销毁时定时器随之销毁、回调不再投递，与三参
        重载语义等价；``timeout.connect(slot)`` 会保活槽对象。
        """
        timer = QTimer(self.mw)
        timer.setSingleShot(True)
        timer.timeout.connect(slot)
        # 与静态 singleShot 语义对齐：触发后自毁，否则每次调用都会在 mw 上
        # 留下一个闲置定时器
        timer.timeout.connect(timer.deleteLater)
        timer.start(max(0, int(msec)))
        return timer

    def _restore_cursor_state_after_reload(self):
        """延迟恢复 cursor 状态（在 UI 稳定后调用）"""
        if getattr(self.mw, "_is_being_destroyed", False):
            self._force_unlock_all()
            return

        saved_cursor_mode = getattr(self.mw, "_saved_cursor_mode", None)
        saved_pinned_x_values = getattr(self.mw, "_saved_pinned_x_values", [])

        is_free_cursor = (
            not saved_cursor_mode
            or saved_cursor_mode == "1 free cursor"
            or not saved_pinned_x_values
        )

        try:
            # v5.11: 先将 vline/vline2 重新添加到 scene，再做状态恢复。
            # _begin_data_reload 中将它们从 scene 移除了，此处必须配对恢复。
            # 对 free cursor 也必须恢复——free cursor 也有 vline（只是不可见）。
            for container in getattr(self.mw, "plot_widgets", []):
                widget = getattr(container, "plot_widget", None)
                if widget:
                    self._reattach_vlines_to_scene(widget)

            # v5.x 修复：无条件重连 vline 信号（必须在 free cursor 判断之前）。
            # _begin_data_reload 断开了信号，但原实现把重连放在 is_free_cursor return 之后，
            # 导致 reload 前 free cursor 时信号永远不会被重连，后续开启 anchored cursor
            # 拖动 vline 不会触发 on_vline_position_changed，跨 plot link 失效。
            for container in getattr(self.mw, "plot_widgets", []):
                widget = getattr(container, "plot_widget", None)
                if widget:
                    self._reconnect_vline_signals(widget)

            if is_free_cursor:
                # 防抖：阻止 reload 过渡期内的中间 cursor label 更新，
                # 只在 _post_reload_ui_refresh 中渲染一次
                for container in getattr(self.mw, "plot_widgets", []):
                    widget = getattr(container, "plot_widget", None)
                    if widget and hasattr(widget, "_last_cursor_update_time"):
                        widget._last_cursor_update_time = time.time()
                return

            # v5.x 修复问题 A：钳制 pinned 值到新数据范围。
            # reload 后新数据范围可能变小，旧 pinned 值可能超出范围；setPos 虽会按
            # maxRange 钳制，但各图钳完就是各图的值，先在这里统一钳一次才能让
            # MainWindow 与各 widget 的 pinned_x_values 保持同一个基准。
            # 此处统一钳制，后续代码自动使用钳制后的值。
            saved_pinned_x_values = self._clamp_pinned_to_global_range(saved_pinned_x_values)

            self.mw.cursor_mode = saved_cursor_mode
            self.mw.pinned_x_values = list(saved_pinned_x_values)

            widgets_list = []
            for container in getattr(self.mw, "plot_widgets", []):
                widget = getattr(container, "plot_widget", None)
                if widget:
                    widgets_list.append(widget)

            for widget in widgets_list:
                widget.is_cursor_pinned = True
                widget.pinned_x_values = list(saved_pinned_x_values)
                if saved_pinned_x_values:
                    widget.pinned_x_value = saved_pinned_x_values[0]
                    widget.pinned_index_value = (saved_pinned_x_values[0] - widget.offset) / widget.factor if widget.factor != 0 else None
                widget.pinned_index_values = []
                for x_val in saved_pinned_x_values:
                    if widget.factor != 0:
                        widget.pinned_index_values.append((x_val - widget.offset) / widget.factor)

                # v5.11: vline 已重新添加到 scene，现在可以安全操作
                if hasattr(widget, "vline"):
                    try:
                        vline = widget.vline
                        if vline is not None:
                            if saved_pinned_x_values:
                                with QSignalBlocker(vline):
                                    vline.setPos(saved_pinned_x_values[0])
                            vline.setMovable(True)
                    except RuntimeError:
                        logger.debug("vline C++ 对象已销毁，跳过恢复")

                if hasattr(widget, "vline2"):
                    try:
                        vline2 = widget.vline2
                        if vline2 is not None:
                            if len(saved_pinned_x_values) > 1:
                                with QSignalBlocker(vline2):
                                    vline2.setPos(saved_pinned_x_values[1])
                            vline2.setMovable(True)
                    except RuntimeError:
                        logger.debug("vline2 C++ 对象已销毁，跳过恢复")

                if saved_cursor_mode == "2 anchored cursor":
                    if hasattr(widget, "vline"):
                        try:
                            if widget.vline is not None:
                                widget.vline.setVisible(True)
                        except RuntimeError:
                            logger.debug("vline C++ 对象已销毁，跳过显示")
                    if hasattr(widget, "vline2"):
                        try:
                            if widget.vline2 is not None:
                                widget.vline2.setVisible(True)
                        except RuntimeError:
                            logger.debug("vline2 C++ 对象已销毁，跳过显示")
                elif saved_cursor_mode == "1 anchored cursor":
                    if hasattr(widget, "vline"):
                        try:
                            if widget.vline is not None:
                                widget.vline.setVisible(True)
                        except RuntimeError:
                            logger.debug("vline C++ 对象已销毁，跳过显示")
                    if hasattr(widget, "vline2"):
                        try:
                            if widget.vline2 is not None:
                                widget.vline2.setVisible(False)
                        except RuntimeError:
                            logger.debug("vline2 C++ 对象已销毁，跳过隐藏")

                if hasattr(widget.view_box, "is_cursor_pinned"):
                    widget.view_box.is_cursor_pinned = True

                # 防抖：设置 throttle 时间戳，阻止 reload 过渡期内的中间 cursor label 更新。
                # pin 状态已恢复、vline 可见性已设置，label 渲染留给 _post_reload_ui_refresh 统一处理。
                if hasattr(widget, "_last_cursor_update_time"):
                    widget._last_cursor_update_time = time.time()

            # v5.x: vline 信号重连已上移到 free cursor 判断之前（_reconnect_vline_signals），
            # 此处不再重复连接，避免槽被多次调用。

        except Exception:
            logger.debug("恢复 pin 状态失败", exc_info=True)
        finally:
            # v5.0 修复：不在此处清除锁！
            # 锁的清除延迟到 _post_reload_ui_refresh() 中执行，
            # 在 setUpdatesEnabled(True) 和场景刷新完成后再清除，
            # 消除锁清除与 setUpdatesEnabled 之间约 50ms 的危险窗口。
            # 在此期间 paintEvent 由 _is_loading_new_data 和 _is_updating_data 双锁拦截。
            self._post_reload_pending_version = self.mw._data_version
            # 窗口销毁即取消（定时器 parent 归 mw），语义等价于三参重载的 context
            # 保活；同时避开三参重载在打包形态下抛 TypeError 的老问题
            self._schedule_delayed(50, self._post_reload_ui_refresh)

    def _stop_safety_timer(self):
        """停掉兜底解锁定时器（可重复调用）。

        清理期窗口的 C++ 对象可能已随窗口销毁、只剩 Python 包装器，此时
        ``stop()`` 会抛 RuntimeError；它若发生在异常处理器内部就会逃逸出槽
        函数，被 PySide6 交给 sys.excepthook（pytest-qt 记为 teardown error）。
        """
        timer = getattr(self, "_safety_timer", None)
        if timer is None:
            return
        self._safety_timer = None
        try:
            timer.stop()
        except RuntimeError:
            logger.debug("_stop_safety_timer: QTimer 的 C++ 对象已销毁", exc_info=True)

    def _window_going_away(self) -> bool:
        """窗口是否已进入销毁流程（closeEvent 置位）或已断链。

        不能直接碰 ``self.mw``：它是弱引用 property，断链时自己就会抛
        ``RuntimeError("MainWindow has been garbage collected")``。
        """
        mw = self._mw_ref()
        return mw is None or bool(getattr(mw, "_is_being_destroyed", False))

    def _safety_force_unlock(self):
        """版本感知的安全解锁：仅当版本匹配时才执行解锁，防止跨 reload 污染"""
        current_version = getattr(self.mw, "_data_version", 0)
        safety_version = getattr(self, "_safety_unlock_version", -1)
        if safety_version != current_version:
            return
        self._force_unlock_all()

    def _force_unlock_all(self):
        """紧急解锁：确保所有退出路径都不会留下死锁。

        兜底解锁**不得销毁曲线**：本函数是 3 秒定时器投递的最后手段，触发时
        往往只是收尾调度没跑成（历史上正是打包形态下三参 singleShot 抛
        TypeError 那条链），曲线本身是好的。旧实现调 ``_safe_clear_plot_items``
        把全部曲线清掉，用户看到的是「重载完 3 秒后曲线全消失」，而
        ``curves`` 字典与 cursor 读数还在 —— 状态与画面彻底对不上。
        这里改为只重建真正失效的曲线（``_refresh_curve_paint_path``）。
        """
        if self._live_mw() is None:
            # 窗口正在销毁 → 原地收口，不去清曲线也不去解双锁：本函数由 3 s 兜底
            # 定时器投递，落地时控件往往已经没了，摸下去只会在半途抛 RuntimeError，
            # 把「解锁」这一语义切成做了一半（历史上正是「过一会儿曲线全消失」那类
            # 事故的放大器）。窗口都没了，也就不需要这把锁了。
            logger.info("紧急解锁已跳过：主窗口正在销毁")
            return
        logger.warning(
            "紧急解锁触发：强制解锁 + 修复失效曲线 "
            "(data_version=%d, safety_unlock_version=%d)",
            getattr(self.mw, "_data_version", -1),
            getattr(self, "_safety_unlock_version", -1),
        )
        # v5.12: 紧急解锁时也要释放 reload 互斥锁
        self._reload_in_progress = False
        self.mw.reload_btn.setEnabled(True)
        self.mw._is_loading_new_data = False
        self._safety_unlock_version = -1
        self._stop_safety_timer()
        for container in self._iter_all_containers():
            widget = getattr(container, "plot_widget", None)
            if widget:
                if not _qt_alive(widget):
                    continue
                # v5.11: 紧急解锁时也要把 vline 加回 scene，否则后续 paint 不会绘制光标
                self._reattach_vlines_to_scene(widget)
                widget._is_updating_data = False
                widget._cached_data_version = self.mw._data_version
                widget.setUpdatesEnabled(True)
                # 只重建失效曲线，正常曲线原样保留
                try:
                    self._refresh_curve_paint_path(widget)
                except Exception:
                    logger.debug(
                        "_refresh_curve_paint_path 失败（紧急解锁期间）", exc_info=True
                    )
                # 紧急解锁后触发一次完整刷新，防止 paintEvent 跳过导致白屏
                if hasattr(widget, "_queue_ui_refresh"):
                    widget._queue_ui_refresh(immediate=True)

    def _post_reload_ui_refresh(self):
        """重载完成后的 UI 刷新：分步清除锁，确保场景一致后再放行 paintEvent。

        清除顺序：
        1. _is_updating_data → False（允许回调执行）
        2. setUpdatesEnabled(True)（恢复 Qt 更新）
        3. _is_loading_new_data → False（放行 paintEvent）
        4. _refresh_curve_paint_path（v5.8 safety net：检测失效曲线并重建）
        5. widget.viewport().update()（显式触发 viewport 重绘）
        6. 单一延迟回调（把 cursor 更新推到下一个事件循环迭代，
           合并 13 个 per-widget 回调为 1 个，避免 BSP 树交叉修改）
        """
        if self._window_going_away():
            # widget 的 C++ 对象可能已删（Python 包装器还在，照属性名取值不报错、
            # 真摸控件才炸）：只停掉兜底定时器——留着它 3 秒后还会去摸已删的
            # reload_btn
            self._stop_safety_timer()
            return

        pending_version = getattr(self, "_post_reload_pending_version", -1)
        if pending_version != getattr(self.mw, "_data_version", 0):
            logger.debug(
                "_post_reload_ui_refresh 跳过：版本不匹配 pending=%d current=%d",
                pending_version, self.mw._data_version,
            )
            return
        try:
            widgets_to_refresh = []
            for container in self._iter_all_containers():
                widget = getattr(container, "plot_widget", None)
                if not widget:
                    continue
                if not _qt_alive(widget):
                    # 窗口正在销毁时子控件先没。必须在这里就跳过：后面任何一次
                    # RuntimeError 都会把 _is_loading_new_data 的复位挤进 except，
                    # 而 except 里若再抛，锁就永久残留 —— 下一次点「导入」会被
                    # 入口守卫直接吃掉（表现是按钮点不动）。
                    continue
                widget._is_updating_data = False
                widget._cached_data_version = self.mw._data_version
                widget.setUpdatesEnabled(True)
                if hasattr(widget, "_queue_ui_refresh"):
                    widgets_to_refresh.append(widget)

            self.mw._is_loading_new_data = False
            self._safety_unlock_version = -1
            self._stop_safety_timer()

            # v5.12: reload 流程完全结束，恢复 reload 按钮
            self.mw.reload_btn.setEnabled(True)

            # v5.8 safety net：检测并重建失效曲线
            for widget in widgets_to_refresh:
                try:
                    self._refresh_curve_paint_path(widget)
                except Exception:
                    logger.debug("_refresh_curve_paint_path 失败", exc_info=True)

            # v5.3 修复问题 1：用 viewport().update() 可靠触发 QGraphicsView 重绘
            for widget in widgets_to_refresh:
                widget.viewport().update()

            # v5.3 修复问题 2：合并 13 个 per-widget 延迟回调为 1 个统一回调
            current_version = self.mw._data_version
            self._schedule_delayed(
                0,
                lambda: self._deferred_cursor_refresh_all(widgets_to_refresh, current_version)
            )
        except Exception:
            logger.debug("_post_reload_ui_refresh 执行失败", exc_info=True)
            # 兜底收尾本身也不得再抛：这里每步都用 safe_qt_op / 存活判断包一层，
            # 否则第二次异常会把后面的复位一起带走（锁残留 = 界面永久卡住）。
            timer = getattr(self, "_safety_timer", None)
            if timer is not None:
                safe_qt_op(timer.stop)
            self._safety_timer = None
            self._safety_unlock_version = -1
            mw = self._live_mw()
            if mw is None:
                return
            mw._is_loading_new_data = False
            # v5.12: 异常路径也要恢复 reload 按钮
            safe_qt_op(mw.reload_btn.setEnabled, True)

    def _refresh_curve_paint_path(self, widget):
        """v5.8 safety net：检测失效曲线并重建。

        失效通常由 _force_unlock_all 误触发 _safe_clear_plot_items 导致，
        表现为 PlotDataItem 从 scene 移除（scene=None）或 _dataset 被清空。
        v5.8 已通过取消过期 safety timer 根治此问题，本方法作为兜底防御。
        """
        if not hasattr(widget, "plot_item") or widget.plot_item is None:
            return

        curves_to_check = []
        try:
            if hasattr(widget, "curves") and widget.curves:
                for var_name, ci in widget.curves.items():
                    if ci.curve is not None:
                        curves_to_check.append((var_name, ci.curve))
        except (RuntimeError, AttributeError):
            return

        for var_name, curve in curves_to_check:
            safe_qt_op(lambda: self._safe_check_and_rebuild_curve(widget, var_name, curve))

    def _safe_check_and_rebuild_curve(self, widget, var_name, curve):
        """检查曲线状态并在失效时尝试重建"""
        curve_scene = curve.scene()
        dataset_none = getattr(curve, "_dataset", None) is None
        if curve_scene is None or dataset_none:
            logger.warning(
                "[v5.8] curve[%s] 检测到失效状态 (scene=%s, dataset=%s)，尝试重建",
                var_name,
                "ok" if curve_scene is not None else "None",
                "None" if dataset_none else "ok",
            )
            try:
                if (
                    hasattr(widget, "_multi_curve_manager")
                    and hasattr(widget, "curves")
                    and var_name in widget.curves
                ):
                    widget._multi_curve_manager._recreate_curve(var_name)
                else:
                    logger.warning(
                        "[v5.8] curve[%s] 重建失败：无法确定重建方式", var_name
                    )
            except Exception:
                logger.warning(
                    "[v5.8] curve[%s] 重建失败", var_name, exc_info=True
                )

    def _deferred_cursor_refresh_all(self, widgets, ver):
        """统一的延迟 cursor 刷新回调（v5.3：替代 per-widget QTimer.singleShot）

        串行遍历所有 widget 执行 cursor 更新，避免 13 个并发回调交叉修改 BSP 树。
        每个 widget 更新前后设置/清除 _is_cursor_modifying_scene 护栏，
        阻止异步 paint 事件访问 BSP 中间态。
        """
        if self._window_going_away():
            return
        if getattr(self.mw, "_data_version", 0) != ver:
            return
        if getattr(self.mw, "_is_loading_new_data", False):
            return
        if getattr(self, "_post_reload_cursor_refreshing", False):
            return

        self._post_reload_cursor_refreshing = True
        try:
            for w in widgets:
                if getattr(self.mw, "_data_version", 0) != ver:
                    return
                if hasattr(w, "_last_cursor_update_time"):
                    w._last_cursor_update_time = 0
                # paint 护栏：阻止 paint event 在场景修改期间访问 BSP 中间态
                w._is_cursor_modifying_scene = True
                try:
                    w._queue_ui_refresh(immediate=True)
                finally:
                    w._is_cursor_modifying_scene = False
        except Exception:
            logger.debug("_deferred_cursor_refresh_all 执行失败", exc_info=True)
        finally:
            self._post_reload_cursor_refreshing = False

    def _live_mw(self):
        """返回 Python 引用与 C++ 对象**两侧**都还存活的 MainWindow，否则 ``None``。

        收尾 / 延迟回调专用的存活判据：``self.mw`` 这个弱引用只探得到 Python 侧被
        回收，而 ``close() → deleteLater()`` 之后包装器往往还活着、Python 属性照读
        不误 —— 拿它当守卫会继续往下走并在第一个控件调用上抛 ``RuntimeError``
        （子控件比窗口先没）。
        """
        try:
            mw = self.mw
        except RuntimeError:  # 弱引用已失效
            return None
        return mw if _qt_alive(mw) else None

    def _iter_all_containers(self):
        """遍历所有页的绘图容器；旧式替身/构造期退化为当前页 ``plot_widgets``。

        解锁、释放旧数据这类「数据层 / 全局」路径必须跨页（设计文档 §7.2
        #10/#11），不能只用垫片返回的当前页列表。
        """
        iter_all = getattr(self.mw, "iter_all_plot_widgets", None)
        if callable(iter_all):
            try:
                return list(iter_all())
            except Exception:
                logger.debug("遍历全部页绘图容器失败，回退当前页", exc_info=True)
        return list(getattr(self.mw, "plot_widgets", []))

    def load_csv_file(self, file_path: str):
        # 入口唯一规范化点：文件对话框、拖拽（Qt 会把 UNC 产出成 "//host/share"）、
        # 命令行位置参数、重新加载全部经这里，之后下游看到的都是当前平台的
        # 绝对写法（幂等，不抛异常）。
        file_path = normalize_input_path(file_path)
        logger.info("开始加载文件: %s", file_path)

        if getattr(self.mw, "_is_loading_new_data", False):
            self.mw.load_btn.setEnabled(True)
            return

        if not self._validate_file_path(file_path):
            self.mw.load_btn.setEnabled(True)
            return

        if not self._check_file_size(file_path):
            self.mw.load_btn.setEnabled(True)
            return

        try:
            self._load_file(file_path)
        except MemoryError as e:
            self._report_load_failure("内存不足", "文件太大，内存不足。请尝试加载较小的文件。")
            # 内存不足是唯一允许连带释放旧数据的失败：不腾出内存连重试都做不了
            self._abort_load_after_failure(e, "内存不足退出", release_old=True)
        except Exception as e:
            self._report_load_failure("加载错误", f"加载文件时发生错误: {str(e)}")
            self._abort_load_after_failure(e, "加载失败退出")
        finally:
            mw = self._live_mw()
            if mw is not None:
                mw.raise_()
                mw.activateWindow()

    def _report_load_failure(self, title: str, text: str) -> None:
        """失败弹窗：窗口正在销毁时静默跳过（此时弹窗只会阻塞退出流程）。"""
        mw = self._live_mw()
        if mw is None:
            logger.info("加载失败弹窗已跳过（主窗口正在销毁）: %s", text)
            return
        QMessageBox.critical(mw, title, text)

    def _abort_load_after_failure(
        self, exc: BaseException, stage: str, *, release_old: bool = False
    ) -> None:
        """加载失败出口的收口：记日志 + 放平 UI 状态，**默认保留**旧数据。

        旧实现在这里无条件调 ``_release_old_data()``，而后者的 docstring 明确写着
        「仅在新 loader 成功后调用」。后果是：用户正看着一屏曲线，去打开另一个
        文件失败（检测阶段文件变得不可读、BLF 起线程失败等），原有数据和曲线被
        一起清掉，而且没有任何撤销入口。

        ``release_old=True`` 只留给内存不足这种「不腾出内存就没法重试」的场景。
        """
        logger.warning("%s: %s", stage, exc)
        if release_old:
            try:
                self._release_old_data()
            except Exception:
                logger.debug("%s 清理旧数据失败", stage, exc_info=True)
        # 失败可能发生在 _begin_data_reload() 之后（锁已置位），这里补一次解锁；
        # 没上过锁时 _end_data_reload 自己会早退
        try:
            self._end_data_reload()
        except Exception:
            logger.debug("%s 解锁失败", stage, exc_info=True)
        self._finish_load_failure()

    def _finish_load_failure(
        self, hint: str = "加载失败", is_reload: bool = False
    ) -> None:
        """把加载期抬起的 UI 状态一次放平：忙等光标 / 秒表 / 分析按钮 / reload 锁。

        与后台线程的 ``_on_load_error``、BLF 的 ``_on_blf_error`` 同口径。分析类
        按钮按「当前是否还有有效数据」恢复，而不是无条件置真 —— 失败前若本来
        就没有数据，把它们点亮等于让用户去点必然再报错的入口。
        """
        mw = self._live_mw()
        if mw is None:
            return
        # 本函数是在异常处理器里跑的：自己再抛一次就会把原始错误顶掉，并把
        # 「恢复」切成做了一半（本仓反复修过这一类，见 _stop_safety_timer 的
        # 注释）。所以每一步各自兜住，谁失败都不影响后面的步骤。
        end_feedback = getattr(mw, "end_load_feedback", None)
        if callable(end_feedback):
            safe_qt_op(end_feedback, hint, level="error")
        try:
            has_data = self._has_valid_data
        except Exception:
            has_data = False
        try:
            self.set_button_status(has_data)
        except Exception:
            logger.debug("恢复分析按钮失败", exc_info=True)
        safe_qt_op(mw.load_btn.setEnabled, True)
        if is_reload or getattr(self, "_reload_in_progress", False):
            self._reload_in_progress = False
            safe_qt_op(mw.reload_btn.setEnabled, True)

    def set_button_status(self, status: bool):
        if status is not None:
            # 「时间修正」按钮已撤，入口换成状态栏中段的 x 轴抽屉（抽屉自己按
            # loader/加载中状态把关），这里不再列它
            self.mw.clear_all_plots_btn.setEnabled(status)
            self.mw.auto_range_btn.setEnabled(status)
            self.mw.auto_y_btn.setEnabled(status)
            self.mw.cursor_btn.setEnabled(status)
            self.mw.mark_region_btn.setEnabled(status)
            self.mw.grid_layout_btn.setEnabled(status)

    def reload_data(self):
        # v5.12: 互斥锁 —— 防止快速连续 reload 导致并发冲突
        if getattr(self, "_reload_in_progress", False):
            logger.debug("Reload 正在进行中，忽略本次请求")
            return

        if getattr(self.mw, "_is_loading_new_data", False):
            return

        if not self._has_valid_loader:
            QMessageBox.critical(self.mw, "错误", "没有可重新加载的数据")
            return

        if not hasattr(self.mw.loader, "path") or not self.mw.loader.path:
            QMessageBox.critical(self.mw, "错误", "数据路径无效")
            return

        if not os.path.isfile(self.mw.loader.path):
            QMessageBox.critical(self.mw, "错误", "文件不存在，无法重新加载")
            return

        self._reload_in_progress = True
        self.mw.reload_btn.setEnabled(False)

        # 获取缓存的 sheet_name（如果当前 loader 是 ExcelDataLoader）
        cached_sheet_name = None
        if hasattr(self.mw.loader, '_sheet_name'):
            cached_sheet_name = self.mw.loader._sheet_name

        is_async = False
        try:
            logger.info("重新加载数据: %s", self.mw.loader.path)
            self._load_file(self.mw.loader.path, is_reload=True,
                            cached_sheet_name=cached_sheet_name)
            # 判断是否走了异步路径（后台线程已启动）
            is_async = getattr(self.mw, "_thread", None) is not None and \
                       getattr(self.mw._thread, "isRunning", lambda: False)()
        except Exception as e:
            # 同步路径异常或 _load_file 内部未启动线程就抛异常
            logger.error("数据重新加载失败: %s", e, exc_info=True)
            QMessageBox.warning(
                self.mw, "加载失败",
                f"重新加载数据时出错：\n{e}"
            )
        finally:
            if not is_async:
                # 同步路径（无论成功或失败）：释放互斥锁，防止重复点击。
                # 按钮恢复延迟到 _post_reload_ui_refresh，与 _is_loading_new_data 同步；
                # 但如果加载失败（_post_reload_ui_refresh 不会被调用），必须在此处恢复按钮。
                self._reload_in_progress = False
                if not getattr(self.mw, "_is_loading_new_data", False):
                    # 加载失败路径：_end_data_reload 已清除 _is_loading_new_data，
                    # _post_reload_ui_refresh 不会运行，必须在此恢复按钮
                    self.mw.reload_btn.setEnabled(True)

    def _load_file(self, file_path: str, is_reload: bool = False,
                   cached_sheet_name: str | None = None):
        file_ext = self._extract_file_extension(file_path)
        if file_ext is None:
            # 白名单外此前会一路走到 config_dict 分支，抛 'NoneType' object is not
            # subscriptable，并被 load_csv_file 的兜底 except 判定为加载失败而清掉
            # 当前已加载的数据。
            QMessageBox.warning(
                self.mw,
                "不支持的文件类型",
                f"无法识别文件类型：{os.path.basename(file_path)}\n\n"
                f"支持的扩展名：{'、'.join(SUPPORTED_EXTENSIONS)}\n"
                "（分卷文件按末尾 .0/.1 之前的那段扩展名识别）",
            )
            self.mw.load_btn.setEnabled(True)
            return
        is_mdf_file = file_ext in (".mf4", ".mdf", ".dat")
        is_blf_file = file_ext == ".blf"
        is_excel_file = file_ext in (".xlsx", ".xlsm")

        delimiter_typ = None
        desc_rows = None
        has_unit = None
        encoding = None
        config_used = False
        sheet_name: str | None = None

        if is_mdf_file:
            delimiter_typ = ","
            desc_rows = 0
            has_unit = False
            config_used = True

        if is_blf_file:
            # §5.1：BLF 必须先拿到 CAN 数据库才能导入。拿不到（用户取消）
            # 就回到导入前状态，**不进入空数据态**。
            db_paths = self._resolve_can_databases(file_path, is_reload=is_reload)
            if db_paths is None:
                self._cancel_pending_reload(is_reload)
                return
            self._blf_db_paths = db_paths
            delimiter_typ = ","
            desc_rows = 0
            has_unit = False
            config_used = True

        elif is_excel_file:
            # reload 时复用缓存的 sheet_name，避免重复弹出
            if is_reload and cached_sheet_name:
                sheet_name = cached_sheet_name
            else:
                from src.ui.dialogs.sheet_selector import SheetSelectorDialog
                sheet_name = SheetSelectorDialog.pick_sheet(file_path, self.mw)
                if not sheet_name:
                    self.mw.load_btn.setEnabled(True)
                    return

            # Excel 不需要分隔符/编码检测，desc_rows=None 触发自动检测
            delimiter_typ = ","
            desc_rows = None
            config_used = True

        if not is_mdf_file and not is_blf_file and not is_excel_file:
            config_path = self._resolve_config_path("config_dict.json")
            if config_path is not None and os.path.isfile(config_path):
                try:
                    config_dict = self.load_dict(config_path)
                    ext_dict = config_dict.get(file_ext[1:], {})
                    cfg_sep = ext_dict.get("sep")
                    cfg_skip = ext_dict.get("skiprows")
                    cfg_has_unit = ext_dict.get("has_unit")
                    if (
                        cfg_sep is not None
                        and cfg_skip is not None
                        and cfg_has_unit is not None
                    ):
                        delimiter_typ = cfg_sep
                        desc_rows = int(cfg_skip)
                        has_unit = bool(cfg_has_unit)
                        config_used = True
                except Exception as e:
                    QMessageBox.warning(
                        self.mw,
                        "配置文件错误",
                        f"config_dict.json 读取失败，将使用自动检测方式加载文件。\n\n错误详情: {e}",
                    )
                    logger.warning("config_dict.json 读取失败: %s", e)

        if not config_used:
            try:
                fmt = FastDataLoader.auto_detect(file_path)
                delimiter_typ = fmt.sep
                desc_rows = fmt.header_row
                has_unit = fmt.has_unit
                encoding = fmt.encoding
                logger.debug(
                    "自动检测: sep=%s, header=%d, has_unit=%s, enc=%s",
                    delimiter_typ, desc_rows, has_unit, encoding,
                )
            except AutoDetectError as e:
                QMessageBox.critical(
                    self.mw,
                    "数据解析失败",
                    "无法自动识别文件的标题行和分隔符。\n"
                    "请确认文件格式是否正确。\n"
                    "支持的分隔符：逗号(,)、分号(;)、制表符(Tab)",
                )
                # 调用方（导入按钮 / 拖拽）在进来前已经把 load_btn 灰掉，
                # 这些「还没上锁就返回」的分支必须自己把它点亮，否则用户
                # 连下一次导入都点不动
                self.mw.load_btn.setEnabled(True)
                return
            except OSError as e:
                # auto_detect 内部要 open(file_path, "rb")：文件在 isfile 校验之后
                # 被移动/删除/占用（网盘同步、Excel 独占）时抛 OSError。旧实现只
                # 捕获 AutoDetectError，这个 OSError 会一路逃到 load_csv_file 的
                # 外层 except，连带把当前已加载的数据清掉。
                logger.error("自动检测阶段无法读取文件: %s", e, exc_info=True)
                QMessageBox.critical(
                    self.mw,
                    "文件不可读",
                    f"无法读取文件内容，文件可能已被移动、删除或占用：\n"
                    f"{file_path}\n\n错误详情: {e}",
                )
                self.mw.load_btn.setEnabled(True)
                return

        if not is_excel_file and not is_blf_file and (delimiter_typ is None or desc_rows is None or has_unit is None):
            QMessageBox.critical(
                self.mw,
                "数据解析失败",
                "无法确定文件的分隔符和标题行位置。\n" "请确认文件格式是否正确。",
            )
            self.mw.load_btn.setEnabled(True)
            return

        # P0-4: 取尺寸必须在 _begin_data_reload() 上锁之前 —— 文件在 isfile 校验之后、
        # getsize 之前被删除/占用（网盘同步、Excel 独占、误操作）时 OSError 若逃出
        # _load_file，锁已置位而 _end_data_reload 永不执行，后续所有加载被静默拒绝。
        try:
            file_size = os.path.getsize(file_path)
        except OSError as e:
            logger.error("读取文件大小失败: %s", e, exc_info=True)
            QMessageBox.critical(
                self.mw,
                "文件不可读",
                f"无法读取文件大小，文件可能已被移动或删除：\n{file_path}\n\n错误详情: {e}",
            )
            return

        self._begin_data_reload()
        self._load_started_at = time.perf_counter()
        self.mw.begin_load_feedback(file_path)
        # 加载期禁能分析类按钮：模态框只挡鼠标点击，挡不住链路内部
        # processEvents 触发的重入（本仓历史上多次修过这类丢列/崩溃）
        self.set_button_status(False)
        started_async = False
        _Threshold_Size_Mb = FILE_SIZE_LIMIT_BACKGROUND_LOADING

        if is_blf_file:
            # BLF 一律走后台线程：扫描是逐条构造 Message 的分钟级循环，
            # 文件再小也不该让 UI 等（且只有在线程里才能做可取消）。
            try:
                self._start_blf_load(
                    file_path,
                    getattr(self, "_blf_db_paths", []) or [],
                    is_reload=is_reload,
                )
                started_async = True
            except Exception:
                self.mw.end_load_feedback("加载失败", level="error")
                self._end_data_reload()
                self.mw.load_btn.setEnabled(True)
                if is_reload:
                    self._reload_in_progress = False
                    self.mw.reload_btn.setEnabled(True)
                raise
            return

        try:
            if file_size < _Threshold_Size_Mb * 1024 * 1024:
                logger.info("同步加载文件 (%.1f MB)", file_size / 1024 / 1024)
                try:
                    status = self._load_sync(
                        file_path,
                        desc_rows=desc_rows,
                        sep=delimiter_typ,
                        has_unit=has_unit,
                        encoding=encoding,
                        sheet_name=sheet_name,
                        is_excel=is_excel_file,
                        is_reload=is_reload,
                    )
                finally:
                    self._end_data_reload()
                if status:
                    self.set_button_status(True)
                    self.mw.load_btn.setEnabled(True)
                    # 延迟到下一个事件循环，确保 paint 事件先处理，避免 UI 半成品白屏
                    self._schedule_delayed(
                        0,
                        lambda: self._post_load_actions(file_path, is_reload=is_reload),
                    )
                else:
                    # _load_sync 已经弹过具体错误框；这里只负责把加载期抬起的
                    # 忙等光标 / 秒表 / 6 个分析按钮放平。旧实现只恢复 load/reload，
                    # 结果是失败一次后状态栏永远停在「正在加载 … Ns」、分析按钮
                    # 永久灰掉，只有下一次成功加载才清得掉。
                    self._finish_load_failure(is_reload=is_reload)
            else:
                logger.info("后台加载文件 (%.1f MB)", file_size / 1024 / 1024)

                self.mw._thread = DataLoadThread(
                    file_path,
                    desc_rows=desc_rows,
                    sep=delimiter_typ,
                    has_unit=has_unit,
                    encoding=encoding,
                    sheet_name=sheet_name,
                )

                # ≥2MB 才弹：小文件同步读，弹一次模态反而比加载本身慢
                self.mw._progress = QProgressDialog(
                    f"正在加载数据... [{os.path.basename(file_path)}]",
                    None,
                    0, 0,  # min == max → 自动切换为 indeterminate（来回摆动）
                    self.mw
                )
                self.mw._progress.setCancelButton(None)
                self.mw._progress.setWindowModality(Qt.WindowModality.ApplicationModal)
                self.mw._progress.setAutoClose(True)
                self.mw._progress.show()

                _load_version = self.mw._data_version
                self.mw._thread.finished.connect(
                    lambda loader: self._on_load_done(loader, file_path, _load_version, is_reload)
                )
                self.mw._thread.error.connect(self._on_load_error)
                # 跨线程 GC 死锁防护：worker 内一旦发生全量 GC（显式调用或自动分代
                # 收集），会把主线程的 QObject 包装器拿到 worker 析构，与主线程
                # 锁序倒置。起跑前先在本线程排空引用环垃圾，使 worker 侧无对象可错。
                gc.collect()
                self.mw._thread.start()
                started_async = True
        except Exception:
            if not started_async:
                self.mw.end_load_feedback("加载失败", level="error")
                self._end_data_reload()
            raise

    # ------------------------------------------------------------------
    # BLF / CAN 数据库集成（§5）
    # ------------------------------------------------------------------

    def _is_blf_source(self) -> bool:
        """当前数据源是否 BLF（决定 banner 与「CAN 数据库…」按钮显隐）。"""
        return getattr(getattr(self.mw, "loader", None), "LOADER_TYPE", "") == "blf"

    def _cancel_pending_reload(self, is_reload: bool) -> None:
        """§5.1「取消选择」：不进入空数据态，只把按钮恢复原状。"""
        self.mw.load_btn.setEnabled(True)
        if is_reload:
            self._reload_in_progress = False
            self.mw.reload_btn.setEnabled(True)

    def _usable_db_paths(self, paths):
        """把数据库路径预检后切成 ``(可用, 不可用)``。

        新导入一律弹窗（不允许静默导入），但预检仍有意义：记忆里已被移走 /
        改坏的老库要在弹窗里标红并给出黄条说明，而不是等用户「导入了才发现
        少了很多信号」。
        """
        from src.data.can_database import CanDatabaseSet

        normalized = [normalize_input_path(str(p)) for p in (paths or []) if str(p)]
        if not normalized:
            return [], []
        try:
            reports = CanDatabaseSet().load(normalized).reports
        except Exception as e:  # 预检本身绝不该打断导入
            logger.warning("CAN 数据库预检失败: %s", e)
            return [], normalized
        usable, bad = [], []
        # load() 对每个输入路径恰好产出一条 report，顺序一致
        for path, report in zip(normalized, reports):
            if report.ok and report.frame_count > 0:
                usable.append(path)
            else:
                bad.append(path)
        return usable, bad

    def _resolve_can_databases(self, file_path: str, is_reload: bool = False):
        """§5.1 的 B / C / F 前置路径。返回 ``db_paths``；``None`` = 中止导入。

        新导入**永远弹窗**（B）：不允许静默解析，上次记忆的库列表只作为弹窗里
        的预选项，命中与否由用户确认。唯一的豁免是「重载」——那是对当前数据
        源的再扫描，库本来就是用户之前显式选过的。
        """
        from html import escape

        from src.core.settings import AppSettings

        remembered = [normalize_input_path(p)
                      for p in AppSettings().get_blf_last_db_paths()]

        if is_reload and self._is_blf_source():
            # 「重载」始终复用上次 db_paths，绝不主动弹窗；只有真读不到才走 C
            current = [normalize_input_path(p)
                       for p in (getattr(self.mw.loader, "db_paths", None) or [])]
            usable, bad = self._usable_db_paths(current)
            if usable and not bad:
                return usable
            return self._ask_can_databases(
                file_path, initial=current, missing=bad,
                notice=BLF_TEXT_DB_MISSING.format(
                    paths=escape("\n".join(bad) or "（无）")
                ),
            )

        if getattr(self, "_blf_force_dialog", False):
            # F 路径：扫完 0 个信号，warning 后直接重开对话框（回到 B）
            self._blf_force_dialog = False
            preset = getattr(self, "_blf_preset_db_paths", None) or remembered
            self._blf_preset_db_paths = None
            paths = self._ask_can_databases(file_path, initial=preset)
            if paths is None:
                # 用户在重开的框里按了取消/没留下任何可用库：本轮导入到此为止，
                # 上一轮暂存的报文桶（可能几百 MB）没有下一轮可复用，立刻释放
                # （评审 A7）。
                self._release_pending_blf_loader()
                return None
            if list(paths) == list(preset or []):
                # 答案与上一轮逐个相同 → 报文没变、库也没变，结果必然还是 0 信号。
                # 换库改成毫秒级重绑后，这一环不再被「整文件重扫」天然限速，同一份
                # 错库会被快速提交成连环弹窗，所以这里直接收口：本轮按「未选库」
                # 取消处理，回到导入前状态，要再试由用户主动点「导入」。
                logger.info("CAN 数据库选择未变化，仍与 %s 的报文不匹配，已取消本轮导入",
                            file_path)
                return None
            return paths

        if remembered:
            # C：记忆命中但个别文件缺失/解析失败 → 弹窗 + 黄条说明
            usable, bad = self._usable_db_paths(remembered)
            if bad and usable:
                return self._ask_can_databases(
                    file_path, initial=remembered, missing=bad,
                    notice=BLF_TEXT_DB_MISSING.format(
                        paths=escape("\n".join(bad))
                    ),
                )

        # B：每次新导入都弹窗，预选上次记忆的列表（可能为空）
        return self._ask_can_databases(file_path, initial=remembered)

    def _ask_can_databases(self, file_path: str, initial=None, missing=None,
                           notice: str = ""):
        """弹 §5.2 的选择对话框。取消/未选中任何可用库 → ``None``。"""
        from src.core.settings import AppSettings
        from src.ui.dialogs.can_db_selector import CanDatabaseSelectorDialog

        dialog = CanDatabaseSelectorDialog(
            self.mw,
            data_file=file_path,
            initial_paths=list(initial or []),
            missing_paths=list(missing or []),
        )
        if notice:
            dialog.set_notice(notice)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            logger.info(BLF_TEXT_NO_DB_SELECTED)
            return None
        paths = dialog.get_selected_paths()
        if not paths:
            logger.info(BLF_TEXT_NO_DB_SELECTED)
            return None
        # 记忆写入时机：「导入」成功后才写；取消不写（§5.2）。此处先记住列表，
        # 扫描失败（异常路径）时不会回滚 —— 下次导入仍能在对话框里看到它们。
        # 复选框现在是「记住此列表，下次自动预选」：不勾 = 这一次的选择不入
        # 记忆，同时清掉旧记忆，否则旧列表会在下次弹窗里违背用户意图地出现。
        settings = AppSettings()
        settings.set_blf_last_db_paths(paths if dialog.remember_choice() else [])
        return paths

    def _close_blf_progress(self) -> None:
        dialog = getattr(self, "_blf_progress", None)
        self._blf_progress = None
        if dialog is None:
            return
        try:
            # 实测 Qt 的 ``QProgressDialog.close()`` **也会** emit ``canceled()``
            # （不只用户点取消才发）。先收线再关闭，否则正常扫完的 BLF 会在这
            # 一步被 ``_on_blf_cancel`` 打上取消标记，紧接着 ``_on_blf_done``
            # 把刚扫好的 loader 当过期结果丢弃 —— 表现是「导入完但变量列表空」。
            dialog.canceled.disconnect()
        except (RuntimeError, TypeError):  # 本就没接线 / 已被 Qt 销毁
            pass
        try:
            dialog.close()
        except RuntimeError:  #已被 Qt 销毁
            pass

    def _start_blf_load(self, file_path: str, db_paths, is_reload: bool = False):
        """§5.4：后台扫描 + 可量化进度 + **必须有取消按钮**。"""
        pending = self._blf_pending_loader
        if pending is not None:
            # F 路径回环（上一轮扫完但 0 信号）：文件没变就复用已扫到的报文桶，只
            # 重绑数据库。建线程反而会把同一个文件再扫一遍（分钟级 vs 毫秒级）。
            self._blf_pending_loader = None
            if pending.path == file_path:
                try:
                    pending.bind_databases(list(db_paths or []))
                except Exception as e:
                    logger.warning("复用已扫 BLF 重绑数据库失败，回落重新扫描: %s",
                                   e, exc_info=True)
                    self._discard_loader(pending)
                else:
                    # 代次沿用当前值：本轮虽然没起线程，但 _begin_data_reload 已经
                    # 递增过版本，下面的取消/过期两道守卫照旧有效。传 -1 反而危险 ——
                    # 它正是 _blf_cancelled_version 在「不带版本号的直接调用」下会
                    # 取到的值，撞上就是结果被静默丢弃 + _is_loading_new_data 永久残留。
                    self._on_blf_done(pending, file_path, self.mw._data_version,
                                      is_reload)
                    return
            self._discard_loader(pending)

        name = os.path.basename(file_path)
        thread = DataLoadThread(file_path, db_paths=list(db_paths))
        # QThread 的 Python 包装器一丢最后一条强引用就当场析构 C++ 对象，Qt 直接
        # std::terminate（实测：丢掉运行中线程的唯一引用后继续跑事件循环 →
        # ``QThread: Destroyed while thread is still running`` + SIGABRT）。
        # 取消路径刻意不 wait()（不能让 UI 等一个可能还在写大块内存的线程），
        # 而 ``mw._thread`` 是旧线程的唯一强引用 —— 用户在取消后点「重载」（无
        # 对话框、按钮已被恢复）就会在这一行覆盖它。先把还在跑的旧线程挂进保活
        # 列表，等它真正跑完再丢。与 MainWindow._shutdown_var_info_worker
        # 对变量信息统计线程做的兜底是同一条纪律。
        refs = self._blf_thread_refs
        if refs is None:
            refs = self._blf_thread_refs = []
        refs[:] = [t for t in refs if t.isRunning()]
        old = getattr(self.mw, "_thread", None)
        if old is not None and old is not thread and old.isRunning():
            refs.append(old)
        self.mw._thread = thread

        dialog = QProgressDialog(f"正在扫描 BLF 报文… [{name}]", "取消", 0, 100, self.mw)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        # 不用 autoClose/autoReset：进度会停在 90%（生成信号列表）再跳到 100%，
        # 自动复位会在扫描结束前先把框收掉。关闭一律走 _close_blf_progress。
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)
        dialog.show()

        self._blf_progress = dialog
        load_version = self.mw._data_version
        self._blf_load_version = load_version
        self._blf_cancelled_version = None

        thread.progress.connect(
            lambda value: self._on_blf_progress(value, load_version)
        )
        dialog.canceled.connect(thread.request_cancel)
        dialog.canceled.connect(lambda: self._on_blf_cancel(file_path, load_version))
        thread.finished.connect(
            lambda loader: self._on_blf_done(loader, file_path, load_version, is_reload)
        )
        thread.error.connect(
            lambda msg: self._on_blf_error(msg, is_reload, load_version)
        )
        thread.cancelled.connect(
            lambda path: self._on_blf_cancelled(path, is_reload, load_version)
        )
        thread.start()

    def _on_blf_progress(self, value: int, load_version: int = -1) -> None:
        if load_version >= 0 and load_version != self._blf_load_version:
            # 旧线程迟到的进度：新一轮已经在扫，不能拿旧百分比与新线程无关的
            # 统计去刷新一轮的进度框（评审 A5）。与 done/error/cancel 同口径。
            return
        dialog = getattr(self, "_blf_progress", None)
        if dialog is None:
            return
        value = max(0, min(100, int(value)))
        try:
            if value < 90:
                stats = getattr(self.mw._thread, "scan_stats", None) or {}
                read = int(stats.get("messages", 0))
                dialog.setLabelText(f"正在扫描 BLF 报文… {value}%（已读 {read:,} 帧）")
            else:
                dialog.setLabelText("正在生成信号列表…")
            dialog.setValue(value)
        except RuntimeError:
            self._blf_progress = None

    def _on_blf_cancel(self, file_path: str, load_version: int = -1) -> None:
        """G 路径：立刻静默恢复，后台线程在下个分块边界抛 InterruptedError。"""
        if load_version >= 0 and load_version != self._blf_load_version:
            # 旧线程迟到的取消：新一轮导入已经开始，不能拿它来收新导入的线
            return
        if self._blf_cancelled_version == load_version:
            return
        self._blf_cancelled_version = load_version
        logger.info(BLF_TEXT_CANCELLED.format(path=file_path))
        thread = getattr(self.mw, "_thread", None)
        if thread is not None and hasattr(thread, "request_cancel"):
            thread.request_cancel()
        self._close_blf_progress()
        # 「恢复原数据」必须含曲线：_begin_data_reload 已经把 item 从 scene 摘了，
        # 而取消路径不调 _release_old_data —— 旧 loader 还在、widget.curves 里的
        # 名字/颜色/可见性也还在，所以能原地重建。不重建的话，用户点一下取消
        # 就白白丢掉整张画布，只能重新拖变量。
        self._restore_curves_after_abort()
        self._end_data_reload()
        # 加载期被闸门禁用的分析按钮要还原（与 _on_load_error 同口径），
        # 否则取消一次 BLF 导入就让 6 个按钮永久灰掉。
        self.set_button_status(self._has_valid_data)
        self.mw.end_load_feedback("已取消导入")
        self.mw.load_btn.setEnabled(True)
        self._reload_in_progress = False
        self.mw.reload_btn.setEnabled(True)

    def _on_blf_cancelled(self, file_path: str, is_reload: bool = False,
                          load_version: int = -1) -> None:
        # 取消在进度框已关闭之后才落地（极小概率竞态）：同样走静默恢复
        if self._blf_cancelled_version != load_version:
            self._on_blf_cancel(file_path, load_version)

    def _restore_curves_after_abort(self) -> None:
        """取消 / 0 信号回环后把已绘曲线画回来（§5.1 G「恢复原数据」）。"""
        loader = getattr(self.mw, "loader", None)
        if loader is None or not getattr(loader, "var_names", None):
            return  # 还没有任何数据源，画布本就是空的
        try:
            self.mw.cursor_sync_manager.replots_after_loading(skip_pin_reset=True)
        except Exception:
            logger.debug("取消 BLF 导入后恢复原曲线失败", exc_info=True)

    def _discard_loader(self, loader) -> None:
        if loader is None:
            return
        try:
            if hasattr(loader, "close"):
                loader.close()
        except Exception:
            logger.debug("丢弃 loader 时释放资源失败", exc_info=True)

    def _on_blf_done(self, new_loader, file_path: str, load_version: int = -1,
                     is_reload: bool = False):
        if self._blf_cancelled_version == load_version:
            # 取消与完成撞在一起：以取消为准，丢掉扫好的 loader。
            # 此时进度框已由 _on_blf_cancel 收掉，这里不能再碰它 —— 它可能已经
            # 属于新一轮导入了（迟到 done 关掉新框会让用户失去取消入口）。
            self._discard_loader(new_loader)
            return
        if load_version >= 0 and load_version != self.mw._data_version:
            logger.warning(
                "数据版本不匹配，丢弃过期 BLF 加载结果 (expected=%d, got=%d)",
                self.mw._data_version, load_version,
            )
            self._discard_loader(new_loader)
            return
        # 代次/取消检查之后再收线：迟到的 done 不允许关掉新一轮的进度框。
        self._close_blf_progress()

        if not new_loader.var_names:
            # F 路径：库与报文完全不通 → warning 后重开对话框，不进空数据态
            db_paths = list(getattr(new_loader, "db_paths", None) or [])
            # 扫好的报文桶**不丢**：桶只按 (通道, CAN ID) 分，与数据库无关，下一轮
            # 换库靠 bind_databases() 毫秒级重绑即可（与 §5.6「CAN 数据库…」按钮
            # 同一条承诺）。此前在这里 close() 掉了整轮扫描成果，于是每一轮重试
            # 都把整个文件重扫一遍（实测 655 万帧 9.16 s/轮 vs 重绑 3.7 ms）。
            self._blf_pending_loader = new_loader
            # 与 G 路径同理：这轮导入根本没成立，没理由把用户原有的画布留空
            self._restore_curves_after_abort()
            self._end_data_reload()
            self.set_button_status(self._has_valid_data)
            self.mw.end_load_feedback()
            self.mw.load_btn.setEnabled(True)
            self._reload_in_progress = False
            self.mw.reload_btn.setEnabled(True)
            QMessageBox.warning(self.mw, "未生成任何信号", BLF_TEXT_NO_SIGNAL)
            self._blf_force_dialog = True
            self._blf_preset_db_paths = db_paths
            self._schedule_delayed(0, lambda: self._restart_blf_import(file_path))
            return

        for err in getattr(new_loader, "load_errors", None) or []:
            logger.warning("CAN 数据库: %s", err)
        if getattr(new_loader, "truncated", False):
            if getattr(new_loader, "truncated_reason", None) == "payload":
                QMessageBox.warning(
                    self.mw, "报文数据超上限", BLF_TEXT_TRUNCATED_PAYLOAD
                )
            else:
                QMessageBox.warning(self.mw, "报文数超上限", BLF_TEXT_TRUNCATED)

        self._swap_loader(new_loader, is_reload=is_reload)
        self._end_data_reload()
        self.set_button_status(True)
        self.mw.load_btn.setEnabled(True)
        self._reload_in_progress = False
        self.mw.reload_btn.setEnabled(True)
        # 延迟到下一个事件循环，确保 paint 事件先处理，避免 UI 半成品白屏
        self._schedule_delayed(
            0, lambda: self._post_load_actions(file_path, is_reload=is_reload)
        )

    def _on_blf_error(self, msg: str, is_reload: bool = False,
                      load_version: int = -1) -> None:
        if load_version >= 0 and load_version != self._blf_load_version:
            # 旧一轮线程迟到的错误：新一轮已经开始了，不能收它的进度框、
            # 也不能替它弹「读取失败」或提前解锁 UI（否则用户可再发一轮导入，
            # 与新线程并发扫描）。与 _on_blf_cancel 的代次闸同口径。
            logger.debug(
                "忽略过期 BLF 错误信号 (got=%d, current=%d)",
                load_version, self._blf_load_version,
            )
            return
        if self._blf_cancelled_version == load_version:
            # 已走静默恢复的取消路径，不能再弹一个错误框打扰用户
            self._close_blf_progress()
            return
        logger.error("BLF 后台加载失败: %s", msg)
        self._close_blf_progress()
        self.mw.end_load_feedback(f"加载失败：{msg}", level="error")
        QMessageBox.critical(self.mw, "读取失败", msg)
        # 与本轮取消同口径：_begin_data_reload 已把曲线摘掉，而失败路径不释放
        # 旧 loader（host.loader 仍是导入前那份），必须显式重绘回来（评审 A6）。
        self._restore_curves_after_abort()
        self._end_data_reload()
        self.mw.load_btn.setEnabled(True)
        self.set_button_status(self._has_valid_data)
        self._reload_in_progress = False
        self.mw.reload_btn.setEnabled(True)

    def _restart_blf_import(self, file_path: str, attempt: int = 0) -> None:
        """F 路径重开导入：等上一轮 reload 生命周期完全收尾后再走。

        这是一条纯 ``QTimer`` 延迟回调链（``_schedule_delayed``，窗口销毁即自动
        取消），而 ``mw`` 是弱引用 —— 重试期间窗口可能被销毁，所以任何一个退出口
        都必须把两个临时标志擦干净：否则下一次任意 BLF 导入会被
        ``_blf_force_dialog`` 残留值强制弹窗（表现是「明明记住了数据库却还弹」）。

        存活判据必须是 ``_live_mw()`` 而不是 ``getattr(self.mw, ...)``：后者读的是
        Python 包装器上的属性，C++ 侧已经析构时照样返回旧值，守卫会放行到
        ``load_csv_file`` 里再去碰按钮，抛一条无人认领的 RuntimeError。
        """
        mw = self._live_mw()
        if mw is None:  # MainWindow 的 Python 引用或 C++ 对象已经没了
            self._give_up_blf_restart()
            logger.info("BLF 重开导入已中止：主窗口正在销毁")
            return
        if not bool(getattr(mw, "_is_loading_new_data", False)):
            self.load_csv_file(file_path)
            return
        if attempt < 50:
            self._schedule_delayed(
                20, lambda: self._restart_blf_import(file_path, attempt + 1)
            )
            return
        self._give_up_blf_restart()
        logger.info("BLF 重开导入被放弃：上一轮加载 1s 后仍未收尾")

    def _give_up_blf_restart(self) -> None:
        """放弃 F 路径重开时必须把两个标志复位（它们只对这一轮导入有意义）。"""
        self._blf_force_dialog = False
        self._blf_preset_db_paths = None
        self._release_pending_blf_loader()

    def _release_pending_blf_loader(self) -> None:
        """丢弃 F 路径暂存的「已扫完但 0 信号」loader（它可能占几百 MB）。"""
        pending = self._blf_pending_loader
        self._blf_pending_loader = None
        if pending is not None and pending is not getattr(self.mw, "loader", None):
            self._discard_loader(pending)

    def _update_can_db_ui(self) -> None:
        """按当前数据源刷新状态栏 CAN 段与抽屉 banner 三态（§6.3）。

        段是常驻信息（不点也看得见三态），抽屉是细节与操作；两者由这一处
        同源驱动 —— 段文案与 banner 两行文案必须出自同一组数字，否则会出现
        「段上写 3 库、抽屉里列 4 行」。段放不放得下交给宽度阈值（收纳在
        ``MainWindow._apply_can_segment_fit``），这里只管写文案。
        """
        banner = getattr(self.mw, "can_db_banner", None)
        loader = getattr(self.mw, "loader", None)
        if banner is None:
            return
        if not self._is_blf_source():
            banner.hide_banner()
            self.mw._update_can_segment("")
            return

        database = loader.database
        db_names = [os.path.basename(p) for p in database.source_files]
        if getattr(loader, "truncated", False):
            reason = getattr(loader, "truncated_reason", None) or "messages"
            if reason == "payload":
                limit = BLF_MAX_PAYLOAD_BYTES
                seg_text = (
                    f"CAN：仅前 {loader.total_messages:,}"
                    f"（上限 {BLF_MAX_PAYLOAD_BYTES // (1024 * 1024)} MB）"
                )
            else:
                limit = BLF_MAX_MESSAGES
                seg_text = (
                    f"CAN：仅前 {loader.total_messages:,}（上限 {BLF_MAX_MESSAGES:,}）"
                )
            banner.show_truncated(
                limit=limit, scanned=loader.total_messages,
                db_count=len(db_names), db_names=db_names, reason=reason,
            )
        else:
            unmatched = len(getattr(loader, "unmatched_ids", None) or [])
            if unmatched:
                banner.show_unmatched(
                    unmatched_count=unmatched, db_count=len(db_names),
                    frame_count=database.frame_count,
                    signal_count=database.signal_count, db_names=db_names,
                )
                seg_text = f"CAN：{len(db_names)} 库 · {unmatched} 个 ID 未匹配"
            else:
                banner.show_normal(
                    db_count=len(db_names), frame_count=database.frame_count,
                    signal_count=database.signal_count, db_names=db_names,
                )
                seg_text = f"CAN：{len(db_names)} 库 · {database.signal_count} 信号"
        # tooltip 给 banner 两行全文（含库名清单），段被宽度收纳时仍有出口
        self.mw._update_can_segment(seg_text, tip=banner.info_label.text())

    def reselect_can_database(self) -> None:
        """§5.6：重选 DBC / ARXML。原始 payload 与数据库解耦，**不重扫报文**，
        但 ``bind_databases`` 仍会在本线程同步解析数据库（大 ARXML 可达数秒），
        因此解析期间挂等待光标；真正的线程化留作后续（评审 M3）。"""
        loader = getattr(self.mw, "loader", None)
        if not self._is_blf_source():
            return
        paths = self._ask_can_databases(
            loader.path, initial=list(getattr(loader, "db_paths", None) or [])
        )
        if not paths:
            return
        # 解析大库时主线程确实会卡住（Qt 也来不及重绘）；至少让光标先变成
        # 等待态，用户能区分「在处理」与「卡死」。finally 保证异常/提前返回
        # 都不会把整个应用留在忙碌光标上；错误框放到光标复原之后再弹。
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            stats = loader.bind_databases(paths, rollback_if_empty=True)
            bind_error = None
        except Exception as e:  # noqa: BLE001 - 对话框要把任意解析异常都报出来
            bind_error = e
        finally:
            QApplication.restoreOverrideCursor()
        if bind_error is not None:
            logger.error("重选 CAN 数据库失败: %s", bind_error, exc_info=bind_error)
            QMessageBox.critical(
                self.mw, "重选失败",
                f"重新绑定 CAN 数据库时出错：\n{bind_error}",
            )
            return
        if stats.get("rolled_back"):
            # 新库与当前报文完全对不上：loader 已自动恢复旧库/旧元数据。
            # 此时绝不能继续清缓存、清曲线、_apply_loader —— 否则一次误选就把
            # 用户已经画好的整张画布换成空变量集，且无法一键撤销。
            logger.info("重选 CAN 数据库 0 命中，已回滚到原数据库")
            QMessageBox.warning(
                self.mw, "未生成任何信号",
                BLF_TEXT_NO_SIGNAL + "\n已保留原数据库与已绘制曲线。",
            )
            return
        if not stats.get("var_count"):
            QMessageBox.warning(self.mw, "未生成任何信号", BLF_TEXT_NO_SIGNAL)
        for err in loader.load_errors:
            logger.warning("CAN 数据库: %s", err)
        # 换库后同名变量可能指向完全不同的信号，统计缓存必须整体作废。
        self.mw.var_stats_cache = {}
        # 已画出的曲线引用的也是作废名字，一律清除（沿用 _release_old_data 纪律）。
        try:
            self.mw.cursor_sync_manager.clear_all_plots()
        except Exception:
            logger.debug("重选数据库后清除曲线失败", exc_info=True)
        # 剩下的「变量集变了」下游收敛全部交回 _apply_loader：它是这条链路上
        # 的唯一入口，含 per-widget units、X 轴量程、数值表重建、绘图变量编辑器
        # 搜索候选、变量信息窗刷新、过滤器重放、banner 三态。手写一份精简版
        # 会漏掉后半：开着的数值表继续按旧变量集建模、变量信息窗拿旧名字去新
        # 库要样本，统计任务直接返回错误文本。不调 _swap_loader 是因为它
        # 会 _release_old_data —— 正在用的 loader 不能释放。
        self._apply_loader(is_reload=True)
        logger.info(
            "CAN 数据库已重绑: %d 库 / %d 变量 / %d 未匹配 ID（未重扫报文）",
            len(paths), stats.get("var_count", 0), stats.get("unmatched", 0),
        )

    def show_can_unmatched(self) -> None:
        """§5.5：banner 状态③「查看」入口。"""
        loader = getattr(self.mw, "loader", None)
        if not self._is_blf_source():
            return
        unmatched = list(getattr(loader, "unmatched_ids", None) or [])
        if not unmatched:
            return
        from src.ui.dialogs.can_unmatched_dialog import CanUnmatchedDialog

        dialog = CanUnmatchedDialog(
            unmatched, total_messages=loader.total_messages, parent=self.mw
        )
        dialog.exec()
        if dialog.reselect_requested:
            self.reselect_can_database()

    def _release_old_data(self):
        """显式释放所有对旧 DataFrame 的引用（仅在新 loader 成功后调用）。"""

        try:
            # 0) 统计缓存必须最先、无条件清空（缺-2/缺-3 加固）。
            # 变量信息窗口的统计缓存键为变量名，同名变量在新数据中的数值
            # 可能完全不同，必须整体作废。原实现放在步骤 1/2 之间，若上方
            # 遍历 plot widgets 抛异常会被 except 吞掉而跳过清空；而
            # _begin_data_reload 在 _is_loading_new_data 残留时不递增
            # _data_version（generation 令牌随之失效），两道保险将同时
            # 失效。作为 try 内首条语句执行：后续任何异常都不影响清空，
            # 且自身抛 AttributeError 时仍受下方 except 保护。
            self.mw.var_stats_cache = {}

            # 1) 清理 plot widgets 的 data 引用（所有页：旧 DataFrame 必须真正释放）
            for container in self._iter_all_containers():
                widget = getattr(container, "plot_widget", None)
                if widget is not None:
                    widget.data = None

            # 2) 清理 main window 的重复引用
            self.mw.data = None
            self.mw.var_names = []
            self.mw.units = {}
            self.mw.data_validity = {}

            # 3) 清理 table dialog 的独立 _df
            from src.ui.table_dialog import DataTableDialog
            if DataTableDialog._instance is not None:
                try:
                    DataTableDialog._instance.clear_all_columns()
                except Exception:
                    logger.debug("清理 DataTableDialog 数据失败", exc_info=True)

            # 3.5) 取消变量信息窗口的后台统计任务
            # 必须在步骤 4 释放 loader **之前**：worker 正在运行的那条任务已把
            # loader 取到局部变量，若不取消会继续读一个即将 close() 的文件句柄。
            # 队列条目持 weakref，不会阻止旧 loader 被 GC。
            from src.ui.dialogs.variable_info_dialog import VariableInfoDialog
            try:
                VariableInfoDialog.on_loader_released()
            except Exception:
                logger.debug("取消变量信息统计任务失败", exc_info=True)

            # 4) 释放旧 loader
            if self._has_valid_loader:
                old_loader = self.mw.loader
                if hasattr(old_loader, "close"):
                    try:
                        old_loader.close()
                    except Exception:
                        logger.debug("关闭旧 loader 时发生异常", exc_info=True)
                if hasattr(old_loader, "release_memory"):
                    try:
                        old_loader.release_memory()
                    except Exception:
                        logger.debug("释放旧 loader 内存时发生异常", exc_info=True)
                self.mw.loader = None

            # 4.5) F 路径暂存的「扫完但 0 信号」loader 同样属于旧数据
            pending = self._blf_pending_loader
            if pending is not None:
                self._blf_pending_loader = None
                if hasattr(pending, "close"):
                    try:
                        pending.close()
                    except Exception:
                        logger.debug("关闭暂存 BLF loader 失败", exc_info=True)

            # 5) 触发 GC — 把 numpy 数组真正归还给操作系统
            gc.collect()

        except (AttributeError, TypeError):
            logger.debug("清理旧数据时属性/类型错误（对象可能已销毁）")
        except Exception:
            logger.warning("清理旧数据时发生异常", exc_info=True)
        finally:
            # 无条件清空 value_cache 和 _enum_text_maps：即使上方清理步骤异常，
            # 这两处缓存也不能残留旧数据引用，否则 reload 后 cursor 会显示
            # 陈旧的枚举文本标签或数值。
            try:
                self.mw.value_cache = OrderedDict()
                self.mw._enum_text_maps = {}
            except Exception:
                logger.debug("清空 value_cache/_enum_text_maps 失败", exc_info=True)

    def _refresh_table_dialog(self, loader):
        """变量数值表窗口开着时就地重建内容，并按重建结果决定 show / 关闭。

        单独成方法有两个理由：
        1. update_data 在表空时会自行 close()，而 closeEvent 经类名把
           DataTableDialog._instance 置 None —— 必须取本地引用并先比对
           身份，继续解引用类属性会抛 AttributeError 并中断加载后续步骤；
        2. tab 模式下 _df 恒空，必须用 has_table_content 统一判据，
           否则重建好的 tab 窗口会被误判为空而直接关闭。
        """
        from src.ui.table_dialog import DataTableDialog

        dlg = DataTableDialog._instance
        if dlg is None:
            return
        dlg.update_data(loader)
        if DataTableDialog._instance is not dlg:
            logger.debug("变量数值表重建后无内容，已自行关闭")
        elif dlg.has_table_content():
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        else:
            dlg.set_skip_close_confirmation(True)
            dlg.close()

    def _post_load_actions(self, file_path: str, is_reload: bool = False):
        if self._window_going_away():
            return
        self.mw.loaded_path = file_path
        self._remember_last_open_dir(file_path)

        elapsed_s = max(
            0.0, time.perf_counter() - getattr(self, "_load_started_at", time.perf_counter())
        )
        self.mw.end_load_feedback()
        self.mw.update_file_status(
            file_path,
            row_count=self._current_data_length,
            channel_count=len(self.mw.var_names or []),
            elapsed_s=elapsed_s,
        )
        self.set_button_status(True)
        
        # 自动恢复：replots_after_loading 优先，auto-save 仅作降级兜底
        if hasattr(self.mw, 'plot_config_manager'):
            any_curve_restored = any(
                not container.isHidden() and bool(container.plot_widget.curves)
                for container in self._iter_all_containers()
            )

            if any_curve_restored:
                logger.info("replots_after_loading 已恢复曲线，跳过自动恢复")
            else:
                current_vars = []
                if hasattr(self.mw, 'data') and self.mw.data is not None:
                    current_vars = list(self.mw.data.columns)
                elif hasattr(self.mw, 'loader') and hasattr(self.mw.loader, 'var_names'):
                    current_vars = list(self.mw.loader.var_names)

                should_apply, reason = self.mw.plot_config_manager.auto_save_manager.should_apply_auto_save(current_vars)
                if should_apply:
                    saved_config = (
                        self.mw.plot_config_manager.auto_save_manager.load_auto_save()
                    )
                    success = self.mw.plot_config_manager.apply_config(
                        self.mw, saved_config
                    )
                    if success:
                        logger.info("自动恢复配置成功: %s", reason)
                        self._announce_auto_restore(saved_config, current_vars)
                    else:
                        logger.warning("自动恢复配置失败")
                else:
                    logger.info("不应用自动保存: %s", reason)

    def _announce_auto_restore(self, config, current_vars) -> None:
        """自动恢复也是套用了一套通道映射，文案交给 ``mw._announce_applied``。

        比率在这里现算而不是取 ``should_apply_auto_save`` 的 reason：那个串是
        写给日志看的英文，而 ``check_template_match`` 的分母与它的放行判据同源
        （都是配置里的变量数），屏上这个百分比就是刚才放行用的那一个。配置文件
        上面已经读过，这里不再读盘。
        """
        ratio, _matched, unmatched = (
            self.mw.plot_config_manager.check_template_match(config, current_vars)
        )
        self.mw._announce_applied("已自动恢复上次布局", ratio, unmatched)

    def _remember_last_open_dir(self, file_path: str):
        directory = os.path.dirname(file_path)
        if directory and os.path.isdir(directory):
            self.mw._last_open_dir = directory

    def _get_dialog_initial_directory(self) -> str:
        if getattr(self.mw, "_last_open_dir", None) and os.path.isdir(
            self.mw._last_open_dir
        ):
            return self.mw._last_open_dir
        return self._default_system_directory()

    def _default_system_directory(self) -> str:
        """返回文件对话框的默认目录。

        注意：Windows 上不能使用 CLSID（如 "::{20D04FE0-...}"）作为
        QFileDialog 的 initial_dir，Qt 内部 QUrl::fromLocalFile() 会将其
        解析为畸形 URL "file::/"，导致 SHCreateItemFromParsingName 失败。
        统一使用 QStandardPaths 获取有效路径。
        """
        def _safe_location(location):
            try:
                return QStandardPaths.writableLocation(location)
            except AttributeError:
                return ""

        candidates: list[str | None] = [
            _safe_location(QStandardPaths.StandardLocation.HomeLocation),
            _safe_location(QStandardPaths.StandardLocation.DesktopLocation),
        ]
        # Windows 回退到系统盘根目录，Unix 回退到 /
        if sys.platform.startswith("win"):
            candidates.append(os.environ.get("SystemDrive", "C:") + os.sep)
        else:
            candidates.append(os.path.sep)

        for path in candidates:
            if path:
                return path
        return ""

    def _extract_file_extension(self, file_path: str) -> str | None:
        base_ext = os.path.splitext(file_path)[1].lower()
        if base_ext in SUPPORTED_EXTENSIONS:
            return base_ext

        base_name = os.path.basename(file_path).lower()
        # BLF 不分卷：把 .blf 排除出分卷后缀，避免 xxx.blf.2 这类不存在的
        # 文件被认成 .blf 送进分钟级扫描（分卷只适用于文本型记录）
        volume_exts = [ext for ext in SUPPORTED_EXTENSIONS if ext != ".blf"]
        pattern = (
            r"("
            + "|".join(re.escape(ext) for ext in volume_exts)
            + r")\.\d+$"
        )
        match = re.search(pattern, base_name)

        if match:
            return match.group(1)

        return None

    def _validate_load_parameters(
        self, file_path: str, desc_rows, sep, has_unit
    ) -> tuple[bool, str]:
        if not isinstance(file_path, str) or not file_path.strip():
            return False, "文件路径无效"
        if desc_rows is not None and (not isinstance(desc_rows, int) or desc_rows < 0):
            return False, "描述行数必须是非负整数"
        if sep is not None and (not isinstance(sep, str) or not sep):
            return False, "分隔符无效"
        if has_unit is not None and not isinstance(has_unit, bool):
            return False, "has_unit参数必须是布尔值"
        return True, ""

    def _load_sync(
        self,
        file_path: str,
        desc_rows: int | None = 0,
        sep: str = ",",
        has_unit: bool | None = True,
        encoding: str | None = None,
        sheet_name: str | None = None,
        is_excel: bool = False,
        is_reload: bool = False,
        db_paths: list[str] | None = None,
    ):
        if not is_excel:
            is_valid, error_msg = self._validate_load_parameters(
                file_path, desc_rows, sep, has_unit
            )
            if not is_valid:
                QMessageBox.critical(self.mw, "参数错误", error_msg)
                return False

        new_loader = None
        status = False

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in (".mf4", ".mdf", ".dat"):
                new_loader = MDFLazyLoader(file_path)
            elif ext == ".blf":
                # 正常导入路径下 BLF 一律走后台线程（§5.4），这里保留同步
                # 入口只为给单测 / 脚本一个能直接拿到 loader 的等价通道。
                from src.data.blf_loader import BLFLazyLoader

                if not db_paths:
                    QMessageBox.warning(
                        self.mw, "未选择数据库", BLF_TEXT_NO_DB_SELECTED
                    )
                    return False
                new_loader = BLFLazyLoader(file_path, db_paths=db_paths)
            elif ext in (".xlsx", ".xlsm") or is_excel:
                from src.data.excel_loader import ExcelDataLoader
                new_loader = ExcelDataLoader(
                    file_path,
                    sheet_name=sheet_name or 0,
                    desc_rows=desc_rows,
                    has_unit=has_unit,
                )
            else:
                new_loader = FastDataLoader(
                    file_path,
                    desc_rows=desc_rows,
                    sep=sep,
                    has_unit=has_unit,
                    encoding=encoding,
                )
            # 新 loader 成功 → 释放旧数据 → 应用新数据
            self._swap_loader(new_loader, is_reload=is_reload)
            status = True
            logger.info("文件加载完成: %s (%d 行)", file_path, new_loader.datalength)
        except MemoryError as e:
            QMessageBox.critical(self.mw, "内存不足", f"加载文件时内存不足: {str(e)}")
            logger.error("内存不足: %s", e)
            status = False
        except FileNotFoundError as e:
            QMessageBox.critical(self.mw, "文件未找到", f"无法找到文件: {str(e)}")
            logger.error("文件未找到: %s", e)
            status = False
        except PermissionError as e:
            QMessageBox.critical(self.mw, "权限错误", f"没有文件访问权限: {str(e)}")
            logger.error("权限错误: %s", e)
            status = False
        except Exception as e:
            QMessageBox.critical(self.mw, "读取失败", f"加载文件时发生错误: {str(e)}")
            logger.error("加载文件失败: %s", e, exc_info=True)
            status = False
        return status

    def _on_load_done(self, new_loader, file_path: str, load_version: int = -1, is_reload: bool = False):
        if load_version >= 0 and load_version != self.mw._data_version:
            logger.warning(
                "数据版本不匹配，丢弃过期加载结果 (expected=%d, got=%d)",
                self.mw._data_version, load_version,
            )
            if hasattr(new_loader, "release_memory"):
                new_loader.release_memory()
            return
        logger.info("后台加载完成: %s", file_path)
        self.mw._progress.close()

        # P0-3: _swap_loader → _apply_loader → replots_after_loading 链上任一异常
        # 都不得跳过解锁，否则 _is_loading_new_data / _reload_in_progress 永久卡死
        # （_end_data_reload 未执行 → safety timer 也不会启动，只能重启应用）。
        # 与同步路径（_load_file 的 finally: _end_data_reload()）保持一致。
        applied = True
        try:
            # —— 释放旧数据 → 应用新 loader
            self._swap_loader(new_loader, is_reload=is_reload)
        except Exception as e:
            applied = False
            logger.error("应用后台加载结果失败: %s", e, exc_info=True)
            QMessageBox.critical(self.mw, "读取失败", f"应用数据时发生错误: {str(e)}")
        finally:
            self._end_data_reload()
            if applied:
                self.set_button_status(True)
            else:
                # 应用失败时按「当前是否还有有效数据」恢复，并收掉忙等光标/秒表；
                # 无条件 set_button_status(True) 会在没有数据时点亮必然再报错的入口
                self._finish_load_failure(is_reload=is_reload)
            self.mw.load_btn.setEnabled(True)
            # v5.12: 异步 reload 完成，释放互斥锁
            self._reload_in_progress = False
            self.mw.reload_btn.setEnabled(True)
        if not applied:
            return
        # 延迟到下一个事件循环，确保 paint 事件先处理，避免 UI 半成品白屏
        self._schedule_delayed(
            0, lambda: self._post_load_actions(file_path, is_reload=is_reload)
        )

    def _on_load_error(self, msg):
        logger.error("后台加载失败: %s", msg)
        self.mw._progress.close()
        self.mw.end_load_feedback(f"加载失败：{msg}", level="error")
        QMessageBox.critical(self.mw, "读取失败", msg)
        self._end_data_reload()
        self.mw.load_btn.setEnabled(True)
        # 加载期被闸门禁用的按钮要还原，否则失败一次就永久灰掉
        self.set_button_status(self._has_valid_data)
        # v5.12: 异步 reload 失败，也要释放互斥锁
        self._reload_in_progress = False
        self.mw.reload_btn.setEnabled(True)

    def _swap_loader(self, new_loader, is_reload: bool = False):
        """释放旧数据并应用新 loader（同步/异步路径共用）"""
        self._release_old_data()
        self.mw.loader = new_loader
        self._apply_loader(is_reload=is_reload)

    def _apply_loader(self, is_reload: bool = False):
        self.mw.var_names = self.mw.loader.var_names
        self.mw.units = self.mw.loader.units
        self.mw.time_channels_infos = self.mw.loader.time_channels_info
        self.mw.data_validity = self.mw.loader.df_validity
        self.mw.data = self.mw.loader.df

        self.mw.list_widget.populate(
            self.mw.var_names, self.mw.units, self.mw.data_validity
        )

        if self.mw.placeholder_label.parent():
            self.mw.placeholder_label.setParent(None)

        if not self.mw.plot_widgets:
            self.mw.layout_manager.create_subplots_matrix(
                self.mw._plot_row_max_default, self.mw._plot_col_max_default
            )
            self.mw.layout_manager.set_plots_visible(
                self.mw._plot_row_current, self.mw._plot_col_current
            )

        for container in self._iter_all_containers():
            widget = container.plot_widget
            widget.data = self.mw.loader.df
            widget.units = self.mw.loader.units
            widget.time_channels_info = self.mw.loader.time_channels_info
            widget.time_column_name = self.mw.loader.time_column_name
            widget.time_axis_label = self.mw.loader.time_axis_label
            widget.update_x_axis_label()

        self.mw.cursor_sync_manager._compute_baseline_density()
        self.mw.cursor_sync_manager._sync_min_xrange()
        # BLF：banner 与「CAN 数据库…」按钮跟数据源走（非 BLF 时自动隐去）
        self._update_can_db_ui()

        # v5.11: reload 场景下跳过 pin 状态重置，保留 _restore_cursor_state_after_reload 恢复的 cursor 状态
        self.mw.cursor_sync_manager.replots_after_loading(skip_pin_reset=is_reload)

        self._refresh_table_dialog(self.mw.loader)

        # 就地更新打开的绘图变量编辑器（搜索栏数据源）：
        # 编辑器为非模态多实例（不能用单例模式），逐个通知；
        # 单个失效实例（plot_widget 已销毁）在 refresh_data_source 内静默跳过，
        # 不阻断其他实例的更新。
        from src.ui.plot_variable_editor import PlotVariableEditorDialog

        for editor in self.mw.findChildren(PlotVariableEditorDialog):
            try:
                editor.refresh_data_source()
            except Exception:
                logger.debug("更新绘图变量编辑器数据源失败", exc_info=True)

        # 就地刷新变量信息窗口（单例）：用新 loader 重建所有页面的元数据快照
        # 并重新提交统计。新数据中不存在的变量保留标签页但标注失效。
        # 窗口未打开过时 _instance 为 None，函数内部直接返回，无额外开销。
        from src.ui.dialogs.variable_info_dialog import VariableInfoDialog

        try:
            VariableInfoDialog.refresh_after_reload(self.mw.loader)
        except Exception:
            logger.debug("刷新变量信息窗口失败", exc_info=True)

        if self.mw.filter_input.text() or self.mw.unit_filter_input.text():
            self.mw.cursor_sync_manager.filter_variables()
        if self.mw.mark_region_btn.isChecked():
            self.mw.layout_manager.request_mark_stats_refresh(immediate=True)
