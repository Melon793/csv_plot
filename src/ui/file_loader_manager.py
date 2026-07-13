"""FileLoaderManager - 文件加载管理

负责 MainWindow 的文件加载功能：
- 文件对话框与路径验证
- 同步/异步加载路由
- 数据加载进度与状态管理
- 加载后 UI 同步
"""

from __future__ import annotations
import os
import sys
import time
import warnings

from PySide6.QtCore import Qt, QStandardPaths, QTimer
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox, QProgressDialog

from src.core.config import FILE_SIZE_LIMIT_BACKGROUND_LOADING, safe_qt_op
from src.core.data_types import AutoDetectError
from src.data.loader import DataLoadThread, FastDataLoader
from src.data.mdf_lazy_loader import MDFLazyLoader
from src.ui.main_window_base_manager import MainWindowBaseManager
from src.core.logger import get_logger

logger = get_logger("ui.file_loader")


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
                pass
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
                pass
            except Exception:
                logger.debug("添加 %s 时异常", attr_name, exc_info=True)

    def _begin_data_reload(self):
        if self.mw._is_loading_new_data:
            return
        self.mw._is_loading_new_data = True
        self.mw._data_version += 1

        # v5.8 修复：取消之前的 safety timer，防止跨 reload 的过期 timer 触发 _force_unlock_all。
        # 之前的实现仅靠版本号防污染，但 _safety_unlock_version 会被最新 _end_data_reload 覆盖，
        # 导致旧 timer 触发时版本检查通过，误执行 _force_unlock_all 清除新曲线。
        if hasattr(self, "_safety_timer") and self._safety_timer is not None:
            self._safety_timer.stop()
            self._safety_timer = None
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
                except (TypeError, RuntimeError):
                    pass
            if hasattr(widget, "vline2"):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        widget.vline2.sigPositionChanged.disconnect(
                            widget.on_vline_position_changed
                        )
                except (TypeError, RuntimeError):
                    pass
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
            if hasattr(widget, "_cursor_trash_bin"):
                widget._cursor_trash_bin.clear()
            if hasattr(widget, "_pending_delete_items"):
                for item in widget._pending_delete_items:
                    safe_qt_op(item.setVisible, False)
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

        QTimer.singleShot(0, self._restore_cursor_state_after_reload)

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

            if is_free_cursor:
                # 防抖：阻止 reload 过渡期内的中间 cursor label 更新，
                # 只在 _post_reload_ui_refresh 中渲染一次
                for container in getattr(self.mw, "plot_widgets", []):
                    widget = getattr(container, "plot_widget", None)
                    if widget and hasattr(widget, "_last_cursor_update_time"):
                        widget._last_cursor_update_time = time.time()
                return

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
                                from PySide6.QtCore import QSignalBlocker
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
                                from PySide6.QtCore import QSignalBlocker
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

            # v5.x: 恢复 vline/vline2 信号连接
            for widget in widgets_list:
                if hasattr(widget, "vline"):
                    try:
                        widget.vline.sigPositionChanged.connect(
                            widget.on_vline_position_changed
                        )
                    except (TypeError, RuntimeError):
                        pass
                if hasattr(widget, "vline2"):
                    try:
                        widget.vline2.sigPositionChanged.connect(
                            widget.on_vline_position_changed
                        )
                    except (TypeError, RuntimeError):
                        pass

        except Exception:
            logger.debug("恢复 pin 状态失败", exc_info=True)
        finally:
            # v5.0 修复：不在此处清除锁！
            # 锁的清除延迟到 _post_reload_ui_refresh() 中执行，
            # 在 setUpdatesEnabled(True) 和场景刷新完成后再清除，
            # 消除锁清除与 setUpdatesEnabled 之间约 50ms 的危险窗口。
            # 在此期间 paintEvent 由 _is_loading_new_data 和 _is_updating_data 双锁拦截。
            self._post_reload_pending_version = self.mw._data_version
            QTimer.singleShot(50, self._post_reload_ui_refresh)

    def _safety_force_unlock(self):
        """版本感知的安全解锁：仅当版本匹配时才执行解锁，防止跨 reload 污染"""
        current_version = getattr(self.mw, "_data_version", 0)
        safety_version = getattr(self, "_safety_unlock_version", -1)
        if safety_version != current_version:
            return
        self._force_unlock_all()

    def _force_unlock_all(self):
        """紧急解锁：确保所有退出路径都不会留下死锁"""
        logger.warning(
            "[v5.8] _force_unlock_all 触发：正在清除曲线 + 强制解锁 "
            "(data_version=%d, safety_unlock_version=%d)",
            getattr(self.mw, "_data_version", -1),
            getattr(self, "_safety_unlock_version", -1),
        )
        for container in getattr(self.mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if widget and hasattr(widget, "_safe_clear_plot_items"):
                try:
                    widget._safe_clear_plot_items()
                except Exception:
                    logger.debug("_safe_clear_plot_items 失败（紧急解锁期间）")
        # v5.12: 紧急解锁时也要释放 reload 互斥锁
        self._reload_in_progress = False
        self.mw.reload_btn.setEnabled(True)
        self.mw._is_loading_new_data = False
        self._safety_unlock_version = -1
        if hasattr(self, "_safety_timer") and self._safety_timer is not None:
            self._safety_timer.stop()
            self._safety_timer = None
        for container in getattr(self.mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if widget:
                # v5.11: 紧急解锁时也要把 vline 加回 scene，否则后续 paint 不会绘制光标
                self._reattach_vlines_to_scene(widget)
                widget._is_updating_data = False
                widget._cached_data_version = self.mw._data_version
                widget.setUpdatesEnabled(True)
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
        6. 单一 QTimer.singleShot(0)（延迟 cursor 更新到下一个事件循环迭代，
           合并 13 个 per-widget 回调为 1 个，避免 BSP 树交叉修改）
        """
        pending_version = getattr(self, "_post_reload_pending_version", -1)
        if pending_version != getattr(self.mw, "_data_version", 0):
            logger.debug(
                "_post_reload_ui_refresh 跳过：版本不匹配 pending=%d current=%d",
                pending_version, self.mw._data_version,
            )
            return
        try:
            widgets_to_refresh = []
            for container in getattr(self.mw, "plot_widgets", []):
                widget = getattr(container, "plot_widget", None)
                if not widget:
                    continue
                widget._is_updating_data = False
                widget._cached_data_version = self.mw._data_version
                widget.setUpdatesEnabled(True)
                if hasattr(widget, "_queue_ui_refresh"):
                    widgets_to_refresh.append(widget)

            self.mw._is_loading_new_data = False
            self._safety_unlock_version = -1
            if hasattr(self, "_safety_timer") and self._safety_timer is not None:
                self._safety_timer.stop()
                self._safety_timer = None

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

            # v5.3 修复问题 2：合并 13 个 per-widget QTimer.singleShot(0) 为 1 个统一回调
            current_version = self.mw._data_version
            QTimer.singleShot(
                0,
                lambda: self._deferred_cursor_refresh_all(widgets_to_refresh, current_version)
            )
        except Exception:
            logger.debug("_post_reload_ui_refresh 执行失败", exc_info=True)
            self.mw._is_loading_new_data = False
            self._safety_unlock_version = -1
            if hasattr(self, "_safety_timer") and self._safety_timer is not None:
                self._safety_timer.stop()
                self._safety_timer = None
            # v5.12: 异常路径也要恢复 reload 按钮
            self.mw.reload_btn.setEnabled(True)

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
            elif hasattr(widget, "curve") and widget.curve is not None:
                var_name = getattr(widget, "y_name", "") or "single"
                curves_to_check.append((var_name, widget.curve))
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
                elif (
                    hasattr(widget, "plot_variable")
                    and hasattr(widget, "y_name")
                    and widget.y_name == var_name
                ):
                    widget.plot_variable(var_name, show_duplicate_warning=False)
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

    def load_csv_file(self, file_path: str):
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
        except MemoryError:
            QMessageBox.critical(
                self.mw, "内存不足", "文件太大，内存不足。请尝试加载较小的文件。"
            )
            self._release_old_data()
            self.mw.load_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self.mw, "加载错误", f"加载文件时发生错误: {str(e)}")
            self._release_old_data()
            self.mw.load_btn.setEnabled(True)
        finally:
            self.mw.raise_()
            self.mw.activateWindow()

    def set_button_status(self, status: bool):
        if status is not None:
            self.mw.time_correction_btn.setEnabled(status)
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
        except Exception:
            # 同步路径异常或 _load_file 内部未启动线程就抛异常
            pass
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
        is_mdf_file = file_ext in (".mf4", ".mdf", ".dat")
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

        elif is_excel_file:
            # reload 时复用缓存的 sheet_name，避免重复弹出
            if is_reload and cached_sheet_name:
                sheet_name = cached_sheet_name
            else:
                from src.ui.dialogs.sheet_selector import SheetSelectorDialog
                dialog = SheetSelectorDialog(file_path, self.mw)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    self.mw.load_btn.setEnabled(True)
                    return
                sheet_name = dialog.get_selected_sheet()
                if not sheet_name:
                    self.mw.load_btn.setEnabled(True)
                    return

            # Excel 不需要分隔符/编码检测，desc_rows=None 触发自动检测
            delimiter_typ = ","
            desc_rows = None
            config_used = True

        if not is_mdf_file and not is_excel_file:
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
                return

        if not is_excel_file and (delimiter_typ is None or desc_rows is None or has_unit is None):
            QMessageBox.critical(
                self.mw,
                "数据解析失败",
                "无法确定文件的分隔符和标题行位置。\n" "请确认文件格式是否正确。",
            )
            return

        self._begin_data_reload()
        started_async = False
        _Threshold_Size_Mb = FILE_SIZE_LIMIT_BACKGROUND_LOADING

        file_size = os.path.getsize(file_path)
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
                    QTimer.singleShot(0, lambda: self._post_load_actions(file_path, is_reload=is_reload))
                else:
                    self.mw.load_btn.setEnabled(True)
                    # v5.12: 同步 reload 失败时也要恢复 reload 按钮
                    if is_reload and getattr(self, "_reload_in_progress", False):
                        self._reload_in_progress = False
                        self.mw.reload_btn.setEnabled(True)
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
                self.mw._thread.start()
                started_async = True
        except Exception:
            if not started_async:
                self._end_data_reload()
            raise

    def _release_old_data(self):
        """显式释放所有对旧 DataFrame 的引用（仅在新 loader 成功后调用）。"""
        import gc

        try:
            # 1) 清理 plot widgets 的 data 引用
            for container in getattr(self.mw, "plot_widgets", []):
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
                    logger.debug("清理 DataTableDialog 数据失败")

            # 4) 释放旧 loader
            if self._has_valid_loader:
                old_loader = self.mw.loader
                if hasattr(old_loader, "close"):
                    try:
                        old_loader.close()
                    except Exception:
                        logger.debug("关闭旧 loader 时发生异常")
                if hasattr(old_loader, "release_memory"):
                    try:
                        old_loader.release_memory()
                    except Exception:
                        logger.debug("释放旧 loader 内存时发生异常")
                self.mw.loader = None

            # 5) 触发 GC — 把 numpy 数组真正归还给操作系统
            gc.collect()

        except (AttributeError, TypeError):
            logger.debug("清理旧数据时属性/类型错误（对象可能已销毁）")
        except Exception:
            logger.warning("清理旧数据时发生异常", exc_info=True)

    def _post_load_actions(self, file_path: str, is_reload: bool = False):
        self.mw.loaded_path = file_path
        self._remember_last_open_dir(file_path)

        def truncate_string(file_path, max_length=79):
            filename_length = len(os.path.basename(file_path))
            if len(file_path) <= max_length:
                return file_path
            return "..." + file_path[min(-filename_length - 1, -(max_length - 3)) :]

        self.mw.setWindowTitle(
            f"{getattr(self.mw, 'defaultTitle', '')} ---- 数据文件: [{truncate_string(file_path)}]"
        )
        self.set_button_status(True)
        
        # 自动恢复：replots_after_loading 优先，auto-save 仅作降级兜底
        if hasattr(self.mw, 'plot_config_manager'):
            any_curve_restored = any(
                not container.isHidden() and (
                    container.plot_widget.y_name or
                    (container.plot_widget.is_multi_curve_mode and container.plot_widget.curves)
                )
                for container in self.mw.plot_widgets
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
                    success = self.mw.plot_config_manager.apply_config(
                        self.mw,
                        self.mw.plot_config_manager.auto_save_manager.load_auto_save()
                    )
                    if success:
                        logger.info(f"自动恢复配置成功: {reason}")
                    else:
                        logger.warning(f"自动恢复配置失败")
                else:
                    logger.info(f"不应用自动保存: {reason}")

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
        candidates: list[str | None] = []
        if sys.platform.startswith("win"):
            candidates.append("::{20D04FE0-3AEA-1069-A2D8-08002B30309D}")

        def _safe_location(location):
            try:
                return QStandardPaths.writableLocation(location)
            except AttributeError:
                return ""

        candidates.extend(
            [
                _safe_location(QStandardPaths.StandardLocation.HomeLocation),
                _safe_location(QStandardPaths.StandardLocation.DesktopLocation),
                os.path.sep,
            ]
        )
        for path in candidates:
            if path:
                return path
        return ""

    def _extract_file_extension(self, file_path: str) -> str:
        import re

        supported_extensions = [
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
            ".xlsx",
            ".xlsm",
        ]

        base_ext = os.path.splitext(file_path)[1].lower()
        if base_ext in supported_extensions:
            return base_ext

        base_name = os.path.basename(file_path).lower()
        pattern = (
            r"(" + "|".join(re.escape(ext) for ext in supported_extensions) + r")\.\d+$"
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

        # —— 释放旧数据 → 应用新 loader
        self._swap_loader(new_loader, is_reload=is_reload)

        self._end_data_reload()
        self.set_button_status(True)
        self.mw.load_btn.setEnabled(True)
        # v5.12: 异步 reload 完成，释放互斥锁
        self._reload_in_progress = False
        self.mw.reload_btn.setEnabled(True)
        # 延迟到下一个事件循环，确保 paint 事件先处理，避免 UI 半成品白屏
        QTimer.singleShot(0, lambda: self._post_load_actions(file_path, is_reload=is_reload))

    def _on_load_error(self, msg):
        logger.error("后台加载失败: %s", msg)
        self.mw._progress.close()
        QMessageBox.critical(self.mw, "读取失败", msg)
        self._end_data_reload()
        self.mw.load_btn.setEnabled(True)
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

        for container in self.mw.plot_widgets:
            widget = container.plot_widget
            widget.data = self.mw.loader.df
            widget.units = self.mw.loader.units
            widget.time_channels_info = self.mw.loader.time_channels_info
            widget.time_column_name = self.mw.loader.time_column_name
            widget.time_axis_label = self.mw.loader.time_axis_label
            widget.update_x_axis_label()

        self.mw.cursor_sync_manager._compute_baseline_density()
        self.mw.cursor_sync_manager._sync_min_xrange()

        # v5.11: reload 场景下跳过 pin 状态重置，保留 _restore_cursor_state_after_reload 恢复的 cursor 状态
        self.mw.cursor_sync_manager.replots_after_loading(skip_pin_reset=is_reload)

        from src.ui.table_dialog import DataTableDialog

        if DataTableDialog._instance is not None:
            DataTableDialog._instance.update_data(self.mw.loader)
            if not DataTableDialog._instance._df.empty:
                DataTableDialog._instance.show()
                DataTableDialog._instance.raise_()
                DataTableDialog._instance.activateWindow()
            else:
                DataTableDialog._instance.set_skip_close_confirmation(True)
                DataTableDialog._instance.close()

        if self.mw.filter_input.text() or self.mw.unit_filter_input.text():
            self.mw.cursor_sync_manager.filter_variables()
        if self.mw.mark_region_btn.isChecked():
            self.mw.layout_manager.request_mark_stats_refresh(immediate=True)
