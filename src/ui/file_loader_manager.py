"""
FileLoaderManager —— 文件加载管理类

封装 MainWindow 的所有文件加载逻辑，包括：
- 按钮点击处理
- 文件路径验证与大小检查
- 同步/异步文件加载
- 重载数据
- 加载状态管理
- 变量过滤
"""

from __future__ import annotations
import weakref
import os
import sys
import re

from PyQt6.QtCore import Qt, QTimer, QStandardPaths, QDir
from PyQt6.QtWidgets import (
    QFileDialog, QMessageBox, QProgressDialog,
)

from src.core.config import debug_log, FILE_SIZE_LIMIT_BACKGROUND_LOADING
from src.core.types import AutoDetectError
from src.data.loader import FastDataLoader, DataLoadThread


class FileLoaderManager:
    """文件加载管理器，负责所有数据文件加载相关的逻辑"""

    def __init__(self, main_window):
        self._mw_ref = weakref.ref(main_window)

    @property
    def _mw(self):
        mw = self._mw_ref()
        if mw is None:
            raise RuntimeError("MainWindow has been garbage collected")
        return mw

    def load_btn_click(self):
        if getattr(self._mw, "_is_loading_new_data", False):
            return

        self._mw.load_btn.setEnabled(False)

        try:
            initial_dir = self._get_dialog_initial_directory()
            file_filter = "all File (*.*);;CSV File (*.csv);;m File (*.mfile);;t00 File (*.t00);;t01 File (*.t01);;t10 File (*.t10);;t11 File (*.t11)"

            file_path, _ = QFileDialog.getOpenFileName(
                self._mw,
                "选择数据文件",
                initial_dir,
                file_filter
            )

            if file_path:
                self.load_csv_file(file_path)
            else:
                self._mw.load_btn.setEnabled(True)
        except Exception:
            self._mw.load_btn.setEnabled(True)
            raise

    def _validate_file_path(self, file_path: str) -> bool:
        if not file_path or not isinstance(file_path, str):
            QMessageBox.warning(self._mw, "文件错误", "请选择一个有效的文件")
            return False

        if not os.path.isfile(file_path):
            QMessageBox.warning(self._mw, "文件错误", "文件不存在")
            return False

        return True

    def _check_file_size(self, file_path: str) -> bool:
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                QMessageBox.warning(self._mw, "文件错误", "文件为空")
                return False

            if file_size > 1024 * 1024 * 1024:
                reply = QMessageBox.question(self._mw, "文件过大",
                    f"文件大小 {file_size/(1024*1024*1024):.1f}GB 较大，加载可能需要较长时间，是否继续？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                return reply == QMessageBox.StandardButton.Yes

            return True

        except OSError as e:
            QMessageBox.critical(self._mw, "文件访问错误", f"无法访问文件: {e}")
            return False

    def _begin_data_reload(self):
        if self._mw._is_loading_new_data:
            return
        self._mw._is_loading_new_data = True
        self._mw._data_version += 1

        if hasattr(self._mw, '_crosshair_update_timer'):
            self._mw._crosshair_update_timer.stop()
        self._mw._pending_crosshair_x = None

        pinned = [
            idx for idx, container in enumerate(getattr(self._mw, "plot_widgets", []), start=1)
            if getattr(container, "plot_widget", None)
            and getattr(container.plot_widget, "is_cursor_pinned", False)
        ]
        debug_log("MainWindow.begin_data_reload pinned_plots=%s version=%s", pinned, self._mw._data_version)
        try:
            self._mw.reset_all_pin_states()
        except Exception:
            pass
        for container in getattr(self._mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if not widget:
                continue
            widget._is_updating_data = True
            widget._cached_data_version = self._mw._data_version
            if hasattr(widget, "_cancel_ui_refresh"):
                widget._cancel_ui_refresh()
            if hasattr(widget, '_cursor_refresh_timer'):
                widget._cursor_refresh_timer.stop()
            if hasattr(widget, '_interaction_timer'):
                widget._interaction_timer.stop()

    def _end_data_reload(self):
        if not self._mw._is_loading_new_data:
            return

        for container in getattr(self._mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if not widget:
                continue
            widget._is_updating_data = False

        self._mw._is_loading_new_data = False
        debug_log("MainWindow.end_data_reload resume_ui version=%s", self._mw._data_version)

        QTimer.singleShot(50, self._post_reload_ui_refresh)

    def _post_reload_ui_refresh(self):
        if self._mw._is_loading_new_data:
            return
        for container in getattr(self._mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if widget and hasattr(widget, "_queue_ui_refresh"):
                widget._queue_ui_refresh(immediate=True)

    def load_csv_file(self, file_path: str):
        if getattr(self._mw, "_is_loading_new_data", False):
            debug_log("MainWindow.load_csv_file skipped - already loading")
            self._mw.load_btn.setEnabled(True)
            return

        debug_log("MainWindow.load_csv_file start path=%s is_loading=%s",
                  file_path, getattr(self._mw, "_is_loading_new_data", False))
        if not self._validate_file_path(file_path):
            self._mw.load_btn.setEnabled(True)
            return

        if not self._check_file_size(file_path):
            self._mw.load_btn.setEnabled(True)
            return

        try:
            self._load_file(file_path)
        except MemoryError:
            QMessageBox.critical(self._mw, "内存不足", "文件太大，内存不足。请尝试加载较小的文件。")
            self._cleanup_old_data()
            self._mw.load_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self._mw, "加载错误", f"加载文件时发生错误: {str(e)}")
            self._cleanup_old_data()
            self._mw.load_btn.setEnabled(True)
        finally:
            if self._has_valid_loader:
                self._post_load_actions(file_path)
                self._mw.raise_()
                self._mw.activateWindow()

    def reload_data(self):
        if getattr(self._mw, "_is_loading_new_data", False):
            return

        if not self._has_valid_loader:
            QMessageBox.critical(self._mw, "错误", "没有可重新加载的数据")
            return

        if not hasattr(self._mw.loader, 'path') or not self._mw.loader.path:
            QMessageBox.critical(self._mw, "错误", "数据路径无效")
            return

        if not os.path.isfile(self._mw.loader.path):
            QMessageBox.critical(self._mw, "错误", "文件不存在，无法重新加载")
            return

        self._load_file(self._mw.loader.path, is_reload=True)

    def _load_file(self, file_path: str, is_reload: bool = False):
        file_ext = self._extract_file_extension(file_path)

        is_mdf_file = file_ext in ('.mf4', '.mdf', '.dat')

        delimiter_typ = None
        descRows = None
        hasunit = None
        encoding = None
        config_used = False

        if is_mdf_file:
            delimiter_typ = ','
            descRows = 0
            hasunit = False
            config_used = True

        if not is_mdf_file and os.path.isfile("config_dict.json"):
            try:
                config_dict = self.load_dict("config_dict.json")
                ext_dict = config_dict.get(file_ext[1:], {})
                cfg_sep = ext_dict.get('sep')
                cfg_skip = ext_dict.get('skiprows')
                cfg_hasunit = ext_dict.get('hasunit')
                if cfg_sep is not None and cfg_skip is not None and cfg_hasunit is not None:
                    delimiter_typ = cfg_sep
                    descRows = int(cfg_skip)
                    hasunit = bool(cfg_hasunit)
                    config_used = True
                    debug_log("MainWindow._load_file using config_dict.json sep=%s descRows=%s hasunit=%s",
                              delimiter_typ, descRows, hasunit)
            except Exception as e:
                QMessageBox.warning(
                    self._mw, "配置文件错误",
                    f"config_dict.json 读取失败，将使用自动检测方式加载文件。\n\n错误详情: {e}"
                )

        if not config_used:
            try:
                fmt = FastDataLoader.auto_detect(file_path)
                delimiter_typ = fmt.sep
                descRows = fmt.header_row
                hasunit = fmt.hasunit
                encoding = fmt.encoding
                debug_log("MainWindow._load_file auto-detected encoding=%s sep=%s descRows=%s hasunit=%s",
                          encoding, delimiter_typ, descRows, hasunit)
            except AutoDetectError as e:
                debug_log("MainWindow._load_file auto-detection failed: %s", e)
                QMessageBox.critical(
                    self._mw, "数据解析失败",
                    "无法自动识别文件的标题行和分隔符。\n"
                    "请确认文件格式是否正确。\n"
                    "支持的分隔符：逗号(,)、分号(;)、制表符(Tab)"
                )
                return

        if delimiter_typ is None or descRows is None or hasunit is None:
            QMessageBox.critical(
                self._mw, "数据解析失败",
                "无法确定文件的分隔符和标题行位置。\n"
                "请确认文件格式是否正确。"
            )
            return

        self._begin_data_reload()
        started_async = False
        _Threshold_Size_Mb = FILE_SIZE_LIMIT_BACKGROUND_LOADING

        file_size = os.path.getsize(file_path)
        debug_log("MainWindow._load_file start path=%s size=%.2fMB reload=%s",
                  file_path, file_size/1024/1024, is_reload)
        try:
            if file_size < _Threshold_Size_Mb * 1024 * 1024:
                try:
                    status = self._load_sync(file_path, descRows=descRows, sep=delimiter_typ,
                                             hasunit=hasunit, encoding=encoding)
                finally:
                    self._end_data_reload()
                if status:
                    self._mw.set_button_status(True)
                    self._mw.load_btn.setEnabled(True)
                    self._post_load_actions(file_path)
                else:
                    debug_log("MainWindow._load_file sync load failed path=%s", file_path)
                    self._mw.load_btn.setEnabled(True)
            else:
                debug_log("MainWindow._load_file spawn thread path=%s", file_path)
                self._mw._progress = QProgressDialog("正在读取数据...", "取消", 0, 100, self._mw)
                self._mw._progress.setWindowModality(Qt.WindowModality.ApplicationModal)
                self._mw._progress.setAutoClose(True)
                self._mw._progress.setCancelButton(None)
                self._mw._progress.setMinimumDuration(0)
                self._mw._progress.show()

                self._mw._thread = DataLoadThread(file_path, descRows=descRows, sep=delimiter_typ,
                                                  hasunit=hasunit, encoding=encoding)
                self._mw._thread.progress.connect(self._mw._progress.setValue)
                self._mw._thread.finished.connect(lambda loader: self._on_load_done(loader, file_path))
                self._mw._thread.error.connect(self._on_load_error)
                self._mw._thread.start()
                started_async = True
        except Exception:
            if not started_async:
                self._end_data_reload()
            raise

    @property
    def _has_valid_loader(self) -> bool:
        return hasattr(self._mw, 'loader') and self._mw.loader is not None

    @property
    def _has_valid_data(self) -> bool:
        return (self._has_valid_loader and
                hasattr(self._mw.loader, 'datalength') and
                self._mw.loader.datalength > 0)

    @property
    def _current_data_length(self) -> int:
        return self._mw.loader.datalength if self._has_valid_loader else 0

    def _cleanup_old_data(self):
        try:
            if self._has_valid_loader:
                if hasattr(self._mw.loader, '_df'):
                    del self._mw.loader._df
                del self._mw.loader
                self._mw.loader = None

            self._mw.clear_all_plots()

            import gc
            gc.collect()

        except (AttributeError, TypeError) as e:
            print(f"清理旧数据时出错: {e}")
        except Exception as e:
            print(f"清理旧数据时发生未知错误: {e}")

    def _post_load_actions(self, file_path: str):
        self._mw.loaded_path = file_path
        self._remember_last_open_dir(file_path)

        def truncate_string(file_path, max_length=79):
            filename_length = len(os.path.basename(file_path))
            if len(file_path) <= max_length:
                return file_path
            return "..." + file_path[min(-filename_length-1,-(max_length-3)):]
        self._mw.setWindowTitle(f"{self._mw.defaultTitle} ---- 数据文件: [{truncate_string(file_path)}]")
        self._mw.set_button_status(True)

    def _remember_last_open_dir(self, file_path: str):
        directory = os.path.dirname(file_path)
        if directory and os.path.isdir(directory):
            self._mw._last_open_dir = directory

    def _get_dialog_initial_directory(self) -> str:
        if getattr(self._mw, "_last_open_dir", None) and os.path.isdir(self._mw._last_open_dir):
            return self._mw._last_open_dir
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

        candidates.extend([
            _safe_location(QStandardPaths.StandardLocation.HomeLocation),
            _safe_location(QStandardPaths.StandardLocation.DesktopLocation),
            QDir.rootPath()
        ])
        for path in candidates:
            if path:
                return path
        return ""

    @staticmethod
    def load_dict(path: str, *, default=None) -> dict:
        import ujson as json
        if not os.path.exists(path):
            return {} if default is None else default
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            debug_log("load_dict JSON decode error for %s: %s", path, e)
            raise

    def _extract_file_extension(self, file_path: str) -> str:
        supported_extensions = ['.csv', '.mfile', '.t00', '.t01', '.t10', '.t11', '.txt',
                                '.mf4', '.mdf', '.dat']

        base_ext = os.path.splitext(file_path)[1].lower()
        if base_ext in supported_extensions:
            return base_ext

        base_name = os.path.basename(file_path).lower()

        pattern = r'(' + '|'.join(re.escape(ext) for ext in supported_extensions) + r')\.\d+$'
        match = re.search(pattern, base_name)

        if match:
            return match.group(1)

        return None

    def _validate_load_parameters(self, file_path: str, descRows, sep, hasunit) -> tuple[bool, str]:
        if not isinstance(file_path, str) or not file_path.strip():
            return False, "文件路径无效"
        if descRows is not None and (not isinstance(descRows, int) or descRows < 0):
            return False, "描述行数必须是非负整数"
        if sep is not None and (not isinstance(sep, str) or not sep):
            return False, "分隔符无效"
        if hasunit is not None and not isinstance(hasunit, bool):
            return False, "hasunit参数必须是布尔值"
        return True, ""

    def _load_sync(self,
                   file_path: str,
                   descRows: int = 0,
                   sep: str = ',',
                   hasunit: bool = True,
                   encoding: str | None = None):
        debug_log("MainWindow._load_sync start path=%s descRows=%s sep=%s hasunit=%s encoding=%s",
                  file_path, descRows, sep, hasunit, encoding)
        is_valid, error_msg = self._validate_load_parameters(file_path, descRows, sep, hasunit)
        if not is_valid:
            QMessageBox.critical(self._mw, "参数错误", error_msg)
            return False

        loader = None
        status = False

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ('.mf4', '.mdf', '.dat'):
                from mdf_loader import MDFDataLoader
                loader = MDFDataLoader(file_path)
            else:
                loader = FastDataLoader(file_path, descRows=descRows, sep=sep, hasunit=hasunit,
                                        encoding=encoding)
            self._mw.loader = loader
            self._apply_loader()
            status = True
        except MemoryError as e:
            QMessageBox.critical(self._mw, "内存不足", f"加载文件时内存不足: {str(e)}")
            status = False
        except FileNotFoundError as e:
            QMessageBox.critical(self._mw, "文件未找到", f"无法找到文件: {str(e)}")
            status = False
        except PermissionError as e:
            QMessageBox.critical(self._mw, "权限错误", f"没有文件访问权限: {str(e)}")
            status = False
        except Exception as e:
            QMessageBox.critical(self._mw, "读取失败", f"加载文件时发生错误: {str(e)}")
            status = False
        finally:
            debug_log("MainWindow._load_sync done path=%s status=%s rows=%s",
                      file_path, status,
                      getattr(loader, "datalength", None) if loader is not None else None)
            if loader is not None:
                loader = None
        return status

    def _on_load_done(self, loader, file_path: str):
        self._mw._progress.close()
        debug_log("MainWindow._on_load_done apply new loader path=%s", file_path)
        if hasattr(self._mw, 'loader') and self._mw.loader is not None:
            if hasattr(self._mw.loader, '_df'):
                del self._mw.loader._df
            del self._mw.loader

        self._mw.loader = loader
        self._apply_loader()
        self._post_load_actions(file_path)
        self._end_data_reload()
        self._mw.load_btn.setEnabled(True)

    def _on_load_error(self, msg):
        self._mw._progress.close()
        debug_log("MainWindow._on_load_error %s", msg)
        QMessageBox.critical(self._mw, "读取失败", msg)
        self._end_data_reload()
        self._mw.load_btn.setEnabled(True)

    def _apply_loader(self):
        debug_log("MainWindow._apply_loader datalength=%s columns=%s",
                  getattr(self._mw.loader, "datalength", None),
                  len(getattr(self._mw.loader, "var_names", []) or []))
        self._mw.var_names = self._mw.loader.var_names
        self._mw.units = self._mw.loader.units
        self._mw.time_channels_infos = self._mw.loader.time_channels_info
        self._mw.data_validity = self._mw.loader.df_validity
        self._mw.data = self._mw.loader.df
        self._mw.list_widget.populate(self._mw.var_names, self._mw.units, self._mw.data_validity)

        if self._mw.placeholder_label.parent():
            self._mw.placeholder_label.setParent(None)

        if not self._mw.plot_widgets:
            self._mw.create_subplots_matrix(self._mw._plot_row_max_default, self._mw._plot_col_max_default)
            self._mw.set_plots_visible(self._mw._plot_row_current, self._mw._plot_col_current)

        for container in self._mw.plot_widgets:
            widget = container.plot_widget
            widget.data = self._mw.loader.df
            widget.units = self._mw.loader.units
            widget.time_channels_info = self._mw.loader.time_channels_info
            widget.time_values = self._mw.loader.time_values
            widget.time_column_name = self._mw.loader.time_column_name
            widget.time_axis_label = self._mw.loader.time_axis_label
            widget.update_x_axis_label()

        self._mw._compute_baseline_density()
        self._mw._sync_min_xrange()

        self._mw.replots_after_loading()
        from src.ui.table_dialog import DataTableDialog
        if DataTableDialog._instance is not None:
            DataTableDialog._instance.update_data(self._mw.loader)
            if not DataTableDialog._instance._df.empty:
                DataTableDialog._instance.show()
                DataTableDialog._instance.raise_()
                DataTableDialog._instance.activateWindow()
            else:
                DataTableDialog._instance.set_skip_close_confirmation(True)
                DataTableDialog._instance.close()

        self.filter_variables()
        if self._mw.mark_region_btn.isChecked():
            self._mw.request_mark_stats_refresh(immediate=True)

    def filter_variables(self):
        if self._mw.var_names is None:
            return
        name_text = self._mw.filter_input.text().lower()
        unit_text = self._mw.unit_filter_input.text().lower()
        name_keywords = name_text.split() if name_text else []
        unit_keywords = unit_text.split() if unit_text else []

        filtered_names = []
        for var in self._mw.var_names:
            if not isinstance(var, str):
                continue

            var_lower = var.lower()
            unit = self._mw.units.get(var, '').lower()

            name_match = not name_keywords or any(kw in var_lower for kw in name_keywords)
            unit_match = not unit_keywords or any(kw in unit for kw in unit_keywords)

            if name_match and unit_match:
                filtered_names.append(var)

        self._mw.list_widget.populate(filtered_names, self._mw.units, self._mw.data_validity)
