from __future__ import annotations 
import sys
import os
import weakref
import subprocess
import numpy as np
import pandas as pd
import warnings
from typing import Any

# 抑制pandas日期解析警告
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", message=".*dayfirst.*")
warnings.filterwarnings("ignore", message=".*Could not infer format.*")

# 屏蔽 macOS ICC 警告
# os.environ["QT_LOGGING_RULES"] = (
#     "qt6ct.debug=false; "      # 原来想关的 qt6ct 日志
#     "qt.gui.icc=false"         # 关闭 ICC 解析相关日志
# )

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData, QMargins, QTimer, QEvent, QObject, QAbstractTableModel, QModelIndex, QPoint, QPointF, QSize, QRect, QItemSelectionModel, QStandardPaths, QDir
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QFontMetrics, QDrag, QPen, QColor, QBrush,  QIcon, QFont, QFontDatabase, QPixmap, QPainter, QTextDocument, QTextCursor, QCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QProgressDialog, QGridLayout, QSpinBox, QMenu, QTextEdit,QAction, QActionGroup,
    QFileDialog, QPushButton, QAbstractItemView, QLabel, QLineEdit, QTableView, QStyledItemDelegate, QCheckBox,
    QMessageBox, QDialog, QFormLayout, QSizePolicy, QHeaderView, QRubberBand, QDoubleSpinBox, QTreeWidget, QTreeWidgetItem, QSplitter,
    QGraphicsLinearLayout, QGraphicsProxyWidget, QGraphicsWidget, QTableWidget, QTableWidgetItem, QColorDialog
)
import pyqtgraph as pg
from threading import Lock

global DEFAULT_PADDING_VAL_X,DEFAULT_PADDING_VAL_Y,FILE_SIZE_LIMIT_BACKGROUND_LOADING,RATIO_RESET_PLOTS, FROZEN_VIEW_WIDTH_DEFAULT, BLINK_PULSE, FACTOR_SCROLL_ZOOM, MIN_INDEX_LENGTH, DEFAULT_LINE_WIDTH, THICK_LINE_WIDTH, THIN_LINE_WIDTH, XRANGE_THRESHOLD_FOR_SYMBOLS
DEFAULT_PADDING_VAL_X = 0.05
DEFAULT_PADDING_VAL_Y = 0.1
FILE_SIZE_LIMIT_BACKGROUND_LOADING = 2
RATIO_RESET_PLOTS = 0.3
FROZEN_VIEW_WIDTH_DEFAULT = 180
XRANGE_THRESHOLD_FOR_SYMBOLS = 100.0  # xRange宽度阈值（考虑factor后），小于此值显示symbols（细线+symbol），否则粗线无symbol
BLINK_PULSE = 200
FACTOR_SCROLL_ZOOM = 0.3
MIN_INDEX_LENGTH = 3
DEFAULT_LINE_WIDTH = 2
THICK_LINE_WIDTH = 2
THIN_LINE_WIDTH = 1
DEBOUNCE_DELAY_MS = 200
UI_DEBOUNCE_DELAY_MS = DEBOUNCE_DELAY_MS  # 别名，保持与 PyQt6 一致
FLOAT32_SAFE_MAX = float(np.finfo(np.float32).max)


def safe_callback(func):
    """
    装饰器：捕获回调中的异常，防止崩溃

    【稳定性优化】用于保护关键的信号回调函数，防止因对象已销毁等原因导致的崩溃。
    特别处理RuntimeError（C++对象已删除）和AttributeError。
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            # C++对象已删除
            err_msg = str(e).lower()
            if "deleted" in err_msg or "wrapped" in err_msg or "c++ object" in err_msg:
                return None
            raise
        except AttributeError as e:
            # 属性访问错误（对象可能部分销毁）
            return None
        except Exception as e:
            return None
    return wrapper


class UnifiedUpdateScheduler(QObject):
    """
    统一UI更新调度器，合并style/cursor/stat等更新请求，延迟批量执行避免频繁刷新
    """
    def __init__(self, delay_ms=UI_DEBOUNCE_DELAY_MS, order=None, parent=None):
        super().__init__(parent)
        self._delay_ms = max(0, delay_ms)
        self._order = list(order) if order else []
        self._pending = []
        self._callbacks = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._flush_pending)

    def register(self, name, callback):
        self._callbacks[name] = callback

    def schedule(self, *names):
        scheduled = False
        for name in names:
            if name not in self._callbacks:
                continue
            if name not in self._pending:
                self._pending.append(name)
            scheduled = True
        if scheduled:
            self._timer.start(self._delay_ms)

    def cancel(self, *names):
        if not names:
            self._pending.clear()
            self._timer.stop()
            return
        remaining = [name for name in self._pending if name not in names]
        self._pending[:] = remaining
        if not self._pending:
            self._timer.stop()

    def run_immediately(self, *names):
        tasks = list(names) if names else (self._order or list(self._pending))
        if not tasks:
            return
        for name in tasks:
            if name in self._pending:
                self._pending.remove(name)
            self._invoke(name)

    def _flush_pending(self):
        if not self._pending:
            return
        pending = self._pending.copy()
        self._pending.clear()
        ordered = [name for name in self._order if name in pending]
        ordered += [name for name in pending if name not in ordered]
        for name in ordered:
            self._invoke(name)

    def _invoke(self, name):
        callback = self._callbacks.get(name)
        if not callback:
            return
        try:
            callback()
        except Exception as e:
            print(f"UI更新调度器执行 {name} 出错: {e}")


def _evaluate_float32_safety(values: Any) -> tuple[bool, float | None]:
    """
    判断给定数值是否可以安全转换为float32。

    Args:
        values: pandas Series、NumPy数组或其他可迭代的数值对象。

    Returns:
        tuple[bool, float | None]: (是否安全, 有限值的绝对值最大值)
            当所有值都不是有限数时，返回 (False, None)。
    """
    if values is None:
        return False, None

    try:
        if isinstance(values, pd.Series):
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        else:
            try:
                arr = np.asarray(values, dtype=np.float64)
            except (ValueError, TypeError, OverflowError):
                arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    except Exception:
        return False, None

    if arr.size == 0:
        return True, 0.0

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return False, None

    abs_max = float(np.max(np.abs(arr[finite_mask])))
    return abs_max <= FLOAT32_SAFE_MAX, abs_max

# 主界面
global SCREEN_WITDH_MARGIN,SCREEN_HEIGHT_MARGIN
SCREEN_WITDH_MARGIN = 0.2
SCREEN_HEIGHT_MARGIN = 0.2

# PyInstaller 解包目录
from pathlib import Path

def resource_path(relative_path: str) -> Path:
    """
    获取打包后的资源文件路径 (兼容 PyInstaller/Nuitka/PyOxidizer Standalone)
    """
    if hasattr(sys, "_MEIPASS"):
        # 模式 1: PyInstaller OneFile 模式
        return Path(os.path.join(sys._MEIPASS, relative_path))
    
    elif getattr(sys, "frozen", False):
        # 模式 2: 其他 Standalone 模式 (PyOxidizer/Nuitka)
        # 资源文件通常位于可执行文件所在目录
        return Path(os.path.dirname(sys.executable)) / relative_path
        
    else:
        # 模式 3: 开发环境
        return Path(relative_path)

# 设置应用程序和窗口图标
if sys.platform == "win32": # Windows
    import ctypes
    myappid = 'mycompany.csv_plot.0.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    ico_path = resource_path("icon.ico")  

elif sys.platform == "darwin":  # macOS
    ico_path = resource_path("icon.icns")  

# 窗口实例相关函数
def get_main_window() -> MainWindow | None:
    """
    安全地查找并返回 MainWindow 实例。
    """
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, MainWindow):
            return widget
    return None

def get_main_loader() -> FastDataLoader | None:
    """
    安全地查找并返回 MainWindow 的 loader 实例。
    """
    main_window = get_main_window()
    if main_window and hasattr(main_window, 'loader') and main_window.loader:
        return main_window.loader
    return None


def parse_dragged_var_names(text: str) -> list[str]:
    """
    解析拖放文本中的变量名，支持 "name1;;name2" 形式的多变量。
    """
    if not text:
        return []
    return [name.strip() for name in text.split(";;") if name.strip()]

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("帮助文档")
        self.resize(800, 600)
        layout = QVBoxLayout(self)

        # 把窗口移动到屏幕中心
        screen = QApplication.primaryScreen().availableGeometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 4
        self.move(x, y)

        # 文本区域
        text_edit = QTextEdit(self)
        text_edit.setReadOnly(True)
        
        # 加载 README.md
        readme_path = resource_path("README.md")
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                text_edit.setMarkdown(f.read())
        else:
            text_edit.setPlainText("README.md 文件未找到。")

        layout.addWidget(text_edit)

        # 关闭按钮
        close_btn = QPushButton("关闭", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)


class DataLoadThread(QThread):
    # 信号：发送进度 0-100，或直接发 DataFrame
    progress = pyqtSignal(int)        # 百分比
    finished = pyqtSignal(object)     # FastDataLoader 实例
    error = pyqtSignal(str)

    def __init__(self, file_path: str, parent=None, descRows: int = 0, sep: str = ',', hasunit: bool = True, use_streaming: bool = False):
        super().__init__(parent)
        self.file_path = file_path
        self.descRows = descRows
        self.sep = sep
        self.hasunit = hasunit
        self.use_streaming = use_streaming
        self._should_stop = False
        
    def run(self):
        try:
            def _progress_cb(progress: int):
                if not self._should_stop:
                    self.progress.emit(progress)

            if self._should_stop:
                return

            loader = FastDataLoader(
                self.file_path,
                descRows=self.descRows,
                sep=self.sep,
                hasunit=self.hasunit,
                chunksize=3600,          
                _progress=_progress_cb,
                use_streaming=self.use_streaming,
            )
            
            if not self._should_stop:
                self.finished.emit(loader)
        except Exception as e:
            if not self._should_stop:
                self.error.emit(str(e))
    
    def stop(self):
        """安全停止线程"""
        self._should_stop = True
        self.quit()
        self.wait(3000)  # 等待最多3秒


class StreamingCSVReader:
    """流式CSV读取器，用于处理正在写入的文件"""
    
    def __init__(self, file_path: str, chunk_size: int = 1000, max_retries: int = 3):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.file_handle = None
        self.current_position = 0
        self.temp_files = []  # 记录临时文件，用于清理
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        """关闭文件句柄并清理临时文件"""
        try:
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
        except Exception:
            pass
            
        # 清理临时文件
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        self.temp_files.clear()
    
    def read_stable_chunk(self, encoding: str = 'utf-8') -> str:
        """读取稳定的数据块，避免读取正在写入的部分"""
        import time
        
        for attempt in range(self.max_retries):
            try:
                if not self.file_handle:
                    self.file_handle = open(self.file_path, 'r', encoding=encoding, buffering=8192)
                    self.file_handle.seek(self.current_position)
                
                # 读取到文件末尾
                self.file_handle.seek(0, 2)  # 移动到文件末尾
                current_size = self.file_handle.tell()
                
                if current_size <= self.current_position:
                    return None  # 没有新数据
                
                # 回退到当前位置
                self.file_handle.seek(self.current_position)
                
                # 读取数据
                data = self.file_handle.read(self.chunk_size)
                if not data:
                    return None
                
                # 检查是否读取到完整行
                if not data.endswith('\n'):
                    # 尝试读取到下一个换行符
                    extra_data = self.file_handle.readline()
                    if extra_data:
                        data += extra_data
                
                self.current_position = self.file_handle.tell()
                return data
                
            except (OSError, IOError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(0.1)  # 短暂等待后重试
                    continue
                else:
                    raise e
                    
        return None

class FastDataLoader:
    # 脏数据清单
    _NA_VALUES = [
        "", "nan", "NaN", "NAN", "NULL", "null", "None", "plus infinity", "minus infinity",
        "Inf", "inf", "-inf", "-Inf", "1.#INF", "-1.#INF", "data err", '* *', '----', 'Infinity', 'no value'
    ]
    
    from typing import Callable
    def __init__(
        self,
        csv_path: str,
        *,
        max_rows_infer: int = 200,
        chunksize: int | None = None,
        usecols: list[str] | None = None,
        drop_empty: bool = False,
        downcast_float: bool = True,
        descRows: int = 0,
        sep: str = ",",
        _progress: Callable | None = None,
        do_parse_date: bool = False,
        hasunit: bool = True,
        use_streaming: bool = False
    ):
        self._path = csv_path
        self.file_size = os.path.getsize(csv_path) 
        self.max_rows_infer = max_rows_infer
        self.usecols = usecols
        self.drop_empty = drop_empty
        self.downcast_float = downcast_float
        self.sep = sep
        self.descRows = descRows
        self._progress_cb = _progress
        self.do_parse_date = do_parse_date
        self.hasunit = hasunit
        self.use_streaming = use_streaming

        # 一次性读取 header + 单位行，并回退编码
        self._var_names, self._units, self.encoding_used = self._load_header_units(
            self._path, desc_rows=self.descRows, usecols=self.usecols, sep=self.sep, hasunit=self.hasunit
        )
        if self._progress_cb:
            self._progress_cb(5)

        # 推断 dtype
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sample = pd.read_csv(
                self._path,
                skiprows=(2 + self.descRows) if self.hasunit else (1 + self.descRows),
                nrows=self.max_rows_infer,
                names=self._var_names,
                encoding=self.encoding_used,
                usecols=self.usecols,
                low_memory=False,
                sep=self.sep,
                na_values=self._NA_VALUES,
                keep_default_na=True,
                dayfirst=True,  # 添加dayfirst参数
            )
        dtype_map, parse_dates, date_formats, downcast_ratio = self._infer_schema(sample)
        self.date_formats = date_formats
        self.sample_mem_size = sample.memory_usage(deep=True).sum()
        
        # 安全计算 byte_per_line，避免除零错误
        if sample.shape[0] > 0:
            self.byte_per_line = (0.8 * self.sample_mem_size) / sample.shape[0]
        else:
            # 如果 sample 为空，使用一个合理的默认值
            self.byte_per_line = 100  # 假设每行约100字节
        
        # 安全计算 estimated_lines
        if self.byte_per_line > 0:
            self.estimated_lines = int(self.file_size / self.byte_per_line)
        else:
            self.estimated_lines = 1000  # 默认假设1000行
        
        import gc
        del sample 
        gc.collect()
        if self._progress_cb:
            self._progress_cb(15)
            
        # 计算 chunk 大小，根据可用内存动态调整
        if chunksize is None:
            try:
                import psutil
                available_memory = psutil.virtual_memory().available
                
                # 确保 available_memory 是有效的数值
                if not isinstance(available_memory, (int, float)) or available_memory <= 0:
                    raise ValueError("Invalid available memory value")
                
                # Win7 32位系统特殊优化
                if sys.platform == "win32":
                    import platform
                    arch_info = platform.architecture()
                    if arch_info and len(arch_info) > 0 and arch_info[0] == "32bit":
                        # 32位系统内存限制，使用更保守的策略
                        max_chunk_memory = min(available_memory * 0.1, 100 * 1024 * 1024)  # 最多100MB
                    else:
                        max_chunk_memory = available_memory * 0.25
                else:
                    max_chunk_memory = available_memory * 0.25
                
                # 确保 byte_per_line 不为零
                if self.byte_per_line <= 0:
                    raise ValueError("Invalid byte_per_line value")
                
                # 使用可用内存的一部分作为chunk大小
                estimated_memory_per_chunk = self.byte_per_line * 3600
                if estimated_memory_per_chunk < max_chunk_memory:
                    chunksize = 3600
                else:
                    chunksize = max(1000, int(max_chunk_memory / self.byte_per_line))
            except Exception as e:
                # 如果psutil未安装或出现其他错误，使用默认值
                # 可以在调试时取消下面这行的注释来查看具体错误
                # print(f"Warning: Failed to calculate optimal chunk size: {e}, using default")
                chunksize = 3600
        
        # 正式读取
        if self.use_streaming:
            self._df = self._read_streaming(
                self._path,
                dtype_map,
                parse_dates,
                int(chunksize),
                sep=self.sep,
                descRows=self.descRows,
                hasunit=self.hasunit
            )
        else:
            self._df = self._read_chunks(
                self._path,
                dtype_map,
                parse_dates,
                int(chunksize),
                sep=self.sep,
                descRows=self.descRows,
                hasunit=self.hasunit
            )
        
        # 后处理
        if drop_empty:
            self._df = self._df.dropna(axis=1, how="all")
        if downcast_float:
            self._downcast_numeric()
        self._df_validity = self._check_df_validity()
        
        import gc
        gc.collect()
        if self._progress_cb:
            self._progress_cb(100)

    @staticmethod
    def _load_header_units(
        path: str,
        desc_rows: int = 0,
        usecols: list[str] | None = None,
        sep: str = ",",
        hasunit: bool = True
    ) -> tuple[list[str], dict[str, str], str]:
        nrows = 2 if hasunit else 1
        encodings_default = ["utf-8", "cp1252"]

        # chardet
        import chardet
        from pathlib import Path
        sample_sime = 2000  # bytes
        with Path(path).open('rb') as f:
            raw_sample = f.read(sample_sime)
        r = chardet.detect(raw_sample)
        language_infer = r['language']
        confidence_infer = r['confidence']

        if language_infer is not None and language_infer.lower() == 'chinese' and confidence_infer > 0.7:
            encoding_infer = ['gb18030']
        else:
            encoding_infer = ['utf-8']

        for enc in list(dict.fromkeys(encoding_infer + encodings_default)):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    df = pd.read_csv(
                        path,
                        skiprows=desc_rows,
                        nrows=nrows,
                        header=None,
                        usecols=usecols,
                        sep=sep,
                        encoding=enc,
                        engine="python",
                        dayfirst=True,  # 添加dayfirst参数
                    )
                break
            except UnicodeDecodeError:
                continue
        else:
            raise RuntimeError("无法以任何可用编码读取文件")

        if df.shape[0] < nrows:
            raise ValueError("文件至少需要两行（变量名 + 单位）")

        var_names = df.iloc[0].astype(str).tolist()
        var_names = FastDataLoader._make_unique(var_names) 
        if hasunit:
            units = dict(zip(var_names, df.iloc[1].fillna("").astype(str).tolist()))
        else:
            units = dict(zip(var_names, ['-'] * len(var_names)))
        return var_names, units, enc

    @staticmethod
    def _infer_schema(sample: pd.DataFrame) -> tuple[dict[str, str], list[str], dict[str, str], float]:
        """推断数据类型和时间格式
        
        优化策略：
        1. 只对列名包含时间相关关键字的列进行时间格式推断
        2. 使用精简的日期格式候选列表
        3. 对每列只采样前10行进行格式推断
        """
        dtype_map: dict[str, str] = {}
        parse_dates: list[str] = []
        date_formats: dict[str, str] = {}

        # 精简的日期格式候选列表（最常用的格式，优先级从高到低）
        date_candidates = [
            "%H:%M:%S.%f",   # 带微秒的时间（优先匹配，避免丢失精度）
            "%H:%M:%S",      # 时间格式
            "%d/%m/%Y",      # 欧洲日期格式 (例: 18/11/2017)
            "%Y/%m/%d",      # 日期格式 (例: 2024/10/31)
            "%Y-%m-%d",      # ISO日期格式 (例: 2024-10-31)
        ]
        
        # 时间列的关键字（小写，用于不区分大小写匹配）
        time_keywords = ['time', 'date', 'datetime', 'timestamp', 'zeit', 'tmd', 'tmod']
        
        float_cols = sample.select_dtypes(include=['float', 'float64', 'category'])
        downcast_ratio_est = float_cols.shape[1] / sample.shape[1] if sample.shape[1] > 0 else 0.000001
        
        for col in sample.columns:
            s = sample[col]
            if s.isna().all():
                dtype_map[col] = "category"
                continue
            
            # 只对列名包含时间关键字的列进行时间格式推断
            col_lower = col.lower()
            is_time_candidate = any(keyword.lower() in col_lower for keyword in time_keywords)
            
            if is_time_candidate:
                # 只采样前10行进行格式推断（大大加快速度）
                s_sample = s.head(10).dropna()
                if len(s_sample) > 0:
                    for fmt in date_candidates:
                        try:
                            pd.to_datetime(s_sample, format=fmt, errors="raise")
                            parse_dates.append(col)
                            date_formats[col] = fmt
                            break
                        except (ValueError, TypeError):
                            continue
                    else:
                        # 不是时间格式，按数值处理
                        if pd.api.types.is_numeric_dtype(s):
                            is_safe, _ = _evaluate_float32_safety(s)
                            dtype_map[col] = "float32" if is_safe else "float64"
                        else:
                            dtype_map[col] = "category"
                else:
                    dtype_map[col] = "category"
            else:
                # 非时间列，直接判断数值类型
                if pd.api.types.is_numeric_dtype(s):
                    is_safe, _ = _evaluate_float32_safety(s)
                    dtype_map[col] = "float32" if is_safe else "float64"
                else:
                    dtype_map[col] = "category"
        
        return dtype_map, parse_dates, date_formats, downcast_ratio_est

    def _read_chunks(
        self,
        path: str,
        dtype_map,
        parse_dates: list[str],
        chunksize: int,
        sep: None | str = ",",
        descRows: int = 0,
        hasunit: bool = True,
    ) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        # do not parse date
        if not self.do_parse_date:
            parse_dates = []
        total_chunks_est = max(1, self.estimated_lines // chunksize + (1 if self.estimated_lines % chunksize else 0))
        increment = 80 / total_chunks_est
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for idx, chunk in enumerate(pd.read_csv(
                path,
                skiprows=(2 + descRows) if hasunit else (1 + descRows),
                names=self._var_names,
                dtype=dtype_map,
                parse_dates=parse_dates,
                encoding=self.encoding_used,
                chunksize=chunksize,
                usecols=self.usecols,
                low_memory=False,
                memory_map=True,
                sep=sep,
                na_values=self._NA_VALUES,
                keep_default_na=True,
                on_bad_lines='skip',
                dayfirst=True  # 添加dayfirst参数
            )):
                if self._progress_cb:
                    chunk_progress = min(80, (idx + 1) * increment)
                    self._progress_cb(15 + int(chunk_progress))
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def _read_streaming(
        self,
        path: str,
        dtype_map,
        parse_dates: list[str],
        chunksize: int,
        sep: None | str = ",",
        descRows: int = 0,
        hasunit: bool = True,
    ) -> pd.DataFrame:
        """流式读取CSV文件，适用于正在写入的文件"""
        import tempfile
        import shutil
        
        # 创建临时文件用于稳定读取
        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', prefix='csv_plot_')
        os.close(temp_fd)  # 关闭文件描述符，我们只需要路径
        
        try:
            # 复制文件到临时位置，避免读取正在写入的文件
            shutil.copy2(path, temp_path)
            
            # 使用临时文件进行读取
            chunks: list[pd.DataFrame] = []
            total_chunks_est = max(1, self.estimated_lines // chunksize + (1 if self.estimated_lines % chunksize else 0))
            increment = 80 / total_chunks_est
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for idx, chunk in enumerate(pd.read_csv(
                    temp_path,
                    skiprows=(2 + descRows) if hasunit else (1 + descRows),
                    names=self._var_names,
                    dtype=dtype_map,
                    parse_dates=parse_dates,
                    encoding=self.encoding_used,
                    chunksize=chunksize,
                    usecols=self.usecols,
                    low_memory=False,
                    memory_map=True,
                    sep=sep,
                    na_values=self._NA_VALUES,
                    keep_default_na=True,
                    on_bad_lines='skip',
                    dayfirst=True  # 添加dayfirst参数
                )):
                    if self._progress_cb:
                        chunk_progress = min(80, (idx + 1) * increment)
                        self._progress_cb(15 + int(chunk_progress))
                    chunks.append(chunk)
            
            return pd.concat(chunks, ignore_index=True)
            
        finally:
            # 立即清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    def _downcast_numeric(self) -> None:
        float_cols = self._df.select_dtypes(include=["float32", "float64"]).columns
        for col in float_cols:
            cleaned = pd.to_numeric(self._df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            is_safe, _ = _evaluate_float32_safety(cleaned)
            if is_safe:
                self._df[col] = cleaned.astype("float32")
            else:
                self._df[col] = cleaned.astype("float64", copy=False)

    def _check_df_validity(self) -> dict:
        validity : dict = {}
        for col in self._df.columns:
            # 传入列名和date_formats参数
            validity[col] = self._classify_column(self._df[col], col, self.date_formats)
        
        return validity

    @staticmethod
    def _make_unique(names: list[str]) -> list[str]:
        seen = {}
        unique_names = []
        for name in names:
            if name in seen:
                seen[name] += 1
                new_name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
                new_name = name
            unique_names.append(new_name)
        return unique_names
    
    @staticmethod
    def _classify_column(series: pd.Series, col_name: str, date_formats: dict) -> int:
        """
        1: （全部可转数字，且 ≥2 个不同有效值） 或 （该列是日期格式） 或 （数据长度为1，且可转换为数字）
        0: 数据长度>=2, 且全部可转数字，且唯一有效值
        -1: 存在非数字（不含日期格式） 或 全部 NaN
        """
        # 如果该列是日期格式，则直接返回1（有效）   
        if col_name in date_formats:
            return 1

        # 1) 先尝试整列转 float，失败直接 -1
        try:
            numeric = pd.to_numeric(series, errors="raise")
        except (ValueError, TypeError):
            return -1

        # 2) 去掉 NaN 后看有效值
        valid = numeric.dropna()
        if valid.empty:          # 全 NaN
            return -1

        # 数据长度为1且可转数字 → 返回1
        if len(series) == 1:
            return 1

        unique_vals = valid.unique()
        if len(unique_vals) == 1:
            return 0
        else:
            return 1
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def units(self) -> dict[str, str]:
        return self._units
    
    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def datalength(self) -> int:
        return self._df.shape[0]

    @property
    def var_names(self) -> list[str]:
        return self._df.columns.tolist()
    
    @property
    def row_count(self) -> int:
        return len(self._df)
    
    @property
    def column_count(self) -> int:
        return len(self._df.columns)
    
    @property
    def time_channels_info(self) -> dict[str, str]:
        return self.date_formats
    
    @property
    def df_validity(self) -> dict:
        return self._df_validity
    

class DropOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.Widget)
        self.setStyleSheet("""
            background:rgba(255,255,255,200);   
            border:none;
        """)
        
        self.label = QLabel("请丢入数据", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            background-color: rgba(168, 168, 168, 255);
            color:#333;
            font-size:36px;
            border-radius:12px;
            padding:20px 40px;
            color: rgba(128, 128, 128, 200);
        """)
        self.hide()

    def adjust_text(self, file_type_supported=True):
        if file_type_supported:
            self.label.setText("请丢入数据")
        else:
            self.label.setText("数据格式不支持")
                               
        
    def adjust_font(self):
        # 根据 label 当前尺寸动态字号
        side = min(self.label.width(), self.label.height())
        font_size = max(12, min(int(side * 0.3), 128))
        font = self.label.font()
        font.setPixelSize(font_size)
        font.setBold(True)
        self.label.setFont(font)

    def resizeEvent(self, event):
        w_half = self.width() 
        h_half = self.height() 
        self.label.setFixedSize(w_half, h_half)
        self.adjust_font()

        self.label.move(
            (self.width() - self.label.width()) // 2,
            (self.height() - self.label.height()) // 2
        )


class PandasTableModel(QAbstractTableModel):
    """只读官方虚拟模型，支持千万行秒开"""
    def __init__(self, df: pd.DataFrame, units: dict[str, str], parent=None):
        super().__init__(parent)
        self._df = df
        self._units = units

    # 三个必须实现的纯虚函数
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        value = self._df.iloc[index.row(), index.column()]
        return str(value) if pd.notnull(value) else ""

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        if self._df.columns.empty:
            return None
        if orientation == Qt.Horizontal:
            col_name = str(self._df.columns[section])
            unit = self._units.get(col_name, '')
            return f"{col_name}\n({unit})" if unit else f"{col_name}\n()"
        return str(section + 1)           # 行号 1-based

    def removeColumns(self, column, count, parent=QModelIndex()):
        if column < 0 or column + count > self.columnCount():
            return False
        self.beginRemoveColumns(parent, column, column + count - 1)
        self._df.drop(self._df.columns[column:column + count], axis=1, inplace=True)
        self.endRemoveColumns()
        return True
    

class CustomDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_rows = set()
        self.selected_cols = set()
        self.highlighted_rows = set()  # 新增：用于存储需要高亮的行（来自另一个视图）
        self.highlighted_cols = set()  # 新增：用于存储需要高亮的列（用于闪烁效果）

    def paint(self, painter, option, index):
        from PyQt5.QtWidgets import QStyle
        
        painter.save()
        
        # 判断单元格是否被选中（同时在被选中的行和列中）
        is_selected_cell = (index.row() in self.selected_rows and 
                           index.column() in self.selected_cols)
        
        # 判断单元格是否只在被选中的行或列中（但不是同时）
        is_in_selected_row = index.row() in self.selected_rows
        is_in_selected_col = index.column() in self.selected_cols
        is_in_selected_row_or_col = (is_in_selected_row or is_in_selected_col) and not is_selected_cell
        
        # 被选中的单元格本身：使用系统高亮颜色（和主界面变量列表一致），50%透明度
        if is_selected_cell:
            highlight_color = option.palette.highlight().color()
            # 设置50%透明度（alpha = 128）
            highlight_color.setAlpha(128)
            painter.fillRect(option.rect, highlight_color)
        # 被选中的单元格所在的行或列：使用浅蓝色，提高透明度
        elif is_in_selected_row_or_col:
            painter.fillRect(option.rect, QColor(200, 200, 255, 32))  # 浅蓝高亮，更透明（从64降低到32）
        
        # 新增：高亮来自另一个视图的行
        if index.row() in self.highlighted_rows:
            painter.fillRect(option.rect, QColor(255, 200, 200, 64))  # 淡红色高亮，更透明

        # 新增：高亮指定的列（用于闪烁）
        if index.column() in self.highlighted_cols:
            painter.fillRect(option.rect, QColor(200, 200, 255, 128))  # 淡蓝色高亮，半透明

        super().paint(painter, option, index)
        painter.restore()


class XYScatterPlotDialog(QDialog):
    def __init__(self, x_data, y_data, x_name, y_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("X/Y 散点图")
        self.resize(500, 500)

        # 设置窗口在关闭时释放内存
        self.setAttribute(Qt.WA_DeleteOnClose)

        layout = QVBoxLayout(self)
        
        # 创建 pyqtgraph 绘图组件
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # 绘制散点图
        scatter = pg.ScatterPlotItem(x=x_data, y=y_data, pen='r', brush='r', size=5)
        self.plot_widget.addItem(scatter)
        self.plot_widget.setBackground('w')
        black_pen = pg.mkPen(color='k', width=2)
        self.plot_widget.getViewBox().setBorder(black_pen)   # 外框黑色

        # 文字加粗，但字体家族用系统默认
        bold_font = QFont()
        bold_font.setBold(True)

        # 设置坐标轴标签和标题
        self.plot_widget.setLabel('bottom', text=x_name, color='k', font=bold_font)
        self.plot_widget.setLabel('left', text=y_name, color='k', font=bold_font)

        # 直接设置标签字体
        axis_bottom = self.plot_widget.getAxis('bottom')
        axis_left = self.plot_widget.getAxis('left')
        axis_bottom.label.setFont(bold_font)
        axis_left.label.setFont(bold_font)

        #self.plot_widget.setTitle(f"{y_name} vs. {x_name}")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # 拿到 AxisItem 句柄
        axis_bottom = self.plot_widget.getAxis('bottom')
        axis_left = self.plot_widget.getAxis('left')

        for ax in (axis_bottom, axis_left):
            # 设置轴线和刻度线的颜色为黑色
            ax.setPen('k')
            # 设置刻度文字颜色为黑色
            ax.setTextPen('k')
            # 设置刻度文字的字体
            ax.setTickFont(QFont())


class DataTableDialog(QMainWindow):
    _instance = None
    _saved_scroll_pos = None  # 类级变量存储滚动位置

    @classmethod
    def popup(cls, var_name: str, data, parent=None):
        if cls._instance is None:
            cls._instance = cls(parent)
        else:
            cls._instance._update_owner_from_widget(parent)

        dlg = cls._instance
        dlg.save_geom()
        if dlg.has_column(var_name):
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        else:
            cls._saved_scroll_pos = dlg.main_view.verticalScrollBar().value() if dlg.main_view else None
            dlg.load_geom()
            dlg._add_variable_to_table(var_name, data)  # 使用内部函数
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()

            # 闪烁
        QTimer.singleShot(100, lambda: dlg._blink_column(var_name, pulse=BLINK_PULSE))
        return dlg

    @classmethod
    def add_variables(cls, var_names, parent=None):
        """批量添加变量到数值表"""
        if isinstance(var_names, str):
            candidates = [var_names]
        else:
            candidates = [name for name in (var_names or []) if isinstance(name, str)]

        normalized = []
        seen = set()
        for name in candidates:
            clean = name.strip()
            if not clean or clean in seen:
                continue
            normalized.append(clean)
            seen.add(clean)

        if not normalized:
            return

        if cls._instance is None:
            cls._instance = cls(parent)
        else:
            cls._instance._update_owner_from_widget(parent)

        dlg = cls._instance
        dlg.save_geom()
        dlg.load_geom()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg._handle_dropped_variables(normalized)

    def _update_owner_from_widget(self, widget):
        window = None
        if isinstance(widget, QWidget):
            window = widget.window()
        self._owner_window_ref = weakref.ref(window) if window else None

    def _get_owner_window(self):
        if hasattr(self, "_owner_window_ref") and self._owner_window_ref:
            window = self._owner_window_ref()
            if window:
                return window
        active = QApplication.activeWindow()
        if isinstance(active, MainWindow):
            return active
        return get_main_window()

    def _resolve_loader(self):
        owner = self._get_owner_window()
        if owner and hasattr(owner, "loader"):
            return owner.loader
        return None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("变量数值表")
        self.window_geometry = None 
        self.scatter_plot_windows = []
        self._skip_close_confirmation = False
        self.frozen_columns = []
        self._owner_window_ref = None
        self._update_owner_from_widget(parent)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter
        # Initialize user-preferred left width
        self.user_left_width = FROZEN_VIEW_WIDTH_DEFAULT  # Initial fixed width for frozen_view

        # Connect splitterMoved to update user preference when handle is dragged
        self.splitter.splitterMoved.connect(self._update_user_left_width)

        self.frozen_view = QTableView(self)
        self.frozen_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozen_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.frozen_view.verticalHeader().setVisible(True)
        self.frozen_view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.frozen_view.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.frozen_view.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.frozen_view.horizontalHeader().customContextMenuRequested.connect(self._on_frozen_header_right_click)
        self.frozen_view.horizontalHeader().setSectionsMovable(True)
        self.frozen_view.horizontalHeader().setDragEnabled(True)
        self.frozen_view.horizontalHeader().setDragDropMode(QAbstractItemView.InternalMove)
        self.frozen_view.horizontalHeader().setDragDropOverwriteMode(False)

        self.frozen_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.frozen_view.customContextMenuRequested.connect(self._show_table_context_menu)

        self.main_view = QTableView(self)
        self.main_view.setSortingEnabled(False)
        self.main_view.verticalHeader().setVisible(False)
        self.main_view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.main_view.horizontalHeader().setSectionsMovable(True)
        self.main_view.horizontalHeader().setDragEnabled(True)
        self.main_view.horizontalHeader().setDragDropMode(QAbstractItemView.InternalMove)
        self.main_view.horizontalHeader().setDragDropOverwriteMode(False)

        self.main_view.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.main_view.horizontalHeader().customContextMenuRequested.connect(self._on_main_header_right_click)
        self.main_view.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.main_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.main_view.customContextMenuRequested.connect(self._show_table_context_menu)
        
        fm = QFontMetrics(self.main_view.font())
        safe_height = int(fm.height() * 1.8)
        self.main_view.verticalHeader().setDefaultSectionSize(safe_height)
        self.frozen_view.verticalHeader().setDefaultSectionSize(safe_height)

        self.main_view.setWordWrap(False)
        self.frozen_view.setWordWrap(False)

        splitter.addWidget(self.frozen_view)
        splitter.addWidget(self.main_view)
        splitter.setSizes([self.user_left_width, 400])

        main_layout.addWidget(splitter)

        self._df = pd.DataFrame()
        self._df_lock = Lock()
        self.model = None
        self.units = {}

        font = self.main_view.horizontalHeader().font()
        font.setBold(True)
        self.main_view.horizontalHeader().setFont(font)
        self.frozen_view.horizontalHeader().setFont(font)

        self.main_view.verticalScrollBar().valueChanged.connect(self.frozen_view.verticalScrollBar().setValue)
        self.frozen_view.verticalScrollBar().valueChanged.connect(self.main_view.verticalScrollBar().setValue)
        self.main_view.verticalHeader().sectionResized.connect(self._sync_row_heights)
        self.frozen_view.verticalHeader().sectionResized.connect(self._sync_row_heights)
        self.main_view.horizontalHeader().setResizeContentsPrecision(1)  # 0: BalanceSpeedAndAccuracy, 試1 (Speed)
        self.frozen_view.horizontalHeader().setResizeContentsPrecision(1)

        self.delegate_frozen = CustomDelegate(self)
        self.delegate_main = CustomDelegate(self)
        self.delegate_frozen.highlighted_rows = set()
        self.delegate_main.highlighted_rows = set()
        self.frozen_view.setItemDelegate(self.delegate_frozen)
        self.main_view.setItemDelegate(self.delegate_main)

        # 添加当前焦点视图跟踪
        self.current_focused_view = None  

        # 为两个视图安装焦点事件过滤器
        self.frozen_view.installEventFilter(self)
        self.main_view.installEventFilter(self)

        # 启用拖放功能
        self.setAcceptDrops(True)
        self.main_view.setAcceptDrops(True)
        self.frozen_view.setAcceptDrops(True)
        
        # 安装事件过滤器处理视图的拖放事件
        self.drop_filter = self.DropFilter(self)
        self.main_view.viewport().installEventFilter(self.drop_filter)
        self.frozen_view.viewport().installEventFilter(self.drop_filter)

        if self.parent() and hasattr(self.parent(), 'data_table_geometry') and self.parent().data_table_geometry:
            self.restoreGeometry(self.parent().data_table_geometry)
        else:
            self.resize(600, 400)
            screen = QApplication.primaryScreen().availableGeometry()
            size = self.geometry()
            x = (screen.width() - size.width()) // 2
            y = (screen.height() - size.height()) // 2
            self.move(x, y)

    # 事件过滤器类处理拖放事件
    class DropFilter(QObject):
        def __init__(self, parent_dialog):
            super().__init__(parent_dialog)
            self.parent_dialog = parent_dialog
            
        def eventFilter(self, obj, event):
            if event.type() == QEvent.DragEnter:
                if event.mimeData().hasText():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.DragMove:
                if event.mimeData().hasText():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Drop:
                if event.mimeData().hasText():
                    var_names_text = event.mimeData().text()
                    var_names = parse_dragged_var_names(var_names_text)
                    if var_names:
                        for name in var_names:
                            self.parent_dialog._handle_dropped_variable(name)
                        event.acceptProposedAction()
                        return True
            return super().eventFilter(obj, event)
        
    def _update_user_left_width(self, pos, index):
        if index == 1:  # Handle for the first splitter section
            self.user_left_width = self.splitter.sizes()[0]

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # On window resize, fix left width to user preference, stretch right
        total_width = sum(self.splitter.sizes())
        self.splitter.setSizes([self.user_left_width, total_width - self.user_left_width])

    # 拖放相关方法
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """处理拖放的变量，将其添加到非冻结区，支持多变量批量添加"""
        var_names = parse_dragged_var_names(event.mimeData().text())

        if not var_names:
            event.ignore()
            return

        if len(var_names) == 1:
            # 单个变量：使用原有逻辑
            self._handle_dropped_variable(var_names[0])
        else:
            # 多个变量：批量处理
            self._handle_dropped_variables(var_names)

        event.acceptProposedAction()

    def _blink_step_on(self, delegate, col_idx, view):
        # 步骤1: 高亮 (持续0.5s)
        delegate.highlighted_cols.add(col_idx)
        view.viewport().update()

    def _blink_step_off(self, delegate, col_idx, view):
        # 步骤2: 取消高亮 (持续0.5s)
        delegate.highlighted_cols.remove(col_idx)
        view.viewport().update()

    def _blink_column(self, var_name, pulse: int = 800):
        if self.has_column(var_name):
            # 启动闪烁动画：淡蓝色底色闪烁2次，频率1次/秒（每个周期1s：高亮0.5s + 正常0.5s）
            col_idx = self._df.columns.get_loc(var_name)  # 获取逻辑列索引
            if var_name in self.frozen_columns:
                delegate = self.delegate_frozen
                view = self.frozen_view
            else:
                delegate = self.delegate_main
                view = self.main_view

            self._blink_step_on(delegate, col_idx, view)
            QTimer.singleShot(pulse, lambda: self._blink_step_off(delegate, col_idx, view))  
        return

    # 内部函数：处理拖放的多个变量
    def _handle_dropped_variables(self, var_names: list):
        """
        处理拖放的多个变量，批量添加到非冻结区

        Args:
            var_names: 要添加的变量名称列表
        """
        loader = self._resolve_loader()
        if loader is None:
            QMessageBox.warning(self, "错误", "没有加载数据")
            return

        existing_vars = []
        invalid_vars = []
        added_vars = []

        # 保存当前的垂直滚动位置
        self._saved_scroll_pos = self.main_view.verticalScrollBar().value() if self.main_view else None

        for var_name in var_names:
            # 检查变量是否已存在
            if self.has_column(var_name):
                existing_vars.append(var_name)
                continue

            # 检查变量是否在数据中存在
            if var_name not in loader.df.columns:
                invalid_vars.append(var_name)
                continue

            # 添加变量
            try:
                series = loader.df[var_name]
                self._add_variable_to_table(var_name, series)
                added_vars.append(var_name)
            except Exception as e:
                invalid_vars.append(f"{var_name} (错误: {str(e)})")

        # 滚动到最后添加的变量
        if added_vars:
            last_var = added_vars[-1]
            QTimer.singleShot(100, lambda: self.scroll_to_column(last_var))
            QTimer.singleShot(100, lambda: self._blink_column(last_var, pulse=BLINK_PULSE))

        # 只在有问题时显示提示
        if existing_vars or invalid_vars:
            msg_parts = []
            if added_vars:
                msg_parts.append(f"成功添加 {len(added_vars)} 个变量")

            if existing_vars:
                msg_parts.append(f"已存在 {len(existing_vars)} 个变量")

            if invalid_vars:
                msg_parts.append(f"无效变量: {', '.join(invalid_vars[:5])}")
                if len(invalid_vars) > 5:
                    msg_parts.append(f"等共 {len(invalid_vars)} 个")

            if invalid_vars:
                QMessageBox.warning(self, "批量添加结果", "\n".join(msg_parts))
            else:
                QMessageBox.information(self, "批量添加结果", "\n".join(msg_parts))

    # 内部函数：处理拖放的变量
    def _handle_dropped_variable(self, var_name: str):
        """处理拖放的变量，添加到非冻结区"""
        # 检查变量是否已存在
        if self.has_column(var_name):
            self.scroll_to_column(var_name)
            self._blink_column(var_name, pulse=BLINK_PULSE)
            return

        # 获取主窗口
        loader = self._resolve_loader()

        if loader is None:
            QMessageBox.warning(self, "错误", "没有加载数据")
            return

        if var_name not in loader.df.columns:  # 改为 loader
            QMessageBox.warning(self, "错误", f"变量 '{var_name}' 不存在")
            return

        # 保存当前的垂直滚动位置，避免添加变量后列表位置变化
        self._saved_scroll_pos = self.main_view.verticalScrollBar().value() if self.main_view else None

        series = loader.df[var_name]  # 改为 loader
        self._add_variable_to_table(var_name, series)

        # 滚动到新添加的列
        QTimer.singleShot(100, lambda: self.scroll_to_column(var_name))
        QTimer.singleShot(100, lambda: self._blink_column(var_name, pulse=BLINK_PULSE))

    # 内部函数：添加变量到表格
    def _add_variable_to_table(self, var_name: str, data: pd.Series):
        """内部函数：将变量添加到表格的非冻结区"""
        prepared = self._prepare_series_for_display(var_name, data)
        self._df[var_name] = prepared
        loader = self._resolve_loader()
        
        if loader:
            self.units = loader.units
            
        self.model = PandasTableModel(self._df, self.units)
        self.main_view.setModel(self.model)
        self.frozen_view.setModel(self.model)
        self._connect_signals()
        self._update_views()
        if self._saved_scroll_pos is not None:
            QTimer.singleShot(0, lambda: self.main_view.verticalScrollBar().setValue(self._saved_scroll_pos))

    def _prepare_series_for_display(self, var_name: str, series: pd.Series) -> pd.Series:
        """根据变量类型为表格显示格式化数据"""
        if series is None:
            return series
        loader = self._resolve_loader()
        if loader is None:
            return series

        fmt = loader.time_channels_info.get(var_name) if loader.time_channels_info else None

        if not fmt or not pd.api.types.is_datetime64_any_dtype(series):
            return series

        # 纯时间列：保留小数精度，同时去掉尾随0和小数点
        if "%H:%M:%S" in fmt:
            formatted = (
                series.dt.strftime("%H:%M:%S.%f")
                .str.rstrip("0")
                .str.rstrip(".")
            )
            return formatted

        # 日期列：按照原格式显示，避免出现 00:00:00
        formatted = series.dt.strftime(fmt)
        return formatted

    def eventFilter(self, obj, event):
        # 处理焦点变化事件
        if event.type() == QEvent.FocusIn:
            if obj in [self.frozen_view, self.main_view]:
                self.current_focused_view = obj
                self._update_highlights_on_focus_change()
        
        return super().eventFilter(obj, event)
    
    def _update_highlights_on_focus_change(self):
        # 根据当前焦点视图更新高亮
        if self.current_focused_view == self.frozen_view:
            # 清除主视图的同步高亮
            self.delegate_main.highlighted_rows = set()
            self.delegate_frozen.highlighted_rows = set()

            # 获取冻结区选中的单元格
            selected_indexes = self.frozen_view.selectionModel().selectedIndexes()
            self.delegate_frozen.selected_rows = set(idx.row() for idx in selected_indexes)
            self.delegate_frozen.selected_cols = set(idx.column() for idx in selected_indexes)
            
            # 设置主视图的高亮行
            self.delegate_main.highlighted_rows = self.delegate_frozen.selected_rows
            
            # 清除主视图的选中状态（只保留高亮行）
            self.delegate_main.selected_rows = set()
            self.delegate_main.selected_cols = set()
            
        elif self.current_focused_view == self.main_view:
            # 清除冻结视图的同步高亮
            self.delegate_frozen.highlighted_rows = set()
            self.delegate_main.highlighted_rows = set()
            # 获取主视图选中的单元格
            selected_indexes = self.main_view.selectionModel().selectedIndexes()
            self.delegate_main.selected_rows = set(idx.row() for idx in selected_indexes)
            self.delegate_main.selected_cols = set(idx.column() for idx in selected_indexes)
            
            # 设置冻结视图的高亮行
            self.delegate_frozen.highlighted_rows = self.delegate_main.selected_rows
            
            # 清除冻结视图的选中状态（只保留高亮行）
            self.delegate_frozen.selected_rows = set()
            self.delegate_frozen.selected_cols = set()
            
        else:
            # 没有焦点，清空所有高亮
            self.delegate_frozen.selected_rows = set()
            self.delegate_frozen.selected_cols = set()
            self.delegate_frozen.highlighted_rows = set()
            
            self.delegate_main.selected_rows = set()
            self.delegate_main.selected_cols = set()
            self.delegate_main.highlighted_rows = set()

        # 更新视图
        self.frozen_view.viewport().update()
        self.main_view.viewport().update()
    
    def _update_highlights_frozen(self, selected, deselected):
        # 设置当前焦点视图为冻结视图
        self.current_focused_view = self.frozen_view
        self._update_highlights_on_focus_change()
    
    def _update_highlights_main(self, selected, deselected):
        # 设置当前焦点视图为主视图
        self.current_focused_view = self.main_view
        self._update_highlights_on_focus_change()
    
    def focusInEvent(self, event):
        # 当对话框获得焦点时，更新高亮
        super().focusInEvent(event)
        self._update_highlights_on_focus_change()
    
    def focusOutEvent(self, event):
        # 当对话框失去焦点时，清除所有高亮
        super().focusOutEvent(event)
        self.current_focused_view = None
        self._update_highlights_on_focus_change()
    
    def _show_table_context_menu(self, pos):
        """表格右键菜单：支持散点图和复制"""
        view = self.sender()
        if not isinstance(view, QTableView):
            return

        selected_indexes = view.selectionModel().selectedIndexes()
        if not selected_indexes:
            return

        frozen_cols = set(self._df.columns.get_loc(col) for col in self.frozen_columns)
        other_view = self.frozen_view if view == self.main_view else self.main_view
        other_selected = other_view.selectionModel().selectedIndexes()
        all_selected = selected_indexes + other_selected

        rows_per_col_all = {}
        for idx in all_selected:
            rows_per_col_all.setdefault(idx.column(), set()).add(idx.row())

        total_cols = set(rows_per_col_all.keys())

        # 复制条件：所选列的行集合一致
        can_copy = False
        ordered_cols = []
        rows_order = []
        if len(total_cols) == 1:
            only_col = next(iter(total_cols))
            ordered_cols = [only_col]
            rows_order = sorted(rows_per_col_all[only_col])
            can_copy = len(rows_order) > 0
        elif len(total_cols) >= 2:
            cols_list = list(total_cols)
            base_rows = rows_per_col_all[cols_list[0]] if cols_list else set()
            if base_rows and all(rows_per_col_all[c] == base_rows for c in cols_list[1:]):
                frozen_header = self.frozen_view.horizontalHeader()
                main_header = self.main_view.horizontalHeader()
                frozen_selected_cols = [c for c in total_cols if c in frozen_cols]
                main_selected_cols = [c for c in total_cols if c not in frozen_cols]
                frozen_selected_cols.sort(key=lambda c: frozen_header.visualIndex(c))
                main_selected_cols.sort(key=lambda c: main_header.visualIndex(c))
                ordered_cols = frozen_selected_cols + main_selected_cols
                rows_order = sorted(base_rows)
                can_copy = True

        # 散点图条件：两列且行集合一致
        plot_enabled = False
        x_col = y_col = None
        plot_rows = []
        if len(total_cols) == 2 and can_copy and len(rows_order) >= 2:
            cols_list = list(total_cols)
            frozen_sel = [c for c in cols_list if c in frozen_cols]
            main_sel = [c for c in cols_list if c not in frozen_cols]

            if len(frozen_sel) == 2:
                header = self.frozen_view.horizontalHeader()
                frozen_sel.sort(key=lambda c: header.visualIndex(c))
                x_col, y_col = frozen_sel[0], frozen_sel[1]
            elif len(main_sel) == 2:
                header = self.main_view.horizontalHeader()
                main_sel.sort(key=lambda c: header.visualIndex(c))
                x_col, y_col = main_sel[0], main_sel[1]
            elif len(frozen_sel) == 1 and len(main_sel) == 1:
                x_col, y_col = frozen_sel[0], main_sel[0]

            if x_col is not None and y_col is not None:
                plot_rows = rows_order
                plot_enabled = True

        menu = QMenu(self)

        scatter_actions_added = False
        plot_candidate_cols = total_cols
        act1 = act2 = None
        if len(plot_candidate_cols) == 2:
            cols_list = sorted(list(plot_candidate_cols))
            if plot_enabled and x_col is not None and y_col is not None:
                x_show, y_show = x_col, y_col
            else:
                x_show, y_show = cols_list[0], cols_list[1]
            x_name = self.model.headerData(x_show, Qt.Horizontal, Qt.DisplayRole).replace('\n', ' ')
            y_name = self.model.headerData(y_show, Qt.Horizontal, Qt.DisplayRole).replace('\n', ' ')

            act1 = QAction(f"绘制x/y图，x={x_name}，y={y_name}", menu)
            act2 = QAction(f"绘制x/y图，x={y_name}，y={x_name}", menu)
            if plot_enabled and x_col is not None and y_col is not None:
                act1.triggered.connect(lambda _checked=False, rows=plot_rows, x=x_col, y=y_col: self._plot_xy_scatter(x, y, rows=rows))
                act2.triggered.connect(lambda _checked=False, rows=plot_rows, x=x_col, y=y_col: self._plot_xy_scatter(y, x, rows=rows))
                act1.setEnabled(True)
                act2.setEnabled(True)
                menu.addAction(act1)
                menu.addAction(act2)
                scatter_actions_added = True
            else:
                act1.setEnabled(False)
                act2.setEnabled(False)

        act_copy_selected = QAction("复制所选数据到剪贴板", menu)
        act_copy_selected.setEnabled(can_copy)
        if can_copy:
            act_copy_selected.triggered.connect(lambda: self._copy_selected_to_clipboard(ordered_cols, rows_order))

        act_copy_all = QAction("复制表内所有数据到剪贴板", menu)
        enable_all = (self._df is not None and self._df.shape[0] > 0 and self._df.shape[1] > 0)
        act_copy_all.setEnabled(enable_all)
        if enable_all:
            act_copy_all.triggered.connect(self._copy_all_to_clipboard)

        if scatter_actions_added:
            menu.addSeparator()
            menu.addAction(act_copy_selected)
            menu.addAction(act_copy_all)
        else:
            menu.addAction(act_copy_selected)
            menu.addAction(act_copy_all)
            if act1 is not None and act2 is not None:
                menu.addSeparator()
                menu.addAction(act1)
                menu.addAction(act2)

        menu.exec(view.mapToGlobal(pos))

    def _show_plot_menu(self, pos, view, x_col, y_col, rows, enabled=True):
        x_name = self.model.headerData(x_col, Qt.Horizontal, Qt.DisplayRole).replace('\n', ' ')
        y_name = self.model.headerData(y_col, Qt.Horizontal, Qt.DisplayRole).replace('\n', ' ')
        menu = QMenu(self)
        act1 = QAction(f"绘制x/y图，x={x_name}，y={y_name}", menu)
        act1.triggered.connect(lambda _checked=False, rows=rows, x=x_col, y=y_col: self._plot_xy_scatter(x, y, rows=rows))
        act1.setEnabled(enabled)
        act2 = QAction(f"绘制x/y图，x={y_name}，y={x_name}", menu)
        act2.triggered.connect(lambda _checked=False, rows=rows, x=x_col, y=y_col: self._plot_xy_scatter(y, x, rows=rows))
        act2.setEnabled(enabled)
        menu.addAction(act1)
        menu.addAction(act2)
        menu.exec(view.mapToGlobal(pos))

    def _plot_xy_scatter(self, x_col_idx, y_col_idx, rows=None, start_row=None, num_rows=None):
        """绘制x/y散点图"""
        try:
            if rows is None:
                if start_row is None or num_rows is None:
                    return
                row_indexer = slice(start_row, start_row + num_rows)
            else:
                row_indexer = rows

            x_data_series = pd.to_numeric(self._df.iloc[row_indexer, x_col_idx], errors='coerce')
            y_data_series = pd.to_numeric(self._df.iloc[row_indexer, y_col_idx], errors='coerce')

            if x_data_series.isnull().any() or y_data_series.isnull().any():
                QMessageBox.warning(self, "提示", "所选数据包含非数值，无法绘制散点图")
                return

            x_header = self.model.headerData(x_col_idx, Qt.Horizontal, Qt.DisplayRole).replace('\n', ' ')
            y_header = self.model.headerData(y_col_idx, Qt.Horizontal, Qt.DisplayRole).replace('\n', ' ')

            plot_dialog = XYScatterPlotDialog(x_data_series.to_numpy(), y_data_series.to_numpy(), x_header, y_header, self)
            self.scatter_plot_windows.append(plot_dialog)
            plot_dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"绘制散点图失败: {e}")

    def _copy_selected_to_clipboard(self, ordered_cols, rows_order):
        """复制所选数据到剪贴板"""
        if not ordered_cols or not rows_order:
            return
        var_names = [str(self._df.columns[c]) for c in ordered_cols]
        units = [self.units.get(name, '') for name in var_names]

        lines = []
        lines.append('\t'.join(var_names))
        lines.append('\t'.join(units))

        for r in rows_order:
            row_vals = []
            for c in ordered_cols:
                val = self._df.iloc[r, c]
                if pd.isna(val):
                    row_vals.append("")
                else:
                    row_vals.append(str(val))
            lines.append('\t'.join(row_vals))

        text = '\n'.join(lines)
        QApplication.clipboard().setText(text)

    def _copy_all_to_clipboard(self):
        """复制表内所有数据到剪贴板（包含单位行）"""
        if self._df is None or self._df.shape[0] == 0 or self._df.shape[1] == 0:
            return

        frozen_cols = set(self._df.columns.get_loc(col) for col in self.frozen_columns)
        frozen_header = self.frozen_view.horizontalHeader()
        main_header = self.main_view.horizontalHeader()

        all_cols = list(range(self._df.shape[1]))
        frozen_list = [c for c in all_cols if c in frozen_cols]
        main_list = [c for c in all_cols if c not in frozen_cols]
        frozen_list.sort(key=lambda c: frozen_header.visualIndex(c))
        main_list.sort(key=lambda c: main_header.visualIndex(c))
        ordered_cols = frozen_list + main_list

        var_names = [str(self._df.columns[c]) for c in ordered_cols]
        units = [self.units.get(name, '') for name in var_names]

        lines = []
        lines.append('\t'.join(var_names))
        lines.append('\t'.join(units))

        for r in range(self._df.shape[0]):
            row_vals = []
            for c in ordered_cols:
                val = self._df.iloc[r, c]
                if pd.isna(val):
                    row_vals.append("")
                else:
                    row_vals.append(str(val))
            lines.append('\t'.join(row_vals))

        text = '\n'.join(lines)
        QApplication.clipboard().setText(text)

    def _connect_signals(self):
        if self.model:
            self.main_view.selectionModel().selectionChanged.connect(self._update_highlights_main)
            self.frozen_view.selectionModel().selectionChanged.connect(self._update_highlights_frozen)

    def save_geom(self):
        if self.parent() and hasattr(self.parent(), 'data_table_geometry'):
            self.parent().data_table_geometry = self.saveGeometry()

    def load_geom(self):
        if self.parent() and hasattr(self.parent(), 'data_table_geometry') and self.parent().data_table_geometry is not None:
            geom = self.parent().data_table_geometry
            self.restoreGeometry(geom)

    def closeEvent(self, event):
        for win in self.scatter_plot_windows[:]:
            try:
                # 尝试访问窗口属性来检查是否有效
                if hasattr(win, 'isVisible'):
                    win.close()
            except RuntimeError:
                # 窗口已经被删除，跳过
                pass
        # print(self._skip_close_confirmation)  # 调试用，已注释
        if not (self._skip_close_confirmation) and len(self._df.columns) >= 4:
            reply = QMessageBox.question(self, "确认关闭", "是否清除所有列表，并关闭数值变量表窗口？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            # if user did not confirm to close the window
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        self.set_skip_close_confirmation(False)

        self.scatter_plot_windows.clear()
        
        # 其他清理代码保持不变...
        self.save_geom()
        self._df = pd.DataFrame()
        self.main_view.setModel(None)
        self.frozen_view.setModel(None)
        self._instance = None
        self._saved_scroll_pos = None
        self.frozen_columns = []  
        self.hide()
        event.accept()

    def set_skip_close_confirmation(self, status: bool):
        self._skip_close_confirmation = status

    def has_column(self, var_name: str) -> bool:
        return var_name in self._df.columns

    def add_series(self, var_name: str, data: pd.Series):
        self._add_variable_to_table(var_name, data)

    def _update_views(self):
        if self.model is None:
            return
        frozen_count = 0
        for col in range(self.model.columnCount()):
            var_name = self._df.columns[col]
            if var_name in self.frozen_columns:
                self.main_view.setColumnHidden(col, True)
                self.frozen_view.setColumnHidden(col, False)
                frozen_count += 1
            else:
                self.main_view.setColumnHidden(col, False)
                self.frozen_view.setColumnHidden(col, True)
        if frozen_count == 0:
            self.frozen_view.hide()
        else:
            self.frozen_view.show()
        if frozen_count > 0:
            self.frozen_view.verticalHeader().setVisible(True)
            self.main_view.verticalHeader().setVisible(False)
        else:
            self.frozen_view.verticalHeader().setVisible(False)
            self.main_view.verticalHeader().setVisible(True)

    def _get_full_visual_order(self):
        """获取所有列的完整视觉顺序（从左到右）"""
        full_order = []
        
        # 获取冻结区的视觉顺序
        frozen_header = self.frozen_view.horizontalHeader()
        for visual_idx in range(frozen_header.count()):
            logical_idx = frozen_header.logicalIndex(visual_idx)
            if not self.frozen_view.isColumnHidden(logical_idx):
                col_name = self._df.columns[logical_idx]
                full_order.append(col_name)
        
        # 获取非冻结区的视觉顺序
        main_header = self.main_view.horizontalHeader()
        for visual_idx in range(main_header.count()):
            logical_idx = main_header.logicalIndex(visual_idx)
            if not self.main_view.isColumnHidden(logical_idx):
                col_name = self._df.columns[logical_idx]
                full_order.append(col_name)
        
        return full_order
    
    def _restore_visual_order_after_model_change(self, old_visual_order, new_logical_order):
        """在模型改变后恢复视觉顺序"""
        # 创建从列名到新逻辑索引的映射
        name_to_new_logical = {name: idx for idx, name in enumerate(new_logical_order)}
        
        # 获取冻结区和非冻结区的表头
        frozen_header = self.frozen_view.horizontalHeader()
        main_header = self.main_view.horizontalHeader()
        
        # 按照旧的视觉顺序重新排列列
        current_visual_index = 0
        
        # 处理冻结区的列
        for col_name in old_visual_order:
            if col_name in self.frozen_columns:
                logical_idx = name_to_new_logical[col_name]
                current_visual_idx = frozen_header.visualIndex(logical_idx)
                if current_visual_idx != current_visual_index:
                    frozen_header.moveSection(current_visual_idx, current_visual_index)
                current_visual_index += 1
        
        # 重置视觉索引计数器，开始处理非冻结区
        current_visual_index = 0
        
        # 处理非冻结区的列
        for col_name in old_visual_order:
            if col_name not in self.frozen_columns:
                logical_idx = name_to_new_logical[col_name]
                current_visual_idx = main_header.visualIndex(logical_idx)
                if current_visual_idx != current_visual_index:
                    main_header.moveSection(current_visual_idx, current_visual_index)
                current_visual_index += 1

    def freeze_column(self, logical_col):
        var_name = self._df.columns[logical_col]
        if var_name not in self.frozen_columns:
            # --- 开始修改 ---
            col_width = self.main_view.columnWidth(logical_col)
            current_sizes = self.splitter.sizes()
            frozen_width, main_width = current_sizes[0], current_sizes[1]

            # 新增逻辑：检查这是否是第一个被冻结的列
            if not self.frozen_columns:
                # 如果是第一个，设置固定宽度为 self.user_left_width
                global FROZEN_VIEW_WIDTH_DEFAULT
                new_frozen_width = FROZEN_VIEW_WIDTH_DEFAULT
                total_width = frozen_width + main_width
                new_main_width = total_width - new_frozen_width
            else:
                # 如果不是第一个，则累加列宽
                new_frozen_width = frozen_width + col_width
                new_main_width = main_width - col_width
            
            self.splitter.setSizes([new_frozen_width, new_main_width])
            self.user_left_width = new_frozen_width

            # 获取当前所有列的完整视觉顺序
            full_visual_order = self._get_full_visual_order()
            
            # 将要冻结的列添加到冻结列列表
            self.frozen_columns.append(var_name)
            
            # 重新构建列顺序
            new_column_order = []
            
            # 按照完整视觉顺序添加列，但冻结列在前，非冻结列在后
            for col in full_visual_order:
                if col in self.frozen_columns and col not in new_column_order:
                    new_column_order.append(col)
            
            for col in full_visual_order:
                if col not in self.frozen_columns and col not in new_column_order:
                    new_column_order.append(col)
            
            # 重新排列DataFrame
            self._df = self._df[new_column_order]
            self.model = PandasTableModel(self._df, self.units)
            self.main_view.setModel(self.model)
            self.frozen_view.setModel(self.model)
            self._connect_signals()
            self._update_views()
            
            # 重新设置模型后，恢复用户调整的视觉顺序
            self._restore_visual_order_after_model_change(full_visual_order, new_column_order)


    def unfreeze_column(self, logical_col):
        var_name = self._df.columns[logical_col]
        if var_name in self.frozen_columns:
            # --- 开始修改 ---
            col_width = self.frozen_view.columnWidth(logical_col)
            current_sizes = self.splitter.sizes()
            frozen_width, main_width = current_sizes[0], current_sizes[1]

            # 新增逻辑：检查解冻后是否只剩一列
            # 注意：此时 self.frozen_columns 尚未移除元素，所以判断长度是否为 2
            if len(self.frozen_columns) == 2:
                # 如果解冻后只剩一列，将宽度重置为 self.user_left_width
                global FROZEN_VIEW_WIDTH_DEFAULT
                new_frozen_width = FROZEN_VIEW_WIDTH_DEFAULT
                total_width = frozen_width + main_width
                new_main_width = total_width - new_frozen_width
            else:
                # 其他情况（解冻后剩 0 列或多于 1 列），则减去列宽
                new_frozen_width = max(0, frozen_width - col_width)
                new_main_width = main_width + col_width

            self.splitter.setSizes([new_frozen_width, new_main_width])
            self.user_left_width = new_frozen_width

            # 获取当前所有列的完整视觉顺序
            full_visual_order = self._get_full_visual_order()
            
            # 将要解冻的列从冻结列列表中移除
            self.frozen_columns.remove(var_name)
            
            # 重新构建列顺序
            new_column_order = []
            
            # 按照完整视觉顺序添加列，但冻结列在前，非冻结列在后
            for col in full_visual_order:
                if col in self.frozen_columns and col not in new_column_order:
                    new_column_order.append(col)
            
            for col in full_visual_order:
                if col not in self.frozen_columns and col not in new_column_order:
                    new_column_order.append(col)
            
            # 重新排列DataFrame
            self._df = self._df[new_column_order]
            self.model = PandasTableModel(self._df, self.units)
            self.main_view.setModel(self.model)
            self.frozen_view.setModel(self.model)
            self._connect_signals()
            self._update_views()
            
            # 重新设置模型后，恢复用户调整的视觉顺序
            self._restore_visual_order_after_model_change(full_visual_order, new_column_order)


    def _sync_row_heights(self, logicalIndex, oldSize, newSize):
        sender = self.sender()
        if sender == self.main_view.verticalHeader():
            self.frozen_view.setRowHeight(logicalIndex, newSize)
        elif sender == self.sender() == self.frozen_view.verticalHeader():
            self.main_view.setRowHeight(logicalIndex, newSize)
    
    def _sync_column_order(self, logicalIndex, oldVisualIndex, newVisualIndex):
        """
        保持冻结区和非冻结区的列顺序一致。
        注意：moveSection 的参数是 visualIndex，不是 logicalIndex。
        """
        frozen_header = self.frozen_view.horizontalHeader()
        main_header = self.main_view.horizontalHeader() 
        # 如果拖动发生在主表头，就同步冻结表头
        sender_header = self.sender()
        if sender_header is main_header:
            frozen_header.blockSignals(True)
            try:
                frozen_header.moveSection(oldVisualIndex, newVisualIndex)
            finally:
                frozen_header.blockSignals(False)

        elif sender_header is frozen_header:
            main_header.blockSignals(True)
            try:
                main_header.moveSection(oldVisualIndex, newVisualIndex)
            finally:
                main_header.blockSignals(False)

    def _on_frozen_header_right_click(self, pos):
        self._on_header_right_click(pos, self.frozen_view)

    def _on_main_header_right_click(self, pos):
        self._on_header_right_click(pos, self.main_view)

    def _on_header_right_click(self, pos, view):
        header = view.horizontalHeader()
        logical_col = header.logicalIndexAt(pos)
        if logical_col < 0:
            return

        var_name = self._df.columns[logical_col]

        menu = QMenu(self)
        act_delete = menu.addAction(f"删除列 \"{var_name}\"")
        if var_name in self.frozen_columns:
            act_freeze = menu.addAction("解除冻结列")
        else:
            act_freeze = menu.addAction("冻结列")

        # 新增: 复制变量名
        act_copy = menu.addAction("复制变量名")
        act_copy.triggered.connect(lambda: QApplication.clipboard().setText(var_name))

        # 新增: 清空列表（全局操作，不依赖具体列）
        act_clear = menu.addAction("清空列表")
        act_clear.triggered.connect(self._clear_all_columns)

        selected = menu.exec(header.mapToGlobal(pos))
        if selected == act_delete:
            self._remove_column(logical_col)
        elif selected == act_freeze:
            if var_name in self.frozen_columns:
                self.unfreeze_column(logical_col)
            else:
                self.freeze_column(logical_col)

    def _clear_all_columns(self):
        reply = QMessageBox.question(self, "确认", "是否清空所有列？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        # 清空所有列
        while self.model.columnCount() > 0:
            self.model.removeColumns(0, 1)
        self._df = pd.DataFrame()
        self.frozen_columns = []
        self._update_views()

    def scroll_to_column(self, var_name: str):
        """滚动到指定变量名的列（可能在冻结区或普通区），不影响垂直滚动位置"""
        if var_name not in self._df.columns:
            return False
        
        # 获取列的索引
        col_idx = self._df.columns.get_loc(var_name)
        
        # 确定列在哪个视图（冻结区或普通区）
        if var_name in self.frozen_columns:
            view = self.frozen_view
        else:
            view = self.main_view
        
        # 获取水平头部
        header = view.horizontalHeader()
        
        # 获取列的视觉位置
        visual_idx = header.visualIndex(col_idx)
        
        # 计算列的位置和大小
        col_pos = 0
        for i in range(visual_idx):
            col_pos += header.sectionSize(header.logicalIndex(i))
        
        col_width = header.sectionSize(col_idx)
        
        # 获取当前水平滚动位置
        scroll_pos = view.horizontalScrollBar().value()
        
        # 计算需要的滚动位置，使列在视图中可见
        viewport_width = view.viewport().width()
        
        # 如果列在视图左侧之外
        if col_pos < scroll_pos:
            view.horizontalScrollBar().setValue(col_pos)
        # 如果列在视图右侧之外
        elif col_pos + col_width > scroll_pos + viewport_width:
            view.horizontalScrollBar().setValue(col_pos + col_width - viewport_width)
        
        return True
    
    def _remove_column(self, logical_col):
        var_name = self._df.columns[logical_col]

        # 如果要删除的列在冻结区，则执行与解冻相同的宽度调整策略
        if var_name in self.frozen_columns:
            # 1. 从 frozen_view 获取列宽
            col_width = self.frozen_view.columnWidth(logical_col)
            current_sizes = self.splitter.sizes()
            frozen_width, main_width = current_sizes[0], current_sizes[1]

            # 2. 应用特殊宽度逻辑：
            #    如果删除后只剩一列（即删除前有两列），则将剩余的冻结区宽度设为 150
            if len(self.frozen_columns) == 2:
                new_frozen_width = 150
                total_width = frozen_width + main_width
                new_main_width = total_width - new_frozen_width
            else:
                #    否则，直接减去被删除列的宽度
                new_frozen_width = max(0, frozen_width - col_width)
                new_main_width = main_width + col_width

            # 3. 应用新尺寸并更新用户偏好宽度
            self.splitter.setSizes([new_frozen_width, new_main_width])
            self.user_left_width = new_frozen_width

        # 从冻结列表中移除
        if var_name in self.frozen_columns:
            self.frozen_columns.remove(var_name)
        
        # 从DataFrame中删除列
        self._df.drop(columns=[var_name], inplace=True)

        # 刷新模型和视图
        self.model = PandasTableModel(self._df, self.units)
        self.main_view.setModel(self.model)
        self.frozen_view.setModel(self.model)
        self._connect_signals()
        self._update_views()

    def update_data(self, loader):
        """当主窗口重载数据时，更新此对话框中的数据"""
        if self.model is None or self._df.empty:
            return
        
        scroll_pos = self.main_view.verticalScrollBar().value()
        frozen_cols = self.frozen_columns.copy()
        current_cols = list(self._df.columns)
        
        # --- BUG修复 START ---
        
        # 创建一个新的DataFrame来保存更新后的数据
        new_df = pd.DataFrame()
        removed = []

        # 遍历当前表中的列
        for col in current_cols:
            if col in loader.df.columns:
                # 从新的加载器数据中复制完整的列
                # 这是关键修复：确保新DataFrame获得完整行数
                series = loader.df[col]
                new_df[col] = self._prepare_series_for_display(col, series)
            else:
                # 该列已从源文件中移除
                                              
                                                   
                removed.append(col)

        # 用新的、行数正确的DataFrame替换旧的
        self._df = new_df
        
        # --- BUG修复 END ---

        self.units = loader.units
        self.model = PandasTableModel(self._df, self.units)
        self.main_view.setModel(self.model)
        self.frozen_view.setModel(self.model)
        self._connect_signals()
        
        # 重新应用冻结列，确保它们仍然存在
        self.frozen_columns = [col for col in frozen_cols if col in self._df.columns]
        
        self._update_views()
        QTimer.singleShot(0, lambda: self.main_view.verticalScrollBar().setValue(scroll_pos))
        
        if removed:
            msg = f"以下变量已从数据中移除：{', '.join(removed)}"
            QMessageBox.information(self, "更新通知", msg)
        
        if self._df.empty:
            # 增加这行，避免在表格变空并关闭时弹出烦人的确认框
            self.set_skip_close_confirmation(True)
            self.close()


class LayoutInputDialog(QDialog):
    def __init__(self, 
                 max_rows: int = 4, 
                 max_cols: int = 2, 
                 cur_rows: int = 1, 
                 cur_cols: int = 1,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置图表的行、列数")
        self.max_rows = max_rows
        self.max_cols = max_cols

        form = QFormLayout(self)

        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, max_rows)
        self.row_spin.setValue(cur_rows)

        self.col_spin = QSpinBox()
        self.col_spin.setRange(1, max_cols)
        self.col_spin.setValue(cur_cols)

        form.addRow("行数：", self.row_spin)
        form.addRow("列数：", self.col_spin)

        btns = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        form.addRow(btns)

        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        QTimer.singleShot(0, self.row_spin.selectAll)
        
    def values(self):
        return self.row_spin.value(), self.col_spin.value()
    

class AxisDialog(QDialog):
    def __init__(self, axis, view_box, axis_type: str, plot_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"调整 {axis_type} 轴")
        self.axis = axis
        self.view_box = view_box
        self.axis_type = axis_type
        self.plot_widget = plot_widget

        # 创建输入字段
        self.min_input = QLineEdit(str(view_box.viewRange()[0 if axis_type == "X" else 1][0]))
        self.max_input = QLineEdit(str(view_box.viewRange()[0 if axis_type == "X" else 1][1]))
        self.tick_count_input = QLineEdit()
        self.tick_count_input.setPlaceholderText("留空自动计算")

        # 创建布局
        layout = QFormLayout()
        layout.addRow("最小值:", self.min_input)
        layout.addRow("最大值:", self.max_input)
        tick_label = QLabel("刻度数量:")
        layout.addRow(tick_label, self.tick_count_input)
        tick_label.setVisible(False)
        self.tick_count_input.setVisible(False)

        # 确定和取消按钮
        button_layout = QVBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(self.apply_changes)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # 添加全选第一个文本框的功能
        QTimer.singleShot(0, lambda: self.min_input.selectAll())

    def apply_changes(self):
        try:
            min_val = float(self.min_input.text())
            max_val = float(self.max_input.text())
            if min_val >= max_val:
                QMessageBox.warning(self, "错误", "最小值必须小于最大值")
                return

            # 处理刻度数量
            tick_count_text = self.tick_count_input.text().strip()
            if tick_count_text:
                tick_count = int(tick_count_text)
                if tick_count < 2:
                    QMessageBox.warning(self, "错误", "刻度数量必须大于等于 2")
                    return
            else:
                tick_count = None  # 自动
            global DEFAULT_PADDING_VAL_X, DEFAULT_PADDING_VAL_Y
            # 设置范围
            if self.axis_type == "X":
                self.view_box.setXRange(min_val, max_val, padding=DEFAULT_PADDING_VAL_X)
            else:
                self.view_box.setYRange(min_val, max_val, padding=DEFAULT_PADDING_VAL_Y)

            # 设置固定刻度
            if tick_count:
                step = (max_val - min_val) / (tick_count - 1)
                ticks = [(min_val + i * step, str(round(min_val + i * step, 6)))
                         for i in range(tick_count)]
                self.axis.setTicks([ticks])
            else:
                self.axis.setTicks(None)
            self.accept()

        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数值（最小值、最大值、刻度数量）")


# ---------------- 自定义 QTableWidget ----------------
class NoHoverDelegate(QStyledItemDelegate):
    """
    变量表自定义委托类（PyQt5版本）
    
    功能：
    1. 禁用鼠标悬停和焦点的视觉反馈，避免干扰用户操作
    2. 在变量名列左侧绘制彩色方块标识符（绿色=有效，橙色=常数，红色=无效）
    3. 方块不占用文本显示空间，最大化变量名显示长度
    4. 确保选中行文本高对比度显示（白色文字）
    """
    
    def paint(self, painter, option, index):
        """自定义单元格绘制逻辑"""
        from PyQt5.QtWidgets import QStyle
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QColor, QPen
        
        # 移除悬停和焦点状态，避免出现白色块和焦点框
        option.state &= ~QStyle.State_MouseOver
        option.state &= ~QStyle.State_HasFocus
        
        # 绘制背景（选中状态用高亮色，未选中用基础色）
        self._draw_background(painter, option)
        
        # 变量名列（第0列）：绘制彩色方块 + 文本
        if index.column() == 0:
            self._draw_variable_name_column(painter, option, index)
        # 其他列（单位、序号）：仅绘制文本
        else:
            self._draw_text_column(painter, option, index)
    
    def _draw_background(self, painter, option):
        """绘制单元格背景"""
        from PyQt5.QtWidgets import QStyle
        
        painter.save()
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.palette.base())
        painter.restore()
    
    def _get_validity_color(self, valid):
        """根据有效性返回对应的颜色"""
        from PyQt5.QtGui import QColor
        
        if valid == 1:
            return QColor(0, 200, 0)      # 鲜艳绿色（有效）
        elif valid == 0:
            return QColor(255, 140, 0)    # 橙色（常数）
        elif valid == -1:
            return QColor(255, 0, 0)      # 红色（无效）
        return None
    
    def _draw_variable_name_column(self, painter, option, index):
        """绘制变量名列（包含彩色方块标识符）"""
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QPen, QColor
        from PyQt5.QtWidgets import QStyle
        
        # 获取有效性和颜色
        valid = index.data(Qt.UserRole)
        color = self._get_validity_color(valid) if valid is not None else None
        
        # 获取原始变量名（存储在UserRole+1中）
        original_name = index.data(Qt.UserRole + 1)
        if not original_name:
            return
        
        # 如果有效性标识，绘制彩色方块
        text_rect = option.rect
        if color:
            # 计算方块位置和大小
            square_size = min(option.rect.height() - 4, 12)
            square_x = option.rect.left() + 3
            square_y = option.rect.top() + (option.rect.height() - square_size) // 2
            
            # 绘制方块
            painter.save()
            painter.setPen(QPen(color, 1))
            painter.setBrush(color)
            painter.drawRect(square_x, square_y, square_size, square_size)
            painter.restore()
            
            # 调整文本区域，为方块留出空间
            text_rect = QRect(option.rect)
            text_rect.setLeft(option.rect.left() + square_size + 8)
        else:
            # 无方块时，左侧留出小边距
            text_rect = option.rect.adjusted(6, 0, -6, 0)
        
        # 绘制文本
        self._draw_text(painter, option, original_name, text_rect)
    
    def _draw_text_column(self, painter, option, index):
        """绘制普通文本列（单位、序号）"""
        text = index.data(Qt.DisplayRole)
        if text is not None:
            text_rect = option.rect.adjusted(3, 0, -3, 0)
            self._draw_text(painter, option, str(text), text_rect)
    
    def _draw_text(self, painter, option, text, text_rect):
        """绘制文本，自动处理选中状态的颜色和文本省略"""
        from PyQt5.QtGui import QColor
        from PyQt5.QtWidgets import QStyle
        
        painter.save()
        
        # 选中时使用白色文字（高对比度），否则使用默认颜色
        if option.state & QStyle.State_Selected:
            painter.setPen(QColor(255, 255, 255))
        else:
            painter.setPen(option.palette.text().color())
        
        # 绘制文本，自动省略过长部分（...）
        elided_text = painter.fontMetrics().elidedText(
            text, 
            Qt.ElideRight, 
            text_rect.width()
        )
        painter.drawText(
            text_rect, 
            Qt.AlignVCenter | Qt.AlignLeft,
            elided_text
        )
        
        painter.restore()

class MyTableWidget(QTableWidget):
    """
    自定义表格控件类（PyQt5版本）
    扩展QTableWidget功能，支持拖拽、排序、彩色标识等自定义交互
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["变量名", "单位", "序号"])
        self.original_indices = {}  # 存储原始索引
        self._column_sort_order = {}  # 记录每列的当前排序状态：{column_index: order}

        # 设置自定义委托，从绘制层面彻底禁用悬停和焦点效果
        self.setItemDelegate(NoHoverDelegate(self))

        # 字体 
        header_font = self.horizontalHeader().font()
        header_font.setBold(False)  
        self.horizontalHeader().setFont(header_font)
        
        # 设置表格选择行为的样式
        # 保留未选中行的自定义背景色，仅在选中时使用高亮色
        self.setStyleSheet("""
            QTableWidget::item:selected {
                font-weight: normal;         /* 确保选中项字体也不加粗 */
            }
        """)

        # 默认列宽度：变量名:单位:序号 = 5:2:1
        total = 300
        self.setColumnWidth(0, int(total * 0.625))  # 变量名列
        self.setColumnWidth(1, int(total * 0.25))   # 单位列
        self.setColumnWidth(2, int(total * 0.125))  # 序号列

        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(False)  # 关闭自动拉伸最后一列
        self.verticalHeader().setVisible(False)  # 隐藏行号
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 允许多选
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # 禁用鼠标追踪和视口追踪，避免白色块跟随鼠标移动
        self.setMouseTracking(False)
        self.viewport().setMouseTracking(False)  # 同时禁用视口的鼠标跟踪
        
        # 禁用焦点指示器
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 禁用自动滚动（当鼠标向表格边缘移动时不会自动滚动）
        self.setAutoScroll(False)
        
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        
        # 禁用Qt自动排序，使用自定义排序
        self.setSortingEnabled(False)
        self.horizontalHeader().setSortIndicatorShown(True)
        self.horizontalHeader().sectionClicked.connect(self._handle_header_click)
          
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
    
    def _handle_header_click(self, logicalIndex):
        """自定义排序处理，确保有效性始终是第一优先级"""
        # 如果该列之前没有排序过，或者上次是降序，则设为升序；否则设为降序
        if logicalIndex not in self._column_sort_order:
            new_order = Qt.AscendingOrder  # 第一次点击：升序
        else:
            current_order = self._column_sort_order[logicalIndex]
            # 切换排序顺序
            new_order = Qt.DescendingOrder if current_order == Qt.AscendingOrder else Qt.AscendingOrder
        
        # 记录当前列的排序状态
        self._column_sort_order[logicalIndex] = new_order
        
        # 直接调用自定义排序方法（不使用sortByColumn，因为setSortingEnabled=False）
        self.sortItems(logicalIndex, new_order)
    
    def sortItems(self, column, order=Qt.AscendingOrder):
        """重写排序方法，确保有效性始终是第一优先级"""
        # 收集所有行数据（整行移动）
        rows = []
        for row in range(self.rowCount()):
            name_item = self.item(row, 0)
            unit_item = self.item(row, 1)
            index_item = self.item(row, 2)
            
            if name_item and unit_item and index_item:
                valid = name_item.data(Qt.UserRole)
                # 获取原始变量名（存储在UserRole+1中）
                original_name = name_item.data(Qt.UserRole + 1)
                rows.append({
                    'name': original_name if original_name else '',
                    'unit': unit_item.text(),
                    'index': index_item.data(Qt.DisplayRole),
                    'valid': valid if valid is not None else -999,
                })
        
        # 排序逻辑：
        # Level 1: 有效性降序（1 → 0 → -1，即有效的在前）
        # Level 2: 按选择的列升序或降序
        
        # 使用分组排序：先按有效性分组，再在组内排序
        from itertools import groupby
        
        # 先按有效性降序排序（保证有效的在前）
        rows.sort(key=lambda x: -x['valid'])
        
        # 按有效性分组，然后在每组内按第二级字段排序
        rows_sorted = []
        for valid_value, group in groupby(rows, key=lambda x: x['valid']):
            group_list = list(group)
            
            # 在组内按选择的列排序
            if column == 0:  # 变量名
                group_list.sort(
                    key=lambda x: x['name'].lower(),
                    reverse=(order == Qt.DescendingOrder)
                )
            elif column == 1:  # 单位
                group_list.sort(
                    key=lambda x: x['unit'].lower(),
                    reverse=(order == Qt.DescendingOrder)
                )
            elif column == 2:  # 序号
                group_list.sort(
                    key=lambda x: x['index'],
                    reverse=(order == Qt.DescendingOrder)
                )
            
            rows_sorted.extend(group_list)
        
        rows = rows_sorted
        
        # 重新填充表格（整行移动，包括颜色）
        for row, data in enumerate(rows):
            # 创建新的item（不含emoji，彩色方块由delegate绘制）
            valid_value = data['valid']
            original_name = data['name']
            
            name_item = QTableWidgetItem()  # 变量名列（文本留空，由delegate绘制）
            unit_item = QTableWidgetItem(data['unit'])
            index_item = QTableWidgetItem()
            index_item.setData(Qt.DisplayRole, data['index'])
            
            # 存储原始变量名到UserRole+1（用于delegate绘制和所有操作）
            name_item.setData(Qt.UserRole + 1, original_name)
            
            # 设置有效性数据（用于排序和delegate绘制彩色方块）
            name_item.setData(Qt.UserRole, valid_value)
            unit_item.setData(Qt.UserRole, valid_value)
            index_item.setData(Qt.UserRole, valid_value)
            
            # 设置到表格
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, unit_item)
            self.setItem(row, 2, index_item)
        
        # 更新排序指示器
        self.horizontalHeader().setSortIndicator(column, order)

    def _show_context_menu(self, pos):
        index = self.indexAt(pos)
        if not index.isValid():
            return

        item = self.item(index.row(), 0)  # 变量名在第0列
        # 获取原始变量名（不含彩色方框标识符）
        var_name = item.data(Qt.UserRole + 1)
        if not var_name:
            # 兼容旧数据
            var_name = item.text()

        menu = QMenu(self)
        
        # a. 添加至数值变量表
        act_add_table = QAction("添加至数值变量表", menu)
        act_add_table.triggered.connect(lambda: self._add_to_data_table(var_name))
        menu.addAction(act_add_table)
        
        # b. 添加至空白绘图区
        act_add_blank_plot = QAction("添加至空白绘图区", menu)
        act_add_blank_plot.triggered.connect(lambda: self._add_to_blank_plot(var_name))
        menu.addAction(act_add_blank_plot)
        
        # c. 复制变量名
        act_copy = QAction("复制变量名", menu)
        act_copy.triggered.connect(lambda: QApplication.clipboard().setText(var_name))
        menu.addAction(act_copy)
        
        menu.exec(self.mapToGlobal(pos))

    def _add_to_data_table(self, var_name: str):
        main_window = self.window()
        loader = None
        if isinstance(main_window, MainWindow) and hasattr(main_window, "loader"):
            loader = main_window.loader
        if loader is None:
            loader = get_main_loader()

        if loader is None:
            QMessageBox.warning(self, "错误", "没有加载数据")
            return
        if var_name not in loader.df.columns:
            QMessageBox.warning(self, "错误", f"变量 '{var_name}' 不存在")
            return
        data = loader.df[var_name]
        DataTableDialog.popup(var_name, data, self)

    def _add_to_blank_plot(self, var_name: str):
        main_window = self.window()
        if not isinstance(main_window, MainWindow):
            main_window = get_main_window()
        if main_window is None:
            QMessageBox.warning(self, "错误", "未找到主窗口实例")
            return

        loader = getattr(main_window, "loader", None)

        if loader is None:
            QMessageBox.warning(self, "错误", "没有加载数据")
            return
        if var_name not in loader.df.columns:
            QMessageBox.warning(self, "错误", f"变量 '{var_name}' 不存在")
            return

        # 1. 首先判断该变量是否含有有效值
        if loader.df_validity.get(var_name, -1) < 0:
            QMessageBox.warning(self, "错误", f"变量 '{var_name}' 没有足够有效数值")
            return

        # 2. 在用户设置的当前布局(mxn)中查找空白绘图区，无论绘图区整体是否可见
        blank_plot = None
        rows, cols = main_window._plot_row_current, main_window._plot_col_current
        max_cols = main_window._plot_col_max_default  # 这是完整网格的列数，用于计算索引

        for idx, container in enumerate(main_window.plot_widgets):
            # 根据一维索引计算其在完整网格(pxq)中的二维坐标(r, c)
            r = idx // max_cols
            c = idx % max_cols

            # 判断这个坐标是否在用户当前的(mxn)布局内
            if r < rows and c < cols:
                # 如果在布局内，再判断是否为空白
                if container.plot_widget.y_name == '' and container.plot_widget.curve is None:
                    blank_plot = container.plot_widget
                    break  # 找到第一个可用的就退出

        if blank_plot is None:
            QMessageBox.warning(self, "提示", "当前布局中已无空白绘图区")
            return

        # 3. 判断绘图区域整体是否被隐藏，并提示用户        
        _delay = 0
        if not main_window._plot_area_visible:
            reply = QMessageBox.Yes
            if reply == QMessageBox.Yes:
                # 激活绘图区，同步按钮状态
                main_window.toggle_plot_btn.setChecked(False)
                _delay = 300
            else:
                return  # 用户选择不激活，则不执行后续操作
        
        def _job():
            # 4. 将变量添加至空白图中
            success = blank_plot.plot_variable(var_name)
            if success:
                main_window.update_mark_stats()

        QTimer.singleShot(_delay, _job) 

    def startDrag(self, supportedActions):
        indexes = self.selectedIndexes()
        if not indexes:
            return
        
        rows = sorted({idx.row() for idx in indexes})
        var_names: list[str] = []
        for row in rows:
            item = self.item(row, 0)
            if item is None:
                continue
            var_name = item.data(Qt.UserRole + 1)
            if not var_name:
                var_name = item.text()
            if var_name:
                var_names.append(var_name)
        
        if not var_names:
            return
        
        drag_text = ";;".join(var_names)
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(drag_text)
        drag.setMimeData(mime_data)

        preview_pixmap = self._create_drag_pixmap(var_names)
        if preview_pixmap:
            drag.setPixmap(preview_pixmap)
            hot_spot = QPoint(preview_pixmap.width() // 2, preview_pixmap.height() // 2)
            drag.setHotSpot(hot_spot)

        drag.exec(Qt.MoveAction)

    def _create_drag_pixmap(self, var_names: list[str]) -> QPixmap | None:
        if not var_names:
            return None

        font = self.font()
        metrics = QFontMetrics(font)
        bullet_names = [f"• {name}" for name in var_names]
        max_visible = 8
        display_lines = bullet_names[:max_visible]
        if len(bullet_names) > max_visible:
            display_lines.append(f"… 共{len(var_names)}项")

        text_width = max((metrics.horizontalAdvance(line) for line in display_lines), default=80)
        margin = 12
        line_height = metrics.lineSpacing()
        width = max(140, text_width + margin * 2)
        height = line_height * len(display_lines) + margin * 2

        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#2b6def"), 2))
        painter.setBrush(QColor(255, 255, 255, 240))
        painter.drawRoundedRect(QRectF(1, 1, width - 2, height - 2), 10, 10)
        painter.setPen(QColor("#1f1f1f"))

        y = margin + metrics.ascent()
        for line in display_lines:
            painter.drawText(QPointF(margin, y), line)
            y += line_height

        painter.end()
        return pixmap

    def mouseDoubleClickEvent(self, event):
        index = self.indexAt(event.pos())
        if not index.isValid():
            super().mouseDoubleClickEvent(event)
            return

        row = index.row()
        item = self.item(row, 0)
        # 获取原始变量名（存储在UserRole+1中）
        var_name = item.data(Qt.UserRole + 1)
        if not var_name:
            # 兼容旧数据
            var_name = item.text()
        
        main_window = self.window()
        if not hasattr(main_window, 'loader') or main_window.loader is None:
            return

        series = main_window.loader.df[var_name]

        # 弹出数值变量表
        dlg = DataTableDialog.popup(var_name, series, parent=main_window)
        
        # 滚动到新添加的列
        QTimer.singleShot(100, lambda: dlg.scroll_to_column(var_name))

        super().mouseDoubleClickEvent(event)

    def populate(self, var_names, units, validity):
        """填充变量列表（三列格式：变量名、单位、序号）"""
        self.clearContents()
        self.setRowCount(len(var_names))
        self.original_indices = {}  # 存储原始索引

        # 创建列表并排序: 先按validity降序, 然后按原顺序 (使用stable sort)
        items = list(zip(var_names, [units.get(v, '') for v in var_names], [validity.get(v, -1) for v in var_names]))
        # 为了保持相同validity的原顺序, 我们用enumerate添加index
        indexed_items = [(valid, idx, name, unit) for idx, (name, unit, valid) in enumerate(items)]
        indexed_items.sort(key=lambda x: (-x[0], x[1]))  # valid desc (-valid), then original idx asc
        sorted_names = [name for valid, idx, name, unit in indexed_items]
        sorted_units = [unit for valid, idx, name, unit in indexed_items]
        sorted_valids = [valid for valid, idx, name, unit in indexed_items]
        sorted_indices = [idx for valid, idx, name, unit in indexed_items]

        for row, (name, unit, valid, original_idx) in enumerate(zip(sorted_names, sorted_units, sorted_valids, sorted_indices)):
            # 存储原始索引
            self.original_indices[name] = original_idx + 1  # 序号从1开始
            
            # 创建item（不含emoji，彩色方块由delegate绘制）
            name_item = QTableWidgetItem()  # 变量名列（文本留空，由delegate绘制）
            unit_item = QTableWidgetItem(unit)
            index_item = QTableWidgetItem()
            index_item.setData(Qt.DisplayRole, original_idx + 1)  # 序号从1开始
            
            # 存储原始变量名到UserRole+1（用于delegate绘制和所有操作）
            name_item.setData(Qt.UserRole + 1, name)
            
            # 设置有效性数据（用于排序和delegate绘制彩色方块）
            name_item.setData(Qt.UserRole, valid)
            unit_item.setData(Qt.UserRole, valid)
            index_item.setData(Qt.UserRole, valid)
            
            # 设置到表格
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, unit_item)
            self.setItem(row, 2, index_item)


# 新增自定义 ViewBox 类
class CustomViewBox(pg.ViewBox):
    """
    自定义视图框类
    扩展pyqtgraph的ViewBox功能，添加自定义右键菜单
    支持跳转到数据表格、清除图表等操作
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_x = None  # 记录右键点击时的 x 坐标
        self.plot_widget = None  # 将在 DraggableGraphicsLayoutWidget 中设置
        self.is_cursor_pinned = False  # 记录cursor是否被固定

    def getMenu(self, ev):
        # 记录鼠标位置的 x 值
        scene_pos = ev.scenePos()
        view_pos = self.mapSceneToView(scene_pos)
        self.context_x = view_pos.x()

        # 获取默认菜单
        menu = super().getMenu(ev)
        if menu is None:
            return None

        # 屏蔽 "Mouse Mode"
        for act in menu.actions():
            if act.text() == 'Mouse Mode':
                act.setVisible(False)
            elif act.text() == 'Plot Options':
                # 子菜单下屏蔽 "Transforms"（注意：实际文本为 "Transforms"）
                submenu = act.menu()
                if submenu:
                    for subact in submenu.actions():
                        if subact.text() == 'Transforms':
                            subact.setVisible(False)

        existing_texts = [act.text() for act in menu.actions()]

        # 添加新 action: "Jump to Data" (检查是否已存在以避免重复)
        if "Jump to Data" not in existing_texts:
            jump_act = QAction("Jump to Data", menu)
            jump_act.triggered.connect(self.trigger_jump_to_data)
            if menu.actions():
                menu.insertAction(menu.actions()[0], jump_act)
            else:
                menu.addAction(jump_act)
        
        # 添加"自动调节y轴"选项（第二个位置，作用于所有plot）
        if "Autoscale in x-Range" not in existing_texts:
            auto_y_act = QAction("Autoscale in x-Range", menu)
            auto_y_act.triggered.connect(self.trigger_auto_y_axis)
            # 在Jump to Data之后插入（第二个位置）
            if len(menu.actions()) >= 1:
                menu.insertAction(menu.actions()[1] if len(menu.actions()) > 1 else None, auto_y_act)
            else:
                menu.addAction(auto_y_act)
        actions_to_remove = []
        for action in menu.actions():
            if action.text() in ["Pin Cursor", "Free Cursor", "Cursor Mode"]:
                actions_to_remove.append(action)
        for action in actions_to_remove:
            menu.removeAction(action)

        cursor_enabled = False
        if self.plot_widget and self.plot_widget.window() and hasattr(self.plot_widget.window(), 'cursor_btn'):
            cursor_enabled = self.plot_widget.window().cursor_btn.isChecked()

        cursor_menu = QMenu("Cursor Mode", menu)
        cursor_menu.setEnabled(cursor_enabled)
        cursor_group = QActionGroup(cursor_menu)
        cursor_group.setExclusive(True)
        current_mode = "1 free cursor"
        if self.plot_widget and self.plot_widget.window() and hasattr(self.plot_widget.window(), 'cursor_mode'):
            current_mode = self.plot_widget.window().cursor_mode

        for mode_text in ["1 free cursor", "1 anchored cursor", "2 anchored cursor"]:
            mode_act = QAction(mode_text, cursor_menu)
            mode_act.setCheckable(True)
            mode_act.setChecked(mode_text == current_mode)
            mode_act.setEnabled(cursor_enabled)
            mode_act.triggered.connect(lambda checked, m=mode_text: self.trigger_cursor_mode(m))
            cursor_group.addAction(mode_act)
            cursor_menu.addAction(mode_act)

        if len(menu.actions()) >= 2:
            menu.insertMenu(menu.actions()[2] if len(menu.actions()) > 2 else None, cursor_menu)
        else:
            menu.addMenu(cursor_menu)

        #  Cursor Value / (,Pin Cursor)
        # 
        actions_to_remove = []
        for action in menu.actions():
            if action.text() in ["Show Cursor Value", "Hide Cursor Value"]:
                actions_to_remove.append(action)
        for action in actions_to_remove:
            menu.removeAction(action)

        # cursor
        cursor_enabled = False
        if self.plot_widget and self.plot_widget.window() and hasattr(self.plot_widget.window(), 'cursor_btn'):
            cursor_enabled = self.plot_widget.window().cursor_btn.isChecked()

        # 
        values_hidden = False
        if self.plot_widget and self.plot_widget.window() and hasattr(self.plot_widget.window(), 'cursor_values_hidden'):
            values_hidden = self.plot_widget.window().cursor_values_hidden

        if values_hidden:
            cursor_value_act = QAction("Show Cursor Value", menu)
            cursor_value_act.triggered.connect(self.trigger_show_cursor_value)
        else:
            cursor_value_act = QAction("Hide Cursor Value", menu)
            cursor_value_act.triggered.connect(self.trigger_hide_cursor_value)

        # cursor
        cursor_value_act.setEnabled(cursor_enabled)

        # Pin Cursor()
        if len(menu.actions()) >= 3:
            menu.insertAction(menu.actions()[3] if len(menu.actions()) > 3 else None, cursor_value_act)
        else:
            menu.addAction(cursor_value_act)

        copy_act = None
        for act in menu.actions():
            if act.text() == "Copy Name":
                copy_act = act
                break
        if copy_act is None:
            copy_act = QAction("Copy Name", menu)
            copy_act.triggered.connect(self.trigger_copy_name)
            menu.addAction(copy_act)

        curves = getattr(self.plot_widget, 'curves', {}) if self.plot_widget else {}
        has_single = bool(
            self.plot_widget
            and getattr(self.plot_widget, 'curve', None) is not None
            and bool(getattr(self.plot_widget, 'y_name', ''))
        )
        has_multi = bool(curves)
        has_data = has_single or has_multi
        copy_act.setEnabled(has_data)
        
        # 添加"绘图变量编辑器"选项
        if "Plot Variable Editor" not in existing_texts:
            editor_act = QAction("Plot Variable Editor", menu)
            editor_act.triggered.connect(self.trigger_variable_editor)
            # 检查是否有数据可以编辑
            # has_data = bool(
            #     self.plot_widget 
            #     and (getattr(self.plot_widget, 'curve', None) is not None 
            #          or getattr(self.plot_widget, 'curves', None))
            # )
            # editor_act.setEnabled(has_data)
            menu.addAction(editor_act)
                
        # 将 "Clear Plot" action 添加到菜单末尾
        if "Clear Plot" not in existing_texts:
            menu.addSeparator()  # 在末尾添加一个分隔符
            clear_act = QAction("Clear Plot", menu)
            clear_act.triggered.connect(self.trigger_clear_plot)
            menu.addAction(clear_act)  # addAction 会将按钮添加到末尾

        return menu

    def trigger_jump_to_data(self):
        if self.plot_widget:
            self.plot_widget.jump_to_data_impl(self.context_x)
            
    def trigger_clear_plot(self):
        if self.plot_widget:
            self.plot_widget.clear_plot_item()
            if self.plot_widget.window():
                self.plot_widget.window().update_mark_stats()
    
    def trigger_pin_cursor(self):
        """固定cursor到最近的数据点"""
        if self.plot_widget:
            self.plot_widget.pin_cursor(self.context_x)
    
    def trigger_free_cursor(self):
        """解除cursor固定，恢复跟随鼠标"""
        if self.plot_widget:
            self.plot_widget.free_cursor()

    def trigger_cursor_mode(self, mode: str):
        if self.plot_widget and self.plot_widget.window():
            self.plot_widget.window().set_cursor_mode(
                mode,
                source_plot=self.plot_widget,
                context_x=self.context_x,
            )


    def trigger_copy_name(self):
        """复制当前绘图变量名到剪贴板（无数据则不执行）"""
        if not self.plot_widget:
            return
        
        # 收集所有变量名
        var_names = []
        
        # 检查是否是多曲线模式
        if hasattr(self.plot_widget, 'is_multi_curve_mode') and self.plot_widget.is_multi_curve_mode:
            # 多曲线模式：复制所有曲线的变量名
            if hasattr(self.plot_widget, 'curves') and self.plot_widget.curves:
                var_names = list(self.plot_widget.curves.keys())
        else:
            # 单曲线模式：复制单个变量名
            var_name = getattr(self.plot_widget, 'y_name', '')
            if var_name:
                var_names = [var_name]
        
        if not var_names:
            return
        
        # 将变量名用空格分隔
        clipboard_text = ' '.join(var_names)
        QApplication.clipboard().setText(clipboard_text)
    
    def trigger_auto_y_axis(self):
        """
        触发自动调节y轴功能（右键菜单）
        
        功能：根据当前可见的x轴范围，自动调整所有plot的y轴范围，
              使当前可见数据的y值完整显示（与顶部"自动调节y轴"按钮功能一致）
        
        应用范围：所有plot（不仅仅是右键点击的plot）
        """
        if self.plot_widget and self.plot_widget.window():
            main_window = self.plot_widget.window()
            # 调用主窗口的auto_y_in_x_range方法，功能和顶部按钮完全一致
            if hasattr(main_window, 'auto_y_in_x_range'):
                main_window.auto_y_in_x_range()
    
    def trigger_show_cursor_value(self):
        """显示cursor值（包括圆圈和y值标签）- 同步所有plot
        
        在多plot环境中，同步所有plot的cursor显示状态。
        如果cursor已启用，显示完整的cursor（vline + x值 + 圆圈 + y值）。
        """
        if self.plot_widget and self.plot_widget.window() and hasattr(self.plot_widget.window(), 'plot_widgets'):
            main_window = self.plot_widget.window()
            # 设置全局状态：cursor值不隐藏
            main_window.cursor_values_hidden = False
            
            # 检查cursor按钮状态
            cursor_enabled = main_window.cursor_btn.isChecked() if hasattr(main_window, 'cursor_btn') else False
            
            # 同步所有plot的cursor显示状态
            for container in main_window.plot_widgets:
                if container.plot_widget and cursor_enabled:
                    container.plot_widget.toggle_cursor(True)
        elif self.plot_widget:
            # 单plot模式
            self.plot_widget.toggle_cursor(True)
    
    def trigger_hide_cursor_value(self):
        """隐藏cursor值（只隐藏圆圈和y值，保留vline和x值）- 同步所有plot
        
        在多plot环境中，同步所有plot的cursor显示状态。
        只隐藏y值标签和圆圈，保留垂直线和x值显示。
        """
        if self.plot_widget and self.plot_widget.window() and hasattr(self.plot_widget.window(), 'plot_widgets'):
            main_window = self.plot_widget.window()
            # 设置全局状态：cursor值隐藏
            main_window.cursor_values_hidden = True
            
            # 检查cursor按钮状态
            cursor_enabled = main_window.cursor_btn.isChecked() if hasattr(main_window, 'cursor_btn') else False
            
            # 同步所有plot的cursor隐藏状态
            for container in main_window.plot_widgets:
                if container.plot_widget and cursor_enabled:
                    container.plot_widget.toggle_cursor(False, hide_values_only=True)
        elif self.plot_widget:
            # 单plot模式
            self.plot_widget.toggle_cursor(False, hide_values_only=True)
    
    def trigger_variable_editor(self):
        """打开绘图变量编辑器"""
        if self.plot_widget:
            dialog = PlotVariableEditorDialog(self.plot_widget, self.plot_widget.window())
            dialog.show()
            dialog.raise_()

class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    """
    可拖拽的图形布局控件类
    支持图表区域的拖拽重排和动态布局调整
    提供灵活的图表排列和交互功能
    """
    def __init__(self, units_dict, dataframe, time_channels_info={},synchronizer=None):
        super().__init__()
        self.factor = 1.0
        self.offset = 0.0
        self.original_index_x = None
        self.original_y = None
        self.mark_region = None
        self.is_cursor_pinned = False  # 记录cursor是否被固定
        self.pinned_x_value = None  # 记录固定的x值
        self.pinned_x_values = []
        self.pinned_index_values = []
        self.pinned_index_value = None  # 记录固定的索引值
        self._is_updating_data = False  # 标志：正在更新数据，禁止某些操作
        self._is_being_destroyed = False  # 标志：对象正在被销毁
        self._suppress_pin_update = False  # 标志：临时禁止pin状态自动更新
        self._pending_delete_items = []  # 
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.setSingleShot(True)
        self._cleanup_timer.timeout.connect(self._process_pending_deletes)
        self.setup_ui(units_dict, dataframe, time_channels_info, synchronizer)
        
    def setup_ui(self, units_dict, dataframe, time_channels_info={},synchronizer=None):
        """
        初始化UI组件和布局
        
        设置图形布局控件的基本配置和数据结构
        初始化绘图相关的属性和同步器
        
        Args:
            units_dict: 单位字典
            dataframe: 数据框
            time_channels_info: 时间通道信息
            synchronizer: 同步器实例
        """
        self.setAcceptDrops(True)
        self.units = units_dict
        self.data = dataframe
        self.time_channels_info = time_channels_info
        self.synchronizer = synchronizer
        self.curve = None
        #self.ci.layout.setContentsMargins(0, 0, 0, 5)
        
        # 多曲线支持
        self.curves = {}  # 存储所有曲线 {var_name: curve_info}
        self.is_multi_curve_mode = False  # 是否处于多曲线模式
        self._batch_adding = False  # 是否正在批量添加变量
        self.curve_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # 默认颜色列表
        self.current_color_index = 0  # 当前颜色索引
        
        self.y_name = ''
        self.y_format = ''
        self.x_name = ''
        self.x_format = ''

        self.xMin:int =0 
        self.xMax:int =1 
        # 添加顶部文本区域
        self.setup_header()
        # 主绘图区域设置
        self.setup_plot_area()
        # 坐标轴设置
        self.setup_axes()
        # 交互元素设置
        self.setup_interaction()

        # 布局比例设置 (绘图区占90%)
        self.ci.layout.setContentsMargins(0, 0, 10, 5)  # 消除所有边距
        self.ci.layout.setSpacing(0)
        self.ci.layout.setRowStretchFactor(1, 1)  # 主区域完全拉伸

        # 初始化框选功能
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()


    def setup_header(self):
        """完全修正的顶部文本区域设置方法"""
        header = pg.GraphicsWidget()
        layout = QGraphicsLinearLayout(Qt.Horizontal)
        

        # 计算固定Y轴宽度
        font = QApplication.font()
        fm = QFontMetrics(font)
        base_spacing=fm.horizontalAdvance("-10000.01")
        header.setFixedHeight(fm.height() * 2) 

        # 添加左边距（空项）
        left_margin = QGraphicsWidget()        
        layout.addItem(left_margin)
        layout.setItemSpacing(0, base_spacing*0) 
        
        # 左侧文本（使用代理窗口部件）
        self.label_left = QLabel("channel name")
        self.label_left.setStyleSheet("""
            color: #000;
            font-weight: bold;
            background-color: transparent;
        """)
        self.label_left.setSizePolicy(QSizePolicy.Policy.Minimum,
                                      QSizePolicy.Policy.Preferred)
        self.label_left.setTextFormat(Qt.RichText)  # 支持HTML格式
        self.label_left.setTextInteractionFlags(Qt.TextBrowserInteraction)  # 支持交互
        self.label_left.setContextMenuPolicy(Qt.NoContextMenu)  # 禁用右键菜单
        self.label_left.mousePressEvent = self._on_legend_clicked  # 绑定点击事件
        #self.label_left.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        proxy_left = QGraphicsProxyWidget()
        proxy_left.setWidget(self.label_left)

        # 只添加左侧文本到布局
        layout.addItem(proxy_left)
        layout.setStretchFactor(proxy_left, 1)
        layout.setAlignment(proxy_left, Qt.AlignBottom| Qt.AlignLeft)

        #layout.setAlignment(Qt.AlignBottom)

        header.setLayout(layout)
        #header.setAlignment(Qt.AlignBottom)
        self.addItem(header, row=0, col=0, colspan=2)

    def setup_plot_area(self):
        """
        配置绘图区域基本属性
        
        创建和配置主要的绘图区域
        设置视图框、坐标轴和基本绘图属性
        
        性能优化（基于iOS/Android浏览器缩放优化经验）：
        1. 智能降采样（peak模式保留峰值）
        2. 视图裁剪（只渲染可见区域）
        3. 交互期间性能降级（类似iOS快照技术）
        4. 智能防抖延迟（根据数据量动态调整）
        """
        self.plot_item = self.addPlot(row=1, col=0, colspan=2, viewBox=CustomViewBox())
        self.view_box = self.plot_item.vb
        self.view_box.plot_widget = self  # 设置 plot_widget 以确保 trigger_jump_to_data 能调用 jump_to_data_impl
        
        # ========== 性能优化 1: 防抖定时器 ==========
        # 优化缩放性能，避免频繁重绘
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._delayed_update_plot_style)
        self._debounce_delay_ms = DEBOUNCE_DELAY_MS
        self._pending_post_interaction_tasks = False
        
        # ========== 性能优化 2: 交互状态管理 ==========
        # 类似iOS的快照技术，在交互期间使用降级渲染
        self._is_interacting = False  # 标记是否正在交互（拖动/缩放）
        self._interaction_timer = QTimer()
        self._interaction_timer.setSingleShot(True)
        self._interaction_timer.timeout.connect(self._end_interaction)
        
        # ========== 性能优化 3.1: 同步缩放标志 ==========
        # 防止XLink同步时递归更新导致的性能问题
        self._is_syncing_range = False  # 标记是否正在同步范围（避免递归更新）

        # ========== 性能优化 3.2: 统一UI刷新调度 ==========
        self._init_ui_refresh_coordinator()
        
        # 移除 self._customize_plot_menu()，因为现在用 CustomViewBox 实现菜单定制
        
        self.view_box.setAutoVisible(x=False, y=True)  # 自动适应可视区域
        self.plot_item.setTitle(None)
        self.plot_item.hideButtons()
        
        # ========== 性能优化 3: 视图裁剪和降采样 ==========
        # 类似网页的懒加载和虚拟化技术
        self.plot_item.setClipToView(True)  # 只渲染可见区域
        # 使用peak模式保留峰值，自动降采样支持百万级数据点
        # 当auto=True时，pyqtgraph会根据可见区域自动计算合适的降采样因子
        # 无需指定ds参数，auto模式会自动处理
        self.plot_item.setDownsampling(mode='peak', auto=True)
        
        self.setBackground('w')

        pen = pg.mkPen('#f00',width=1)
        self.plot_item.getAxis('left').setGrid(255) 
        self.plot_item.getAxis('bottom').setGrid(255) 
        self.plot_item.showGrid(x=True, y=True, alpha=0.1)

        # 基于点数修改曲线风格
        # 使用防抖机制来优化缩放性能
        self.view_box.sigRangeChanged.connect(self._on_range_changed)
        
    def jump_to_data_impl(self, x):
        # 检查是否有数据
        has_data = False
        var_names = []
        
        # 收集所有要显示的变量名
        if self.is_multi_curve_mode and self.curves:
            # 多曲线模式：使用curves字典中的所有变量
            var_names = list(self.curves.keys())
            has_data = len(var_names) > 0
        elif self.curve and self.y_name:
            # 单曲线模式
            var_names = [self.y_name]
            has_data = True
        
        if not has_data:
            # 没有曲线，直接返回
            return

        main_window = self.window()
        if not hasattr(main_window, 'loader') or main_window.loader is None:
            return

        # a. 打开/激活数值变量表，并添加所有变量
        dlg = None
        for var_name in var_names:
            if var_name not in main_window.loader.df.columns:
                continue
            dlg = DataTableDialog.popup(var_name, main_window.loader.df[var_name], parent=main_window)

        # 如果没有成功打开任何dialog，直接返回
        if dlg is None:
            return

        # 判断"数值变量表"窗口是否被最小化了，如果是，则恢复正常状态
        if dlg.isMinimized():
            dlg.showNormal()
            
        # b. popup 已处理：如果已在（冻结或非冻结），不添加；否则添加到非冻结区域

        # c. 计算行索引（0-based）
        if self.factor == 0:
            return  # 避免除零

        index = (x - self.offset) / self.factor
        index = int(round(index)) - 1  # 转换为 0-based 行索引
        index = max(0, min(index, len(main_window.loader.df) - 1))  # 夹到有效范围

        # 使用第一个变量来定位和选中
        first_var_name = var_names[0]
        
        # 获取模型和列索引
        model = dlg.model
        col_idx = dlg._df.columns.get_loc(first_var_name)  # 逻辑列索引

        # 确定使用哪个视图（冻结或主视图）
        if first_var_name in dlg.frozen_columns:
            view = dlg.frozen_view
        else:
            view = dlg.main_view

        # 获取视觉列索引（因为列可拖动）
        header = view.horizontalHeader()
        visual_col = header.visualIndex(col_idx)

        # 创建 QModelIndex
        qindex = model.index(index, col_idx)

        # 跳转并居中，使用 QTimer 确保在窗口显示后执行
        QTimer.singleShot(0, lambda: view.scrollTo(qindex, QAbstractItemView.PositionAtCenter))

        # 选中该单元格
        QTimer.singleShot(0, lambda: view.selectionModel().select(qindex, QItemSelectionModel.ClearAndSelect))

    def auto_range(self):
        # 检查是否有数据可显示
        if not self.curve and not self.curves:
            return False
            
        # 清除X轴和Y轴的刻度
        self.axis_x.setTicks(None)
        self.axis_y.setTicks(None)

        # 获取x_values
        if self.is_multi_curve_mode and self.curves:
            # 多曲线模式：使用第一个曲线的x_data
            first_curve_info = next(iter(self.curves.values()))
            if 'x_data' in first_curve_info:
                x_values = first_curve_info['x_data']
            else:
                return False
        else:
            # 单曲线模式
            if self.original_index_x is not None:
                x_values = self.offset + self.factor * self.original_index_x
            elif self.curve:
                # 如果original_index_x为None，尝试从curve获取数据
                x_data, _ = self.curve.getData()
                if x_data is not None:
                    x_values = x_data
                else:
                    return False
            else:
                return False
        
        global DEFAULT_PADDING_VAL_X,DEFAULT_PADDING_VAL_Y, FILE_SIZE_LIMIT_BACKGROUND_LOADING, RATIO_RESET_PLOTS, FROZEN_VIEW_WIDTH_DEFAULT, BLINK_PULSE, FACTOR_SCROLL_ZOOM
        
        padding_xVal = DEFAULT_PADDING_VAL_X  
        padding_yVal = 0.5

        # 计算X轴范围
        min_x = np.min(x_values)
        max_x = np.max(x_values)
        
        # 计算Y轴范围
        if self.is_multi_curve_mode and self.curves:
            # 多曲线模式：计算所有可见曲线的Y轴范围
            all_y_values = []
            for var_name, curve_info in self.curves.items():
                if curve_info.get('visible', True) and 'y_data' in curve_info:
                    all_y_values.extend(curve_info['y_data'])
            
            if all_y_values:
                min_y = np.nanmin(all_y_values)
                max_y = np.nanmax(all_y_values)
            else:
                min_y, max_y = 0, 1  # 默认范围
        else:
            # 单曲线模式
            if self.original_y is not None:
                special_limits = self.handle_single_point_limits(x_values, self.original_y)
                if special_limits:
                    min_x, max_x, min_y, max_y = special_limits
                else:
                    min_y = np.nanmin(self.original_y)
                    max_y = np.nanmax(self.original_y)
            elif self.curve:
                # 如果original_y为None，尝试从curve获取数据
                _, y_data = self.curve.getData()
                if y_data is not None:
                    min_y = np.nanmin(y_data)
                    max_y = np.nanmax(y_data)
                else:
                    min_y, max_y = 0, 1
            else:
                min_y, max_y = 0, 1
        
        limits_xMin = min_x - padding_xVal * (max_x - min_x)
        limits_xMax = max_x + padding_xVal * (max_x - min_x)

        # 新增：显式设置 XRange（与 YRange 一致，使用 padding=0.05）
        self.view_box.setXRange(min_x, max_x, padding=DEFAULT_PADDING_VAL_X)  # 重置到全范围
        self._set_safe_y_range(min_y, max_y)

        global MIN_INDEX_LENGTH 
        minXRange_val = min(MIN_INDEX_LENGTH,len(x_values)-1 if len(x_values)>1 else 1)*self.factor
        self.plot_item.setLimits(xMin=limits_xMin, xMax=limits_xMax, minXRange=minXRange_val)

        # self.window().sync_all_x_limits(limits_xMin, limits_xMax, min(3,len(x_values))*self.factor)
        self._set_vline_bounds([min_x, max_x])

        # 在设置完新范围后，立即直接调用样式更新函数。
        self.update_plot_style(self.view_box, self.view_box.viewRange(), None)
        self.plot_item.update()
        self._update_cursor_after_plot(min_x, max_x)

        return True

    def auto_y_in_x_range(self):
        vb=self.view_box
        vb.enableAutoRange(axis=vb.YAxis, enable=True)
        vb.plot_widget.axis_y.setTicks(None)

    def update_left_header(self, left_text=None):
        """更新顶部文本内容"""
        if left_text is not None:
            self.label_left.setText(left_text)

    def update_right_header(self, right_text=None):
        """更新顶部文本内容（已移除右侧label）"""
        # 右侧label已被移除，此方法保留以兼容现有代码
        pass

    def _get_safe_x_range(self, min_x: float, max_x: float) -> tuple[float, float]:
        """
        确保X轴范围非零，如果 min_x == max_x，则基于 factor 扩展。
        """
        if min_x == max_x:
            # 在中心点两侧各扩展 0.5 * factor
            min_x_safe = min_x - 0.5 * self.factor
            max_x_safe = max_x + 0.5 * self.factor
            return min_x_safe, max_x_safe
        return min_x, max_x
    
    def _get_min_x_range_value(self) -> float:
        """
        根据数据长度计算最小的可缩放 X 范围 (minXRange)。
        """
        global MIN_INDEX_LENGTH
        if self.data is None or self.data.empty:
            data_len = 0
        else:
            data_len = self.data.shape[0]
            
        # 计算有效的数据点最小间隔
        effective_min_len = min(MIN_INDEX_LENGTH, data_len - 1 if data_len > 1 else 1)
        
        # 最小范围 = 最小间隔 * 比例因子
        return effective_min_len * self.factor

    def _set_x_limits_with_min_range(self, limits_xMin: float | None, limits_xMax: float | None):
        """
        统一设置 X 轴的 limits 和 minXRange。
        """
        minXRange_val = self._get_min_x_range_value()
        self.plot_item.setLimits(xMin=limits_xMin, xMax=limits_xMax, minXRange=minXRange_val)

    def _set_safe_y_range(self, min_y: float, max_y: float, set_limits: bool = True):
        """
        设置 Y 轴的 viewRange 和 limits，自动处理 NaN 或恒定值。
        
        Args:
            min_y: Y轴最小值
            max_y: Y轴最大值
            set_limits: 是否同时设置y轴limits，默认为True。当为False时只设置viewRange。
        """
        global DEFAULT_PADDING_VAL_Y
        
        # Y 轴 limit 的内外边距 (0.5 表示上下各扩展 50%)
        padding_yVal_limit = 0.5 

        if np.isnan(min_y) or np.isnan(max_y) or min_y == max_y:
            # 如果是 NaN 或恒定值
            y_center = min_y if not np.isnan(min_y) else 0
            # 保证最小范围为 1.0，或者为中心值的 20%
            y_range_half = (1.0 if y_center == 0 else abs(y_center) * 0.2)
            
            y_min_view = y_center - y_range_half
            y_max_view = y_center + y_range_half
            
            # Limits 也使用这个扩展后的范围
            y_min_limit = y_min_view
            y_max_limit = y_max_view
        else:
            # 如果是正常范围
            y_min_view = min_y
            y_max_view = max_y
            
            # Limits 应用 50% 的外边距
            y_range = max_y - min_y
            y_min_limit = min_y - padding_yVal_limit * y_range
            y_max_limit = max_y + padding_yVal_limit * y_range

        # 只在需要时设置limits
        if set_limits:
            self.plot_item.setLimits(yMin=y_min_limit, yMax=y_max_limit)
        # ViewRange 使用 PADDING_Y (默认0.1) 的内边距
        self.view_box.setYRange(y_min_view, y_max_view, padding=DEFAULT_PADDING_VAL_Y)

    def reset_plot(self,index_xMin,index_xMax):

        self.plot_item.setLimits(xMin=None, xMax=None)  # 解除X轴限制
        self.plot_item.setLimits(yMin=None, yMax=None)  # 解除Y轴限制
        
        xMin = self.offset + self.factor * index_xMin
        xMax = self.offset + self.factor * index_xMax
        
        global DEFAULT_PADDING_VAL_X, DEFAULT_PADDING_VAL_Y

        if not (np.isnan(xMax) or np.isinf(xMax)):
            xMin, xMax = self._get_safe_x_range(xMin, xMax)

            self.view_box.setXRange(xMin, xMax, padding=DEFAULT_PADDING_VAL_X)
            padding_xVal=DEFAULT_PADDING_VAL_X
            limits_xMin = xMin - padding_xVal * (xMax - xMin)
            limits_xMax = xMax + padding_xVal * (xMax - xMin)
            self._set_x_limits_with_min_range(limits_xMin, limits_xMax)

        self.view_box.setYRange(0,1,padding=DEFAULT_PADDING_VAL_Y) 
        self._set_vline_bounds([None, None]) 

        self.xMin = xMin
        self.xMax = xMax
        self.y_name = ''
        self.y_format = ''
        #self.plot_item.update()
        # 先清除cursor items（包括scene中的items）
        # 重置plot时完全清除对象池，避免复用异常状态的items
        self._clear_cursor_items(hide_only=False)
        self._safe_clear_plot_items() 
        self.axis_y.setLabel(text="")
        self.update_left_header("channel name")
        self.update_right_header("")

        self.curve = None
        self.original_index_x = None
        self.original_y = None
        self.curves.clear()
        self.is_multi_curve_mode = False
        self.current_color_index = 0



    def setup_axes(self):
        """配置坐标轴样式和范围"""
        # X轴配置
        self.axis_x = self.plot_item.getAxis('bottom')
        self.axis_x.setTextPen('black')
        self.axis_x.setPen(QPen(QColor('black'), 1))
        self.axis_x.setRange(0, 10)
        
        # Y轴配置
        self.axis_y = self.plot_item.getAxis('left')
        self.axis_y.enableAutoSIPrefix(False)
        self.axis_y.setTextPen('black')
        self.axis_y.setPen(QPen(QColor('black'), 1))
        

        # 其他边框配置
        for pos in ('top', 'right'):
            ax = self.plot_item.getAxis(pos)
            ax.setVisible(True)
            ax.setTicks([])
            ax.setStyle(showValues=False, tickLength=0)
            ax.setPen(QPen(QColor('black'), 1))
        
        # 计算固定Y轴宽度
        font = QApplication.font()
        fm = QFontMetrics(font)
        self.axis_y.setWidth(fm.horizontalAdvance("-10000.01") )

        # 基于应用程序基础字体大小，增加2像素作为标签字体大小
        font_family = font.family() 
        # 使用 font.pixelSize() 保证跨平台一致性，并略微增大
        pixel_size = font.pixelSize() + 2

        # Y轴标签
        # self.axis_y.setLabel(
        #     color='black',
        #     angle=-90,
        #     **{'font-family': 'Arial', 'font-size': '12pt', 'font-weight': 'bold'}
        # )
        self.axis_y.setLabel(
            color='black',
            angle=-90,
            # 修正：使用像素大小 'px' 代替点大小 'pt'，并使用系统字体
            **{'font-family': font_family, 'font-size': f'{pixel_size}px', 'font-weight': 'bold'}
        )

    def setup_interaction(self):
        """配置交互元素"""
        # 光标线
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 0, 0, 100), width=4))
        self.vline2 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 0, 0, 100), width=4))
        self.vline.cursor_index = 0
        self.vline2.cursor_index = 1
        self.vline2.setZValue(100)
        self.vline.setZValue(100)
        self.cursor_label = pg.TextItem("", anchor=(1, 1), color="red")
        self.plot_item.addItem(self.vline, ignoreBounds=True)
        self.plot_item.addItem(self.vline2, ignoreBounds=True)
        self.plot_item.addItem(self.cursor_label, ignoreBounds=True)
        self.vline.setVisible(False)
        self.vline2.setVisible(False)
        self.cursor_label.setVisible(False)
        
        # 多曲线cursor元素
        self.multi_cursor_items = []  # 存储多曲线cursor的可视化元素
        self.show_values_only = True  # 是否只显示x值（不显示圆圈和y值）
        
        # 【内存优化】对象池 - 复用ScatterPlotItem和TextItem，避免重复创建
        self._cursor_item_pool = {
            'circles': [],  # ScatterPlotItem
            'labels': [],   # TextItem(y)
            'x_labels': []  # X()
        }
        
        # 信号连接
        # 【性能优化】控制cursor更新频率，减少CPU占用
        # 多曲线时降低频率可显著提升响应速度
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=20, slot=self.mouse_moved)
        self.vline.sigPositionChanged.connect(self.on_vline_position_changed)
        self.vline2.sigPositionChanged.connect(self.on_vline_position_changed)
        self.setAntialiasing(False)
        
        # 【性能优化】cursor更新节流控制
        self._last_cursor_update_time = 0
        self._cursor_update_throttle = 0.033  # 基础节流：33ms（约30fps）
        self._adaptive_throttle_enabled = True  # 启用自适应节流
        self._cursor_refresh_timer = QTimer(self)
        self._cursor_refresh_timer.setSingleShot(True)
        self._cursor_refresh_timer.timeout.connect(self._refresh_cursor_geometry)

    def _extract_var_names_from_text(self, text: str) -> list[str]:
        return [name.strip() for name in (text or '').split(';;') if name.strip()]

    def _notify_drag_indicator(self, var_names: list[str] | None = None, hide: bool = False):
        main_window = self.window()
        if not main_window or not hasattr(main_window, "_show_drag_indicator_for_plot"):
            return
        if hide:
            main_window._hide_drag_indicator_for_plot(self)
        else:
            main_window._show_drag_indicator_for_plot(self, var_names or [])

    def handle_single_point_limits(self, x_values, y_values):
        """处理单点或所有点x坐标相同的特殊情况，避免x轴范围为0
        
        Args:
            x_values: x坐标数组
            y_values: y坐标数组
            
        Returns:
            tuple: (min_x, max_x, min_y, max_y) 或 None（正常情况不需要特殊处理）
        """
        if len(x_values) == 1:
            # 单点情况：扩展x轴范围
            x = x_values[0]
            min_x, max_x = self._get_safe_x_range(x, x)
            if len(y_values) == 1:
                y = y_values[0]
                min_y = y - 0.5 if y != 0 else -0.5
                max_y = y + 0.5 if y != 0 else 0.5
            else:
                min_y = np.nanmin(y_values)
                max_y = np.nanmax(y_values)
            return min_x, max_x, min_y, max_y
        else:
            # 检查是否所有x值都相同（多点但x坐标相同的情况）
            unique_x = set(x_values)
            if len(unique_x) == 1:
                # 所有点的x坐标相同，扩展x轴范围
                x = list(unique_x)[0]
                min_x, max_x = self._get_safe_x_range(x, x)
                min_y = np.nanmin(y_values)
                max_y = np.nanmax(y_values)
                return min_x, max_x, min_y, max_y
            else:
                # 正常情况：有多个不同的x值
                return None

    def _is_cursor_update_locked(self) -> bool:
        """判定当前是否应跳过cursor更新（例如加载/刷新过程中）"""
        if getattr(self, "_is_updating_data", False):
            return True
        main_window = self.window()
        if main_window and getattr(main_window, "_is_loading_new_data", False):
            return True
        return False

    def wheelEvent(self, ev):
        vb = self.plot_item.getViewBox()
        delta = ev.angleDelta().y()
        # 只在没有按下任何修饰键（Ctrl/Shift/Alt…）时才执行缩放
        if ev.modifiers() == Qt.NoModifier:
            if delta != 0:
                # 获取鼠标位置
                mouse_pos = ev.position().toPoint()
                scene_pos = self.mapToScene(mouse_pos)
                view_pos = vb.mapSceneToView(scene_pos)
                mouse_x = view_pos.x()
                mouse_y = view_pos.y()

                global FACTOR_SCROLL_ZOOM
                factor = max(0.000001,1-FACTOR_SCROLL_ZOOM)if delta > 0 else (1+FACTOR_SCROLL_ZOOM)
                vb.scaleBy((factor, 1), center=(mouse_x, mouse_y))
                ev.accept()  # 确保事件被处理
            else:
                super().wheelEvent(ev)
        else:
            # 有按键按下，交给父类默认处理（或自己写别的逻辑）
            super().wheelEvent(ev)
        
        #ev.accept()  # 确保事件被处理
    
    def mouse_moved(self, evt):
        """鼠标移动事件处理 - 稳定性加固版"""
        # 【稳定性加固】额外的安全检查
        if not hasattr(self, 'plot_item') or self.plot_item is None:
            return
        if getattr(self, '_is_being_destroyed', False):
            return

        try:
            pos = evt[0]
            if not self.plot_item.sceneBoundingRect().contains(pos):
                return
        except (RuntimeError, AttributeError):
            # C++ 对象已被删除
            return

        if self._is_cursor_update_locked():
            return

        # 【稳定性加固】检查是否正在执行pin操作，避免冲突
        if getattr(self, '_is_pinning_cursor', False):
            return

        try:
            mousePoint = self.plot_item.vb.mapSceneToView(pos)
        except (RuntimeError, AttributeError):
            return

        # 如果cursor被固定，不跟随鼠标移动
        if self.is_cursor_pinned:
            # 在pin状态下，cursor保持固定位置，不跟随鼠标
            pass
        else:
            # 正常跟随鼠标模式
            if hasattr(self.window(), 'sync_crosshair'):
                self.window().sync_crosshair(mousePoint.x(), self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_cursor_geometry_update()

    def on_vline_position_changed(self, line_obj=None):
        """
        vline
        
        vline,pinpinplotcursor
        cursor
        """
        if self._is_cursor_update_locked():
            return
        if self.is_cursor_pinned:
            if getattr(self, "_suppress_pin_update", False):
                return
            line = line_obj if line_obj is not None else self.vline
            cursor_index = getattr(line, 'cursor_index', 0)
            x_pos = line.value()

            if len(self.pinned_x_values) <= cursor_index:
                self.pinned_x_values += [x_pos] * (cursor_index + 1 - len(self.pinned_x_values))
            self.pinned_x_values[cursor_index] = x_pos

            if cursor_index == 0:
                self.pinned_x_value = x_pos
                if self.factor != 0:
                    self.pinned_index_value = (x_pos - self.offset) / self.factor
                else:
                    self.pinned_index_value = None

            self.pinned_index_values = []
            for x_val in self.pinned_x_values:
                if self.factor != 0:
                    self.pinned_index_values.append((x_val - self.offset) / self.factor)

            if self.window() and hasattr(self.window(), 'pinned_x_values'):
                self.window().pinned_x_values = list(self.pinned_x_values)

            if self.window() and hasattr(self.window(), 'plot_widgets'):
                for container in self.window().plot_widgets:
                    widget = container.plot_widget
                    if widget.is_cursor_pinned and widget != self:
                        target_line = widget.vline if cursor_index == 0 else getattr(widget, 'vline2', None)
                        if target_line is not None:
                            target_line.blockSignals(True)
                            try:
                                target_line.setPos(x_pos)
                            finally:
                                target_line.blockSignals(False)
                        if len(widget.pinned_x_values) <= cursor_index:
                            widget.pinned_x_values += [x_pos] * (cursor_index + 1 - len(widget.pinned_x_values))
                        widget.pinned_x_values[cursor_index] = x_pos
                        if cursor_index == 0:
                            widget.pinned_x_value = x_pos
                            if widget.factor != 0:
                                widget.pinned_index_value = (x_pos - widget.offset) / widget.factor
                            else:
                                widget.pinned_index_value = None
                        widget.update_cursor_label()

            self.update_cursor_label()
        else:
            if self.show_values_only:
                self._show_x_position_only()
            else:
                self.update_cursor_label()

    def sInt_to_fmtStr(self, value: int):
        """将秒数转换为时间字符串 HH:MM:SS.SS - 优化版避免内存泄漏"""
        # 【优化】直接计算而不创建pandas对象，避免内存累积
        total = value % (24*3600)  # 一天内的秒数
        hh = int(total // 3600)
        mm = int((total % 3600) // 60)
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:05.2f}"
    
    def dateInt_to_fmtStr(self, value: int):
        """将时间戳转换为日期字符串 - 优化版避免内存泄漏"""
        # 【优化】直接使用datetime而不创建pandas Series，避免内存累积
        from datetime import datetime
        try:
            dt = datetime.fromtimestamp(value)
            return dt.strftime('%Y/%m/%d')
        except:
            return str(value)
    
    def _significant_decimal_format_str(self,value: float, ref: float, max_dp:int | None = None) -> str:
        """
        根据 ref 的“显示精度”自动决定 value 的字符串格式。
        """
        # check length
        s = format(ref, 'f').rstrip('0').rstrip('.')
        if '.' not in s:
            dp = 0
        else:
            dp = len(s.split('.')[1])

        if max_dp is None or max_dp < 0:
            pass
        else:
            dp = min(max_dp,dp)

        if dp == 0:                       # ref 本身按整数显示
            return str(int(round(value)))
        
        fmt = f'{{:.{dp}f}}'              # 例如保留 2 位 -> "{:.2f}"
        return fmt.format(value).rstrip('0').rstrip('.')  # 去掉无意义的 0
    


    def set_xrange_with_link_handling(self, xmin, xmax,padding:float = 0):
        plot=self.plot_item
        # 1. 记录当前联动对象
        linked = plot.getViewBox().linkedView(0)
        
        # 2. 临时断开联动
        if linked is not None:
            plot.setXLink(None)
        
        # 3. 安全设置范围
        plot.getViewBox().enableAutoRange(x=False)
        plot.setXRange(xmin, xmax, padding=max(0,padding))
        
        # 4. 恢复联动
        if linked is not None:
            plot.setXLink(linked)

    def update_cursor_label(self):
        """更新光标标签位置和内容 - 稳定性加固版"""
        # 【稳定性加固】重入保护
        if getattr(self, '_is_updating_cursor_label', False):
            return
        if self._is_cursor_update_locked():
            return

        self._is_updating_cursor_label = True
        try:
            # 统一使用多曲线样式的cursor显示
            self._update_multi_curve_cursor_label()
        except (RuntimeError, AttributeError) as e:
            # C++ 对象可能已被删除，忽略此类错误
            pass
        finally:
            self._is_updating_cursor_label = False
    
    def _update_single_curve_cursor_label(self):
        """更新单曲线模式的光标标签"""
        if len(self.plot_item.listDataItems()) == 0:
            self.update_right_header("")
            return
        
        try:
            x = self.vline.value()           
            curve = self.plot_item.listDataItems()[0]
            x_data, y_data = curve.getData()
            if x_data is None or len(x_data) == 0:
                self.update_right_header("")
                return
            x = np.clip(x, x_data.min(), x_data.max())
            idx = np.argmin(np.abs(x_data - x))
            y_val = y_data[idx]
            x_str = self._significant_decimal_format_str(value=float(x),ref=self.factor)
            if self.y_format == 's':
                time_str=self.sInt_to_fmtStr(y_val)
                self.update_right_header(f"x={x_str}, y={time_str}")
            elif self.y_format == 'date':
                date_str=self.dateInt_to_fmtStr(y_val)
                self.update_right_header(f"x={x_str}, y={date_str}")
            else:
                self.update_right_header(f"x={x_str}, y={y_val:.5g}")

        except Exception as e:
            print(f"Cursor update error: {e}")
            self.update_right_header("")
    
    def _get_circle_from_pool(self, index):
        """从对象池获取ScatterPlotItem，如果不存在则创建
        
        使用对象池复用ScatterPlotItem，避免重复创建导致内存泄漏。
        每个索引位置对应一个ScatterPlotItem实例，用于在cursor交点处显示圆圈标记。
        
        Args:
            index: 对象池索引位置
            
        Returns:
            ScatterPlotItem: 从池中获取或新创建的圆圈标记对象
        """
        pool = self._cursor_item_pool['circles']
        
        # 如果池中已有该索引的对象，直接复用
        if index < len(pool):
            return pool[index]
        
        # 否则创建新对象并加入池
        circle = pg.ScatterPlotItem(
            symbol='o',
            size=8,
            brush=None
        )
        pool.append(circle)
        return circle
    
    def _get_label_from_pool(self, index):
        """从对象池获取TextItem，如果不存在则创建
        
        使用对象池复用TextItem，避免重复创建导致内存泄漏。
        每个索引位置对应一个TextItem实例，用于显示cursor交点处的y值标签。
        
        Args:
            index: 对象池索引位置
            
        Returns:
            TextItem: 从池中获取或新创建的文本标签对象
        """
        pool = self._cursor_item_pool['labels']
        
        # 如果池中已有该索引的对象，直接复用
        if index < len(pool):
            return pool[index]
        
        # 否则创建新对象并加入池
        label = pg.TextItem(
            color=(0, 0, 0),
            fill=pg.mkBrush(255, 255, 255, 220),
            anchor=(0.5, 0.5)
        )
        # label.setFont(QFont('Arial', 8))

        font = QApplication.font()  # 获取App的默认字体
        font.setPixelSize(11)     # 设置一个跨平台一致的逻辑像素大小 (11px)
        label.setFont(font)

        pool.append(label)
        return label
    
    def _get_x_label_from_pool(self, index=0):
        """X(,)
        
        Xplotcursorx,
        TextItem,
        
        Returns:
            TextItem: x()
        """
        pool = self._cursor_item_pool.get('x_labels', [])
        if index < len(pool):
            return pool[index]

        x_label = pg.TextItem(
            color=(255, 255, 255),
            fill=pg.mkBrush(64, 64, 64, 230),
            border=pg.mkPen(128, 128, 128, width=1),
            anchor=(0.5, 0)
        )
        font = QApplication.font()
        font.setPixelSize(12)
        x_label.setFont(font)
        pool.append(x_label)
        self._cursor_item_pool['x_labels'] = pool
        return x_label

    def _clear_cursor_items(self, hide_only=True):
        """清除或隐藏所有cursor可视化元素
        
        默认模式下只隐藏元素（供下次复用），完全清除模式下才会删除对象。
        这种策略通过对象池复用机制避免频繁创建/销毁对象导致的内存泄漏。
        
        Args:
            hide_only: 如果为True，只隐藏所有元素（默认，用于复用）
                      如果为False，完全删除所有元素和对象池（用于切换数据文件等场景）
        """
        # 【安全检查】确保关键对象存在
        if not hasattr(self, 'multi_cursor_items') or not hasattr(self, 'plot_item'):
            return
        
        # 【修复QPainter错误】先隐藏所有item，避免清理过程中触发绘制
        for item in self.multi_cursor_items:
            try:
                if item is not None:
                    item.setVisible(False)
            except:
                pass
        
        # 分类处理：对象池中的元素清除数据，非池对象删除
        for item in self.multi_cursor_items:
            try:
                item_type = type(item).__name__
                if item_type == 'ScatterPlotItem':
                    # 对象池中的圆圈标记：清除数据
                    try:
                        item.clear()  # 清除ScatterPlotItem的数据，释放内存
                    except:
                        pass
                elif item in self._cursor_item_pool.get('x_labels', []):
                    # X轴标签：清空文本
                    try:
                        item.setText("")  # 清空文本，释放字符串占用的内存
                    except:
                        pass
                elif item in self._cursor_item_pool.get('labels', []):
                    # 对象池中的y值标签：清空文本
                    try:
                        item.setText("")  # 清空文本，释放字符串占用的内存
                    except:
                        pass
                else:
                    # ():
                    self._queue_item_for_deletion(item)
            except Exception as e:
                # 忽略清理过程中的错误
                pass
        
        # 清空当前使用列表
        self.multi_cursor_items.clear()
        
        if not hide_only:
            for circle in self._cursor_item_pool.get('circles', []):
                self._queue_item_for_deletion(circle)

            for label in self._cursor_item_pool.get('labels', []):
                self._queue_item_for_deletion(label)

            for x_label in self._cursor_item_pool.get('x_labels', []):
                self._queue_item_for_deletion(x_label)

            self._cursor_item_pool = {
                'circles': [],
                'labels': [],
                'x_labels': []
            }

            if self._pending_delete_items and not self._cleanup_timer.isActive():
                self._cleanup_timer.start(100)

    def _queue_item_for_deletion(self, item):
        """将item加入延迟删除队列"""
        if item is not None and item not in self._pending_delete_items:
            try:
                item.setVisible(False)
            except (RuntimeError, AttributeError):
                pass
            self._pending_delete_items.append(item)

    def _process_pending_deletes(self):
        """处理待删除队列中的item"""
        if self._is_updating_data or self._is_being_destroyed:
            if self._pending_delete_items:
                self._cleanup_timer.start(100)
            return

        items_to_delete = self._pending_delete_items.copy()
        self._pending_delete_items.clear()

        for item in items_to_delete:
            try:
                if item is None:
                    continue
                try:
                    scene = item.scene()
                    if scene is not None:
                        scene.removeItem(item)
                except (RuntimeError, AttributeError):
                    pass
                try:
                    if hasattr(item, 'deleteLater'):
                        item.deleteLater()
                except (RuntimeError, AttributeError):
                    pass
            except Exception:
                pass

    def _safe_clear_plot_items(self):
        """安全地清理所有plot items，避免scene不匹配问题
        
        【内存优化】清除曲线时释放其数据和样式缓存
        """
        try:
            # 【安全检查】确保plot_item存在且有效
            if not hasattr(self, 'plot_item') or self.plot_item is None:
                return
            
            current_scene = self.plot_item.scene()
            
            if current_scene is not None:
                # 获取所有items
                all_items = current_scene.items()
                
                # 手动清理所有items，避免使用clearPlots()
                items_removed = 0
                for i, item in enumerate(all_items):
                    try:
                        # 检查item是否仍然有效
                        item_scene = item.scene()
                        if item_scene == current_scene:
                            # 只移除数据曲线，不移除cursor相关items（由_clear_cursor_items管理）
                            should_remove = False
                            item_type = type(item).__name__
                            
                            # 检查是否是数据曲线（PlotDataItem）
                            if hasattr(item, 'getData') and hasattr(item, 'opts'):
                                # 确保不是坐标轴
                                if not hasattr(item, 'setLabel'):
                                    should_remove = True
                                    
                                    # 清除曲线的缓存数据，释放内存
                                    if hasattr(item, '_cached_pen_key'):
                                        delattr(item, '_cached_pen_key')
                                    if hasattr(item, '_has_symbols'):
                                        delattr(item, '_has_symbols')
                                    
                                    # 清除曲线数据
                                    try:
                                        item.clear()
                                    except:
                                        pass
                            
                            # 注意：不在这里清理TextItem和ScatterPlotItem
                            # 这些cursor相关的items由_clear_cursor_items()管理
                            
                            if should_remove:
                                current_scene.removeItem(item)
                                items_removed += 1
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass

                # 兜底：确保所有 PlotDataItem 被移除，避免重复曲线残留
                try:
                    data_items = list(self.plot_item.listDataItems())
                    for item in data_items:
                        try:
                            item_scene = item.scene()
                            if item_scene is not None and item_scene == current_scene:
                                self.plot_item.removeItem(item)
                        except Exception:
                            pass
                        try:
                            item.clear()
                        except Exception:
                            pass
                except Exception:
                    pass
                
            
            
        except Exception as e:
            pass
    
    def _update_multi_curve_cursor_label(self):
        """更新多曲线cursor标签"""
        # 更新cursor标签，避免过度刷新
        if getattr(self, '_is_interacting', False):
            return  # 正在交互时跳过
        
        # 计算更新节流
        import time
        current_time = time.time()
        
        # :,
        if self._adaptive_throttle_enabled and hasattr(self, 'curves'):
            curve_count = len(self.curves)
            # 16ms + 2ms,100ms
            adaptive_throttle = min(0.033 + curve_count * 0.004, 0.2)
        else:
            adaptive_throttle = self._cursor_update_throttle
        
        if hasattr(self, '_last_cursor_update_time'):
            time_since_last = current_time - self._last_cursor_update_time
            if time_since_last < adaptive_throttle:
                return  # 
        self._last_cursor_update_time = current_time
        
        # cursor
        self._clear_cursor_items()

        mode = self._get_cursor_mode()
        if mode == "2 anchored cursor":
            vline_visible = bool(self.vline.isVisible() or self.vline2.isVisible())
        else:
            vline_visible = self.vline.isVisible()
        if not vline_visible:
            self.update_right_header("")
            return
        
        # x,x()
        has_attr = hasattr(self, 'show_values_only')
        show_values_only = self.show_values_only if has_attr else False
        if has_attr and show_values_only:
            self._show_x_position_only()
            return
        
        # 
        if not self.curves and not self.curve:
            self.update_right_header("")
            return

        x_positions = self._get_cursor_x_positions()
        if not x_positions:
            self.update_right_header("")
            return
            
        try:
            cursor_values = []
            
            # 
            (x_min, x_max), (y_min, y_max) = self.view_box.viewRange()
            
            # 
            curves_to_process = []
            
            if self.curves:
                # curves:curves()
                for var_name, curve_info in self.curves.items():
                    if not curve_info.get('visible', True):
                        continue
                    curves_to_process.append({
                        'var_name': var_name,
                        'x_data': curve_info['x_data'],
                        'y_data': curve_info['y_data'],
                        'color': curve_info['color'],
                        'y_format': curve_info.get('y_format', ''),
                        'unit': self.units.get(var_name, '')
                    })
            elif not self.is_multi_curve_mode and self.curve and self.y_name:
                # 
                x_data, y_data = self.curve.getData()
                if x_data is not None and len(x_data) > 0:
                    # 
                    curve_color = 'blue'
                    try:
                        if hasattr(self.curve, 'opts') and 'pen' in self.curve.opts:
                            pen = self.curve.opts['pen']
                            if hasattr(pen, 'color'):
                                curve_color = pen.color().name()
                    except:
                        pass
                    
                    curves_to_process.append({
                        'var_name': self.y_name,
                        'x_data': x_data,
                        'y_data': y_data,
                        'color': curve_color,
                        'y_format': self.y_format,
                        'unit': self.units.get(self.y_name, '')
                    })
            
            # y
            for x in x_positions:
                if x < x_min or x > x_max:
                    continue
                for curve_data in curves_to_process:
                    var_name = curve_data['var_name']
                    x_data = curve_data['x_data']
                    y_data = curve_data['y_data']
                    color = curve_data['color']
                    y_format = curve_data['y_format']
                    
                    if x_data is None or len(x_data) == 0:
                        continue
                    
                    # x
                    if x < x_data.min() or x > x_data.max():
                        continue
                    
                    # argmin,10-100
                    # x_data(),searchsorted
                    try:
                        idx = np.searchsorted(x_data, x, side='left')
                        # 
                        if idx >= len(x_data):
                            idx = len(x_data) - 1
                        elif idx > 0:
                            # 
                            if abs(x_data[idx - 1] - x) < abs(x_data[idx] - x):
                                idx = idx - 1
                    except (ValueError, TypeError):
                        # x_data,argmin
                        idx = np.argmin(np.abs(x_data - x))
                        
                    y_val = y_data[idx]
                    x_actual = x_data[idx]
                    
                    # NaN(All-NaN slice encountered)
                    if np.isnan(x_actual) or np.isnan(y_val):
                        continue
                    
                    # y
                    if y_val < y_min or y_val > y_max:
                        continue
                    
                    # y
                    if y_format == 's':
                        y_str = self.sInt_to_fmtStr(y_val)
                    elif y_format == 'date':
                        y_str = self.dateInt_to_fmtStr(y_val)
                    else:
                        y_str = f"{y_val:.5g}"
                    
                    # 
                    cursor_values.append({
                        'var_name': var_name,
                        'x_pos': x_actual,
                        'y_pos': y_val,
                        'y_value': y_str,
                        'color': color
                    })
                    
                    # ScatterPlotItem,cursor
                    circle = self._get_circle_from_pool(len(cursor_values) - 1)
                    
                    # 
                    circle.clear()
                    circle.setData([x_actual], [y_val])
                    
                    # 
                    # pen
                    if not hasattr(circle, '_cached_color') or circle._cached_color != color:
                        pen = pg.mkPen(color, width=1.5)
                        circle.setPen(pen)
                        circle._cached_color = color
                    
                    # 
                    circle.setVisible(True)
                    circle.setZValue(200)
                    # itemplot()
                    circle_scene = circle.scene()
                    plot_scene = self.plot_item.scene()
                    
                    if circle_scene != plot_scene:
                        # itemscene,
                        if circle_scene is not None:
                            circle_scene.removeItem(circle)
                        # :ignoreBounds=True y
                        # ,y,
                        self.plot_item.addItem(circle, ignoreBounds=True)
                    
                    self.multi_cursor_items.append(circle)
            
            # ,
            self._position_labels_avoid_overlap(cursor_values, x_min, x_max, y_min, y_max)

            for idx, x in enumerate(x_positions):
                if x < x_min or x > x_max:
                    continue
                x_str = self._significant_decimal_format_str(value=float(x), ref=self.factor)
                x_info_item = self._get_x_label_from_pool(idx)
                x_info_item.setText(x_str)
                x_info_item.setVisible(True)

                view_rect = self.plot_item.vb.sceneBoundingRect()
                scene_point = self.plot_item.vb.mapViewToScene(pg.Point(x, y_min))
                scene_x = scene_point.x()
                scene_y = view_rect.bottom()
                x_info_item.setPos(scene_x, scene_y)
                x_info_item.setZValue(100000)

                scene = self.plot_item.scene()
                x_scene = x_info_item.scene()
                if x_scene != scene:
                    if x_scene is not None:
                        x_scene.removeItem(x_info_item)
                    scene.addItem(x_info_item)

                self.multi_cursor_items.append(x_info_item)

            # ,

        except Exception as e:
            print(f"Multi-curve cursor update error: {e}")
            self.update_right_header("")

    def _position_labels_avoid_overlap(self, cursor_values, x_min, x_max, y_min, y_max):
        """优化的标签定位，使用对角线位置避免遮挡曲线
        
        【内存优化】使用对象池复用TextItem
        """
        if not cursor_values:
            return
        
        # 计算视图范围，用于边界检查
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # 标签尺寸估算（像素转数据坐标）用于边界检查
        label_width_data = 80 * x_range / 500
        label_height_data = 25 * y_range / 400
        
        # 使用场景坐标（真实屏幕像素）定位，确保在任何布局下距离一致
        view_box = self.plot_item.getViewBox()
        
        # 设置固定的屏幕像素偏移
        gap_pixels = 5  # 文本框左边缘距离cursor的水平像素间隔
        vertical_gap_pixels = 10  # 垂直像素间隔
        
        # 获取TextItem的字体来动态计算标签尺寸（缓存font_metrics避免重复创建）
        if not hasattr(self, '_cached_font_metrics'):
            sample_text_item = self._get_label_from_pool(0)
            text_font = sample_text_item.textItem.font()
            self._cached_font_metrics = QFontMetrics(text_font)
            self._cached_label_height_pixels = self._cached_font_metrics.height() + 6
        
        font_metrics = self._cached_font_metrics
        label_height_pixels = self._cached_label_height_pixels
        
        for idx, item in enumerate(cursor_values):
            var_name = item['var_name']
            x_pos = item['x_pos']
            y_pos = item['y_pos']
            y_value = item['y_value']
            color = item['color']
            
            # 从对象池获取TextItem并更新其属性
            text_item = self._get_label_from_pool(idx)
            text_item.setText(y_value)
            
            # 复用pen对象或只在颜色变化时创建
            if not hasattr(text_item, '_cached_border_color') or text_item._cached_border_color != color:
                border_pen = pg.mkPen(color, width=1.5)
                text_item.border = border_pen
                text_item._cached_border_color = color
            text_item.setVisible(True)
            
            # 根据实际文本内容动态计算标签宽度
            text_width = font_metrics.horizontalAdvance(y_value)
            label_width_pixels = text_width + 12
            
            # 将数据坐标转换为场景坐标（屏幕像素）
            cursor_scene_pos = view_box.mapViewToScene(QPointF(x_pos, y_pos))
            cursor_scene_x = cursor_scene_pos.x()
            cursor_scene_y = cursor_scene_pos.y()
            
            # 计算文本框中心的偏移（TextItem的anchor=(0.5, 0.5)）
            offset_x_right = gap_pixels + label_width_pixels / 2
            offset_x_left = -(gap_pixels + label_width_pixels / 2)
            offset_y_up = -(vertical_gap_pixels + label_height_pixels / 2)
            offset_y_down = vertical_gap_pixels + label_height_pixels / 2
            
            # 尝试4个候选位置（右上、左上、右下、左下）
            strategies = [
                (offset_x_right, offset_y_up, "右上"),
                (offset_x_left, offset_y_up, "左上"),
                (offset_x_right, offset_y_down, "右下"),
                (offset_x_left, offset_y_down, "左下"),
            ]
            
            label_scene_x, label_scene_y = None, None
            selected_strategy = None
            
            for strategy_idx, (dx_pixels, dy_pixels, name) in enumerate(strategies):
                candidate_scene_x = cursor_scene_x + dx_pixels
                candidate_scene_y = cursor_scene_y + dy_pixels
                
                # 转换回数据坐标检查边界
                candidate_data_pos = view_box.mapSceneToView(QPointF(candidate_scene_x, candidate_scene_y))
                candidate_x = candidate_data_pos.x()
                candidate_y = candidate_data_pos.y()
                
                # 检查是否在数据范围内
                left_ok = candidate_x - label_width_data * 0.5 >= x_min
                right_ok = candidate_x + label_width_data * 0.5 <= x_max
                bottom_ok = candidate_y - label_height_data * 0.5 >= y_min
                top_ok = candidate_y + label_height_data * 0.5 <= y_max
                
                in_bounds = left_ok and right_ok and bottom_ok and top_ok
                
                if in_bounds:
                    label_scene_x = candidate_scene_x
                    label_scene_y = candidate_scene_y
                    label_x = candidate_x
                    label_y = candidate_y
                    selected_strategy = name
                    break
            
            # 如果所有策略都超界，默认使用右上并约束在边界内
            if label_scene_x is None:
                label_scene_x = cursor_scene_x + offset_x_right
                label_scene_y = cursor_scene_y + offset_y_up
                
                label_data_pos = view_box.mapSceneToView(QPointF(label_scene_x, label_scene_y))
                label_x = label_data_pos.x()
                label_y = label_data_pos.y()
                
                label_x = max(x_min + label_width_data * 0.5, 
                             min(x_max - label_width_data * 0.5, label_x))
                label_y = max(y_min + label_height_data * 0.5, 
                             min(y_max - label_height_data * 0.5, label_y))
                selected_strategy = "约束右上"
            
            # 边缘避让逻辑：防止标签在边缘抖动
            edge_margin_strict = label_height_data * 1.5
            y_center = (y_min + y_max) / 2
            y_quarter_upper = y_min + (y_max - y_min) * 0.25
            y_quarter_lower = y_max - (y_max - y_min) * 0.25
            
            data_point_near_bottom = (y_pos - y_min) < edge_margin_strict
            data_point_near_top = (y_max - y_pos) < edge_margin_strict
            
            if data_point_near_bottom:
                label_y = max(y_quarter_upper, label_y)
                label_y = min(label_y, y_center)
            elif data_point_near_top:
                label_y = min(y_quarter_lower, label_y)
                label_y = max(label_y, y_center)
            else:
                edge_margin_soft = label_height_data * 2.0
                if label_y - y_min < edge_margin_soft:
                    label_y = y_min + edge_margin_soft
                elif y_max - label_y < edge_margin_soft:
                    label_y = y_max - edge_margin_soft
            
            text_item.setPos(label_x, label_y)
            text_item.setZValue(201)
            
            text_scene = text_item.scene()
            plot_scene = self.plot_item.scene()
            if text_scene != plot_scene:
                if text_scene is not None:
                    text_scene.removeItem(text_item)
                self.plot_item.addItem(text_item, ignoreBounds=True)
            
            self.multi_cursor_items.append(text_item)

    def toggle_cursor(self, show: bool, hide_values_only: bool = False):
        """
        
        Args:
            show: cursor
            hide_values_only: True,y,vlinex
        """
        mode = self._get_cursor_mode()

        if hide_values_only:
            self._set_vline_visibility_for_mode(True, mode)
            self.cursor_label.setVisible(False)
            self.show_values_only = True
            self._clear_cursor_items()
            self._show_x_position_only()
        else:
            self._set_vline_visibility_for_mode(show, mode)
            self.cursor_label.setVisible(show)
            self.show_values_only = not show

            if not show:
                self._clear_cursor_items()
                self.update_right_header("")
                self.is_cursor_pinned = False
                self.pinned_x_value = None
                self.pinned_index_value = None
                self.pinned_x_values = []
                self.pinned_index_values = []
            else:
                self.update_cursor_label()

    def _show_x_position_only(self, x_positions=None):
        """仅显示x位置（隐藏y值）"""
        try:
            if not self._has_visible_curve_data():
                self._clear_cursor_items()
                self.update_right_header("")
                return
            x_positions = x_positions if x_positions is not None else self._get_cursor_x_positions()
            if not x_positions:
                return

            (x_min, x_max), (y_min, y_max) = self.view_box.viewRange()
            self._clear_cursor_items()

            for idx, x in enumerate(x_positions):
                if x < x_min or x > x_max:
                    continue
                x_str = self._significant_decimal_format_str(value=float(x), ref=self.factor)
                x_info_item = self._get_x_label_from_pool(idx)
                x_info_item.setText(x_str)
                x_info_item.setVisible(True)

                view_rect = self.plot_item.vb.sceneBoundingRect()
                scene_point = self.plot_item.vb.mapViewToScene(pg.Point(x, y_min))
                scene_x = scene_point.x()
                scene_y = view_rect.bottom()
                x_info_item.setPos(scene_x, scene_y)
                x_info_item.setZValue(100000)

                scene = self.plot_item.scene()
                x_scene = x_info_item.scene()
                if x_scene != scene:
                    if x_scene is not None:
                        x_scene.removeItem(x_info_item)
                    scene.addItem(x_info_item)

                self.multi_cursor_items.append(x_info_item)

        except Exception as e:
            import traceback
            traceback.print_exc()

    def _get_cursor_mode(self):
        window = self.window()
        if window and hasattr(window, 'cursor_mode'):
            return window.cursor_mode
        return "1 free cursor"

    def _get_cursor_x_positions(self):
        mode = self._get_cursor_mode()
        if mode == "2 anchored cursor":
            if self.pinned_x_values and len(self.pinned_x_values) >= 2:
                return list(self.pinned_x_values[:2])
            positions = []
            if hasattr(self, 'vline') and self.vline.isVisible():
                positions.append(self.vline.value())
            if hasattr(self, 'vline2') and self.vline2.isVisible():
                positions.append(self.vline2.value())
            return positions
        if mode == "1 anchored cursor":
            if self.pinned_x_values:
                return [self.pinned_x_values[0]]
            if self.pinned_x_value is not None:
                return [self.pinned_x_value]
        if hasattr(self, 'vline'):
            return [self.vline.value()]
        return []

    def _set_vline_visibility_for_mode(self, visible: bool, mode: str):
        if not hasattr(self, 'vline'):
            return
        if mode == "2 anchored cursor":
            self.vline.setVisible(visible)
            if hasattr(self, 'vline2'):
                self.vline2.setVisible(visible)
        else:
            self.vline.setVisible(visible)
            if hasattr(self, 'vline2'):
                self.vline2.setVisible(False)

    def _set_vline_bounds(self, bounds):
        if hasattr(self, 'vline'):
            self.vline.setBounds(bounds)
        if hasattr(self, 'vline2'):
            self.vline2.setBounds(bounds)

    def apply_cursor_mode(self, mode, pinned_x_values):
        if mode == "1 free cursor":
            self.is_cursor_pinned = False
            self.pinned_x_value = None
            self.pinned_index_value = None
            self.pinned_x_values = []
            self.pinned_index_values = []
            if hasattr(self, 'vline'):
                self.vline.setMovable(False)
            if hasattr(self, 'vline2'):
                self.vline2.setMovable(False)
            if hasattr(self.view_box, 'is_cursor_pinned'):
                self.view_box.is_cursor_pinned = False
            self._set_vline_visibility_for_mode(True, mode)
            return

        if mode == "1 anchored cursor":
            self.is_cursor_pinned = True
            self.pinned_x_values = list(pinned_x_values[:1]) if pinned_x_values else self.pinned_x_values[:1]
            if self.pinned_x_values:
                self.pinned_x_value = self.pinned_x_values[0]
            if self.factor != 0 and self.pinned_x_value is not None:
                self.pinned_index_value = (self.pinned_x_value - self.offset) / self.factor
            else:
                self.pinned_index_value = None
            self.pinned_index_values = [self.pinned_index_value] if self.pinned_index_value is not None else []
            if hasattr(self, 'vline') and self.pinned_x_value is not None:
                self.vline.setMovable(True)
                self.vline.blockSignals(True)
                try:
                    self.vline.setPos(self.pinned_x_value)
                finally:
                    self.vline.blockSignals(False)
            if hasattr(self, 'vline2'):
                self.vline2.setMovable(False)
            if hasattr(self.view_box, 'is_cursor_pinned'):
                self.view_box.is_cursor_pinned = True
            self._set_vline_visibility_for_mode(True, mode)
            return

        if mode == "2 anchored cursor":
            self.is_cursor_pinned = True
            if pinned_x_values and len(pinned_x_values) >= 2:
                self.pinned_x_values = list(pinned_x_values[:2])
            elif len(self.pinned_x_values) >= 2:
                self.pinned_x_values = list(self.pinned_x_values[:2])
            elif len(self.pinned_x_values) == 1:
                self.pinned_x_values = [self.pinned_x_values[0], self.pinned_x_values[0]]
            else:
                view_min, view_max = self.view_box.viewRange()[0]
                if view_min is not None and view_max is not None:
                    x1 = view_min + (view_max - view_min) / 3
                    x2 = view_min + 2 * (view_max - view_min) / 3
                    self.pinned_x_values = [x1, x2]
                else:
                    self.pinned_x_values = [0.0, 0.0]
            self.pinned_x_value = self.pinned_x_values[0]
            self.pinned_index_values = []
            for x_val in self.pinned_x_values:
                if self.factor != 0:
                    self.pinned_index_values.append((x_val - self.offset) / self.factor)
            if hasattr(self, 'vline'):
                self.vline.setMovable(True)
            if hasattr(self, 'vline2'):
                self.vline2.setMovable(True)
            if hasattr(self, 'vline') and self.pinned_x_values:
                self.vline.blockSignals(True)
                try:
                    self.vline.setPos(self.pinned_x_values[0])
                finally:
                    self.vline.blockSignals(False)
            if hasattr(self, 'vline2') and len(self.pinned_x_values) > 1:
                self.vline2.blockSignals(True)
                try:
                    self.vline2.setPos(self.pinned_x_values[1])
                finally:
                    self.vline2.blockSignals(False)
            if hasattr(self.view_box, 'is_cursor_pinned'):
                self.view_box.is_cursor_pinned = True
            self._set_vline_visibility_for_mode(True, mode)
            return

    def _has_visible_curve_data(self) -> bool:
        try:
            if self.curves:
                for curve_info in self.curves.values():
                    if not curve_info.get('visible', True):
                        continue
                    x_data = curve_info.get('x_data')
                    if x_data is not None and len(x_data) > 0:
                        return True
                return False
            if self.curve:
                x_data, _ = self.curve.getData()
                return x_data is not None and len(x_data) > 0
            return False
        except Exception:
            return False

    def pin_cursor(self, x_value):
        """
        cursorxplot - 

        plotcursorx,plotcursor
        cursor,vline

        Args:
            x_value (float): x ()
        """
        if getattr(self, '_is_pinning_cursor', False):
            return
        self._is_pinning_cursor = True

        try:
            if not self.window() or not hasattr(self.window(), 'cursor_btn') or not self.window().cursor_btn.isChecked():
                self.window().cursor_btn.setChecked(True)
                self.window().toggle_cursor_all(True)

            curves_x_data = []
            if self.is_multi_curve_mode and self.curves:
                for _, curve_info in self.curves.items():
                    if curve_info.get('visible', True) and 'x_data' in curve_info and curve_info['x_data'] is not None:
                        curves_x_data.append(curve_info['x_data'])
            elif self.curve:
                x_data, _ = self.curve.getData()
                if x_data is not None:
                    curves_x_data.append(x_data)

            if not curves_x_data:
                return

            globally_closest_x = None
            min_distance = float('inf')
            for x_data in curves_x_data:
                if x_data is None or len(x_data) == 0:
                    continue
                try:
                    idx = np.searchsorted(x_data, x_value, side='left')
                    if idx > 0 and idx < len(x_data):
                        dist_left = abs(x_data[idx - 1] - x_value)
                        dist_right = abs(x_data[idx] - x_value)
                        if dist_left < dist_right:
                            idx = idx - 1
                    elif idx == len(x_data):
                        idx = len(x_data) - 1
                except (ValueError, TypeError):
                    idx = np.argmin(np.abs(x_data - x_value))

                nearest_x_in_curve = x_data[idx]
                distance = abs(nearest_x_in_curve - x_value)

                if distance < min_distance:
                    min_distance = distance
                    globally_closest_x = nearest_x_in_curve

            if globally_closest_x is not None:
                display_x = globally_closest_x
                if self.window() and hasattr(self.window(), 'plot_widgets'):
                    main_window = self.window()
                    if hasattr(main_window, 'cursor_mode'):
                        main_window.cursor_mode = "1 anchored cursor"
                    if hasattr(main_window, 'pinned_x_values'):
                        main_window.pinned_x_values = [display_x]
                    widgets_to_update = []
                    for container in main_window.plot_widgets:
                        widget = container.plot_widget
                        widget.apply_cursor_mode("1 anchored cursor", [display_x])
                        if hasattr(widget, '_last_cursor_update_time'):
                            widget._last_cursor_update_time = 0
                        if hasattr(widget.view_box, 'is_cursor_pinned'):
                            widget.view_box.is_cursor_pinned = True
                        widgets_to_update.append(widget)

                    def _delayed_label_update(widgets=widgets_to_update):
                        for widget in widgets:
                            if not getattr(widget, '_is_being_destroyed', False):
                                try:
                                    widget.update_cursor_label()
                                except (RuntimeError, AttributeError):
                                    pass

                    QTimer.singleShot(0, _delayed_label_update)

        finally:
            self._is_pinning_cursor = False


    def free_cursor(self):
        """
        cursor,
        
        plotcursor,cursor
        vline
        """
        self.is_cursor_pinned = False
        self.pinned_x_value = None
        self.pinned_index_value = None
        self.pinned_x_values = []
        self.pinned_index_values = []
        
        if hasattr(self, 'vline'):
            self.vline.setMovable(False)
        if hasattr(self, 'vline2'):
            self.vline2.setMovable(False)
        
        # ViewBoxpin
        if hasattr(self.view_box, 'is_cursor_pinned'):
            self.view_box.is_cursor_pinned = False
        
        # plot
        if self.window() and hasattr(self.window(), 'plot_widgets'):
            main_window = self.window()
            if hasattr(main_window, 'cursor_mode'):
                main_window.cursor_mode = "1 free cursor"
            if hasattr(main_window, 'pinned_x_values'):
                main_window.pinned_x_values = []
            for container in main_window.plot_widgets:
                widget = container.plot_widget
                widget.apply_cursor_mode("1 free cursor", [])
                if hasattr(widget.view_box, 'is_cursor_pinned'):
                    widget.view_box.is_cursor_pinned = False

    def reset_pin_state(self):
        """
        plotpin
        
        plotcursor,:
        - pin
        - vline
        - ViewBoxpin
        """
        self.is_cursor_pinned = False
        self.pinned_x_value = None
        self.pinned_index_value = None
        self.pinned_x_values = []
        self.pinned_index_values = []
        if hasattr(self, 'vline'):
            self.vline.setMovable(False)
        if hasattr(self, 'vline2'):
            self.vline2.setMovable(False)
            self.vline2.setVisible(False)
        if hasattr(self.view_box, 'is_cursor_pinned'):
            self.view_box.is_cursor_pinned = False

    def _update_vline_bounds_from_data(self):
        """根据当前绘制的数据更新vline bounds
        
        这个函数计算当前所有可见曲线的x范围，并更新vline的移动边界。
        在多曲线模式下，使用所有曲线的范围；在单曲线模式下，使用当前曲线的范围。
        """
        try:
            if self.is_multi_curve_mode and self.curves:
                # 多曲线模式：收集所有可见曲线的x值
                all_x_values = []
                for curve_info in self.curves.values():
                    if curve_info.get('visible', True) and curve_info.get('x_data') is not None:
                        all_x_values.extend(curve_info['x_data'])
                
                if all_x_values:
                    min_x, max_x = np.min(all_x_values), np.max(all_x_values)
                    self._set_vline_bounds([min_x, max_x])
                    return min_x, max_x
            elif self.curve is not None:
                # 单曲线模式：使用当前曲线的x值
                x_data, _ = self.curve.getData()
                if x_data is not None and len(x_data) > 0:
                    min_x, max_x = np.min(x_data), np.max(x_data)
                    self._set_vline_bounds([min_x, max_x])
                    return min_x, max_x
            
            # 如果没有数据，使用默认bounds
            if hasattr(self, 'xMin') and hasattr(self, 'xMax'):
                self._set_vline_bounds([self.xMin, self.xMax])
                return self.xMin, self.xMax
            else:
                self._set_vline_bounds([None, None])
                return None, None
        except Exception as e:
            print(f"Warning: Error updating vline bounds: {e}")
            self._set_vline_bounds([None, None])
            return None, None
    
    def _update_cursor_after_plot(self, min_x_bound: float, max_x_bound: float):
        """在绘图或自动缩放后，更新光标线的边界和可见性
        
        根据数据范围更新cursor的移动边界，并根据主窗口的cursor状态决定显示模式。
        
        Args:
            min_x_bound: cursor允许的最小x值
            max_x_bound: cursor允许的最大x值
        """
        main_window = self.window()
        if main_window and hasattr(main_window, 'cursor_btn'):
            # 设置cursor的移动边界
            self._set_vline_bounds([min_x_bound, max_x_bound])
            if getattr(main_window, '_suppress_cursor_refresh', False):
                self.toggle_cursor(False)
                return
            cursor_enabled = main_window.cursor_btn.isChecked()
            cursor_values_hidden = getattr(main_window, 'cursor_values_hidden', False)
            
            # 根据全局cursor状态决定显示模式
            if cursor_enabled and cursor_values_hidden:
                # cursor启用但只显示vline和x值
                self.toggle_cursor(False, hide_values_only=True)
            else:
                # cursor完全启用或禁用
                self.toggle_cursor(cursor_enabled)
        else:
            # 无主窗口或cursor按钮，禁用cursor
            self._set_vline_bounds([None, None])
            self.toggle_cursor(False)

    def clear_value_cache(self):
        #self._value_cache: dict[str, tuple] = {}
        pass
    def datetime_to_unix_seconds(self, series: pd.Series) -> pd.Series:
        """将datetime Series转换为Unix时间戳（秒，float64精度）"""
        if "ns" in str(series.dtype):
            return (series.astype("int64") / 10**9).astype("float64")
        elif "us" in str(series.dtype):
            return (series.astype("int64") / 10**6).astype("float64")
        elif "ms" in str(series.dtype):
            return (series.astype("int64") / 10**3).astype("float64")
        else:
            raise ValueError(f"Unsupported datetime dtype: {series.dtype}")
        
    def get_value_from_name(self,var_name)-> tuple | None:
        main_window = self.window()
        if var_name in main_window.value_cache:
            return main_window.value_cache[var_name]

        raw_values = self.data[var_name]
        dtype_kind = raw_values.dtype.kind
        y_values = None
        y_format = 'number'

        if dtype_kind in "iuf":
            y_values = raw_values
        elif dtype_kind == "b":
            y_values = raw_values.astype(np.int32)
        elif var_name in self.time_channels_info:
            fmt = self.time_channels_info[var_name]
            try:
                if "%H:%M:%S" in fmt:
                    # 时间格式：提取时间部分并转换为Unix时间戳
                    times = pd.to_datetime(raw_values, format=fmt, errors="coerce")
                    today = pd.Timestamp.today().normalize()
                    # 提取从午夜开始的时间差（保留毫秒/微秒精度）
                    time_deltas = times - times.dt.normalize()
                    dt_values = today + time_deltas
                    y_values = self.datetime_to_unix_seconds(dt_values)
                    y_format = 's'
                else:
                    # 日期格式：直接转换为Unix时间戳
                    dt_values = pd.to_datetime(raw_values, format=fmt, errors='coerce')
                    y_values = self.datetime_to_unix_seconds(dt_values)
                    y_format = 'date'
            except (ValueError, TypeError):
                # 无法解析时间格式
                return None, None
        else:
            # 非时间通道：尝试将object等类型转换为数字，只要存在至少一个有效值就接受
            try:
                numeric_values = pd.to_numeric(raw_values, errors='coerce')
            except Exception:
                numeric_values = None

            if numeric_values is not None:
                finite_mask = np.isfinite(numeric_values.to_numpy(dtype=np.float64))
                if finite_mask.any():
                    y_values = numeric_values
                else:
                    return None, None
            else:
                return None, None
        
        if y_values is None:
            return None, None

        main_window.value_cache[var_name] = (y_values, y_format)
        return y_values, y_format
    
    def update_time_correction(self, new_factor, new_offset):
        self._suppress_pin_update = True
        try:
            old_factor = self.factor
            old_offset = self.offset
            self.factor = new_factor
            self.offset = new_offset
            
            if self.is_multi_curve_mode:
                # 多曲线模式：更新所有曲线的x数据
                for var_name, curve_info in self.curves.items():
                    if 'curve' in curve_info and 'x_data' in curve_info:
                        curve = curve_info['curve']
                        # 从旧的x_data反推原始索引，然后应用新的factor和offset
                        old_x = curve_info['x_data']
                        if old_factor != 0:
                            # 反推原始索引：original_index = (old_x - old_offset) / old_factor
                            original_index = (old_x - old_offset) / old_factor
                        else:
                            # 如果old_factor为0（不应该发生），使用默认索引
                            original_index = np.arange(1, len(old_x) + 1)
                        # 重新计算x数据
                        # 【性能优化】使用in-place操作避免创建临时数组（对于百万级数据点）
                        # 注意：original_index已经是计算后的数组，这里直接使用
                        new_x = self.offset + self.factor * original_index
                        # pyqtgraph的setData会内部处理数据，这里直接传入即可
                        curve.setData(new_x, curve_info['y_data'])
                        # 更新存储的x_data
                        curve_info['x_data'] = new_x
            else:
                # 单曲线模式
                if self.original_index_x is not None:
                    # 【性能优化】对于百万级数据点，numpy的向量化操作已经足够高效
                    # 直接计算，无需额外优化（pyqtgraph内部会处理数据）
                    new_x = self.offset + self.factor * self.original_index_x
                    self.curve.setData(new_x, self.original_y)

            # 计算数据长度
            if self.is_multi_curve_mode and self.curves:
                # 多曲线模式：从任一曲线获取数据长度
                first_curve_info = next(iter(self.curves.values()))
                datalength = len(first_curve_info['y_data']) if 'y_data' in first_curve_info else 0
            elif self.original_index_x is not None:
                # 单曲线模式：使用original_index_x
                datalength = len(self.original_index_x)
            else:
                # 回退：从loader获取
                datalength = self.window().loader.datalength if hasattr(self.window(), 'loader') else 0

            global DEFAULT_PADDING_VAL_X
            padding_xVal = DEFAULT_PADDING_VAL_X

            index_min = 1 - padding_xVal * datalength
            index_max = datalength + padding_xVal * datalength
            limits_xMin = self.offset + self.factor * index_min
            limits_xMax = self.offset + self.factor * index_max
            self._set_x_limits_with_min_range(limits_xMin, limits_xMax)

            data_min_x = self.offset + self.factor * 1 if datalength > 0 else 0
            data_max_x = self.offset + self.factor * datalength if datalength > 0 else 1
            self._set_vline_bounds([data_min_x, data_max_x])

            # 更新标记区域（仅在第一个plot上更新，避免重复）
            if self.mark_region is not None and self is self.window().plot_widgets[0].plot_widget:
                old_min, old_max = self.mark_region.getRegion()
                if old_factor != 0:
                    index_min = (old_min - old_offset) / old_factor
                    index_max = (old_max - old_offset) / old_factor
                    new_min = new_offset + new_factor * index_min
                    new_max = new_offset + new_factor * index_max
                    self.mark_region.setRegion([new_min, new_max])
                    self.window().sync_mark_regions(self.mark_region)  # 同步到其他plot
        finally:
            def safe_update_mark_stats():
                main_window = self.window() if hasattr(self, 'window') else None
                if main_window is None or not hasattr(main_window, 'update_mark_stats'):
                    main_window = get_main_window()
                if (
                    main_window is not None
                    and hasattr(main_window, 'update_mark_stats')
                    and not getattr(self, '_is_being_destroyed', False)
                ):
                    main_window.update_mark_stats()
            QTimer.singleShot(0, safe_update_mark_stats)
            self._suppress_pin_update = False


    # ---------------- 拖拽相关 ----------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            var_names = self._extract_var_names_from_text(event.mimeData().text())
            if var_names:
                self._notify_drag_indicator(var_names, hide=False)
                event.acceptProposedAction()
            else:
                self._notify_drag_indicator(hide=True)
                event.ignore()
        else:
            self._notify_drag_indicator(hide=True)
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            var_names = self._extract_var_names_from_text(event.mimeData().text())
            if var_names:
                self._notify_drag_indicator(var_names, hide=False)
                event.acceptProposedAction()
            else:
                self._notify_drag_indicator(hide=True)
                event.ignore()
        else:
            self._notify_drag_indicator(hide=True)
            event.ignore()

    def dragLeaveEvent(self, event):
        self._notify_drag_indicator(hide=True)
        event.accept()

    def dropEvent(self, event):
        """处理变量拖放事件，支持单个或多个变量同时拖入
        
        拖入多个变量时，自动激活多曲线模式并批量添加所有变量。
        批量添加过程中会抑制重复提示，最后统一显示失败的变量。
        
        Args:
            event: 拖放事件对象，包含拖入的变量名（用;;分隔）
        """
        var_names = parse_dragged_var_names(event.mimeData().text())
        self._notify_drag_indicator(hide=True)
        
        if not var_names:
            event.ignore()
            return

        if len(var_names) > 1:
            # 多个变量：批量添加优化
            failed_vars = []
            success_vars = []
            
            # 阶段1：验证所有变量，准备数据
            variables_data = []
            for var_name in var_names:
                is_valid, error_msg = self._validate_plot_data(var_name)
                if not is_valid:
                    failed_vars.append(var_name)
                    continue
                
                success, error_msg, x_array, y_array, y_format = self._prepare_plot_data(var_name)
                if not success:
                    failed_vars.append(var_name)
                    continue
                
                # 检查是否已存在
                if (self.is_multi_curve_mode and var_name in self.curves) or \
                   (not self.is_multi_curve_mode and var_name == self.y_name):
                    failed_vars.append(var_name)
                    continue
                
                variables_data.append((var_name, x_array, y_array, y_format))
            
            # 阶段2：批量添加变量（设置标志避免中间更新）
            self._batch_adding = True
            
            if variables_data:
                # 检查是否需要从单曲线转换
                if not self.is_multi_curve_mode and self.curve and self.y_name:
                    # 迁移单曲线到多曲线字典
                    current_color = 'blue'
                    if hasattr(self.curve, 'opts') and 'pen' in self.curve.opts:
                        current_pen = self.curve.opts['pen']
                        if hasattr(current_pen, 'color'):
                            current_color = current_pen.color().name()
                    
                    self.curves[self.y_name] = {
                        'curve': self.curve,
                        'x_data': self.offset + self.factor * self.original_index_x if self.original_index_x is not None else None,
                        'y_data': self.original_y if self.original_y is not None else None,
                        'color': current_color,
                        'y_format': self.y_format,
                        'visible': True
                    }
                    self.current_color_index = 1
                
                # 批量创建所有曲线
                for var_name, x_array, y_array, y_format in variables_data:
                    x_values = self.offset + self.factor * x_array
                    y_values = y_array
                    
                    # 选择颜色
                    color = self.curve_colors[self.current_color_index % len(self.curve_colors)]
                    self.current_color_index += 1
                    
                    # 创建曲线
                    pen = pg.mkPen(color=color, width=DEFAULT_LINE_WIDTH)
                    curve = self.plot_item.plot(x_values, y_values, pen=pen, name=var_name)
                    
                    # 存储曲线信息
                    self.curves[var_name] = {
                        'curve': curve,
                        'x_data': x_values,
                        'y_data': y_values,
                        'color': color,
                        'y_format': y_format or '',
                        'visible': True
                    }
                    
                    success_vars.append(var_name)
                
                # 更新模式状态
                self.is_multi_curve_mode = len(self.curves) > 1
            
            # 阶段3：批量添加完成，恢复标志
            self._batch_adding = False
            
            # 阶段4：统一更新边界、坐标轴和legend（只执行一次）
            if success_vars:
                # 更新legend
                if self.is_multi_curve_mode:
                    self.update_legend()
                
                # 统一更新坐标轴
                self._update_axes_for_multi_curve(update_x_range=False)
                
                # 统一更新cursor边界
                all_x_values = []
                for curve_info in self.curves.values():
                    if curve_info['visible'] and curve_info['x_data'] is not None:
                        all_x_values.extend(curve_info['x_data'])
                if all_x_values:
                    min_x, max_x = np.min(all_x_values), np.max(all_x_values)
                    self._set_vline_bounds([min_x, max_x])
                    self._update_cursor_after_plot(min_x, max_x)
                
                # 更新cursor标签
                if self.vline.isVisible():
                    self.update_cursor_label()
            
            # 提示失败的变量
            if failed_vars:
                QMessageBox.information(self, "提示", f"以下变量已在绘图中:\n" + "\n".join(failed_vars))
        elif len(var_names) == 1:
            # 单个变量：正常处理
            self.plot_variable(var_names[0])
        
        event.acceptProposedAction()
        self.window().update_mark_stats()

    def _validate_plot_data(self, var_name: str) -> tuple[bool, str]:
        """
        验证绘图数据的有效性
        
        检查变量名和数据源的有效性
        确保数据可以安全地用于绘图
        
        Args:
            var_name: 要验证的变量名称
            
        Returns:
            tuple: (是否有效, 错误信息)
        """
        if not isinstance(var_name, str) or not var_name.strip():
            return False, "变量名无效"
            
        if not hasattr(self, 'data') or self.data is None:
            return False, "没有可用的数据"
            
        if not hasattr(self.data, 'columns'):
            return False, "数据格式无效"
            
        if var_name not in self.data.columns:
            return False, f"变量 {var_name} 不存在"
            
        return True, ""

    def _prepare_plot_data(self, var_name: str) -> tuple[bool, str, np.ndarray, np.ndarray, str]:
        """
        准备绘图数据
        
        从数据源中提取指定变量的数据，进行格式化和预处理
        生成用于绘图的x和y数组
        
        Args:
            var_name: 变量名称
            
        Returns:
            tuple: (是否成功, 错误信息, x数组, y数组, y格式)
        """
        try:
            y_values, y_format = self.get_value_from_name(var_name=var_name)
            
            if y_values is None or len(y_values) == 0:
                return False, f"变量 {var_name} 没有有效数据", None, None, ""
            
            # 转换为numpy数组，根据数据类型选择合适的精度
            # 时间数据（Unix时间戳）使用float64以保留毫秒精度
            # 其他数据使用float32以减少内存
            if isinstance(y_values, pd.Series):
                array_source = y_values.to_numpy()
                safety_source = y_values
            else:
                array_source = np.asarray(y_values)
                safety_source = array_source

            float32_safe, abs_max = _evaluate_float32_safety(safety_source)
            # 检查时间数据：Unix时间戳通常 > 1e8
            is_time_data = bool(abs_max is not None and abs_max > 1e8)
            prefer_float64 = is_time_data or not float32_safe
            target_dtype = np.float64 if prefer_float64 else np.float32

            try:
                if isinstance(y_values, pd.Series):
                    y_array = y_values.to_numpy(dtype=target_dtype)
                else:
                    y_array = np.asarray(array_source, dtype=target_dtype)
            except (OverflowError, ValueError, TypeError):
                if isinstance(y_values, pd.Series):
                    y_array = y_values.to_numpy(dtype=np.float64)
                else:
                    y_array = np.asarray(array_source, dtype=np.float64)

            if target_dtype == np.float32 and np.any(np.isinf(y_array)):
                if isinstance(y_values, pd.Series):
                    y_array = y_values.to_numpy(dtype=np.float64)
                else:
                    y_array = np.asarray(array_source, dtype=np.float64)

            # 检查数据是否全为NaN
            if np.all(np.isnan(y_array)):
                return False, f"变量 {var_name} 的数据全为无效值", None, None, ""
                
            # 【NumPy优化】使用float32类型的索引数组
            x_array = np.arange(1, len(y_array) + 1, dtype=np.float32)
            
            return True, "", x_array, y_array, y_format
            
        except Exception as e:
            return False, f"处理数据时出错: {str(e)}", None, None, ""

    def plot_variable(self, var_name: str, show_duplicate_warning: bool = True) -> bool:
        """
        绘制变量到图表
        
        将指定的数据变量绘制到当前图表中
        包括数据验证、格式化和图形渲染
        
        Args:
            var_name: 要绘制的变量名称
            show_duplicate_warning: 是否显示重复变量警告
            
        Returns:
            bool: 绘制是否成功
        """
        # 验证输入
        is_valid, error_msg = self._validate_plot_data(var_name)
        if not is_valid:
            QMessageBox.warning(self, "错误", error_msg)
            return False
        
        # 准备数据
        success, error_msg, x_array, y_array, y_format = self._prepare_plot_data(var_name)
        if not success:
            QMessageBox.warning(self, "错误", error_msg)
            return False
        
        try:
            # 检查是否是多曲线模式
            if self.is_multi_curve_mode:
                # 多曲线模式：直接添加新曲线
                x_values = self.offset + self.factor * x_array
                return self.add_variable_to_plot(var_name, x_values, y_array, y_format, show_duplicate_warning=show_duplicate_warning)
            
            # 单曲线模式：设置绘图数据
            self.y_format = y_format
            self.y_name = var_name
            # x_array是索引数组，使用float32足够
            self.original_index_x = np.asarray(x_array, dtype=np.float32)
            safe_for_float32, abs_max_plot = _evaluate_float32_safety(y_array)
            keep_float64 = (
                y_format in ['s', 'date']
                or not safe_for_float32
                or (abs_max_plot is not None and abs_max_plot > 1e8)
            )
            target_y_dtype = np.float64 if keep_float64 else np.float32
            self.original_y = np.asarray(y_array, dtype=target_y_dtype)
            x_values = self.offset + self.factor * self.original_index_x
            
            # 单曲线模式：清除旧图并绘制新图
            # 先清除cursor items（包括scene中的items）
            # 绘制新变量时完全清除对象池，避免复用异常状态的items
            self._clear_cursor_items(hide_only=False)
            
            # 手动清理所有图形项，避免PyQtGraph的clearPlots scene不匹配问题
            self._safe_clear_plot_items()
            self.curves.clear()  # 清空多曲线数据
            
            # ========== 性能优化：创建单曲线 ==========
            _pen = pg.mkPen(color='blue', width=DEFAULT_LINE_WIDTH)
            self.curve = self.plot_item.plot(
                x_values, self.original_y, 
                pen=_pen, 
                name=var_name
            )
            
            # 性能优化说明：
            # - 自动降采样：plot_item.setDownsampling(mode='peak', auto=True) 已配置
            # - 视图裁剪：plot_item.setClipToView(True) 已配置
            # - 智能防抖：根据数据量动态调整延迟
            # 这些设置会自动应用到曲线，无需OpenGL也能获得良好性能
            
            # 延迟更新样式（带安全检查）
            self.schedule_style_refresh()

            # 更新标题
            full_title = f"{var_name} ({self.units.get(var_name, '')})".strip()
            self.update_left_header(full_title)
            
            # 设置坐标轴范围
            # 始终保持x轴范围不变，只更新y轴范围
            # 因为所有plot的x轴都是linked的，改变x轴会影响其他plot
            
            # 处理单点或所有点x坐标相同的特殊情况
            special_limits = self.handle_single_point_limits(x_values, self.original_y)
            if special_limits:
                # 单点数据：使用特殊处理的范围
                # handle_single_point_limits已经返回了扩展后的x范围，直接使用
                min_x, max_x, min_y, max_y = special_limits
                # 更新y轴范围和limits
                self._set_safe_y_range(min_y, max_y)
                # 更新x轴limits（不再额外扩展，因为handle_single_point_limits已经扩展过了）
                self._set_x_limits_with_min_range(min_x, max_x)
            else:
                # 正常数据：
                # 1. 基于数据的全范围设置y轴limits（允许用户缩放到所有数据）
                data_min_y = np.nanmin(self.original_y)
                data_max_y = np.nanmax(self.original_y)
                self._set_safe_y_range(data_min_y, data_max_y, set_limits=True)
                
                # 2. 基于当前x轴范围内的数据设置y轴viewRange（初始显示范围）
                current_x_range = self.view_box.viewRange()[0]
                x_min, x_max = current_x_range
                min_y, max_y = self._get_y_range_in_x_window(x_values, self.original_y, x_min, x_max)
                self._set_safe_y_range(min_y, max_y, set_limits=False)
                
                # 3. 更新x轴limits（允许的最大范围），确保可以平移/缩放到数据的范围
                data_min_x = np.min(x_values)
                data_max_x = np.max(x_values)
                padding_x = DEFAULT_PADDING_VAL_X
                limits_xMin = data_min_x - padding_x * (data_max_x - data_min_x)
                limits_xMax = data_max_x + padding_x * (data_max_x - data_min_x)
                self._set_x_limits_with_min_range(limits_xMin, limits_xMax)
            
            # 更新光标 - 在单曲线模式下使用当前数据范围即可
            min_x, max_x = np.min(x_values), np.max(x_values)
            self._set_vline_bounds([min_x, max_x])
            self.plot_item.update()
            self._update_cursor_after_plot(min_x, max_x)

            return True
            
        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"绘制变量时发生错误: {str(e)}")
            return False

    def _compute_valid_min_max(self, values) -> tuple[float | None, float | None]:
        """安全计算数据的最小值和最大值，忽略NaN/INF"""
        if values is None:
            return None, None

        try:
            if isinstance(values, pd.Series):
                arr = pd.to_numeric(values, errors='coerce').to_numpy(dtype=np.float64)
            else:
                arr = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError):
            try:
                arr = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=np.float64)
            except Exception:
                return None, None

        if arr.size == 0:
            return None, None

        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return None, None

        finite_values = arr[finite_mask]
        return float(np.min(finite_values)), float(np.max(finite_values))

    def _get_y_range_in_x_window(self, x_values: np.ndarray, y_values: np.ndarray, x_min: float, x_max: float):
        """计算在指定x轴范围内的y值范围
        
        Args:
            x_values: X轴数据数组
            y_values: Y轴数据数组
            x_min: X轴范围最小值
            x_max: X轴范围最大值
            
        Returns:
            tuple: (min_y, max_y) 在x范围内的y值最小值和最大值
        """
        try:
            # 找到在x范围内的数据点
            mask = (x_values >= x_min) & (x_values <= x_max)
            if not np.any(mask):
                # 如果没有数据点在范围内，返回全部数据的范围
                bounds = self._compute_valid_min_max(y_values)
            else:
                y_in_range = y_values[mask]
                bounds = self._compute_valid_min_max(y_in_range)
                if bounds[0] is None or bounds[1] is None:
                    bounds = self._compute_valid_min_max(y_values)

            if bounds[0] is None or bounds[1] is None:
                return 0.0, 1.0
            return bounds
        except Exception:
            # 出错时返回全部数据范围
            bounds = self._compute_valid_min_max(y_values)
            if bounds[0] is None or bounds[1] is None:
                return 0.0, 1.0
            return bounds
    
    def _setup_plot_axes(self, x_values: np.ndarray, y_values: np.ndarray, update_x_range: bool = True):
        """设置绘图坐标轴
        
        根据数据范围设置X和Y轴的显示范围和限制范围。
        对于单点或所有点x坐标相同的特殊情况，会自动扩展x轴范围。
        
        Args:
            x_values: X轴数据数组
            y_values: Y轴数据数组
            update_x_range: 是否更新X轴范围，默认为True
        """
        try:
            # 处理特殊情况（单点或所有点x坐标相同）
            special_limits = self.handle_single_point_limits(x_values, y_values)
            if special_limits:
                min_x, max_x, min_y, max_y = special_limits
            else:
                min_x = np.min(x_values)
                max_x = np.max(x_values)
                min_y = np.nanmin(y_values)
                max_y = np.nanmax(y_values)
                
            # 计算X轴的限制范围（允许的最大范围）
            padding_x = DEFAULT_PADDING_VAL_X
            limits_xMin = min_x - padding_x * (max_x - min_x)
            limits_xMax = max_x + padding_x * (max_x - min_x)
            
            # 只在update_x_range为True时设置X轴的viewRange（显示范围）
            if update_x_range:
                self.view_box.setXRange(min_x, max_x, padding=DEFAULT_PADDING_VAL_X)
            
            # 设置Y轴范围和X轴limits
            self._set_safe_y_range(min_y, max_y)
            self._set_x_limits_with_min_range(limits_xMin, limits_xMax)
            
        except Exception as e:
            # 出错时使用默认范围
            self._set_safe_y_range(0, 1)
            self._set_x_limits_with_min_range(0, 1)

    def _reset_plot_limits(self):
        """重置绘图限制"""
        try:
            self.plot_item.setLimits(yMin=None, yMax=None)
            self.view_box.setYRange(0, 1, padding=DEFAULT_PADDING_VAL_Y)
            self._set_vline_bounds([None, None])
        except Exception as e:
            print(f"重置绘图限制时出错: {e}")

    def _clear_plot_data(self):
        """清除绘图数据"""
        try:
            # 先清除cursor items（包括scene中的items）
            # 重要：清除plot时需要完全清除对象池（hide_only=False）
            # 这样可以避免对象池中的items处于异常状态（scene=None但PyQtGraph仍认为它属于PlotItem）
            self._clear_cursor_items(hide_only=False)
            
            # 清除所有plot items
            self._safe_clear_plot_items()
            self.axis_y.setLabel(text="")
            self.y_name = ''
            self.y_format = ''
            self.update_left_header("channel name")
            self.update_right_header("")
            
            # 清除单曲线的缓存数据
            if self.curve:
                if hasattr(self.curve, '_cached_pen_key'):
                    delattr(self.curve, '_cached_pen_key')
                if hasattr(self.curve, '_has_symbols'):
                    delattr(self.curve, '_has_symbols')
                try:
                    self.curve.clear()
                except:
                    pass
            
            self.curve = None
            self.original_index_x = None
            self.original_y = None
            
            # 清除多曲线数据
            # 先清除每个曲线的缓存数据
            for var_name, curve_info in self.curves.items():
                if 'curve' in curve_info:
                    curve = curve_info['curve']
                    # 清除样式缓存
                    if hasattr(curve, '_cached_pen_key'):
                        delattr(curve, '_cached_pen_key')
                    if hasattr(curve, '_has_symbols'):
                        delattr(curve, '_has_symbols')
                    # 清除数据
                    try:
                        curve.clear()
                    except:
                        pass
            
            self.curves.clear()
            self.is_multi_curve_mode = False
            self.current_color_index = 0
            
            # 强制垃圾回收，释放内存
            import gc
            gc.collect()
        except Exception as e:
            print(f"清除绘图数据时出错: {e}")

    def clear_plot_item(self):
        """清除绘图项"""
        self._reset_plot_limits()
        self._clear_plot_data()
        
    def add_variable_to_plot(self, var_name: str, x_values: np.ndarray = None, y_values: np.ndarray = None, y_format: str = None, skip_existence_check: bool = False, show_duplicate_warning: bool = True, preferred_color: str | None = None) -> bool:
        """添加变量到多曲线绘图
        
        这是多曲线绘图的核心方法，支持以下功能：
        1. 自动处理单曲线到多曲线模式的转换
        2. 防止重复添加相同变量
        3. 支持批量添加模式（抑制中间坐标轴更新）
        4. 自动颜色分配和曲线样式优化
        
        工作流程：
        - 检查变量是否已存在（可选）
        - 如果是从单曲线模式转换，将现有单曲线迁移到curves字典
        - 创建新曲线并设置性能优化选项
        - 更新坐标轴范围（非批量模式）
        - 更新cursor显示
        
        Args:
            var_name: 变量名称
            x_values: X轴数据（可选，如果为None则从dataframe准备）
            y_values: Y轴数据（可选，如果为None则从dataframe准备）
            y_format: Y轴格式（可选，如's'时间格式、'date'日期格式等）
            skip_existence_check: 是否跳过存在性检查（内部使用）
            show_duplicate_warning: 是否显示重复变量警告（批量添加时设为False）
            
        Returns:
            bool: 添加是否成功。失败原因可能是：变量已存在、数据无效等
        """
        try:
            # 如果数据未提供，则准备数据
            if x_values is None or y_values is None:
                success, error_msg, x_array, y_array, y_format = self._prepare_plot_data(var_name)
                if not success:
                    QMessageBox.warning(self, "错误", error_msg)
                    return False
                x_values = self.offset + self.factor * x_array
                y_values = y_array
            
            # 检查变量是否已存在（除非跳过检查）
            if not skip_existence_check:
                if (self.is_multi_curve_mode and var_name in self.curves) or \
                   (not self.is_multi_curve_mode and var_name == self.y_name):
                    if show_duplicate_warning:
                        QMessageBox.information(self, "提示", f"变量 {var_name} 已在绘图中")
                    return False
            
            # 特殊情况：多曲线模式但curves为空，需要迁移单曲线
            # 说明正在从单曲线过渡到多曲线，需要先将self.curve迁移到curves字典
            if self.is_multi_curve_mode and len(self.curves) == 0 and self.curve and self.y_name:
                # 将当前单曲线添加到curves字典
                current_color = 'blue'
                if hasattr(self.curve, 'opts') and 'pen' in self.curve.opts:
                    current_pen = self.curve.opts['pen']
                    if hasattr(current_pen, 'color'):
                        current_color = current_pen.color().name()
                
                self.curves[self.y_name] = {
                    'curve': self.curve,
                    'x_data': self.offset + self.factor * self.original_index_x if self.original_index_x is not None else None,
                    'y_data': self.original_y if self.original_y is not None else None,
                    'color': current_color,
                    'y_format': self.y_format,
                    'visible': True
                }
                self.current_color_index = 1  # 从第二个颜色开始
                
                # 如果要添加的变量与已迁移的相同，直接返回
                if var_name == self.y_name:
                    if not skip_existence_check and show_duplicate_warning:
                        QMessageBox.information(self, "提示", f"变量 {var_name} 已在绘图中")
                    return False
            
            # 如果当前是单曲线模式，需要先转换到多曲线模式
            if not self.is_multi_curve_mode and self.curve and self.y_name:
                # 检查要添加的变量是否与当前单曲线相同
                if var_name == self.y_name:
                    # 相同变量，不需要转换模式，直接返回
                    if not skip_existence_check and show_duplicate_warning:
                        QMessageBox.information(self, "提示", f"变量 {var_name} 已在绘图中")
                    return False
                
                # 将当前单曲线添加到curves字典
                current_color = 'blue'  # 默认颜色
                if hasattr(self.curve, 'opts') and 'pen' in self.curve.opts:
                    current_pen = self.curve.opts['pen']
                    if hasattr(current_pen, 'color'):
                        current_color = current_pen.color().name()
                
                self.curves[self.y_name] = {
                    'curve': self.curve,
                    'x_data': self.offset + self.factor * self.original_index_x if self.original_index_x is not None else x_values,
                    'y_data': self.original_y if self.original_y is not None else y_values,
                    'color': current_color,
                    'y_format': self.y_format,
                    'visible': True
                }
                self.current_color_index = 1  # 从第二个颜色开始
            
            # 选择颜色
            default_color = self.curve_colors[self.current_color_index % len(self.curve_colors)]
            self.current_color_index += 1
            color = preferred_color or default_color
            
            # ========== 性能优化：创建曲线并配置渲染选项 ==========
            pen = pg.mkPen(color=color, width=DEFAULT_LINE_WIDTH)
            
            # 创建曲线（保持简单参数以确保兼容性）
            curve = self.plot_item.plot(
                x_values, y_values, 
                pen=pen, 
                name=var_name
            )
            
            # 性能优化说明：
            # - 自动降采样：plot_item.setDownsampling(mode='peak', auto=True) 已在setup_plot_area中配置
            # - 视图裁剪：plot_item.setClipToView(True) 已在setup_plot_area中配置
            # - 智能防抖：根据数据量和曲线数动态调整延迟
            # 这些设置会自动应用到所有曲线，无需OpenGL也能获得良好性能
            
            # 存储曲线信息到curves字典
            self.curves[var_name] = {
                'curve': curve,
                'x_data': x_values,
                'y_data': y_values,
                'color': color,
                'y_format': y_format or '',
                'visible': True
            }
            
            # 更新多曲线模式
            self.update_multi_curve_mode()
            
            # 更新坐标轴范围（批量添加时跳过，避免重复更新）
            batch_adding = getattr(self, '_batch_adding', False)
            if not batch_adding:
                # 始终保持x轴范围不变，只更新y轴范围
                # 因为所有plot的x轴都是linked的，改变x轴会影响其他plot
                
                # 1. 先计算所有曲线的全范围y值，用于设置y轴limits
                all_y_values = []
                for curve_info in self.curves.values():
                    if curve_info['visible']:
                        all_y_values.extend(curve_info['y_data'])
                
                if all_y_values:
                    all_data_min_y = np.nanmin(all_y_values)
                    all_data_max_y = np.nanmax(all_y_values)
                    # 设置y轴limits为所有数据的范围
                    self._set_safe_y_range(all_data_min_y, all_data_max_y, set_limits=True)
                
                # 2. 再根据当前x范围设置y轴viewRange
                # 检查是否是单点数据
                special_limits = self.handle_single_point_limits(x_values, y_values)
                if special_limits:
                    # 单点数据：使用特殊处理
                    min_x, max_x, min_y, max_y = special_limits
                    
                    # 检查是否是第一个曲线
                    has_other_curves = len(self.curves) > 1
                    
                    if not has_other_curves:
                        # 第一次添加曲线：直接设置y轴viewRange
                        self._set_safe_y_range(min_y, max_y, set_limits=False)
                    else:
                        # 已有曲线：根据新曲线扩展y轴viewRange
                        current_y_range = self.view_box.viewRange()[1]
                        current_min_y, current_max_y = current_y_range
                        final_min_y = min(current_min_y, min_y)
                        final_max_y = max(current_max_y, max_y)
                        self._set_safe_y_range(final_min_y, final_max_y, set_limits=False)
                    
                    # 更新x轴limits（不再额外扩展，因为handle_single_point_limits已经扩展过了）
                    self._set_x_limits_with_min_range(min_x, max_x)
                else:
                    # 正常数据
                    current_x_range = self.view_box.viewRange()[0]
                    x_min, x_max = current_x_range
                    
                    # 计算新曲线在当前x轴范围内的y值范围
                    new_min_y, new_max_y = self._get_y_range_in_x_window(x_values, y_values, x_min, x_max)
                    
                    # 检查是否是第一个曲线
                    has_other_curves = len(self.curves) > 1
                    
                    if not has_other_curves:
                        # 第一次添加曲线：直接设置y轴viewRange为新曲线在当前x范围内的范围
                        self._set_safe_y_range(new_min_y, new_max_y, set_limits=False)
                    else:
                        # 已有曲线：根据新曲线扩展y轴viewRange
                        current_y_range = self.view_box.viewRange()[1]
                        current_min_y, current_max_y = current_y_range
                        
                        # 扩展y轴viewRange（只考虑新曲线的min/max）
                        final_min_y = min(current_min_y, new_min_y)
                        final_max_y = max(current_max_y, new_max_y)
                        
                        # 更新y轴viewRange
                        self._set_safe_y_range(final_min_y, final_max_y, set_limits=False)
                    
                    # 3. 更新x轴limits（允许的最大范围）以包含所有曲线的数据
                    all_x_values = []
                    for curve_info in self.curves.values():
                        if curve_info['visible']:
                            all_x_values.extend(curve_info['x_data'])
                    if all_x_values:
                        data_min_x = np.min(all_x_values)
                        data_max_x = np.max(all_x_values)
                        padding_x = DEFAULT_PADDING_VAL_X
                        limits_xMin = data_min_x - padding_x * (data_max_x - data_min_x)
                        limits_xMax = data_max_x + padding_x * (data_max_x - data_min_x)
                        self._set_x_limits_with_min_range(limits_xMin, limits_xMax)
            
            # 更新cursor边界 - 使用所有曲线的x范围（而不仅仅是当前添加的变量）
            all_x_values = []
            for curve_info in self.curves.values():
                if curve_info['visible'] and curve_info['x_data'] is not None:
                    all_x_values.extend(curve_info['x_data'])
            if all_x_values:
                min_x, max_x = np.min(all_x_values), np.max(all_x_values)
            else:
                # 如果没有其他曲线，使用当前变量的范围
                min_x, max_x = np.min(x_values), np.max(x_values)
            self._set_vline_bounds([min_x, max_x])
            
            # 应用全局cursor值显示状态
            self._update_cursor_after_plot(min_x, max_x)
            
            # 如果cursor可见，立即更新cursor标签以显示新添加的曲线
            if self.vline.isVisible():
                self.update_cursor_label()
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"添加变量时发生错误: {str(e)}")
            return False
    
    def update_multi_curve_mode(self):
        """更新多曲线模式状态"""
        curve_count = len(self.curves)
        
        # 如果正在批量添加，不要自动切换模式
        if not hasattr(self, '_batch_adding'):
            self._batch_adding = False
            
        if not self._batch_adding:
            self.is_multi_curve_mode = curve_count > 1
        
        if self.is_multi_curve_mode:
            # 多曲线模式：显示legend
            self.update_legend()
        else:
            # 单曲线模式：显示传统标题
            if curve_count == 1:
                var_name = list(self.curves.keys())[0]
                full_title = f"{var_name} ({self.units.get(var_name, '')})".strip()
                self.update_left_header(full_title)
            else:
                self.update_left_header("channel name")
                self.update_right_header("")
    
    def update_legend(self):
        """更新图例显示
        
        在多曲线模式下，在左上角显示所有曲线的图例。
        图例样式：
        - 可见曲线：实心方块(■) + 曲线颜色 + 变量名(单位)
        - 隐藏曲线：空心方块(□) + 半透明颜色 + 灰色文字
        
        点击图例中的曲线名可以切换该曲线的显示/隐藏状态。
        """
        if not self.is_multi_curve_mode:
            return
            
        # 构建图例文本（包含所有曲线，不管是否可见）
        legend_items = []
        for var_name, curve_info in self.curves.items():
            color = curve_info['color']
            unit = self.units.get(var_name, '')
            legend_text = f"{var_name} ({unit})" if unit else var_name
            
            # 根据可见性调整显示样式
            if curve_info['visible']:
                # 可见：实心方块 + 加粗文字
                legend_items.append(f"<span style='color: {color}; font-weight: bold;'>■</span> {legend_text}")
            else:
                # 隐藏：空心方块 + 灰色文字
                legend_items.append(f"<span style='color: {color}; opacity: 0.5;'>□</span> <span style='color: gray;'>{legend_text}</span>")
        
        if legend_items:
            legend_text = " | ".join(legend_items)
            self.update_left_header(legend_text)
        else:
            self.update_left_header("channel name")
    
    def toggle_curve_visibility_by_name(self, var_name):
        """通过变量名切换曲线可见性
        
        点击图例中的曲线名时调用，切换该曲线的显示/隐藏状态。
        如果曲线对象失效（不在scene中），会尝试重新创建。
        
        Args:
            var_name: 要切换可见性的变量名
        """
        if var_name not in self.curves:
            return
            
        curve_info = self.curves[var_name]
        # 切换可见性状态
        curve_info['visible'] = not curve_info['visible']
        new_visible = curve_info['visible']
        
        # 更新曲线对象的可见性
        if 'curve' in curve_info:
            curve_obj = curve_info['curve']
            
            try:
                # 检查曲线对象是否仍然有效
                if curve_obj.scene() is not None:
                    curve_obj.setVisible(new_visible)
                else:
                    # 曲线对象已经不在scene中，重新创建
                    self._recreate_curve(var_name)
            except Exception:
                # 尝试重新创建曲线
                self._recreate_curve(var_name)
        
        # 更新图例显示
        self.update_legend()
        
        # 更新cursor显示（如果cursor可见）
        if self.vline.isVisible():
            self.update_cursor_label()
    
    def _recreate_curve(self, var_name):
        """重新创建失效的曲线"""
        try:
            if var_name in self.curves:
                curve_info = self.curves[var_name]
                # 重新绘制曲线
                success = self.add_variable_to_plot(var_name, skip_existence_check=True)
                if success:
                    pass
                else:
                    pass
            else:
                pass
        except Exception as e:
            pass
    
    def _on_legend_clicked(self, event):
        """Legend点击事件处理
        
        使用QTextDocument进行精确的hitTest，定位用户点击的是哪条曲线，
        然后切换该曲线的显示/隐藏状态。
        
        处理流程：
        1. 将legend HTML文本解析为QTextDocument
        2. 使用hitTest找到点击位置对应的文本位置
        3. 根据文本位置确定对应的曲线索引
        4. 调用toggle_curve_visibility_by_name切换曲线可见性
        
        Args:
            event: 鼠标点击事件
        """
        if not self.is_multi_curve_mode:
            return
        
        # 获取点击位置
        pos = event.pos()
        click_x = pos.x()
        
        # 改进的点击检测：基于实际legend文本内容进行更精确的匹配
        if not self.curves:
            return
            
        # 获取当前曲线列表（按legend显示顺序）
        curve_list = list(self.curves.items())
        
        if not curve_list:
            return
        
        # 使用QTextDocument + hitTest精确定位点击位置
        # 构建完整的legend HTML（与update_legend完全一致）
        legend_parts = []
        for var_name, curve_info in curve_list:
            color = curve_info['color']
            unit = self.units.get(var_name, '')
            legend_text = f"{var_name} ({unit})" if unit else var_name
            
            if curve_info['visible']:
                symbol = f"<span style='color: {color}; font-weight: bold;'>■</span>"
                legend_parts.append(f"{symbol} {legend_text}")
            else:
                # 隐藏时：空心方格 + 灰色文字（与update_legend一致）
                symbol = f"<span style='color: {color}; opacity: 0.5;'>□</span>"
                legend_parts.append(f"{symbol} <span style='color: gray;'>{legend_text}</span>")
        
        full_html = " | ".join(legend_parts)
        
        # 创建QTextDocument来进行hitTest
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(self.label_left.font())
        doc.setHtml(full_html)
        
        # 使用hitTest找到点击位置对应的字符位置
        layout = doc.documentLayout()
        hit_pos = layout.hitTest(QPointF(click_x, pos.y()), Qt.HitTestAccuracy.ExactHit)
        
        # 计算每个legend部分在HTML中的字符位置范围
        clicked_index = -1
        char_pos = 0
        item_ranges = []
        
        for i, part in enumerate(legend_parts):
            if i > 0:
                # 加上分隔符" | "的长度（注意：纯文本长度，不是HTML长度）
                char_pos += 3  # " | " = 3个字符
            
            part_start = char_pos
            # 计算这个part的纯文本长度（去除HTML标签）
            part_doc = QTextDocument()
            part_doc.setHtml(part)
            part_text_length = len(part_doc.toPlainText())
            part_end = part_start + part_text_length
            
            item_ranges.append({
                'index': i,
                'start': part_start,
                'end': part_end,
                'var_name': curve_list[i][0]
            })
            
            if hit_pos >= part_start and hit_pos < part_end:
                clicked_index = i
                break
            
            char_pos = part_end
        
        # 如果hitTest没有精确匹配（点击在分隔符区域或文本范围外），找距离最近的item
        if clicked_index == -1:
            # 如果hitTest失败（返回-1），说明点击在文本范围外
            if hit_pos < 0:
                # 根据实际点击像素位置判断：左侧选第一个，右侧选最后一个
                total_text_width = doc.size().width()
                if click_x < total_text_width / 2:
                    clicked_index = 0
                else:
                    clicked_index = len(curve_list) - 1
            else:
                # 计算到每个item的距离，选择最近的
                min_distance = float('inf')
                for item in item_ranges:
                    if hit_pos < item['start']:
                        distance = item['start'] - hit_pos
                    elif hit_pos >= item['end']:
                        distance = hit_pos - item['end']
                    else:
                        distance = 0
                    
                    if distance < min_distance:
                        min_distance = distance
                        clicked_index = item['index']
        
        # 确保索引在有效范围内
        clicked_index = max(0, min(clicked_index, len(curve_list) - 1))
        
        # 切换对应曲线的可见性
        var_name = curve_list[clicked_index][0]
        self.toggle_curve_visibility_by_name(var_name)
    
    def _update_axes_for_multi_curve(self, update_x_range: bool = False):
        """为多曲线更新坐标轴范围
        
        计算所有可见曲线的数据范围，并更新坐标轴显示范围。
        只考虑visible=True的曲线，忽略隐藏的曲线。
        
        Args:
            update_x_range: 是否更新X轴范围。默认为False，保持当前x轴范围不变。
                           当为True时（通常是第一次添加曲线或批量添加完成），会设置x轴范围为数据的全范围。
        """
        if not self.curves:
            return
            
        # 计算所有可见曲线的数据范围
        all_x_values = []
        all_y_values = []
        
        for var_name, curve_info in self.curves.items():
            if curve_info['visible']:
                all_x_values.extend(curve_info['x_data'])
                all_y_values.extend(curve_info['y_data'])
        
        if all_x_values and all_y_values:
            x_values = np.array(all_x_values)
            y_values = np.array(all_y_values)
            
            if update_x_range:
                # 更新x和y轴范围（第一次添加曲线或批量添加完成）
                self._setup_plot_axes(x_values, y_values, update_x_range=True)
            else:
                # 保持x轴范围不变，只更新y轴范围
                
                # 1. 先基于所有数据的全范围设置y轴limits
                all_data_min_y = np.nanmin(y_values)
                all_data_max_y = np.nanmax(y_values)
                self._set_safe_y_range(all_data_min_y, all_data_max_y, set_limits=True)
                
                # 2. 再根据当前x范围设置y轴viewRange
                # 检查是否是单点数据
                special_limits = self.handle_single_point_limits(x_values, y_values)
                if special_limits:
                    # 单点数据：使用特殊处理
                    # handle_single_point_limits已经返回了扩展后的x范围，直接使用
                    min_x, max_x, min_y, max_y = special_limits
                    self._set_safe_y_range(min_y, max_y, set_limits=False)
                    # 更新x轴limits（不再额外扩展，因为handle_single_point_limits已经扩展过了）
                    self._set_x_limits_with_min_range(min_x, max_x)
                else:
                    # 正常数据
                    current_x_range = self.view_box.viewRange()[0]
                    x_min, x_max = current_x_range
                    
                    # 计算所有曲线在当前x轴范围内的y值范围
                    all_y_in_range = []
                    for var_name, curve_info in self.curves.items():
                        if curve_info['visible']:
                            min_y, max_y = self._get_y_range_in_x_window(
                                np.array(curve_info['x_data']), 
                                np.array(curve_info['y_data']), 
                                x_min, 
                                x_max
                            )
                            all_y_in_range.extend([min_y, max_y])
                    
                    if all_y_in_range:
                        final_min_y = np.nanmin(all_y_in_range)
                        final_max_y = np.nanmax(all_y_in_range)
                        self._set_safe_y_range(final_min_y, final_max_y, set_limits=False)
                    
                    # 3. 更新x轴limits（允许的最大范围）
                    data_min_x = np.min(x_values)
                    data_max_x = np.max(x_values)
                    padding_x = DEFAULT_PADDING_VAL_X
                    limits_xMin = data_min_x - padding_x * (data_max_x - data_min_x)
                    limits_xMax = data_max_x + padding_x * (data_max_x - data_min_x)
                    self._set_x_limits_with_min_range(limits_xMin, limits_xMax)

    # ---------------- 双击轴弹出对话框 ----------------
    def mouseDoubleClickEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.MiddleButton):
            super().mouseDoubleClickEvent(event)
            return
        
        if event.button() == Qt.MiddleButton:
            self.clear_plot_item()
            self.window().update_mark_stats()
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            
            # 获取坐标轴区域
            y_axis_rect_scene = self.axis_y.mapToScene(self.axis_y.boundingRect()).boundingRect()
            x_axis_rect_scene = self.axis_x.mapToScene(self.axis_x.boundingRect()).boundingRect()
            
            # 获取绘图区域的实际范围（排除坐标轴区域）- 使用view_box而不是plot_item
            view_box_rect = self.view_box.boundingRect()
            view_box_rect_scene = self.view_box.mapToScene(view_box_rect).boundingRect()
            
            # 缩小X轴检测区域，只检测X轴标签区域（底部部分）
            x_axis_label_rect = QRectF(x_axis_rect_scene.left(), x_axis_rect_scene.bottom() - 30, x_axis_rect_scene.width(), 30)

            global DEFAULT_PADDING_VAL_X
            
            # 优先检测X轴标签区域（最具体）
            if x_axis_label_rect.contains(scene_pos):
                dialog = AxisDialog(self.axis_x, self.view_box, "X", self)
                if dialog.exec():
                    min_val, max_val = self.view_box.viewRange()[0]
                    for view in self.window().findChildren(DraggableGraphicsLayoutWidget):
                        #view.view_box.setXRange(min_val, max_val, padding=0.00)
                        #view.plot_item.setXRange(min_val, max_val, padding=0.00)
                        self.set_xrange_with_link_handling(xmin=min_val,xmax=max_val,padding=DEFAULT_PADDING_VAL_X)
                        view.plot_item.update()
                return
            # 然后检测绘图区域（在检测Y轴之前）
            elif view_box_rect_scene.contains(scene_pos):
                # 双击绘图区域（网格内部），弹出变量编辑器
                dialog = PlotVariableEditorDialog(self, self.window())
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
                return
            # 最后检测Y轴区域（最后兜底）
            elif y_axis_rect_scene.contains(scene_pos):
                dialog = AxisDialog(self.axis_y, self.view_box, "Y", self)
                if dialog.exec():
                    self.plot_item.update()
                return
        return super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier:
            self.origin = event.pos()
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rubberBand.isVisible():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rubberBand.isVisible() and event.button() == Qt.LeftButton:
            self.rubberBand.hide()
            rect = self.rubberBand.geometry()
            if rect.width() > 10 and rect.height() > 10:  # 避免误触
                topLeft = self.mapToScene(rect.topLeft())
                bottomRight = self.mapToScene(rect.bottomRight())

                p1 = self.view_box.mapSceneToView(topLeft)
                p2 = self.view_box.mapSceneToView(bottomRight)

                x_min = min(p1.x(), p2.x())
                x_max = max(p1.x(), p2.x())
                y_min = min(p1.y(), p2.y())
                y_max = max(p1.y(), p2.y())

                # 添加10% margin
                dx = x_max - x_min
                dy = y_max - y_min
                margin = 0.1
                x_min -= margin * dx
                x_max += margin * dx
                y_min -= margin * dy
                y_max += margin * dy

                self.view_box.setXRange(x_min, x_max, padding=0)
                self.view_box.setYRange(y_min, y_max, padding=0)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def add_mark_region(self, min_x, max_x):
        self.mark_region = pg.LinearRegionItem([min_x, max_x], movable=True)
        #self.mark_region.setBrush(pg.mkBrush(181,196,177, 80))
        for line in self.mark_region.lines:
            line.setHoverPen(pg.mkPen(color='r', width=10)) 
            
        self.plot_item.addItem(self.mark_region)
        self.mark_region.sigRegionChanged.connect(self.window().sync_mark_regions)

    def remove_mark_region(self):
        if self.mark_region and self.mark_region.scene() is not None:
            self.plot_item.removeItem(self.mark_region)
        self.mark_region = None

    def update_mark_region(self):
        if self.mark_region:
            old_min, old_max = self.mark_region.getRegion()
            # 更新基于新factor/offset，但由于x是scaled的，不需要额外缩放
            self.mark_region.setRegion([old_min, old_max])  # 实际不需要变，因为x已scale

    def get_mark_stats(self):
        """获取标记区域的统计信息
        
        【NumPy优化】使用NumPy掩码数组批量计算统计值，避免循环过滤
        """
        if not self.mark_region:
            return None
        
        min_x, max_x = self.mark_region.getRegion()
        
        if self.is_multi_curve_mode:
            # 多曲线模式：返回每个曲线的统计信息
            stats_list = []
            for var_name, curve_info in self.curves.items():
                if not curve_info.get('visible', True):
                    continue
                
                if 'curve' not in curve_info:
                    continue
                
                # 【NumPy优化】优先使用缓存的x_data和y_data，如果没有则从curve获取
                if 'x_data' in curve_info and 'y_data' in curve_info:
                    # 使用缓存的x_data和y_data（更准确且更快）
                    x_data = curve_info['x_data']
                    y_data = curve_info['y_data']
                elif 'y_data' in curve_info:
                    # 只有y_data，需要重新计算x_data（向后兼容）
                    x_data = self.offset + self.factor * np.arange(1, len(curve_info['y_data']) + 1, dtype=np.float32)
                    y_data = curve_info['y_data']
                else:
                    # 从curve获取（最慢但最可靠）
                    curve = curve_info['curve']
                    x_data, y_data = curve.getData()
                    if x_data is None or len(x_data) == 0:
                        continue
                
                # 确保是NumPy数组（保持原有精度，不强制转换）
                x_data = np.asarray(x_data)
                y_data = np.asarray(y_data)
                # 如果是整数类型且幅值适合，则转换为float32以减少内存
                if x_data.dtype.kind in 'iu':
                    safe_x, _ = _evaluate_float32_safety(x_data)
                    x_dtype = np.float32 if safe_x else np.float64
                    x_data = x_data.astype(x_dtype)
                if y_data.dtype.kind in 'iu':
                    safe_y, _ = _evaluate_float32_safety(y_data)
                    y_dtype = np.float32 if safe_y else np.float64
                    y_data = y_data.astype(y_dtype)
                
                # 计算边界点
                idx_left = np.argmin(np.abs(x_data - min_x))
                idx_right = np.argmin(np.abs(x_data - max_x))
                x1 = x_data[idx_left]
                y1 = y_data[idx_left]
                x2 = x_data[idx_right]
                y2 = y_data[idx_right]
                dx = x2 - x1
                dy = y2 - y1
                slope = float('inf') if dx == 0 else dy / dx
                
                # 【NumPy优化】使用掩码数组批量计算统计值
                mask = (x_data >= min_x) & (x_data <= max_x)
                if not np.any(mask):
                    y_avg = y_max = y_min = np.nan
                else:
                    y_masked = y_data[mask]
                    valid_y = y_masked[~np.isnan(y_masked)]
                    if len(valid_y) > 0:
                        y_avg = np.mean(valid_y)
                        y_max = np.max(valid_y)
                        y_min = np.min(valid_y)
                    else:
                        y_avg = y_max = y_min = np.nan
                
                # 添加变量名到标签
                unit = self.units.get(var_name, '')
                label = f"{var_name} ({unit})" if unit else var_name
                
                stats_list.append((x1, x2, y1, y2, dx, dy, slope, label, y_avg, y_max, y_min))
            
            return stats_list if stats_list else None
        else:
            # 单曲线模式：优先使用original_index_x和original_y
            if not self.curve:
                return None
            
            # 【NumPy优化】优先使用original_index_x和original_y
            if hasattr(self, 'original_index_x') and hasattr(self, 'original_y') and self.original_index_x is not None:
                x_data = self.offset + self.factor * self.original_index_x
                y_data = self.original_y
            else:
                x_data, y_data = self.curve.getData()
                if x_data is None or len(x_data) == 0:
                    return None
            
            # 确保是NumPy数组（保持原有精度，不强制转换）
            x_data = np.asarray(x_data)
            y_data = np.asarray(y_data)
            # 如果是整数类型且幅值适合，则转换为float32以减少内存
            if x_data.dtype.kind in 'iu':
                safe_x, _ = _evaluate_float32_safety(x_data)
                x_dtype = np.float32 if safe_x else np.float64
                x_data = x_data.astype(x_dtype)
            if y_data.dtype.kind in 'iu':
                safe_y, _ = _evaluate_float32_safety(y_data)
                y_dtype = np.float32 if safe_y else np.float64
                y_data = y_data.astype(y_dtype)
            
            idx_left = np.argmin(np.abs(x_data - min_x))
            idx_right = np.argmin(np.abs(x_data - max_x))
            x1 = x_data[idx_left]
            y1 = y_data[idx_left]
            x2 = x_data[idx_right]
            y2 = y_data[idx_right]
            dx = x2 - x1
            dy = y2 - y1
            slope = float('inf') if dx == 0 else dy / dx
            
            # 【NumPy优化】使用掩码数组批量计算统计值
            mask = (x_data >= min_x) & (x_data <= max_x)
            if not np.any(mask):
                y_avg = y_max = y_min = np.nan
            else:
                y_masked = y_data[mask]
                valid_y = y_masked[~np.isnan(y_masked)]
                if len(valid_y) > 0:
                    y_avg = np.mean(valid_y)
                    y_max = np.max(valid_y)
                    y_min = np.min(valid_y)
                else:
                    y_avg = y_max = y_min = np.nan
            
            return [(x1, x2, y1, y2, dx, dy, slope, self.label_left.text(), y_avg, y_max, y_min)]

    def _apply_plot_style(self, show_symbols: bool):
        """应用绘图样式 - 基于xrange只有两种搭配：细线+symbol 或 粗线无symbol

        【内存优化】缓存pen对象，避免zoom时重复创建导致内存泄漏
        """
        try:
            # 优先检查curves字典（多曲线模式或从多曲线删到单曲线的情况）
            if self.curves:
                # 有curves字典：遍历所有曲线应用样式
                for var_name, curve_info in self.curves.items():
                    if 'curve' not in curve_info:
                        continue

                    curve = curve_info['curve']
                    color = curve_info.get('color', 'blue')

                    # 缓存pen对象：检查当前样式是否匹配，避免重复创建
                    if show_symbols:
                        # xrange小：细线 + 符号
                        cache_key = f'thin_{color}'
                        if not hasattr(curve, '_cached_pen_key') or curve._cached_pen_key != cache_key:
                            pen = pg.mkPen(color=color, width=THIN_LINE_WIDTH)
                            curve.setPen(pen)
                            curve._cached_pen_key = cache_key

                        if not hasattr(curve, '_has_symbols') or not curve._has_symbols:
                            curve.setSymbol('s')
                            curve.setSymbolSize(3)
                            curve.setSymbolPen(color)
                            curve.setSymbolBrush(color)
                            curve._has_symbols = True
                    else:
                        # xrange大：粗线无符号
                        cache_key = f'thick_{color}'
                        if not hasattr(curve, '_cached_pen_key') or curve._cached_pen_key != cache_key:
                            pen = pg.mkPen(color=color, width=THICK_LINE_WIDTH)
                            curve.setPen(pen)
                            curve._cached_pen_key = cache_key

                        if not hasattr(curve, '_has_symbols') or curve._has_symbols:
                            curve.setSymbol(None)
                            curve._has_symbols = False
            elif self.curve:
                # 没有curves字典但有单曲线：使用self.curve
                # 获取当前曲线的颜色
                current_pen = self.curve.opts.get('pen', pg.mkPen('blue'))
                color = current_pen.color().name() if hasattr(current_pen, 'color') else 'blue'

                # 缓存pen对象：检查当前样式是否匹配，避免重复创建
                if show_symbols:
                    # xrange小：细线 + 符号
                    cache_key = f'thin_{color}'
                    if not hasattr(self.curve, '_cached_pen_key') or self.curve._cached_pen_key != cache_key:
                        pen = pg.mkPen(color=color, width=THIN_LINE_WIDTH)
                        self.curve.setPen(pen)
                        self.curve._cached_pen_key = cache_key

                    if not hasattr(self.curve, '_has_symbols') or not self.curve._has_symbols:
                        self.curve.setSymbol('s')
                        self.curve.setSymbolSize(3)
                        self.curve.setSymbolPen(color)
                        self.curve.setSymbolBrush(color)
                        self.curve._has_symbols = True
                else:
                    # xrange大：粗线无符号
                    cache_key = f'thick_{color}'
                    if not hasattr(self.curve, '_cached_pen_key') or self.curve._cached_pen_key != cache_key:
                        pen = pg.mkPen(color=color, width=THICK_LINE_WIDTH)
                        self.curve.setPen(pen)
                        self.curve._cached_pen_key = cache_key

                    if not hasattr(self.curve, '_has_symbols') or self.curve._has_symbols:
                        self.curve.setSymbol(None)
                        self.curve._has_symbols = False
        except Exception as e:
            print(f"应用绘图样式时出错: {e}")

    def _calculate_visible_points(self, range):
        """计算当前可见范围的点数估算
        
        Args:
            range: 视图范围 [[x_min, x_max], [y_min, y_max]]
            
        Returns:
            tuple: (index_range_width, visible_points)
                - index_range_width: 索引范围宽度（考虑factor）
                - visible_points: 可见点数估算（考虑曲线数量）
        """
        # 获取当前视图的xRange
        x_min, x_max = range[0]
        x_range_width = x_max - x_min
        
        # 考虑factor的影响 - 将xRange转换为索引范围
        # x = offset + factor * index，所以 index_range = x_range / factor
        if hasattr(self, 'factor') and self.factor != 0:
            index_range_width = x_range_width / abs(self.factor)
        else:
            index_range_width = x_range_width
        
        # 计算曲线数量（考虑单曲线和多曲线两种模式）
        if hasattr(self, 'is_multi_curve_mode') and self.is_multi_curve_mode:
            # 多曲线模式：使用 self.curves 字典的长度
            curve_count = len(self.curves) if hasattr(self, 'curves') and self.curves else 0
        else:
            # 单曲线模式：检查是否有曲线
            curve_count = 1 if hasattr(self, 'curve') and self.curve is not None else 0
        
        # 至少按1条曲线计算（避免除0或无意义的计算）
        curve_count = max(curve_count, 1)
        
        # 计算可见点数：索引范围 × 曲线数量
        visible_points = index_range_width * curve_count
        
        return index_range_width, visible_points
    
    def update_plot_style(self, view_box, range, rect=None):
        """更新绘图样式 - 基于xRange宽度判断，只有两种搭配：细线+symbol 或 粗线无symbol
        
        【性能优化】交互期间降低样式更新频率，支持百万级数据点流畅绘制
        """
        try:
            # 【安全检查】如果正在更新数据或对象被销毁，跳过样式更新
            if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
                return
            
            # 【安全检查】确保关键对象存在
            if not hasattr(self, 'factor') or not hasattr(self, 'plot_item'):
                return
            
            # 【性能优化】交互期间：完全跳过样式更新，避免遍历所有曲线导致卡顿
            # 样式更新只在交互结束后执行一次，保证缩放时的流畅性
            is_interacting = getattr(self, '_is_interacting', False)
            if is_interacting:
                return  # 交互期间完全跳过样式更新，避免卡顿
            
            # 使用共用方法计算索引范围
            index_range_width, visible_points = self._calculate_visible_points(range)
            
            # 基于索引范围宽度判断样式：小于阈值显示symbol（细线+symbol），否则粗线无symbol
            global XRANGE_THRESHOLD_FOR_SYMBOLS
            show_symbols = index_range_width < XRANGE_THRESHOLD_FOR_SYMBOLS
            
            # 应用样式到所有曲线
            self._apply_plot_style(show_symbols)
            
        except Exception as e:
            print(f"更新绘图样式时出错: {e}")


    @safe_callback
    def _on_range_changed(self, view_box, range, changed=None):
        """ViewBox范围变化回调处理"""
        try:
            if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
                self._cancel_ui_refresh()
                return

            if getattr(self, '_is_syncing_range', False):
                return

            if not self._is_interacting:
                self._is_interacting = True
                self._start_interaction()

            if hasattr(self, '_interaction_timer'):
                self._interaction_timer.stop()
                self._interaction_timer.start(UI_DEBOUNCE_DELAY_MS)

            if self._is_interacting:
                self._cancel_ui_refresh('style', 'cursor')
                return

            self._queue_ui_refresh()
        except Exception as e:
            print(f"范围变化处理出错: {e}")

    def _start_interaction(self):
        """开始交互时的优化处理
        
        类似iOS的快照策略：在交互期间临时降低渲染质量
        """
        try:
            # 【性能优化】交互期间临时提高降采样阈值，减少渲染的点数
            # 这样可以显著提升缩放时的流畅度
            if hasattr(self, 'plot_item'):
                # 保存原始降采样设置
                if not hasattr(self, '_original_downsample_ds'):
                    # 获取当前降采样设置（如果有）
                    self._original_downsample_ds = getattr(self.plot_item, '_downsample', None)
                
                # 临时提高降采样阈值：交互期间使用更激进的降采样
                # 通过设置更大的ds值来减少渲染的点数
                # 注意：pyqtgraph的auto模式会自动处理，这里主要是确保降采样更激进
                # 实际上，pyqtgraph的auto模式已经会根据可见区域自动调整
                # 但我们可以通过临时禁用某些昂贵的操作来提升性能
                pass  # pyqtgraph的auto模式已经足够智能，无需手动调整
            
            # 【性能优化】交互期间禁用样式更新（已在update_plot_style中实现）
            # 这样可以避免在缩放时遍历所有曲线并更新样式
        except Exception as e:
            print(f"开始交互优化时出错: {e}")
    
    def _end_interaction(self):
        """交互结束时恢复正常渲染质量

        类似iOS在缩放结束后重新渲染页面内容
        """
        try:
            self._is_interacting = False
            # 交互结束后，立即刷新样式、cursor和统计
            self._queue_ui_refresh(immediate=True)
        except Exception as e:
            print(f"结束交互优化时出错: {e}")
    
    def _update_cursor_after_interaction(self):
        """交互结束后更新cursor标签"""
        try:
            # 检查cursor是否可见且不在交互状态
            main_window = self.window()
            cursor_enabled = True
            if main_window and hasattr(main_window, 'cursor_btn'):
                cursor_enabled = main_window.cursor_btn.isChecked()
            
            if (not getattr(self, '_is_interacting', False) and 
                hasattr(self, 'vline') and self.vline.isVisible() and
                cursor_enabled):
                self.update_cursor_label()
        except Exception as e:
            pass  # 忽略cursor更新错误

    def _delayed_update_plot_style(self):
        """延迟更新绘图样式
        
        性能优化：缩放/拖动时不更新cursor，避免多曲线模式下的性能问题。
        Cursor只在真正移动时才更新（通过鼠标移动或点击触发）。
        """
        try:
            # 【安全检查】如果正在更新数据或对象被销毁，跳过
            if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
                return
            
            if hasattr(self, 'view_box') and hasattr(self, 'plot_item'):
                self.update_plot_style(self.view_box, self.view_box.viewRange(), None)
                self._run_post_interaction_tasks()
                # 【性能优化】不在缩放/拖动时更新cursor标签
                # cursor标签只在cursor真正移动时更新（mouseMoveEvent等）
        except Exception as e:
            print(f"延迟更新绘图样式时出错: {e}")

    def _init_ui_refresh_coordinator(self):
        """初始化统一UI刷新调度器"""
        self._ui_refresh = UnifiedUpdateScheduler(
            delay_ms=UI_DEBOUNCE_DELAY_MS,
            order=("style", "cursor", "stats"),
            parent=self,
        )
        self._ui_refresh.register("style", self._run_style_refresh)
        self._ui_refresh.register("cursor", self._run_cursor_refresh)
        self._ui_refresh.register("stats", self._run_stats_refresh)

    def _queue_ui_refresh(self, style=True, cursor=True, stats=True, immediate=False):
        """统一队列UI刷新请求

        【修复说明】改进了pin状态下的刷新逻辑：
        - 正常状态：刷新所有请求的任务
        - 数据更新/加载期间：跳过所有任务（数据不完整）
        - cursor被pin时：允许cursor刷新，跳过style和stats
          原因：zoom/时间修正会改变坐标系统，需要重新计算cursor对应的y值和显示位置
        """
        if not hasattr(self, '_ui_refresh'):
            return
        tasks = []
        if style:
            tasks.append("style")
        if cursor:
            tasks.append("cursor")
        if stats:
            tasks.append("stats")
        if not tasks:
            return

        # 【修复】更精确的跳过条件检查
        # 数据更新期间或加载新数据期间：跳过所有刷新（数据不完整，防止访问不一致的数据结构）
        skip_all = any((
            getattr(self, '_is_updating_data', False),
            bool(self.window() and getattr(self.window(), '_is_loading_new_data', False)),
        ))
        if skip_all:
            return

        # 【关键修复】cursor被pin时：允许cursor刷新，但跳过style和stats
        # 这样zoom/时间修正等操作可以正确更新cursor标签的位置和数值
        if getattr(self, 'is_cursor_pinned', False):
            # 在pin状态下，只保留cursor刷新任务，移除style和stats
            tasks = [t for t in tasks if t in ('cursor', 'style')]
            if not tasks:
                return

        if immediate:
            self._ui_refresh.run_immediately(*tasks)
        else:
            self._ui_refresh.schedule(*tasks)

    def _cancel_ui_refresh(self, *tasks):
        """取消UI刷新"""
        if hasattr(self, '_ui_refresh'):
            if tasks:
                self._ui_refresh.cancel(*tasks)
            else:
                self._ui_refresh.cancel()

    def _run_style_refresh(self):
        """执行样式刷新"""
        if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
            return
        try:
            self.update_plot_style(self.view_box, self.view_box.viewRange(), None)
        except Exception as e:
            print(f"样式刷新出错: {e}")

    def _run_cursor_refresh(self):
        """执行光标刷新"""
        try:
            self._update_cursor_after_interaction()
        except Exception:
            pass

    def _run_stats_refresh(self):
        """执行统计刷新"""
        try:
            main_window = self.window()
            if main_window and not getattr(self, '_is_being_destroyed', False):
                if hasattr(main_window, 'update_mark_stats'):
                    main_window.update_mark_stats()
        except Exception:
            pass

    def schedule_style_refresh(self):
        """统一触发样式、光标和统计的延迟刷新"""
        if not hasattr(self, '_update_timer'):
            return
        self._pending_post_interaction_tasks = True
        self._update_timer.stop()
        self._update_timer.start(self._debounce_delay_ms)

    def _run_post_interaction_tasks(self):
        """样式更新后统一执行光标和统计刷新"""
        if not getattr(self, '_pending_post_interaction_tasks', False):
            return
        self._pending_post_interaction_tasks = False
        try:
            self._update_cursor_after_interaction()
        except Exception:
            pass
        main_window = self.window()
        if main_window and not getattr(self, '_is_being_destroyed', False):
            if hasattr(main_window, 'update_mark_stats'):
                main_window.update_mark_stats()

    def _schedule_cursor_geometry_update(self):
        timer = getattr(self, '_cursor_refresh_timer', None)
        if timer is None:
            return
        if not getattr(self, 'vline', None) or not self.vline.isVisible():
            return
        timer.start(0)

    def _refresh_cursor_geometry(self):
        if not getattr(self, 'vline', None) or not self.vline.isVisible():
            return
        if self.show_values_only:
            self._show_x_position_only()
        else:
            self.update_cursor_label()
# ---------------- 主窗口 ----------------
class MarkStatsWindow(QDialog):
    _instance = None  # Singleton instance

    @classmethod
    def get_instance(cls, parent=None):
        if cls._instance is None:
            cls._instance = cls(parent)
        return cls._instance
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_geometry = None  # 存储几何信息
        self.setWindowTitle("标记区域统计")

        # 取消关闭按钮
        flags = self.windowFlags()
        flags |= Qt.CustomizeWindowHint
        flags &= ~Qt.WindowCloseButtonHint
        self.setWindowFlags(flags)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Plot", "x1", "x2", "y1", "y2", "dx", "dy", "slope", "y_avg", "y_max", "y_min"])
        self.tree.setColumnWidth(0, 200)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        self.no_curve_item = QTreeWidgetItem(self.tree, ["No Curve"])
        self.no_curve_item.setExpanded(False)
        if self.parent().mark_stats_geometry:
            self.restoreGeometry(self.parent().mark_stats_geometry)
        else:
            self.resize(1200, 300)

    def update_stats(self, stats_list):
        self.tree.clear()
        self.no_curve_item = QTreeWidgetItem(self.tree, ["No Curve"])
        self.no_curve_item.setExpanded(False)
        has_no_curve = False
        for idx, stats in enumerate(stats_list):
            if stats:
                entries = stats if isinstance(stats, list) else [stats]
                if isinstance(stats, list) and len(entries) > 1:
                    parent_item = QTreeWidgetItem(self.tree, [
                        f"Plot {idx+1} (多线模式)", "", "", "", "", "", "", "", "", "", ""
                    ])
                    parent_item.setExpanded(True)
                    for entry in entries:
                        if not entry or len(entry) < 11:
                            continue
                        label = entry[7]
                        slope_val = entry[6]
                        slope_str = f"{slope_val:.2f}" if not np.isinf(slope_val) else "inf"
                        QTreeWidgetItem(parent_item, [
                            f"  - {label}",
                            f"{entry[0]:.2f}", f"{entry[1]:.2f}",
                            f"{entry[2]:.2f}", f"{entry[3]:.2f}",
                            f"{entry[4]:.2f}", f"{entry[5]:.2f}",
                            slope_str,
                            f"{entry[8]:.2f}", f"{entry[9]:.2f}", f"{entry[10]:.2f}"
                        ])
                else:
                    entry = entries[0] if entries else None
                    if not entry or len(entry) < 11:
                        continue
                    label = entry[7]
                    slope_val = entry[6]
                    slope_str = f"{slope_val:.2f}" if not np.isinf(slope_val) else "inf"
                    QTreeWidgetItem(self.tree, [
                        f"Plot {idx+1} -> {label}",
                        f"{entry[0]:.2f}", f"{entry[1]:.2f}",
                        f"{entry[2]:.2f}", f"{entry[3]:.2f}",
                        f"{entry[4]:.2f}", f"{entry[5]:.2f}",
                        slope_str,
                        f"{entry[8]:.2f}", f"{entry[9]:.2f}", f"{entry[10]:.2f}"
                    ])
            else:
                has_no_curve = True
                QTreeWidgetItem(self.no_curve_item, [f"Plot {idx+1}", "", "", "", "", "", "", "", "", "", ""])
        if not has_no_curve:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(self.no_curve_item))
    
    def save_geom(self):
        self.parent().mark_stats_geometry = self.saveGeometry()

    def load_geom(self):
        if self.parent().mark_stats_geometry is not None:
            geom = self.parent().mark_stats_geometry
            self.restoreGeometry(geom)

    def closeEvent(self, event):
        self.save_geom()
        super().closeEvent(event)


class TimeCorrectionDialog(QDialog):
    def __init__(self, cur_factor=1.0, cur_offset=0.0, parent=None):
        super().__init__(parent)
        self.window_geometry = None  # 存储几何信息
        self.setWindowTitle("时间修正")

        form = QFormLayout(self)

        self.factor_spin = QDoubleSpinBox()
        self.factor_spin.setRange(0.0001, 1e6)
        self.factor_spin.setValue(cur_factor)
        self.factor_spin.setDecimals(6)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1e9, 1e9)
        self.offset_spin.setValue(cur_offset)
        self.offset_spin.setDecimals(6)

        form.addRow("比例因子:", self.factor_spin)
        form.addRow("偏移量:", self.offset_spin)

        btns = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        form.addRow(btns)

        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        QTimer.singleShot(0, lambda: self.factor_spin.selectAll())

        if self.parent().time_correction_geometry:
            self.restoreGeometry(self.parent().time_correction_geometry)

    def values(self):
        return self.factor_spin.value(), self.offset_spin.value()

    def closeEvent(self, event):
        self.parent().time_correction_geometry = self.saveGeometry()
        super().closeEvent(event)


class PlotContainerWidget(QWidget):
    """包装单个 Plot, 负责显示拖拽提示"""
    def __init__(self, plot_widget: DraggableGraphicsLayoutWidget, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self._forced_indicator_text: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(QMargins(0, 0, 5, 5))
        layout.addWidget(plot_widget)
        self._init_indicator()

    def _init_indicator(self):
        self._indicator = QWidget(self)
        self._indicator.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._indicator.hide()
        self._indicator.setStyleSheet(
            "background-color: rgba(0, 120, 215, 40);"
            "border: 2px dashed #0078d7;"
            "border-radius: 12px;"
        )
        layout = QVBoxLayout(self._indicator)
        layout.setContentsMargins(16, 16, 16, 16)
        self._indicator_label = QLabel("", self._indicator)
        self._indicator_label.setAlignment(Qt.AlignCenter)
        self._indicator_label.setWordWrap(True)
        self._indicator_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._indicator_label.setStyleSheet(
            "color: #0b365a; font-size: 16px; font-weight: bold; background: transparent; border: none;"
        )
        layout.addWidget(self._indicator_label, alignment=Qt.AlignCenter)

    def _build_indicator_text(self, var_names: list[str]) -> str:
        if self._forced_indicator_text:
            return self._forced_indicator_text

        has_curve = bool(getattr(self.plot_widget, "curve", None) or getattr(self.plot_widget, "curves", None))
        multi_mode = bool(getattr(self.plot_widget, "is_multi_curve_mode", False))
        if multi_mode:
            return "释放以添加"
        if has_curve:
            return "释放以替换"
        return "释放以添加"

    def show_drag_indicator(self, var_names: list[str] | None = None, forced_text: str | None = None):
        self._forced_indicator_text = forced_text
        text = self._build_indicator_text(var_names or [])
        self._indicator_label.setText(text)
        self._indicator.setGeometry(self.rect())
        self._indicator.raise_()
        self._indicator.show()

    def hide_drag_indicator(self):
        self._forced_indicator_text = None
        self._indicator.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._indicator.isVisible():
            self._indicator.setGeometry(self.rect())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._drop_event_filter_registered = False
        self.defaultTitle = "数据快速查看器(PyQt5), Alpha版本"

        # 设置应用程序图标（影响Dock图标）
        global ico_path        
        if sys.platform == "darwin":  # macOS
            if os.path.exists(ico_path):
                app_icon = QIcon(str(ico_path))
                app = QApplication.instance()
                app.setWindowIcon(app_icon)
                self.setWindowIcon(app_icon)

        elif sys.platform == "win32":  # Windows
            if os.path.exists(ico_path):
                self.setWindowIcon(QIcon(str(ico_path))) 
       
        self.setWindowTitle(self.defaultTitle)
        
        # 临时文件清理
        self.temp_files = set()
        self._cleanup_done = False
        self._cleanup_callback = self._create_cleanup_callback()
        self._is_loading_new_data = False
        import atexit
        atexit.register(self._cleanup_callback)
        
        # 添加应用程序退出时的清理
        self._app = QApplication.instance()
        if self._app:
            self._app.aboutToQuit.connect(self._cleanup_callback)
        
        self._factor_default = 1
        self._offset_default = 0
        self.factor = self._factor_default
        self.offset = self._offset_default
        self._active_drag_container: PlotContainerWidget | None = None

        # try to load config json files
        _read_status = False
        if os.path.isfile("config_dict.json"):            
            try:
                config_dict = self.load_dict("config_dict.json")
                layout_config_dict = config_dict.get("layout_config", {})
                _width = int(layout_config_dict.get('window_width', 0))
                _height = int(layout_config_dict.get('window_height', 0))
                _max_row = int(layout_config_dict.get('max_row', 0))
                _max_col = int(layout_config_dict.get('max_col', 0))
                _default_row = int(layout_config_dict.get('default_row', 0))
                _default_col = int(layout_config_dict.get('default_col', 0))
                _hide_plot_area = bool(layout_config_dict.get('hide_plot_area', None))
                _read_status = all(x > 0 for x in (_width, _height, _max_row, _max_col, _default_row, _default_col)) and _hide_plot_area is not None
            except Exception as e:     
                print(f"配置文件读取失败: {e}")

        if _read_status == True:
            self._window_width_default = max(600, _width)
            self._window_height_default = max(400, _height)
            self.resize(self._window_width_default, self._window_height_default)
            self._plot_row_max_default = max(1, _max_row)
            self._plot_col_max_default = max(1, _max_col)
            self._plot_row_current = max(1, min(_default_row, _max_row))
            self._plot_col_current = max(1, min(_default_col, _max_col))
        else:
            CANDIDATES = [
                (1920, 1080),
                (1600, 900),
                (1366, 768),
                (1280, 720),
                (1024, 600),
                (800, 600),
                (640, 480),
            ]

            def best_resolution() -> tuple[int, int]:
                desk = QApplication.primaryScreen().size()
                for w, h in sorted(CANDIDATES, key=lambda t: t[0] * t[1], reverse=True):
                    if w < desk.width() * (1 - SCREEN_WITDH_MARGIN) and h < desk.height() * (1 - SCREEN_HEIGHT_MARGIN):
                        return w, h
                return desk.width(), desk.height()

            self._window_width_default, self._window_height_default = best_resolution()
            self.resize(self._window_width_default, self._window_height_default)
            # put default plots into the window
            self._plot_row_max_default = 4
            self._plot_col_max_default = 3
            self._plot_row_current = 3
            self._plot_col_current = 1
            _hide_plot_area = False

        self.loaded_path = ''
        self._last_open_dir: str | None = None
        self.var_names = None
        self.units = None
        self.time_channels_infos = None
        self.data = None
        self.data_validity = None

        # 窗口几何信息
        self.data_table_geometry = None
        self.mark_stats_geometry = None
        self.time_correction_geometry = None

        # 标记区域与统计窗口（需在任何可能使用前初始化）
        self.saved_mark_range = None
        self.mark_stats_window = None

        # value cache
        self.value_cache = {}

        # ---------------- 中央控件 ----------------
        central = QWidget()
        self.setCentralWidget(central)

        self.setAcceptDrops(True)

        # 总水平布局：左侧变量列表 + 右侧绘图区
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for precise alignment

        # ---------------- 左侧变量列表 ----------------
        left_widget = QWidget()
        left_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 0, 5, 0)

        # 左侧大标题
        # 创建一个水平布局来放置标题和帮助按钮
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧大标题
        left_layout_title = QLabel("变量列表")
        font = left_layout_title.font()
        font.setBold(True)
        left_layout_title.setFont(font)
        title_layout.addWidget(left_layout_title)

        # 添加弹簧，将按钮推到右侧
        title_layout.addStretch(1)

        self.clone_btn = QPushButton("分身")
        self.clone_btn.setToolTip("启动独立实例")
        self.clone_btn.clicked.connect(self.spawn_clone_window)
        self.clone_btn.setVisible(True)
        title_layout.addWidget(self.clone_btn)

        # 帮助按钮（使用小图标按钮样式）
        self.help_btn_small = QPushButton("?")
        self.help_btn_small.setToolTip("帮助文档")  # 添加提示
        self.help_btn_small.clicked.connect(self.show_help)
        title_layout.addWidget(self.help_btn_small)

        # 将标题布局添加到左侧布局
        left_layout.addLayout(title_layout)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("输入变量名关键词（空格分隔）")
        self.filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.filter_input)

        self.unit_filter_input = QLineEdit()
        self.unit_filter_input.setPlaceholderText("输入单位关键词（空格分隔）")
        self.unit_filter_input.setContentsMargins(60, 0, 0, 0)
        self.unit_filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.unit_filter_input)

        # 创建一个水平布局来放置这两个按钮
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距

        # 创建按钮
        self.load_btn = QPushButton("导入数据文件")
        self.load_btn.clicked.connect(self.load_btn_click)

        self.reload_btn = QPushButton("重载")
        self.reload_btn.clicked.connect(self.reload_data)

        # 设置按钮的拉伸比例（4:1）
        button_layout.addWidget(self.load_btn, 4)  # 导入按钮占4份
        button_layout.addWidget(self.reload_btn, 1)  # 重新加载按钮占1份

        # 将按钮布局添加到左侧布局
        left_layout.addLayout(button_layout)
        self.list_widget = MyTableWidget()
        left_layout.addWidget(self.list_widget)

        self.toggle_plot_btn = QPushButton("隐藏绘图区")
        self.toggle_plot_btn.setCheckable(True)
        self.toggle_plot_btn.toggled.connect(self.toggle_plot_area)
        left_layout.addWidget(self.toggle_plot_btn)
        left_layout.setSpacing(2)
        self.left_widget = left_widget
        main_layout.addWidget(left_widget, 0)

        # 添加成员变量来保存窗口状态
        self._plot_area_visible = True
        self._saved_geometry = None

        # ---------------- 右侧绘图区 ----------------
        self.plot_widget = QWidget()
        root_layout = QVBoxLayout(self.plot_widget)

        root_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        root_layout.setSpacing(0)  # Remove spacing

        # 顶部按钮栏：弹簧 + 光标按钮（右对齐）
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 5, 5)
        
        # 左侧按钮
        self.time_correction_btn = QPushButton("时间修正")
        self.time_correction_btn.clicked.connect(self.open_time_correction_dialog)
        top_bar.addWidget(self.time_correction_btn)

        self.clear_all_plots_btn = QPushButton("清除绘图")
        self.clear_all_plots_btn.clicked.connect(self.clear_all_plots)
        top_bar.addWidget(self.clear_all_plots_btn)

        # 中键占位
        top_bar.addStretch(1)

        # 右侧按钮        
        self.auto_range_btn = QPushButton("自动缩放")
        self.auto_range_btn.clicked.connect(self.auto_range_all_plots)
        
        self.auto_y_btn = QPushButton("仅调节y轴")
        self.auto_y_btn.clicked.connect(self.auto_y_in_x_range)
        
        self.cursor_btn = QPushButton("显示光标")
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.clicked.connect(self.toggle_cursor_all)
        self.cursor_values_hidden = False
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []

        self.mark_region_btn = QPushButton("标记区域")
        self.mark_region_btn.setCheckable(True)
        self.mark_region_btn.clicked.connect(self.toggle_mark_region)
        
        self.grid_layout_btn = QPushButton("修改布局")
        self.grid_layout_btn.clicked.connect(self.open_layout_dialog)

        self.set_button_status(False)
        
        top_bar.addWidget(self.grid_layout_btn)
        top_bar.addWidget(self.cursor_btn)
        top_bar.addWidget(self.mark_region_btn)
        top_bar.addWidget(self.auto_y_btn)
        top_bar.addWidget(self.auto_range_btn)
        
        # 添加布局
        root_layout.addLayout(top_bar)

        # 真正容纳子图的布局
        self.plot_layout = QGridLayout()
        self.plot_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        self.plot_layout.setSpacing(0)  # No spacing
        root_layout.addLayout(self.plot_layout, 1)    # 1 表示可伸缩
        main_layout.addWidget(self.plot_widget, 4)

        # ---------------- 子图 ----------------
        self.plot_widgets = []

        self.placeholder_label = QLabel("请导入 CSV 文件以查看数据", self.plot_widget)
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 24px; color: gray;")
        self.plot_layout.addWidget(self.placeholder_label, 0, 0)

        self.drop_overlay = DropOverlay(self.centralWidget())
        self.drop_overlay.lower()  # 初始在最下层
        self.drop_overlay.hide()

        # 全局拖拽过滤器
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)
            self._drop_event_filter_registered = True
   
        if _hide_plot_area:
            # 临时设置离屏属性，模拟显示以计算布局尺寸（无闪烁）
            self.setAttribute(Qt.WA_DontShowOnScreen, True)
            self.show()  # 此处仅计算布局，不实际显示在屏幕上
            _geometry = self.geometry()

            # 更新按钮文本、checked状态和可见性标志
            self.toggle_plot_btn.setChecked(True)
            self.toggle_plot_btn.setText("显示绘图区")
            self._plot_area_visible = False

            # 隐藏右侧绘图区域
            self.plot_widget.hide()

            # 计算左侧部件的实际宽度（包括边距和框架）
            left_width = self.left_widget.width()
            main_margin = self.centralWidget().layout().contentsMargins()
            left_width += main_margin.left() + main_margin.right()
            frame_width = self.frameGeometry().width() - self.width()
            new_width = left_width + frame_width

            # 设置固定宽度，并关闭离屏模拟
            self.setFixedWidth(new_width)
            self.move(_geometry.topLeft())
            self.close()  # 关闭模拟窗口
            self.setAttribute(Qt.WA_DontShowOnScreen, False)

            # 保存原最大宽度（用于后续显示时恢复）
            self._old_max_width = self._window_width_default  # 或实际原宽

        # ---------------- 命令行直接加载文件 ----------------
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            self.load_csv_file(file_path)

    def closeEvent(self, event):
        # 在主窗口关闭前，设置DataTableDialog的_skip_close_confirmation为True
        if DataTableDialog._instance is not None:
            DataTableDialog._instance.set_skip_close_confirmation(True)

        self._unregister_global_event_filter()

        # 安全清理资源
        self._safe_cleanup()

        super().closeEvent(event)
        
    def toggle_plot_area(self, checked):
        if checked:
            self._saved_geometry = self.saveGeometry()
            self.plot_widget.hide()
            self.toggle_plot_btn.setText("显示绘图区")
            
            # 保存当前的最大宽度策略，然后设置固定宽度
            self._old_max_width = self.maximumWidth()
            # 计算固定宽度
            left_width = self.left_widget.width()
            # 加上主布局的左右边距
            main_margin = self.centralWidget().layout().contentsMargins()
            left_width += main_margin.left() + main_margin.right()
            # 加上窗口框架的宽度
            frame_width = self.frameGeometry().width() - self.width()
            new_width = left_width + frame_width
            self.setFixedWidth(new_width)
            self._plot_area_visible = False
        else:
            # 恢复窗口大小策略
            self.setMaximumWidth(self._old_max_width)
            self.setMinimumWidth(0)
            self.plot_widget.show()
            self.toggle_plot_btn.setText("隐藏绘图区")
            if self._saved_geometry:
                self.restoreGeometry(self._saved_geometry)
            self._plot_area_visible = True

    def show_help(self):
        dlg = HelpDialog(self)
        dlg.exec()

    def _get_plot_container(self, plot_widget) -> PlotContainerWidget | None:
        parent = plot_widget.parentWidget()
        return parent if isinstance(parent, PlotContainerWidget) else None

    def _show_drag_indicator_for_plot(self, plot_widget, var_names: list[str], forced_text: str | None = None):
        container = self._get_plot_container(plot_widget)
        if not container:
            return
        if self._active_drag_container and self._active_drag_container is not container:
            self._active_drag_container.hide_drag_indicator()
        container.show_drag_indicator(var_names, forced_text=forced_text)
        self._active_drag_container = container

    def _hide_drag_indicator_for_plot(self, plot_widget):
        container = self._get_plot_container(plot_widget)
        if not container:
            return
        container.hide_drag_indicator()
        if self._active_drag_container is container:
            self._active_drag_container = None

    def _clear_active_plot_highlight(self):
        """Hide the current drag indicator, if any."""
        if self._active_drag_container:
            self._active_drag_container.hide_drag_indicator()
            self._active_drag_container = None

    def _is_cursor_over_plot_area(self) -> bool:
        """Check whether the global cursor is over any plot container."""
        widget = QApplication.widgetAt(QCursor.pos())
        while widget:
            if isinstance(widget, PlotContainerWidget):
                return True
            widget = widget.parentWidget()
        return False

    def spawn_clone_window(self):
        try:
            if getattr(sys, "frozen", False):
                args = [sys.executable]
            else:
                script_path = os.path.abspath(__file__)
                args = [sys.executable, script_path]

            if sys.platform == "win32":
                subprocess.Popen(
                    args,
                    cwd=os.getcwd(),
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NO_WINDOW
                    ),
                    close_fds=True,
                )
            else:
                subprocess.Popen(
                    args,
                    cwd=os.getcwd(),
                    start_new_session=True,
                    close_fds=True,
                )
        except Exception as e:
            QMessageBox.warning(self, "错误", f"启动独立实例失败: {e}")

    def load_btn_click(self):
        initial_dir = self._get_dialog_initial_directory()
        dialog = QFileDialog(self, "选择数据文件")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setNameFilter("All Files (*.*);;CSV File (*.csv);;m File (*.mfile);;t00 File (*.t00);;t01 File (*.t01);;t10 File (*.t10);;t11 File (*.t11)")
        if initial_dir:
            dialog.setDirectory(initial_dir)

        if dialog.exec():
            selected_files = dialog.selectedFiles()
            if selected_files:
                self.load_csv_file(selected_files[0])

    def load_csv_file(self, file_path: str):
        if not file_path or not os.path.isfile(file_path):
            return
        
        # 检查是否是同一个文件路径，如果是则强制重新加载
        is_same_file = (hasattr(self, 'loader') and 
                       self.loader and 
                       hasattr(self.loader, 'path') and 
                       os.path.abspath(self.loader.path) == os.path.abspath(file_path))
        
        try:
            if is_same_file:
                # 同路径文件，强制重新加载
                self._cleanup_old_data()
                self._load_file(file_path, is_reload=True)
            else:
                # 新文件，正常加载
                self._load_file(file_path, is_reload=False)
        except Exception as e:
            QMessageBox.critical(self, "加载错误", str(e))
        finally:
            if hasattr(self, 'loader') and self.loader:  # 如果加载成功
                self._post_load_actions(file_path)
                self.raise_()  # 加载完成后前置
                self.activateWindow()

    def set_button_status(self, status: bool):
        if status is not None:
            self.time_correction_btn.setEnabled(status)
            self.clear_all_plots_btn.setEnabled(status)
            self.auto_range_btn.setEnabled(status)
            self.auto_y_btn.setEnabled(status) 
            self.cursor_btn.setEnabled(status)
            self.mark_region_btn.setEnabled(status)
            self.grid_layout_btn.setEnabled(status)

    def reload_data(self):
        if not hasattr(self, 'loader') or not self.loader or not self.loader.path or not os.path.isfile(self.loader.path):
            QMessageBox.critical(self, "错误", "文件路径无效，无法重新加载")
            return

        # 强制重新读取文件，确保获取最新数据
        file_path = self.loader.path
        
        # 清理旧数据
        self._cleanup_old_data()
        
        # 重新加载文件
        self._load_file(file_path, is_reload=True)

    def _load_file(self, file_path: str, is_reload: bool = False):
        self._begin_data_reload()
        try:
            # 在加载新数据前，先释放旧数据以节省内存
            if not is_reload and hasattr(self, 'loader') and self.loader is not None:
                self._cleanup_old_data()

            file_ext = self._extract_file_extension(file_path)
            if file_ext is None:
                QMessageBox.critical(self, "读取失败", "无法识别的文件格式")
                self._end_data_reload()
                return

            # load default
            delimiter_typ = ','
            descRows = 0
            hasunit = True

            # try to load config json files
            if os.path.isfile("config_dict.json"):
                try:
                    config_dict = self.load_dict("config_dict.json")
                    ext_dict = config_dict.get(file_ext[1:], {})
                    delimiter_typ = ext_dict.get('sep')
                    descRows = int(ext_dict.get('skiprows'))
                    hasunit = bool(ext_dict.get('hasunit'))

                except Exception as e:
                    print(f"配置文件读取失败: {e}")
            else:
                if file_ext in ['.csv',]:
                    delimiter_typ = ','
                    descRows = 0
                    hasunit = True
                elif file_ext in ['.mfile', '.t00', '.t01', '.t10', '.t11']:
                    delimiter_typ = '\t'
                    descRows = 2
                    hasunit = True
                elif file_ext in ['.txt',]:
                    delimiter_typ = '\t'
                    descRows = 0
                    hasunit = True
                else:
                    QMessageBox.critical(self, "读取失败", f"无法读取后缀为:'{file_ext}'的文件")
                    self._end_data_reload()
                    return

            _Threshold_Size_Mb = FILE_SIZE_LIMIT_BACKGROUND_LOADING

            # 获取文件大小
            file_size = os.path.getsize(file_path)

            # 检查文件是否正在被写入（通过文件锁定检测）
            use_streaming = self._is_file_being_written(file_path)

            # 机械硬盘优化：如果文件很大且不是流式读取，建议使用流式读取
            if not use_streaming and file_size > 50 * 1024 * 1024:  # 50MB以上
                # 检查是否是机械硬盘（通过文件系统类型判断）
                if self._is_mechanical_drive(file_path):
                    use_streaming = True

            # < 5 MB 直接读
            if file_size < _Threshold_Size_Mb * 1024 * 1024:
                status = self._load_sync(file_path, descRows=descRows, sep=delimiter_typ, hasunit=hasunit, use_streaming=use_streaming)
                if status:
                    self.set_button_status(True)
                    self._post_load_actions(file_path)
                self._end_data_reload()
            else:
                # 5 MB 以上走线程
                progress_text = "正在读取数据..." + ("(流式)" if use_streaming else "")
                self._progress = QProgressDialog(progress_text, "取消", 0, 100, self)
                self._progress.setWindowModality(Qt.ApplicationModal)
                self._progress.setAutoClose(True)
                self._progress.setCancelButton(None)  # 不可取消
                self._progress.setMinimumDuration(0)  # 立即显示，避免延迟
                self._progress.show()

                self._thread = DataLoadThread(file_path, descRows=descRows, sep=delimiter_typ, hasunit=hasunit, use_streaming=use_streaming)
                self._thread.progress.connect(self._progress.setValue)
                self._thread.finished.connect(lambda loader: self._on_load_done(loader, file_path))
                self._thread.error.connect(self._on_load_error)
                self._thread.start()

                # 设置线程为可中断
                self._thread.setTerminationEnabled(True)
        except Exception:
            self._end_data_reload()
            raise

    def _post_load_actions(self, file_path: str):
        self.loaded_path = file_path
        self._remember_last_open_dir(file_path)

        def truncate_string(file_path, max_length=79):
            # directory = os.path.dirname(file_path)
            filename_length = len(os.path.basename(file_path))
            if len(file_path) <= max_length:
                return file_path
            return "..." + file_path[min(-filename_length-1,-(max_length-3)):]
        self.setWindowTitle(f"{self.defaultTitle} ---- 数据文件: [{truncate_string(file_path)}]")
        self.set_button_status(True)

    def _remember_last_open_dir(self, file_path: str):
        """记录最近一次成功加载的目录（仅在运行期有效）"""
        directory = os.path.dirname(file_path)
        if directory and os.path.isdir(directory):
            self._last_open_dir = directory

    def _get_dialog_initial_directory(self) -> str:
        """返回导入对话框应使用的初始目录"""
        if getattr(self, "_last_open_dir", None) and os.path.isdir(self._last_open_dir):
            return self._last_open_dir
        return self._default_system_directory()

    def _default_system_directory(self) -> str:
        """根据平台提供一个合理的默认目录"""
        candidates: list[str | None] = []
        if sys.platform.startswith("win"):
            # Windows 上优先“此电脑”，Qt 支持该 shell 路径
            candidates.append("::{20D04FE0-3AEA-1069-A2D8-08002B30309D}")

        def _safe_location(location):
            try:
                return QStandardPaths.writableLocation(location)
            except AttributeError:
                return ""

        candidates.extend([
            _safe_location(QStandardPaths.HomeLocation),
            _safe_location(QStandardPaths.DesktopLocation),
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
        except json.JSONDecodeError:
            return {} if default is None else default
    
    def _extract_file_extension(self, file_path: str) -> str | None:
        """
        解析文件后缀名，支持诸如 't00.1' 这样的数字变体。
        """
        import re

        supported_extensions = ['.csv', '.mfile', '.t00', '.t01', '.t10', '.t11', '.txt']

        base_ext = os.path.splitext(file_path)[1].lower()
        if base_ext in supported_extensions:
            return base_ext

        base_name = os.path.basename(file_path).lower()
        pattern = r'(' + '|'.join(re.escape(ext) for ext in supported_extensions) + r')\.\d+$'
        match = re.search(pattern, base_name)

        if match:
            return match.group(1)
        return None
        
    def _load_sync(self, 
                   file_path: str,
                   descRows: int = 0,
                   sep: str = ',',
                   hasunit: bool = True,
                   use_streaming: bool = False):
        """小文件直接读"""
        loader = None
        status = False
        try:
            loader = FastDataLoader(file_path, descRows=descRows, sep=sep, hasunit=hasunit, use_streaming=use_streaming)
            self.loader = loader
            self._apply_loader()
            status = True
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            status = False
        finally:
            if loader is not None:
                loader = None
            return status

    def _on_load_done(self, loader, file_path: str):
        self._progress.close()
        self.loader = loader
        self._apply_loader()
        self._post_load_actions(file_path)
        self._end_data_reload()

    def _on_load_error(self, msg):
        self._progress.close()
        QMessageBox.critical(self, "读取失败", msg)
        self._end_data_reload()

    def _cleanup_old_data(self):
        """清理旧数据以释放内存"""
        self.reset_all_pin_states()
        if hasattr(self, 'loader') and self.loader is not None:
            try:
                # 清理DataFrame
                if hasattr(self.loader, '_df') and self.loader._df is not None:
                    try:
                        self.loader._df = None
                    except (AttributeError, TypeError):
                        # 如果无法设置属性，尝试删除
                        try:
                            delattr(self.loader, '_df')
                        except:
                            pass
                
                # 清理其他大对象 - 使用更安全的方式
                for attr_name in ['var_names', 'units', 'time_channels_info', 'df_validity']:
                    if hasattr(self.loader, attr_name):
                        try:
                            setattr(self.loader, attr_name, None)
                        except (AttributeError, TypeError):
                            # 如果无法设置属性，尝试删除
                            try:
                                delattr(self.loader, attr_name)
                            except:
                                pass
                
                # 清理plot widgets中的数据引用
                if hasattr(self, 'plot_widgets') and self.plot_widgets:
                    for container in self.plot_widgets:
                        if hasattr(container, 'plot_widget'):
                            for attr_name in ['data', 'units']:
                                if hasattr(container.plot_widget, attr_name):
                                    try:
                                        setattr(container.plot_widget, attr_name, None)
                                    except (AttributeError, TypeError):
                                        pass
                
            except Exception as e:
                print(f"清理数据时出错: {e}")
            finally:
                # 删除loader对象
                self.loader = None
                
                # 强制垃圾回收
                import gc
                gc.collect()

    def reset_all_pin_states(self):
        """重置所有plot的光标固定状态"""
        if not getattr(self, "plot_widgets", None):
            return
        for container in self.plot_widgets:
            plot_widget = getattr(container, "plot_widget", None)
            if plot_widget is not None:
                plot_widget.reset_pin_state()

    def _begin_data_reload(self):
        """开始重载时重置pin状态"""
        self._is_loading_new_data = True
        if hasattr(self, 'cursor_btn'):
            self._cursor_state_before_reload = self.cursor_btn.isChecked()
        else:
            self._cursor_state_before_reload = False
        self._suppress_cursor_refresh = True
        self.reset_all_pin_states()
        if hasattr(self, 'cursor_mode'):
            self.cursor_mode = "1 free cursor"
        if hasattr(self, 'pinned_x_values'):
            self.pinned_x_values = []
        if hasattr(self, '_apply_cursor_mode_to_plots'):
            self._apply_cursor_mode_to_plots()

    def _end_data_reload(self):
        """标记数据加载完成"""
        self._is_loading_new_data = False
        self._suppress_cursor_refresh = False
        if hasattr(self, '_cursor_state_before_reload'):
            self.toggle_cursor_all(self._cursor_state_before_reload)
        QTimer.singleShot(0, self._refresh_cursor_labels_after_reload)

    def _refresh_cursor_labels_after_reload(self):
        """重载完成后刷新cursor标签，避免显示状态不同步"""
        if not hasattr(self, 'cursor_btn') or not self.cursor_btn.isChecked():
            return
        for container in getattr(self, 'plot_widgets', []):
            widget = getattr(container, 'plot_widget', None)
            if widget is None or not container.isVisible():
                continue
            try:
                widget.update_cursor_label()
            except (RuntimeError, AttributeError):
                pass


    def _create_cleanup_callback(self):
        """Return a weakref-based cleanup callable."""
        self_ref = weakref.ref(self)

        def _cleanup_wrapper():
            instance = self_ref()
            if instance is not None:
                instance._safe_cleanup()

        return _cleanup_wrapper

    def _safe_cleanup(self):
        """安全清理所有资源，防止退出时崩溃"""
        if getattr(self, "_cleanup_done", False):
            return
        self._cleanup_done = True

        try:
            # 清理线程
            if hasattr(self, '_thread') and self._thread is not None:
                if self._thread.isRunning():
                    self._thread.stop()  # 使用自定义的stop方法
                self._thread = None
            
            # 清理进度对话框
            if hasattr(self, '_progress') and self._progress is not None:
                self._progress.close()
                self._progress = None
            
            # 清理数据
            self._cleanup_old_data()
            
            # 清理临时文件
            self._cleanup_temp_files()
            
            # 清理plot widgets
            if hasattr(self, 'plot_widgets') and self.plot_widgets:
                for container in self.plot_widgets:
                    if hasattr(container, 'plot_widget'):
                        try:
                            container.plot_widget.clear()
                        except:
                            pass
                self.plot_widgets.clear()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"清理资源时出错: {e}")

    def _cleanup_temp_files(self):
        """清理所有临时文件"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        self.temp_files.clear()

    def _add_temp_file(self, temp_file: str):
        """添加临时文件到清理列表"""
        self.temp_files.add(temp_file)

    def _is_file_being_written(self, file_path: str) -> bool:
        """检测文件是否正在被写入"""
        try:
            # 方法1: 检查文件大小变化
            size1 = os.path.getsize(file_path)
            import time
            time.sleep(0.1)  # 短暂等待
            size2 = os.path.getsize(file_path)
            
            # 如果文件大小在变化，可能正在写入
            if size1 != size2:
                return True
                
            # 方法2: 检查文件修改时间
            mtime1 = os.path.getmtime(file_path)
            time.sleep(0.05)  # 更短的等待时间
            mtime2 = os.path.getmtime(file_path)
            
            # 如果修改时间在变化，文件可能正在被写入
            if mtime1 != mtime2:
                return True
                
            # 方法3: 尝试以独占模式打开文件
            try:
                with open(file_path, 'a'):
                    pass
                return False  # 可以以追加模式打开，说明没有其他进程在写入
            except (OSError, IOError):
                return True  # 无法以追加模式打开，可能有其他进程在写入
                
        except Exception:
            return False  # 出错时默认不使用流式读取

    def _is_mechanical_drive(self, file_path: str) -> bool:
        """检测文件是否在机械硬盘上"""
        try:
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes
                
                # 获取文件所在驱动器
                drive = os.path.splitdrive(file_path)[0]
                if not drive:
                    return True  # 默认假设是机械硬盘
                
                # 获取驱动器类型
                drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)
                # DRIVE_FIXED = 3, 固定硬盘
                if drive_type == 3:
                    # 进一步检查是否是SSD（通过查询磁盘属性）
                    try:
                        # 这里可以添加更复杂的SSD检测逻辑
                        # 暂时返回True，假设是机械硬盘
                        return True
                    except:
                        return True
                else:
                    return True  # 非固定硬盘，可能是机械硬盘
            else:
                # Linux/Mac系统，通过文件系统类型判断
                import subprocess
                try:
                    result = subprocess.run(['df', '-T', file_path], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            fs_type = lines[1].split()[1]
                            # 常见的SSD文件系统
                            ssd_fs = ['ext4', 'xfs', 'btrfs', 'zfs']
                            return fs_type not in ssd_fs
                except:
                    pass
                return True  # 默认假设是机械硬盘
        except Exception:
            return True  # 出错时默认假设是机械硬盘

    def _apply_loader(self):
        """把 loader 的内容同步到 UI"""
        self.var_names = self.loader.var_names
        self.units = self.loader.units
        self.time_channels_infos = self.loader.time_channels_info
        self.data_validity = self.loader.df_validity
        self.list_widget.populate(self.var_names, self.units, self.data_validity)

        # 移除占位符
        if self.placeholder_label.parent():
            self.placeholder_label.setParent(None)

        # 如果尚未创建子图矩阵，则创建
        if not self.plot_widgets:
            self.create_subplots_matrix(self._plot_row_max_default, self._plot_col_max_default)
            self.set_plots_visible(self._plot_row_current, self._plot_col_current)

        # 更新所有 plot_widgets 的数据
        for container in self.plot_widgets:
            widget = container.plot_widget
            widget.data = self.loader.df
            widget.units = self.loader.units
            widget.time_channels_info = self.loader.time_channels_info

        # 清除cache
        self.replots_after_loading()
        # 更新数值变量表（如果存在）
        if DataTableDialog._instance is not None:
            DataTableDialog._instance.update_data(self.loader)
            # 只show如果有列
            if not DataTableDialog._instance._df.empty:
                DataTableDialog._instance.show()  # 确保窗口显示
                DataTableDialog._instance.raise_()
                DataTableDialog._instance.activateWindow()
            else:
                DataTableDialog._instance.set_skip_close_confirmation(True)
                DataTableDialog._instance.close()

        self.filter_variables() 
        if self.mark_region_btn.isChecked():
            self.update_mark_stats()

    def filter_variables(self):
        if self.var_names is None:
            return
        name_text = self.filter_input.text().lower()
        unit_text = self.unit_filter_input.text().lower()
        name_keywords = name_text.split() if name_text else []
        unit_keywords = unit_text.split() if unit_text else []

        filtered_names = []
        for var in self.var_names:
            var_lower = var.lower()
            unit = self.units.get(var, '').lower()

            name_match = not name_keywords or any(kw in var_lower for kw in name_keywords)
            unit_match = not unit_keywords or any(kw in unit for kw in unit_keywords)

            if name_match and unit_match:
                filtered_names.append(var)

        self.list_widget.populate(filtered_names, self.units, self.data_validity)

    def toggle_mark_region(self, checked):
        if checked:
            self.mark_region_btn.setText("关闭标记")
            # 添加标记区域
            if len(self.plot_widgets) == 0:
                self.mark_region_btn.setChecked(False)
                return
            if self.saved_mark_range:
                min_x, max_x = self.saved_mark_range
                view_min, view_max = self.plot_widgets[0].plot_widget.view_box.viewRange()[0]
                if min_x >= view_min and max_x <= view_max:
                    pass  # 沿用
                else:
                    # 新位置：中间1/3
                    width = view_max - view_min
                    min_x = view_min + width / 3
                    max_x = view_min + 2 * width / 3
            else:
                # 默认中间1/3
                view_min, view_max = self.plot_widgets[0].plot_widget.view_box.viewRange()[0]
                width = view_max - view_min
                min_x = view_min + width / 3
                max_x = view_min + 2 * width / 3

            for container in self.plot_widgets:
                if container.isVisible():
                    container.plot_widget.add_mark_region(min_x, max_x)

            # 打开统计窗口
            self.mark_stats_window = MarkStatsWindow.get_instance(self)
            geom = self.mark_stats_window.load_geom()
            if geom:
                self.mark_stats_window.restoreGeometry(geom)

            self.mark_stats_window.show()
            self.update_mark_stats()
        else:
            self.mark_region_btn.setText("标记区域")
            # 保存当前范围
            if self.plot_widgets and self.plot_widgets[0].plot_widget.mark_region:
                self.saved_mark_range = self.plot_widgets[0].plot_widget.mark_region.getRegion()
            for container in self.plot_widgets:
                container.plot_widget.remove_mark_region()
            if self.mark_stats_window:
                self.mark_stats_window.save_geom()
                self.mark_stats_window.hide()  # Hide instead of close to preserve state
                # Do not set to None to maintain singleton

    def sync_mark_regions(self, region_item):
        min_x, max_x = region_item.getRegion()
        for container in self.plot_widgets:
            if container.isVisible() and container.plot_widget.mark_region and container.plot_widget.mark_region != region_item:
                container.plot_widget.mark_region.setRegion([min_x, max_x])
        self.update_mark_stats()

    def update_mark_stats(self):
        if hasattr(self, 'mark_stats_window') and self.mark_stats_window:
            stats_list = []
            for container in self.plot_widgets:
                if container.isVisible():
                    stats = container.plot_widget.get_mark_stats()
                    stats_list.append(stats)
            self.mark_stats_window.update_stats(stats_list)

    def open_layout_dialog(self):
        dlg = LayoutInputDialog(max_rows=self._plot_row_max_default, 
                                max_cols=self._plot_col_max_default, 
                                cur_rows=self._plot_row_current,
                                cur_cols=self._plot_col_current,
                                parent=self)
        if dlg.exec() == QDialog.Accepted:
            r, c = dlg.values()
            self.set_plots_visible(r, c)
            self.update_mark_regions_on_layout_change()

    def open_time_correction_dialog(self):
        dialog = TimeCorrectionDialog(self.factor, self.offset, self)
        if dialog.window_geometry:
            dialog.restoreGeometry(dialog.window_geometry)
        if dialog.exec() == QDialog.Accepted:
            new_factor, new_offset = dialog.values()
            if new_factor <= 0:
                QMessageBox.warning(self, "错误", "比例因子 (Factor) 必须是正数")
                return
            
            old_factor = self.factor
            old_offset = self.offset
            self.factor = new_factor
            self.offset = new_offset

            # 如果没有加载数据或图表，则直接返回
            if not self.plot_widgets or not hasattr(self, 'loader') or self.loader is None:
                return

            # 获取应用修正前的X轴视图范围 (以第一个图表为准)
            curr_min, curr_max = self.plot_widgets[0].plot_widget.view_box.viewRange()[0]

            # 1. 更新所有图表的数据和内部限制
            for container in self.plot_widgets:
                container.plot_widget.update_time_correction(new_factor, new_offset)

            # 2. 根据旧范围和新旧参数，计算新的X轴范围
            if old_factor != 0:
                index_min = (curr_min - old_offset) / old_factor
                index_max = (curr_max - old_offset) / old_factor
                new_min = new_offset + new_factor * index_min
                new_max = new_offset + new_factor * index_max
            else:
                # 如果旧因子为0，提供一个安全的回退方案
                datalength = self.loader.datalength
                new_min = new_offset + new_factor * 1
                new_max = new_offset + new_factor * datalength

            # 3. 关键：只对第一个（主）图表设置X轴范围，链接的图表会自动同步
            global DEFAULT_PADDING_VAL_X
            first_plot = self.plot_widgets[0].plot_widget
            first_plot.view_box.enableAutoRange(x=False) # 禁用X轴自动范围
            first_plot.view_box.setXRange(new_min, new_max, padding=0) # 设置新范围，明确设置 padding=0

            # 4. 重新对齐已固定的cursor
            self._realign_pinned_cursor_after_time_correction(old_factor, old_offset, new_factor, new_offset)

            # 5. 更新标记统计（如果可见）
            QTimer.singleShot(0, self.update_mark_stats)


    def sync_crosshair(self, x, source):
        """同步所有plot的cursor位置 - 稳定性加固版

        使用blockSignals防止信号级联，使用延迟批量更新避免卡顿
        """
        if not self.cursor_btn.isChecked():
            return

        # 【稳定性加固】重入保护
        if getattr(self, '_is_syncing_crosshair', False):
            return
        self._is_syncing_crosshair = True

        try:
            # 收集需要更新的widget
            widgets_to_update = []

            # 首先更新源绘图区域的光标位置
            # 【关键修复】阻断vline信号，防止触发级联的sigPositionChanged
            source.vline.blockSignals(True)
            try:
                source.vline.setValue(x)
            finally:
                source.vline.blockSignals(False)
            widgets_to_update.append(source)

            # 然后同步到其他绘图区域
            for container in self.plot_widgets:
                if container.isVisible():
                    widget = container.plot_widget
                    if widget != source:  # 跳过源绘图区域
                        # 【关键修复】阻断信号防止递归
                        widget.vline.blockSignals(True)
                        try:
                            widget.vline.setValue(x)
                        finally:
                            widget.vline.blockSignals(False)
                        widgets_to_update.append(widget)

            # 【稳定性加固】延迟批量更新标签，避免同步更新造成的卡顿
            # 使用闭包捕获widget列表，通过singleShot让Qt有时间处理其他事件
            def _batch_update_labels(widgets=widgets_to_update):
                for widget in widgets:
                    if not getattr(widget, '_is_being_destroyed', False):
                        try:
                            widget.update_cursor_label()
                        except (RuntimeError, AttributeError):
                            pass  # 对象可能已被销毁

            QTimer.singleShot(0, _batch_update_labels)

        finally:
            self._is_syncing_crosshair = False

    def _get_cursor_view_range(self, plot):
        if plot is None or not hasattr(plot, 'view_box'):
            return None, None
        try:
            view_min, view_max = plot.view_box.viewRange()[0]
            return view_min, view_max
        except Exception:
            return None, None

    @staticmethod
    def _clamp_value(value, min_val, max_val):
        return max(min_val, min(max_val, value))

    def _calc_second_cursor_position(self, pinned_x, view_min, view_max):
        if view_min is None or view_max is None:
            return pinned_x
        if view_min > view_max:
            view_min, view_max = view_max, view_min
        clamped = self._clamp_value(pinned_x, view_min, view_max)
        threshold = view_min + 0.6 * (view_max - view_min)
        if clamped <= threshold:
            return clamped + (view_max - clamped) / 2
        return view_min + (clamped - view_min) / 2

    def _select_farthest_cursor_index(self, context_x):
        if not self.pinned_x_values:
            return None
        if context_x is None:
            return len(self.pinned_x_values) - 1
        distances = [abs(x - context_x) for x in self.pinned_x_values]
        return int(np.argmax(distances))

    def _apply_cursor_mode_to_plots(self):
        cursor_on = hasattr(self, 'cursor_btn') and self.cursor_btn.isChecked()
        for container in getattr(self, 'plot_widgets', []):
            widget = getattr(container, 'plot_widget', None)
            if widget is None:
                continue
            if not cursor_on:
                widget.toggle_cursor(False)
                continue
            widget.apply_cursor_mode(self.cursor_mode, self.pinned_x_values)

    def set_cursor_mode(self, mode, *, source_plot=None, context_x=None):
        if not hasattr(self, 'cursor_btn') or not self.cursor_btn.isChecked():
            return
        if mode not in ('1 free cursor', '1 anchored cursor', '2 anchored cursor'):
            return

        prev_mode = getattr(self, 'cursor_mode', '1 free cursor')
        view_min, view_max = self._get_cursor_view_range(source_plot)

        if mode == '1 free cursor':
            self.cursor_mode = mode
            self.pinned_x_values = []
        elif mode == '1 anchored cursor':
            if prev_mode == '2 anchored cursor':
                remove_idx = self._select_farthest_cursor_index(context_x)
                if remove_idx is not None:
                    remaining = [x for idx, x in enumerate(self.pinned_x_values) if idx != remove_idx]
                    self.pinned_x_values = remaining[:1]
            if not self.pinned_x_values:
                if source_plot is not None and hasattr(source_plot, 'vline'):
                    self.pinned_x_values = [source_plot.vline.value()]
            self.cursor_mode = mode
        elif mode == '2 anchored cursor':
            if prev_mode == '1 free cursor' or not self.pinned_x_values:
                pinned = context_x
                if pinned is None and source_plot is not None and hasattr(source_plot, 'vline'):
                    pinned = source_plot.vline.value()
                if pinned is not None:
                    second = self._calc_second_cursor_position(pinned, view_min, view_max)
                    self.pinned_x_values = [pinned, second]
            elif prev_mode == '1 anchored cursor':
                pinned = self.pinned_x_values[0] if self.pinned_x_values else None
                if pinned is None and source_plot is not None and hasattr(source_plot, 'vline'):
                    pinned = source_plot.vline.value()
                if pinned is not None:
                    second = self._calc_second_cursor_position(pinned, view_min, view_max)
                    self.pinned_x_values = [pinned, second]
            else:
                if len(self.pinned_x_values) == 1:
                    second = self._calc_second_cursor_position(self.pinned_x_values[0], view_min, view_max)
                    self.pinned_x_values = [self.pinned_x_values[0], second]
            self.cursor_mode = mode

        self._apply_cursor_mode_to_plots()
        for container in getattr(self, 'plot_widgets', []):
            widget = getattr(container, 'plot_widget', None)
            if widget is not None:
                widget.update_cursor_label()

    def toggle_cursor_all(self, checked):
        for container in self.plot_widgets:
            widget = container.plot_widget
            if checked and self.cursor_values_hidden:
                widget.toggle_cursor(False, hide_values_only=True)
            else:
                widget.toggle_cursor(checked)
        if checked:
            self.cursor_mode = "1 free cursor"
            self.pinned_x_values = []
            self._apply_cursor_mode_to_plots()
        else:
            self.cursor_mode = "1 free cursor"
            self.pinned_x_values = []
        self.cursor_btn.setText("隐藏光标" if checked else "显示光标")

    def _realign_pinned_cursor_after_time_correction(self, old_factor, old_offset, new_factor, new_offset):
        """时间修正后重对齐cursor"""
        if not self.plot_widgets:
            return

        if getattr(self, 'cursor_mode', '1 free cursor') == '1 free cursor':
            return

        pinned_values = list(getattr(self, 'pinned_x_values', []) or [])
        if not pinned_values:
            return

        if old_factor == 0:
            return

        pinned_indices = []
        for pinned_value in pinned_values:
            if pinned_value is None or not np.isfinite(pinned_value):
                continue
            index_pos = (pinned_value - old_offset) / old_factor
            if np.isfinite(index_pos):
                pinned_indices.append(index_pos)

        if not pinned_indices:
            return

        datalength = 0
        if hasattr(self, 'loader') and self.loader is not None:
            datalength = max(int(self.loader.datalength), 0)
        elif self.plot_widgets[0].plot_widget.original_index_x is not None:
            datalength = len(self.plot_widgets[0].plot_widget.original_index_x)

        new_display_values = []
        for index_pos in pinned_indices:
            if datalength > 0:
                index_pos = min(max(index_pos, 1), datalength)
            new_display_x = new_offset + new_factor * index_pos
            if np.isfinite(new_display_x):
                new_display_values.append(new_display_x)

        if not new_display_values:
            return

        self.pinned_x_values = new_display_values

        widgets_to_update = []
        for container in self.plot_widgets:
            widget = container.plot_widget
            widget.is_cursor_pinned = True
            widget.pinned_x_values = list(new_display_values)
            widget.pinned_x_value = new_display_values[0]
            if widget.factor != 0:
                widget.pinned_index_value = (new_display_values[0] - widget.offset) / widget.factor
                widget.pinned_index_values = [(x - widget.offset) / widget.factor for x in new_display_values]
            else:
                widget.pinned_index_value = None
                widget.pinned_index_values = []

            widget.vline.blockSignals(True)
            try:
                widget._set_vline_bounds([None, None])
                widget.vline.setMovable(True)
                widget.vline.setPos(new_display_values[0])
            finally:
                widget.vline.blockSignals(False)

            if hasattr(widget, 'vline2'):
                widget.vline2.blockSignals(True)
                try:
                    widget.vline2.setMovable(True)
                    if len(new_display_values) > 1:
                        widget.vline2.setPos(new_display_values[1])
                finally:
                    widget.vline2.blockSignals(False)

            widget._update_vline_bounds_from_data()

            if hasattr(widget.view_box, 'is_cursor_pinned'):
                widget.view_box.is_cursor_pinned = True
            if hasattr(widget, '_last_cursor_update_time'):
                widget._last_cursor_update_time = 0

            widgets_to_update.append(widget)

        def _delayed_label_update(widgets=widgets_to_update):
            for widget in widgets:
                if not getattr(widget, '_is_being_destroyed', False):
                    try:
                        widget.update_cursor_label()
                    except (RuntimeError, AttributeError):
                        pass

        QTimer.singleShot(0, _delayed_label_update)

    def auto_range_all_plots(self):
        for container in self.plot_widgets:
            if container.isVisible():
                container.plot_widget.auto_range()

    def auto_y_in_x_range(self):
        for container in self.plot_widgets:
            if container.isVisible():
                container.plot_widget.auto_y_in_x_range()

    def clear_all_plots(self):
        reply = QMessageBox.question(self, "确认", "是否清除所有绘图？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        for container in self.plot_widgets:
            container.plot_widget.clear_plot_item()
        self.reset_all_pin_states()
        self.update_mark_stats()

    def create_subplots_matrix(self, rows, cols):
        # 清空现有子图
        for container in self.plot_widgets:
            container.setParent(None)
        self.plot_widgets.clear()

        # 创建新的子图矩阵
        first_plot_widget = None
        for i in range(rows * cols):
            plot_widget = DraggableGraphicsLayoutWidget(
                units_dict=self.units if hasattr(self, 'units') else {},
                dataframe=self.loader.df if hasattr(self, 'loader') else pd.DataFrame(),
                time_channels_info=self.time_channels_infos if hasattr(self, 'time_channels_infos') else {},
                synchronizer=self.sync_crosshair
            )
            container = PlotContainerWidget(plot_widget)
            
            # 为容器添加 plot_widget 属性
            container.plot_widget = plot_widget
            
            # 设置 X 轴联动
            if first_plot_widget is None:
                # 第一个子图作为联动目标
                first_plot_widget = plot_widget
                plot_widget.plot_item.setXLink(None)  # 第一个图不链接到任何图
            else:
                # 其他子图都链接到第一个子图
                plot_widget.plot_item.setXLink(first_plot_widget.plot_item)

            self.plot_widgets.append(container)

        self._plot_row_max_default = rows
        self._plot_col_max_default = cols

    def set_plots_visible(self, rows, cols):
        # 更新当前布局的行列记录
        self._plot_row_current = rows
        self._plot_col_current = cols

        if not self.plot_widgets:
            return

        # --- 1. 从主绘图控件（总是第一个）获取真实的状态信息 ---
        master_plot = self.plot_widgets[0].plot_widget
        master_x_limits = master_plot.plot_item.getViewBox().state['limits']['xLimits']
        master_x_range = master_plot.view_box.viewRange()[0]

        # --- 2. 准备工作：先将所有控件的链接断开，避免操作过程中出现意外联动 ---
        for container in self.plot_widgets:
            container.plot_widget.plot_item.setXLink(None)

        # --- 3. 正确地清空布局 ---
        # 使用 takeAt(0) 将控件从布局中取出，但不会从窗口中删除它们。
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget():
                # 隐藏取出的控件，等待后续重新显示
                child.widget().hide()

        first_visible_plot = None
        # --- 4. 重新布局：遍历新布局的每个位置，放置并设置控件 ---
        for i in range(rows):
            for j in range(cols):
                idx = i * self._plot_col_max_default + j
                if idx < len(self.plot_widgets):
                    container = self.plot_widgets[idx]
                    widget = container.plot_widget

                    # 对于将要显示的控件，强制同步其缩放限制
                    widget.plot_item.setLimits(xMin=master_x_limits[0], xMax=master_x_limits[1])

                    # 重新建立链接关系
                    if first_visible_plot is None:
                        first_visible_plot = widget  # 找到第一个可见的控件作为新的主控件
                    else:
                        widget.plot_item.setXLink(first_visible_plot.plot_item)

                    # 将控件添加到新布局的正确位置并显示它
                    self.plot_layout.addWidget(container, i, j)
                    container.show()

        # --- 5. 恢复主控件的视图范围，此操作会自动通过链接传递给所有其他可见控件 ---
        if first_visible_plot is not None:
            first_visible_plot.view_box.setXRange(master_x_range[0], master_x_range[1], padding=0)

        # After the layout is finalized, force an update of the cursor state for all visible plots.
        self.toggle_cursor_all(self.cursor_btn.isChecked())
        
    def update_mark_regions_on_layout_change(self):
        if self.mark_region_btn.isChecked():
            if self.plot_widgets and self.plot_widgets[0].plot_widget.mark_region:
                self.saved_mark_range = self.plot_widgets[0].plot_widget.mark_region.getRegion()
            if self.saved_mark_range:
                min_x, max_x = self.saved_mark_range
            else:
                view_min, view_max = self.plot_widgets[0].plot_widget.view_box.viewRange()[0]
                width = view_max - view_min
                min_x = view_min + width / 3
                max_x = view_min + 2 * width / 3

            for container in self.plot_widgets:
                plot_widget = container.plot_widget
                if plot_widget.mark_region:
                    plot_widget.remove_mark_region()

            for container in self.plot_widgets:
                if container.isVisible():
                    container.plot_widget.add_mark_region(min_x, max_x)
            self.update_mark_stats()

    def reset_plots_after_loading(self, index_xMin, index_xMax):
        """Helper function to reset all plots to a new data range."""
        self.reset_all_pin_states()
        # 【安全标志】设置所有widget为更新中状态
        for container in self.plot_widgets:
            container.plot_widget._is_updating_data = True
        
        try:
            for container in self.plot_widgets:
                container.plot_widget.reset_plot(index_xMin, index_xMax)
            self.value_cache.clear()
            self.saved_mark_range = None
            if hasattr(self, 'mark_stats_window') and self.mark_stats_window:
                self.mark_stats_window.hide()
                self.mark_stats_window.tree.clear()

            if self.mark_region_btn.isChecked():
                self.mark_region_btn.setChecked(False) # This will also trigger toggle_mark_region(False)
        
        finally:
            # 【安全标志】恢复所有widget的正常状态
            for container in self.plot_widgets:
                container.plot_widget._is_updating_data = False
            
            # 【样式同步】主动触发样式更新
            for container in self.plot_widgets:
                widget = container.plot_widget
                try:
                    if hasattr(widget, 'view_box') and hasattr(widget, 'curve') and widget.curve is not None:
                        widget.update_plot_style(widget.view_box, widget.view_box.viewRange(), None)
                except Exception:
                    pass

    def replots_after_loading(self):
        self.reset_all_pin_states()
        # 标记为更新状态，避免信号回调访问到半更新的数据
        for container in self.plot_widgets:
            container.plot_widget._is_updating_data = True
            if hasattr(container.plot_widget, "_update_timer"):
                container.plot_widget._update_timer.stop()

        try:
            if self.loader.datalength == 0:
                return

            all_y_names: list[str] = []
            for container in self.plot_widgets:
                widget = container.plot_widget
                if widget.y_name:
                    all_y_names.append(widget.y_name)
                if widget.is_multi_curve_mode and widget.curves:
                    all_y_names.extend(widget.curves.keys())

            if DataTableDialog._instance is not None:
                all_y_names.extend(DataTableDialog._instance._df.columns.tolist())

            unique_y_names = set(all_y_names)
            if not unique_y_names:
                self.reset_plots_after_loading(1, self.loader.datalength)
                return

            var_names_set = set(self.loader.var_names)
            in_var_names = [y for y in unique_y_names if y in var_names_set]

            if in_var_names:
                validity_values = [self.loader.df_validity.get(y, -1) for y in in_var_names]
                validity_array = np.array(validity_values)
                valid_mask = validity_array == 1
                found = [in_var_names[i] for i in np.where(valid_mask)[0]]
            else:
                found = []

            ratio = len(found) / len(unique_y_names) if unique_y_names else 0
            cleared: list[tuple[int, str]] = []

            if ratio <= RATIO_RESET_PLOTS or len(found) < 1:
                self.reset_plots_after_loading(1, self.loader.datalength)
            else:
                self.value_cache.clear()
                global DEFAULT_PADDING_VAL_X
                for idx, container in enumerate(self.plot_widgets):
                    widget = container.plot_widget

                    original_index_x = np.arange(1, self.loader.datalength + 1, dtype=np.float32)
                    min_x = widget.offset + widget.factor * np.min(original_index_x)
                    max_x = widget.offset + widget.factor * np.max(original_index_x)
                    min_x, max_x = widget._get_safe_x_range(min_x, max_x)
                    limits_xMin = min_x - DEFAULT_PADDING_VAL_X * (max_x - min_x)
                    limits_xMax = max_x + DEFAULT_PADDING_VAL_X * (max_x - min_x)
                    widget._set_x_limits_with_min_range(limits_xMin, limits_xMax)
                    widget._set_vline_bounds([min_x, max_x])

                    if widget.is_multi_curve_mode and widget.curves:
                        current_curves = dict(widget.curves)
                        widget.curves.clear()
                        widget.is_multi_curve_mode = False
                        widget.current_color_index = 0
                        widget._clear_cursor_items(hide_only=False)
                        widget._safe_clear_plot_items()
                        widget.curve = None
                        widget.y_name = ''
                        widget.original_index_x = None
                        widget.original_y = None

                        curves_added = 0
                        visibility_to_restore: dict[str, bool] = {}
                        for var_name, curve_info in current_curves.items():
                            if var_name in self.loader.df.columns and self.loader.df_validity.get(var_name, -1) >= 0:
                                success = widget.add_variable_to_plot(
                                    var_name,
                                    skip_existence_check=True,
                                    show_duplicate_warning=False,
                                    preferred_color=curve_info.get('color')
                                )
                                if success:
                                    curves_added += 1
                                    visibility_to_restore[var_name] = curve_info.get("visible", True)
                            else:
                                cleared.append((idx + 1, f"{var_name} failed to refresh"))

                        widget.update_multi_curve_mode()

                        for var_name, original_visible in visibility_to_restore.items():
                            if var_name in widget.curves:
                                widget.curves[var_name]["visible"] = original_visible
                                curve_obj = widget.curves[var_name].get("curve")
                                if curve_obj is not None:
                                    try:
                                        curve_obj.setVisible(original_visible)
                                    except Exception:
                                        pass

                        if curves_added > 0:
                            widget.update_legend()
                        else:
                            cleared.append((idx + 1, "no curves restored"))
                    else:
                        y_name = widget.y_name
                        if not y_name:
                            continue
                        if y_name in self.loader.df.columns and self.loader.df_validity.get(y_name, -1) >= 0:
                            success = widget.plot_variable(y_name)
                            if not success:
                                widget.clear_plot_item()
                                cleared.append((idx + 1, f"{y_name} refresh failed"))
                        else:
                            widget.clear_plot_item()
                            reason = (
                                f"{y_name} missing from new data"
                                if y_name not in self.loader.df.columns
                                else f"{y_name} has no valid data"
                            )
                            cleared.append((idx + 1, reason))

                if self.plot_widgets:
                    first_plot = self.plot_widgets[0].plot_widget
                    curr_min, curr_max = first_plot.view_box.viewRange()[0]
                    first_plot.view_box.setXRange(curr_min, curr_max, padding=0)

                if cleared:
                    msg = "Plots that could not refresh:\n"
                    for plot_idx, reason in cleared:
                        msg += f"Plot {plot_idx}: {reason}\n"
                    QMessageBox.information(self, "更新通知", msg)

        finally:
            for container in self.plot_widgets:
                container.plot_widget._is_updating_data = False

            for container in self.plot_widgets:
                widget = container.plot_widget
                try:
                    if hasattr(widget, "view_box") and hasattr(widget, "plot_item"):
                        has_data = (widget.curve is not None) or (widget.is_multi_curve_mode and widget.curves)
                        if has_data:
                            widget.update_plot_style(widget.view_box, widget.view_box.viewRange(), None)
                except Exception:
                    pass
                
    def show_drop_overlay(self):
        self.drop_overlay.setGeometry(self.centralWidget().rect())
        self.drop_overlay.raise_()
        self.drop_overlay.show()
        self.drop_overlay.activateWindow()

    def hide_drop_overlay(self):
        self.drop_overlay.hide()

    def _unregister_global_event_filter(self):
        if not getattr(self, "_drop_event_filter_registered", False):
            return
        app = QApplication.instance()
        if app:
            app.removeEventFilter(self)
        self._drop_event_filter_registered = False
        
    def eventFilter(self, obj, event):
            # 1. 基本类型检查
            if not isinstance(obj, QWidget):
                return super().eventFilter(obj, event)

            # 2. 获取事件的目标窗口
            target_window = obj.window()

            # 3. 【关键修正】严格的归属权检查
            # 如果事件的目标窗口不是当前这个实例(self)，
            # 绝对不要拦截(return True)，也不要修改事件状态(event.ignore)。
            # 直接放行，让属于那个窗口的过滤器去处理它。
            if target_window is not self:
                return super().eventFilter(obj, event)

            # --- 以下逻辑仅处理属于本窗口(self)的事件 ---
            
            etype = event.type()

            # A. 拖拽进入 (DragEnter)
            if etype == QEvent.DragEnter:
                # 只有 DragEnter 有 mimeData，安全检查
                if event.mimeData().hasUrls():
                    urls = event.mimeData().urls()
                    # 检查文件后缀
                    supported = any(self._extract_file_extension(u.toLocalFile()) is not None for u in urls)

                    if supported:
                        self.show_drop_overlay()
                        self.drop_overlay.adjust_text(file_type_supported=True)
                        event.acceptProposedAction()
                        return True  # 既然是我们处理了，就拦截掉，不再传递
                    else:
                        self.show_drop_overlay()
                        self.drop_overlay.adjust_text(file_type_supported=False)
                        event.ignore()
                        return True

            # B. 拖拽移动 (DragMove)
            elif etype == QEvent.DragMove:
                # 处理文件拖拽的移动
                if event.mimeData().hasUrls():
                    urls = event.mimeData().urls()
                    supported = any(self._extract_file_extension(u.toLocalFile()) is not None for u in urls)
                    if supported:
                        event.acceptProposedAction()
                        return True
                
                # 处理内部变量拖拽的移动 (Text)
                elif event.mimeData().hasText():
                    if not self._is_cursor_over_plot_area():
                        self._clear_active_plot_highlight()

            # C. 拖拽离开 (DragLeave)
            elif etype == QEvent.DragLeave:
                # 无论是文件离开还是变量离开，都隐藏 Overlay
                self.hide_drop_overlay()
                self._clear_active_plot_highlight()
                return True

            # D. 放置 (Drop)
            elif etype == QEvent.Drop:
                self.hide_drop_overlay()
                self._clear_active_plot_highlight()
                
                # 处理文件 Drop
                if event.mimeData().hasUrls():
                    urls = event.mimeData().urls()
                    for u in urls:
                        path = u.toLocalFile()
                        if self._extract_file_extension(path) is not None:
                            self.load_csv_file(path)
                            event.acceptProposedAction()
                            return True
                
                # 处理非文件 Drop (如内部变量拖拽失败)
                event.ignore()
                return False

            # 4. 其他所有非拖拽事件，或者不属于本逻辑处理的事件，交给父类
            return super().eventFilter(obj, event)    

class PlotVariableEditorDialog(QDialog):
    """
    绘图变量编辑器对话框类
    用于管理plot中的多个曲线，支持添加、删除、颜色自定义等功能
    """
    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self.setWindowTitle("绘图变量编辑器")
        self.setModal(False)  # 改为非模态，允许与主窗口交互
        self.resize(600, 400)
        self.setAcceptDrops(True)  # 启用拖拽功能
        
        # 高DPI支持 - PyQt6中不需要WA_UseHighDpiPixmaps
        # PyQt6默认支持高DPI，通过样式表控制字体大小
        
        self.setup_ui()
        self.load_current_curves()
        self._drag_highlight_active = False
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("绘图变量编辑器")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title_label)
        
        # 创建表格
        self.var_table = QTableWidget()
        self.var_table.setColumnCount(3)
        self.var_table.setHorizontalHeaderLabels(["显示", "变量名", "颜色"])
        
        # 设置表格属性
        self.var_table.setDragDropMode(QTableWidget.DragDropMode.DropOnly)
        self.var_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.var_table.setAlternatingRowColors(True)
        self.var_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 6px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #000000;
            }
            QCheckBox {
                font-size: 12px;
            }
        """)
        
        # 设置列宽
        header = self.var_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # 显示列固定宽度
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # 变量名列自适应
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)  # 颜色列固定宽度
        self.var_table.setColumnWidth(0, 60)   # 显示列
        self.var_table.setColumnWidth(2, 80)   # 颜色列
        
        layout.addWidget(self.var_table)
        
        # 按钮区域
        button_layout = QHBoxLayout()

        # 上移/下移按钮
        self.move_up_btn = QPushButton("上移")
        self.move_up_btn.clicked.connect(lambda: self._move_selected_row(-1))
        self.move_up_btn.setEnabled(False)
        button_layout.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("下移")
        self.move_down_btn.clicked.connect(lambda: self._move_selected_row(1))
        self.move_down_btn.setEnabled(False)
        button_layout.addWidget(self.move_down_btn)

        # 删除按钮
        self.remove_btn = QPushButton("删除选中")
        self.remove_btn.clicked.connect(self.remove_selected_variable)
        self.remove_btn.setEnabled(False)
        button_layout.addWidget(self.remove_btn)

        # 清空按钮
        self.clear_btn = QPushButton("清空所有")
        self.clear_btn.clicked.connect(self.clear_all_variables)
        self.clear_btn.setEnabled(False)
        button_layout.addWidget(self.clear_btn)

        # 重置颜色按钮
        self.reset_color_btn = QPushButton("重置颜色")
        self.reset_color_btn.clicked.connect(self.reset_curve_colors)
        self.reset_color_btn.setEnabled(False)
        button_layout.addWidget(self.reset_color_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 说明文本
        info_label = QLabel("提示：从变量表拖拽变量到此窗口可添加新变量")
        info_label.setStyleSheet("color: gray; font-size: 12px; margin-top: 10px;")
        layout.addWidget(info_label)
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(bottom_layout)
        self.setLayout(layout)
        
        # 连接信号
        self.var_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.var_table.cellClicked.connect(self.on_cell_clicked)
        # 不再需要itemChanged信号，因为使用QCheckBox控件
        
    def load_current_curves(self):
        """加载当前绘图中的曲线"""
        # 先清空表格
        self.var_table.setRowCount(0)
        
        # 检查多曲线模式
        if self.plot_widget.curves:
            # 有curves字典：从curves字典加载（无论是否是多曲线模式）
            for var_name, curve_info in self.plot_widget.curves.items():
                self._add_variable_to_table(var_name, curve_info)
        elif self.plot_widget.curve and self.plot_widget.y_name:
            # 单曲线模式：从curve和y_name加载
            var_name = self.plot_widget.y_name
            
            # 获取曲线的实际可见性状态
            curve_visible = True
            try:
                if hasattr(self.plot_widget.curve, 'isVisible'):
                    curve_visible = self.plot_widget.curve.isVisible()
            except Exception as e:
                print(f"获取曲线可见性失败: {e}")
            
            # 获取曲线的实际颜色
            curve_color = 'blue'
            try:
                if hasattr(self.plot_widget.curve, 'opts') and 'pen' in self.plot_widget.curve.opts:
                    pen = self.plot_widget.curve.opts['pen']
                    if hasattr(pen, 'color'):
                        curve_color = pen.color().name()
            except Exception as e:
                print(f"获取曲线颜色失败: {e}")
            
            curve_info = {
                'color': curve_color,
                'visible': curve_visible,
                'y_format': self.plot_widget.y_format
            }
            self._add_variable_to_table(var_name, curve_info)
        
        self.update_button_states()

    def _highlight_target_plot(self, var_names: list[str] | None = None, hide: bool = False):
        main_window = self.plot_widget.window()
        if not main_window or not hasattr(main_window, "_show_drag_indicator_for_plot"):
            return
        if hide:
            main_window._hide_drag_indicator_for_plot(self.plot_widget)
            self._drag_highlight_active = False
            return
        main_window._show_drag_indicator_for_plot(
            self.plot_widget,
            var_names or [],
            forced_text="释放以添加",
        )
        self._drag_highlight_active = True

    def _get_selected_row(self) -> int | None:
        """获取当前选中的行号"""
        selected_items = self.var_table.selectedItems()
        if not selected_items:
            return None
        return selected_items[0].row()

    def _move_selected_row(self, offset: int):
        """移动选中行的位置"""
        if self.var_table.rowCount() <= 1 or not self.plot_widget.curves:
            return
        current_row = self._get_selected_row()
        if current_row is None:
            return
        target_row = current_row + offset
        if target_row < 0 or target_row >= self.var_table.rowCount():
            return

        order = []
        for row in range(self.var_table.rowCount()):
            name_item = self.var_table.item(row, 1)
            if name_item is not None:
                order.append(name_item.data(Qt.UserRole))

        if len(order) <= 1:
            return

        order[current_row], order[target_row] = order[target_row], order[current_row]
        self._apply_curve_order(order)
        self.load_current_curves()
        self.var_table.selectRow(target_row)

    def _apply_curve_order(self, new_order: list[str]):
        """根据给定顺序重排plot中的曲线"""
        if not self.plot_widget.curves:
            return
        reordered: dict[str, dict] = {}
        for name in new_order:
            if name in self.plot_widget.curves:
                reordered[name] = self.plot_widget.curves[name]
        # 附加遗漏的变量（理论上不会发生）
        for name, info in self.plot_widget.curves.items():
            if name not in reordered:
                reordered[name] = info
        self.plot_widget.curves = reordered
        self.plot_widget.update_legend()
        # 立即刷新光标显示顺序
        self.plot_widget._clear_cursor_items()
        if self.plot_widget.vline.isVisible():
            self.plot_widget.update_cursor_label()
    
    def _add_variable_to_table(self, var_name, curve_info):
        """添加变量到表格"""
        row = self.var_table.rowCount()
        self.var_table.insertRow(row)
        
        # 显示状态复选框 - 使用QCheckBox控件
        checkbox = QCheckBox()
        checkbox.setChecked(curve_info.get('visible', True))
        checkbox.stateChanged.connect(lambda state, name=var_name: self._on_checkbox_changed(name, state))
        self.var_table.setCellWidget(row, 0, checkbox)
        
        # 获取可见性状态
        is_visible = curve_info.get('visible', True)
        
        # 变量名和单位
        unit = self.plot_widget.units.get(var_name, '')
        display_text = f"{var_name} ({unit})" if unit else var_name
        name_item = QTableWidgetItem(display_text)
        name_item.setData(Qt.UserRole, var_name)
        name_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        self.var_table.setItem(row, 1, name_item)
        
        # 颜色 - 使用QWidget显示真实颜色
        color = curve_info.get('color', 'blue')
        color_widget = QWidget()
        color_widget.setStyleSheet(f"background-color: {color}; border: 1px solid #333;")
        color_widget.setFixedSize(30, 20)
        self.var_table.setCellWidget(row, 2, color_widget)
        
        # 同时设置一个隐藏的item来存储数据
        color_item = QTableWidgetItem()
        color_item.setData(Qt.UserRole, var_name)
        color_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        self.var_table.setItem(row, 2, color_item)
        
    def _on_checkbox_changed(self, var_name, state):
        """复选框状态变化处理"""
        # PyQt5 state is int; compare against Qt.Checked to determine visibility
        is_visible = (state == Qt.Checked)
        
        # 更新曲线可见性
        if self.plot_widget.curves and var_name in self.plot_widget.curves:
            self.plot_widget.curves[var_name]['visible'] = is_visible
            curve_info = self.plot_widget.curves[var_name]
            if 'curve' in curve_info:
                curve_obj = curve_info['curve']
                curve_obj.setVisible(is_visible)
            self.plot_widget.update_legend()
        elif not self.plot_widget.is_multi_curve_mode and var_name == self.plot_widget.y_name:
            if self.plot_widget.curve:
                self.plot_widget.curve.setVisible(is_visible)

    def on_selection_changed(self):
        """选择改变时的处理"""
        self.update_button_states()
    
    def on_cell_clicked(self, row, column):
        """单元格点击事件"""
        if column == 2:  # 颜色列
            self.set_variable_color(row)
    
    def toggle_variable_visibility(self, row):
        """切换变量显示状态"""
        var_name = self.var_table.item(row, 0).data(Qt.UserRole)
        visible_item = self.var_table.item(row, 0)
        is_visible = visible_item.checkState() == Qt.CheckState.Checked
        
        if self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves:
            # 多曲线模式：更新curves字典中的可见性
            self.plot_widget.curves[var_name]['visible'] = is_visible
            
            # 更新曲线显示
            curve_info = self.plot_widget.curves[var_name]
            if 'curve' in curve_info:
                try:
                    # 检查曲线对象是否仍然有效
                    if curve_info['curve'].scene() is not None:
                        curve_info['curve'].setVisible(is_visible)
                    else:
                        # 曲线对象已经不在scene中，重新创建
                        self.plot_widget._recreate_curve(var_name)
                except Exception as e:
                    print(f"Warning: Error toggling curve visibility for {var_name}: {e}")
                    # 尝试重新创建曲线
                    self.plot_widget._recreate_curve(var_name)
            
            # 更新legend
            self.plot_widget.update_legend()
        elif not self.plot_widget.is_multi_curve_mode and var_name == self.plot_widget.y_name:
            # 单曲线模式：更新curve的可见性
            if self.plot_widget.curve:
                try:
                    if self.plot_widget.curve.scene() is not None:
                        self.plot_widget.curve.setVisible(is_visible)
                except Exception as e:
                    print(f"Warning: Error toggling single curve visibility: {e}")
        
    def update_button_states(self):
        """更新按钮状态"""
        has_selection = len(self.var_table.selectedItems()) > 0
        has_items = self.var_table.rowCount() > 0
        
        self.remove_btn.setEnabled(has_selection)
        self.clear_btn.setEnabled(has_items)
        self.reset_color_btn.setEnabled(has_items)

        selected_row = self._get_selected_row()
        can_move = (
            has_selection
            and self.var_table.rowCount() > 1
            and bool(self.plot_widget.curves)
        )
        self.move_up_btn.setEnabled(can_move and selected_row is not None and selected_row > 0)
        self.move_down_btn.setEnabled(
            can_move and selected_row is not None and selected_row < self.var_table.rowCount() - 1
        )
        
    def remove_selected_variable(self):
        """删除选中的变量"""
        selected_items = self.var_table.selectedItems()
        if not selected_items:
            return
        
        # 获取所有选中的行号
        selected_rows = set()
        for item in selected_items:
            selected_rows.add(item.row())
        
        # 记录最小的被删除行号，用于后续选中
        min_deleted_row = min(selected_rows)
        
        # 从后往前删除，避免行号变化
        for row in sorted(selected_rows, reverse=True):
            # 获取变量名 - 现在从第二列（变量名列）获取
            var_name_item = self.var_table.item(row, 1)
            if var_name_item is None:
                continue
            var_name = var_name_item.data(Qt.UserRole)
            
            if self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves:
                # 多曲线模式：从curves字典中移除
                curve_info = self.plot_widget.curves[var_name]
                if 'curve' in curve_info and curve_info['curve'].scene() is not None:
                    self.plot_widget.plot_item.removeItem(curve_info['curve'])
                del self.plot_widget.curves[var_name]
            elif var_name in self.plot_widget.curves:
                # 单曲线模式但曲线在curves字典中：从curves字典中移除
                curve_info = self.plot_widget.curves[var_name]
                if 'curve' in curve_info and curve_info['curve'].scene() is not None:
                    self.plot_widget.plot_item.removeItem(curve_info['curve'])
                del self.plot_widget.curves[var_name]
            elif not self.plot_widget.is_multi_curve_mode and var_name == self.plot_widget.y_name:
                # 单曲线模式：清除整个plot
                self.plot_widget.clear_plot_item()
            
            # 从表格中移除
            self.var_table.removeRow(row)
        
        # 删除后自动选中下一条或上一条曲线
        row_count = self.var_table.rowCount()
        if row_count > 0:
            # 优先选中下一条（原来被删除行的位置）
            if min_deleted_row < row_count:
                next_row = min_deleted_row
            else:
                # 如果没有下一条，选中上一条
                next_row = row_count - 1
            
            # 选中整行
            self.var_table.selectRow(next_row)
        
        # 更新多曲线模式
        self.plot_widget.update_multi_curve_mode()
        
        # 更新vline bounds以反映移除变量后的数据范围
        self.plot_widget._update_vline_bounds_from_data()
        
        # 如果删除了所有曲线，确保完全清理
        if not self.plot_widget.curves:
            # 清理所有可能的残留
            if self.plot_widget.curve and self.plot_widget.curve.scene() is not None:
                self.plot_widget.plot_item.removeItem(self.plot_widget.curve)
            self.plot_widget.curve = None
            self.plot_widget.y_name = ''
            self.plot_widget.y_format = ''
            self.plot_widget.original_index_x = None
            self.plot_widget.original_y = None
            self.plot_widget.current_color_index = 0
            self.plot_widget.is_multi_curve_mode = False
            self.plot_widget.update_left_header("channel name")
            self.plot_widget.update_right_header("")
            
            # 清理所有plot item（先清除cursor items）
            # 清空所有变量时完全清除对象池，避免复用异常状态的items
            self.plot_widget._clear_cursor_items(hide_only=False)
            self.plot_widget._safe_clear_plot_items()
        
        self.update_button_states()
        
    def clear_all_variables(self):
        """清空所有变量"""
        reply = QMessageBox.question(self, "确认", "确定要清空所有绘图变量吗？",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            # 清空所有曲线
            if self.plot_widget.is_multi_curve_mode:
                # 多曲线模式：清空curves字典
                for var_name, curve_info in list(self.plot_widget.curves.items()):
                    if 'curve' in curve_info and curve_info['curve'].scene() is not None:
                        self.plot_widget.plot_item.removeItem(curve_info['curve'])
                self.plot_widget.curves.clear()
            else:
                # 单曲线模式：清空整个plot
                self.plot_widget.clear_plot_item()
            
            self.plot_widget.is_multi_curve_mode = False
            self.plot_widget.current_color_index = 0
            
            # 清空表格
            self.var_table.setRowCount(0)
            self.update_button_states()
            
            # 更新显示
            self.plot_widget.update_left_header("channel name")
            self.plot_widget.update_right_header("")
            
            # 重置vline bounds到默认值
            self.plot_widget._update_vline_bounds_from_data()
            
    def reset_curve_colors(self):
        """按照默认顺序重新分配曲线颜色"""
        row_count = self.var_table.rowCount()
        if row_count == 0:
            return
        color_cycle = getattr(self.plot_widget, 'curve_colors', ['blue'])
        if not color_cycle:
            return

        if self.plot_widget.curves:
            for idx, var_name in enumerate(self.plot_widget.curves.keys()):
                color_name = color_cycle[idx % len(color_cycle)]
                self._apply_color_to_curve(var_name, color_name)
        elif self.plot_widget.curve and self.plot_widget.y_name:
            self._apply_color_to_curve(self.plot_widget.y_name, color_cycle[0])

        # 重新加载表格以更新颜色显示
        selected_row = self._get_selected_row()
        self.load_current_curves()
        if selected_row is not None and selected_row < self.var_table.rowCount():
            self.var_table.selectRow(selected_row)

    def _apply_color_to_curve(self, var_name: str, color_name: str):
        """将指定变量的颜色更新为给定颜色"""
        updated = False
        if self.plot_widget.curves and var_name in self.plot_widget.curves:
            curve_info = self.plot_widget.curves[var_name]
            curve_info['color'] = color_name
            if 'curve' in curve_info and curve_info['curve'] is not None:
                curve_obj = curve_info['curve']
                old_pen = curve_obj.opts.get('pen')
                width = DEFAULT_LINE_WIDTH
                if hasattr(old_pen, 'widthF'):
                    width = old_pen.widthF()
                elif hasattr(old_pen, 'width'):
                    width = old_pen.width()
                curve_obj.setPen(pg.mkPen(color=color_name, width=width))

                # 如果curve当前有symbols，也需要更新symbol的颜色
                if hasattr(curve_obj, '_has_symbols') and curve_obj._has_symbols:
                    curve_obj.setSymbolPen(color_name)
                    curve_obj.setSymbolBrush(color_name)

                # 清除缓存标志，强制下次刷新时重新应用样式
                if hasattr(curve_obj, '_cached_pen_key'):
                    delattr(curve_obj, '_cached_pen_key')

            updated = True
        elif var_name == self.plot_widget.y_name and self.plot_widget.curve:
            old_pen = self.plot_widget.curve.opts.get('pen')
            width = DEFAULT_LINE_WIDTH
            if hasattr(old_pen, 'widthF'):
                width = old_pen.widthF()
            elif hasattr(old_pen, 'width'):
                width = old_pen.width()
            self.plot_widget.curve.setPen(pg.mkPen(color=color_name, width=width))

            # 如果curve当前有symbols，也需要更新symbol的颜色
            if hasattr(self.plot_widget.curve, '_has_symbols') and self.plot_widget.curve._has_symbols:
                self.plot_widget.curve.setSymbolPen(color_name)
                self.plot_widget.curve.setSymbolBrush(color_name)

            # 清除缓存标志，强制下次刷新时重新应用样式
            if hasattr(self.plot_widget.curve, '_cached_pen_key'):
                delattr(self.plot_widget.curve, '_cached_pen_key')

            updated = True

        if updated:
            self.plot_widget.update_legend()
            # 重新应用样式以确保symbol/线宽保持一致
            if hasattr(self.plot_widget, 'view_box'):
                self.plot_widget.update_plot_style(
                    self.plot_widget.view_box,
                    self.plot_widget.view_box.viewRange(),
                    None
                )
            self.plot_widget._clear_cursor_items()
            if hasattr(self.plot_widget, '_last_cursor_update_time'):
                self.plot_widget._last_cursor_update_time = 0
            if self.plot_widget.vline.isVisible():
                self.plot_widget.update_cursor_label()

    def set_variable_color(self, row=None):
        """设置变量颜色"""
        if row is None:
            # 从选中项获取行号
            selected_items = self.var_table.selectedItems()
            if not selected_items:
                return
            row = selected_items[0].row()
        
        # 获取变量名 - 现在从第二列（变量名列）获取
        var_name_item = self.var_table.item(row, 1)
        if var_name_item is None:
            return
        var_name = var_name_item.data(Qt.UserRole)
        
        # 打开颜色选择对话框
        current_color = 'blue'  # 默认颜色
        if self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves:
            current_color = self.plot_widget.curves[var_name].get('color', 'blue')
        
        color = QColorDialog.getColor(QColor(current_color), self, "选择颜色")
        
        if color.isValid():
            self._apply_color_to_curve(var_name, color.name())
            # 更新表格项颜色
            color_widget = self.var_table.cellWidget(row, 2)
            if color_widget:
                color_widget.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #333;")
                
    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasText():
            var_names = parse_dragged_var_names(event.mimeData().text())
            self._highlight_target_plot(var_names)
            event.acceptProposedAction()
        else:
            if self._drag_highlight_active:
                self._highlight_target_plot(hide=True)
            event.ignore()
    
    def dragMoveEvent(self, event):
        """拖拽移动事件"""
        if event.mimeData().hasText():
            var_names = parse_dragged_var_names(event.mimeData().text())
            self._highlight_target_plot(var_names)
            event.acceptProposedAction()
        else:
            if self._drag_highlight_active:
                self._highlight_target_plot(hide=True)
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        if self._drag_highlight_active:
            self._highlight_target_plot(hide=True)
        super().dragLeaveEvent(event)
            
    def dropEvent(self, event):
        """拖拽放下事件"""
        if event.mimeData().hasText():
            var_names = parse_dragged_var_names(event.mimeData().text())
            if not var_names:
                if self._drag_highlight_active:
                    self._highlight_target_plot(hide=True)
                event.ignore()
                return
            
            duplicates = []
            errors = []
            added = False
            
            for var_name in var_names:
                if (self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves) or \
                   (not self.plot_widget.is_multi_curve_mode and var_name == self.plot_widget.y_name):
                    duplicates.append(var_name)
                    continue
                
                success = self.plot_widget.add_variable_to_plot(var_name)
                if success:
                    added = True
                else:
                    errors.append(var_name)
            
            if added:
                self.load_current_curves()

            # 只在有问题时显示提示
            if duplicates or errors:
                msg_parts = []
                if duplicates:
                    msg_parts.append("以下变量已在绘图中：\n" + "\n".join(duplicates))
                if errors:
                    msg_parts.append("以下变量无法添加：\n" + "\n".join(errors))

                if errors:
                    QMessageBox.warning(self, "批量添加结果", "\n\n".join(msg_parts))
                else:
                    QMessageBox.information(self, "批量添加结果", "\n\n".join(msg_parts))
            
            event.acceptProposedAction()
        else:
            event.ignore()
        if self._drag_highlight_active:
            self._highlight_target_plot(hide=True)

# ---------------- 主程序 ----------------

if __name__ == "__main__":
    # 设置应用程序属性
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # 启用高DPI缩放
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)     # 启用高DPI图标

    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 设置应用程序属性，确保安全退出
    app.setQuitOnLastWindowClosed(True)
    if sys.platform == "win32":

        def get_windows_chinese_font():
            font_priority = [
                'Microsoft YaHei UI',  # Win10/11默认
                'Microsoft YaHei',     # Win7/8默认
                'SimHei',              # 传统Windows
                'Arial Unicode MS'     # 备用
            ]
            
            font_db = QFontDatabase()  # 创建字体数据库对象
            available_fonts = font_db.families()  # 通过实例调用方法
            
            for font in font_priority:
                if font in available_fonts:
                    return QFont(font)
            
            return QApplication.font()
        
        font = get_windows_chinese_font()        
        font.setPointSize(9)
        app.setFont(font)
        # app.setStyle("Fusion")


    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    window_ref = weakref.ref(window)

    # 添加信号处理器，确保安全退出
    import signal
    def signal_handler(signum, frame):
        target = window_ref()
        if target is not None:
            target._safe_cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 在Windows上添加额外的信号处理
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, signal_handler)

    # 执行应用程序
    try:
        sys.exit(app.exec_())
    except SystemExit:
        # 确保清理资源
        target = window_ref()
        if target is not None:
            target._safe_cleanup()
        raise

# pyinstaller
#     - one file
# pyinstaller csv_plot_pyqt5.py --onefile --name csv_plot_pyqt5 --icon icon.ico --add-data "icon.ico;." --add-data "README.md;." --noconsole --noupx --clean --noconfirm
#     - one dir
# pyinstaller csv_plot_pyqt5.py --onedir --name csv_plot_pyqt5 --icon icon.ico --add-data "icon.ico;." --add-data "README.md;." --noconsole --clean --noconfirm

# nuitka compile
# 
# nuitka --standalone --output-filename=csv_plot_pyqt5 --windows-console-mode=disable --windows-icon-from-ico=icon.ico --enable-plugin=pyqt5 --include-data-file=icon.ico=data --include-data-file=README.md=data --remove-output csv_plot_pyqt5.py
