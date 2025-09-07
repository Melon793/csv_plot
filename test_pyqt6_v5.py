from __future__ import annotations 
import sys
import os
import numpy as np
import pandas as pd

# 屏蔽 macOS ICC 警告
os.environ["QT_LOGGING_RULES"] = (
    "qt6ct.debug=false; "      # 原来想关的 qt6ct 日志
    "qt.gui.icc=false"         # 关闭 ICC 解析相关日志
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData, QMargins, QTimer, QEvent,QObject,QMargins,Qt, QAbstractTableModel, QModelIndex,QModelIndex, QPoint, QSize, QRect,QItemSelectionModel
from PyQt6.QtGui import  QFontMetrics, QDrag, QPen, QColor,QBrush,QAction,QIcon,QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QProgressDialog,QGridLayout,QSpinBox,QMenu,QTextEdit,
    QFileDialog, QPushButton, QAbstractItemView, QLabel, QLineEdit,QTableView,QStyledItemDelegate,
    QMessageBox, QDialog, QFormLayout, QSizePolicy,QGraphicsLinearLayout,QGraphicsProxyWidget,QGraphicsWidget,QTableWidget,QTableWidgetItem,QHeaderView, QRubberBand,QDoubleSpinBox,QTreeWidget,QTreeWidgetItem, QSplitter,
)
import pyqtgraph as pg
from threading import Lock

global DEFAULT_PADDING_VAL,FILE_SIZE_LIMIT_BACKGROUND_LOADING,RATIO_RESET_PLOTS
DEFAULT_PADDING_VAL= 0.02
FILE_SIZE_LIMIT_BACKGROUND_LOADING = 5
RATIO_RESET_PLOTS = 0.3


# PyInstaller 解包目录
from pathlib import Path
def resource_path(relative_path: str) -> Path:
    """获取打包后的资源文件路径"""
    if hasattr(sys, "_MEIPASS"):  
        return Path(os.path.join(sys._MEIPASS, relative_path))
    return Path(relative_path)

# 设置应用程序和窗口图标
if sys.platform == "win32": # Windows
    myappid = 'mycompany.csv_plot.0.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    ico_path = resource_path("icon.ico")  

elif sys.platform == "darwin":  # macOS
    ico_path = resource_path("icon.icns")  


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("帮助文档")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        
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

    def __init__(self, file_path: str, parent=None,descRows:int=0,sep:str=',',hasunit:bool=True):
        super().__init__(parent)
        self.file_path = file_path
        self.descRows=descRows
        self.sep=sep
        self.hasunit=hasunit
    def run(self):
        try:
            # 将 FastDataLoader 的读取过程按块拆进度
            # 这里用文件大小估算百分比，够简单
            #total_bytes = os.path.getsize(self.file_path)#Path(self.file_path).stat().st_size

            def _progress_cb(progress: int):
                self.progress.emit(progress)

            # print("Calling FastDataLoader with _progress:", _progress_cb) 

            # 给 FastDataLoader 打补丁：加一个回调
            loader = FastDataLoader(
                self.file_path,
                # 其他参数照抄
                descRows=self.descRows,
                sep=self.sep,
                hasunit=self.hasunit,
                chunksize=3600,          
                _progress= _progress_cb,
            )
            self.finished.emit(loader)
        except Exception as e:
            self.error.emit(str(e))

class FastDataLoader:
    # 脏数据清单
    _NA_VALUES = [
        "", "nan", "NaN", "NAN", "NULL", "null", "None",
        "Inf", "inf", "-inf", "-Inf", "1.#INF", "-1.#INF", "data err", '* *', '----', 'Infinity', 'no value'
    ]
    from typing import Callable
    def __init__(
        self,
        csv_path: str ,
        *,
        max_rows_infer: int = 200,
        chunksize: int | None = None,
        usecols: list[str] | None = None,
        drop_empty: bool = False,
        downcast_float: bool = True,
        descRows: int = 0,
        sep: str = ",",
        _progress: Callable | None = None,
        do_parse_date: bool =False,
        hasunit:bool = True
    ):
        #print("Calling inside FastDataLoader with _progress:", _progress) 
        self._path = csv_path
        self.file_size = os.path.getsize(csv_path) 
        self.max_rows_infer = max_rows_infer
        self.usecols = usecols
        self.drop_empty = drop_empty
        self.downcast_float = downcast_float
        self.sep = sep
        self.descRows = descRows
        self._progress_cb = _progress
        self.do_parse_date=do_parse_date
        self.hasunit=hasunit

        # 一次性读取 header + 单位行，并回退编码
        self._var_names, self._units, self.encoding_used = self._load_header_units(
            self._path, desc_rows=self.descRows, usecols=self.usecols, sep=self.sep,hasunit=self.hasunit
        )
        if self._progress_cb:
            self._progress_cb(5)

        # 推断 dtype
        
        sample = pd.read_csv(
            self._path,
            skiprows=(2 + self.descRows) if self.hasunit else (1+self.descRows),
            nrows=self.max_rows_infer,
            names=self._var_names,
            encoding=self.encoding_used,
            usecols=self.usecols,
            low_memory=False,
            sep=self.sep,
            na_values=self._NA_VALUES,
            keep_default_na=True,
        )
        dtype_map, parse_dates, date_formats,downcast_ratio = self._infer_schema(sample)
        self.date_formats = date_formats
        self.sample_mem_size = sample.memory_usage(deep=True).sum()
        # print(f"the estimated downcast ratio is {downcast_ratio*100:2f} %, the compression ratio estimated {(0.5*downcast_ratio+1*(1-downcast_ratio))}")
        # print(f"sample of {sample.shape[0]} lines has costed memory {self.sample_mem_size/(1024**2):2f}Mb")
        #self.byte_per_line = ((0.5*downcast_ratio+1*(1-downcast_ratio))*self.sample_mem_size)/sample.shape[0]
        self.byte_per_line = (0.8*self.sample_mem_size)/sample.shape[0]
        self.estimated_lines = int(self.file_size/(self.byte_per_line ))
        # print(f"this file might have lines of {self.estimated_lines}")
        import gc
        del sample 
        gc.collect()
        if self._progress_cb:
            self._progress_cb(15)
            
        # 计算 chunk 大小
        if chunksize is None:
            chunksize = 3600
        
        #print(f"chunk size is {chunksize}")
        # 正式读取
        self._df = self._read_chunks(
            self._path,
            dtype_map,
            parse_dates,
            int(chunksize),
            sep=self.sep,
            descRows=self.descRows,
            hasunit=self.hasunit
        )
        #print(f"actual lines of data files is {self.row_count}")
        # 后处理
        if drop_empty:
            self._df = self._df.dropna(axis=1, how="all")
        if downcast_float:
            self._downcast_numeric()
        self._df_validity=self._check_df_validity()
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
        hasunit:bool=True
    ) -> tuple[list[str], dict[str, str], str]:
        """
        返回 (变量名列表, {变量名: 单位}, 最终编码)
        """
        nrows = 2 if hasunit else 1
        encodings_default = ["utf-8", "cp1252"]

        # chardet
        import chardet
        from pathlib import Path
        sample_sime = 2000 #bytes
        with Path(path).open('rb') as f:
            raw_sample = f.read(sample_sime)
        r = chardet.detect(raw_sample)
        language_infer = r['language']
        confidence_infer = r['confidence']

        if language_infer is not None and language_infer.lower() =='chinese' and confidence_infer > 0.7:
            encoding_infer = ['gb18030']
        else:
            encoding_infer = ['utf-8']

        for enc in list(dict.fromkeys(encoding_infer+encodings_default)):
            try:
                df = pd.read_csv(
                    path,
                    skiprows=desc_rows,
                    nrows=nrows,
                    header=None,
                    usecols=usecols,
                    sep=sep,
                    encoding=enc,
                    engine="python",
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
    def _infer_schema(sample: pd.DataFrame) -> tuple[dict[str, str], list[str], dict[str, str],float]:
        dtype_map: dict[str, str] = {}
        parse_dates: list[str] = []
        date_formats: dict[str, str] = {}

        date_candidates = [
            "%Y/%m/%d","%H:%M:%S", "%H:%M:%S.%f",
            "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d","%d-%m-%Y", "%m-%d-%Y"
        ]
        float_cols = sample.select_dtypes(include=['float', 'float64','category'])
        downcast_ratio_est = float_cols.shape[1] / sample.shape[1] if sample.shape[1] > 0 else 0.000001
        for col in sample.columns:
            s = sample[col]
            if s.isna().all():
                dtype_map[col] = "category"
                continue
            for fmt in date_candidates:
                try:
                    pd.to_datetime(s, format=fmt, errors="raise")
                    parse_dates.append(col)
                    date_formats[col] = fmt
                    break
                except (ValueError, TypeError):
                    continue
            else:
                if pd.api.types.is_numeric_dtype(s):
                    dtype_map[col] = "float32"
                else:
                    dtype_map[col] = "category"
        return dtype_map, parse_dates, date_formats,downcast_ratio_est

    def _read_chunks(
        self,
        path: str,
        dtype_map,
        parse_dates: list[str],
        chunksize: int,
        sep: None | str = ",",
        descRows: int = 0,
        hasunit:bool = True,
    ) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        # do not parse date
        if not self.do_parse_date:
            parse_dates=[]
        total_chunks_est = max(1, self.estimated_lines // chunksize + (1 if self.estimated_lines % chunksize else 0))
        increment = 80 / total_chunks_est
        
        for idx,chunk in enumerate(pd.read_csv(
            path,
            skiprows=(2 + descRows) if hasunit else (1+descRows),
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
            on_bad_lines='skip'
        )):
            #print(f"chunksize is {chunksize}, full size {self.file_size/(1024**2):2f}Mb")
            if self._progress_cb:
                chunk_progress = min(80, (idx + 1) * increment)
                self._progress_cb(15 + int(chunk_progress))
                #print (f"progress {idx} is {bytes_read}")
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def _downcast_numeric(self) -> None:
        float_cols = self._df.select_dtypes(include=["float32", "float64"]).columns
        for col in float_cols:
            self._df[col] = (
                pd.to_numeric(self._df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .astype("float32")
            )


    def _check_df_validity(self) -> dict:
        validity : dict = {}
        for col in self._df.columns:
            validity[col] = self._classify_column(self._df[col])
        
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
    def _classify_column(series: pd.Series) -> int:
        """
        1: 全部可转数字，且 ≥2 个不同有效值
        0: 全部可转数字，且唯一有效值
        -1: 存在非数字 或 全部 NaN
        """
        # 1) 先尝试整列转 float，失败直接 C
        try:
            numeric = pd.to_numeric(series, errors="raise")
        except (ValueError, TypeError):
            return (-1)

        # 2) 去掉 NaN 后看有效值
        valid = numeric.dropna()
        if valid.empty:          # 全 NaN
            return (-1)

        unique_vals = valid.unique()
        if len(unique_vals) == 1:
            return (0)
        return (1)
    
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
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setStyleSheet("""
            background:rgba(255,255,255,200);   
            border:none;
        """)
        
        self.label = QLabel("请丢入数据", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        font = self.label.font() #QFont()
        font.setPixelSize(font_size)
        font.setBold(True)
        self.label.setFont(font)

    def resizeEvent(self, event):
        #self.label.adjustSize()
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

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self._df.iloc[index.row(), index.column()]
        return str(value) if pd.notnull(value) else ""

    def headerData(self, section, orientation, role):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
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

    def paint(self, painter, option, index):
        painter.save()
        # 高亮选中的单元格所在的行和列
        if index.row() in self.selected_rows or index.column() in self.selected_cols:
            painter.fillRect(option.rect, QColor(200, 200, 255, 128))  # 浅蓝高亮，半透明
        
        # 新增：高亮来自另一个视图的行
        if index.row() in self.highlighted_rows:
            painter.fillRect(option.rect, QColor(255, 200, 200, 64))  # 淡红色高亮，更透明
        
        super().paint(painter, option, index)
        painter.restore()

class XYScatterPlotDialog(QDialog):
    def __init__(self, x_data, y_data, x_name, y_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("X/Y 散点图")
        self.resize(400, 300)

        # 设置窗口在关闭时释放内存
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

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
        self.plot_widget.setLabel('left',   text=y_name, color='k', font=bold_font)

        # 直接设置标签字体
        axis_bottom = self.plot_widget.getAxis('bottom')
        axis_left = self.plot_widget.getAxis('left')
        axis_bottom.label.setFont(bold_font)
        axis_left.label.setFont(bold_font)

        self.plot_widget.setTitle(f"{y_name} vs. {x_name}")
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

class DataTableDialog(QDialog):
    _instance = None
    _saved_scroll_pos = None  # 类级变量存储滚动位置

    @classmethod
    def popup(cls, var_name: str, data, parent=None):
        if cls._instance is None:
            cls._instance = cls(parent)

        dlg = cls._instance
        dlg.save_geom()
        if dlg.has_column(var_name):
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return dlg
        
        cls._saved_scroll_pos = dlg.main_view.verticalScrollBar().value() if dlg.main_view else None
        dlg.load_geom()
        dlg._add_variable_to_table(var_name, data)  # 使用内部函数
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    def __init__(self, parent=None):
        super().__init__(parent)


        self.setWindowTitle("变量数值表")
        self.window_geometry = None 
        self.scatter_plot_windows = []

        self.frozen_columns = []

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(5)
        splitter.setChildrenCollapsible(False)

        self.frozen_view = QTableView(self)
        self.frozen_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen_view.verticalHeader().setVisible(True)
        self.frozen_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.frozen_view.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.frozen_view.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.frozen_view.setStyleSheet("QTableView { background-color: rgba(245,245,245,128); }")
        self.frozen_view.horizontalHeader().customContextMenuRequested.connect(self._on_frozen_header_right_click)
        self.frozen_view.horizontalHeader().setSectionsMovable(True)
        self.frozen_view.horizontalHeader().setDragEnabled(True)
        self.frozen_view.horizontalHeader().setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.frozen_view.horizontalHeader().setDragDropOverwriteMode(False)

        self.frozen_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.frozen_view.customContextMenuRequested.connect(self._show_table_context_menu)

        self.main_view = QTableView(self)
        self.main_view.setSortingEnabled(False)
        self.main_view.verticalHeader().setVisible(False)
        self.main_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.main_view.horizontalHeader().setSectionsMovable(True)
        self.main_view.horizontalHeader().setDragEnabled(True)
        self.main_view.horizontalHeader().setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.main_view.horizontalHeader().setDragDropOverwriteMode(False)

        self.main_view.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.main_view.horizontalHeader().customContextMenuRequested.connect(self._on_main_header_right_click)
        self.main_view.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.main_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.main_view.customContextMenuRequested.connect(self._show_table_context_menu)
        
        fm = QFontMetrics(self.main_view.font())
        safe_height = int(fm.height() * 1.6)
        self.main_view.verticalHeader().setDefaultSectionSize(safe_height)
        self.frozen_view.verticalHeader().setDefaultSectionSize(safe_height)

        self.main_view.setWordWrap(False)
        self.frozen_view.setWordWrap(False)

        splitter.addWidget(self.frozen_view)
        splitter.addWidget(self.main_view)
        splitter.setSizes([200, 400])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

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
        self.main_view.horizontalHeader().sectionMoved.connect(self._sync_column_order)
        self.frozen_view.horizontalHeader().sectionMoved.connect(self._sync_column_order)

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

        if self.parent() and self.parent().data_table_geometry:
            self.restoreGeometry(self.parent().data_table_geometry)
        else:
            self.resize(600, 400)

    # 事件过滤器类处理拖放事件
    class DropFilter(QObject):
        def __init__(self, parent_dialog):
            super().__init__(parent_dialog)
            self.parent_dialog = parent_dialog
            
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.DragEnter:
                if event.mimeData().hasText():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Type.DragMove:
                if event.mimeData().hasText():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Type.Drop:
                if event.mimeData().hasText():
                    var_name = event.mimeData().text()
                    self.parent_dialog._handle_dropped_variable(var_name)
                    event.acceptProposedAction()
                    return True
            return super().eventFilter(obj, event)

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
        var_name = event.mimeData().text()
        self._handle_dropped_variable(var_name)
        event.acceptProposedAction()

    # 内部函数：处理拖放的变量
    def _handle_dropped_variable(self, var_name: str):
        """处理拖放的变量，添加到非冻结区"""
        # 检查变量是否已存在
        if self.has_column(var_name):
            #QMessageBox.information(self, "提示", f"变量 '{var_name}' 已存在")
            self.scroll_to_column(var_name)
            return
            
        # 获取数据
        main_window = self.parent()
        if not isinstance(main_window, MainWindow):
            # 尝试通过其他方式获取主窗口，例如通过应用程序的activeWindow
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, MainWindow):
                    main_window = widget
                    break
            else:
                QMessageBox.warning(self, "错误", "无法找到主窗口")
                return
        
        if not hasattr(main_window, 'loader') or main_window.loader is None:
            QMessageBox.warning(self, "错误", "没有可用的数据")
            return
            
        if var_name not in main_window.loader.df.columns:
            QMessageBox.warning(self, "错误", f"变量 '{var_name}' 不存在")
            return
            
        # 添加到表格
        series = main_window.loader.df[var_name]
        self._add_variable_to_table(var_name, series)
        
        # 滚动到新添加的列
        QTimer.singleShot(100, lambda: self.scroll_to_column(var_name))

    # 内部函数：添加变量到表格
    def _add_variable_to_table(self, var_name: str, data: pd.Series):
        """内部函数：将变量添加到表格的非冻结区"""
        self._df[var_name] = data
        if hasattr(self.parent(), 'loader') and self.parent().loader:
            self.units = self.parent().loader.units
        self.model = PandasTableModel(self._df, self.units)
        self.main_view.setModel(self.model)
        self.frozen_view.setModel(self.model)
        self._connect_signals()
        self._update_views()
        if self._saved_scroll_pos is not None:
            QTimer.singleShot(0, lambda: self.main_view.verticalScrollBar().setValue(self._saved_scroll_pos))

    def eventFilter(self, obj, event):
        # 处理焦点变化事件
        if event.type() == QEvent.Type.FocusIn:
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
        """
        根据视觉顺序判断是否显示绘图菜单，并传递正确的列索引。
        """
        view = self.sender()
        if not isinstance(view, QTableView):
            return

        selected_indexes = view.selectionModel().selectedIndexes()
        if not selected_indexes:
            return
        frozen_cols = set(self._df.columns.get_loc(col) for col in self.frozen_columns)
        if view == self.main_view:
            # non-frozen selected
            this_cols = set(idx.column() for idx in selected_indexes)- frozen_cols
            other_view = self.frozen_view
            other_selected = other_view.selectionModel().selectedIndexes()
            other_cols = set(idx.column() for idx in other_selected) & frozen_cols
        else:
            # frozen
            other_view = self.main_view
            this_cols = set(idx.column() for idx in selected_indexes) & frozen_cols
            other_selected = other_view.selectionModel().selectedIndexes()
            other_cols = set(idx.column() for idx in other_selected) - frozen_cols

        all_selected = selected_indexes + other_selected
        total_cols = this_cols | other_cols

        # 先检查该view是否有正好2 cols
        if len(this_cols) == 2:
            cols_list = sorted(list(this_cols))
            rows_per_col = {}
            for col in cols_list:
                rows = set(idx.row() for idx in selected_indexes if idx.column() == col)
                rows_per_col[col] = rows
            if len(rows_per_col[cols_list[0]]) >= 2 and rows_per_col[cols_list[0]] == rows_per_col[cols_list[1]]:
                # 确定顺序：根据视觉索引
                header = view.horizontalHeader()
                vis1 = header.visualIndex(cols_list[0])
                vis2 = header.visualIndex(cols_list[1])
                if vis1 < vis2:
                    x_col, y_col = cols_list[0], cols_list[1]
                else:
                    x_col, y_col = cols_list[1], cols_list[0]
                min_row = min(rows_per_col[cols_list[0]])
                num_rows = len(rows_per_col[cols_list[0]])
                self._show_plot_menu(pos, view, x_col, y_col, min_row, num_rows, enabled=True)
                return

        # 然后检查跨区 1+1
        elif len(this_cols) == 1 and len(other_cols) == 1:
            this_col = list(this_cols)[0]
            other_col = list(other_cols)[0]
            this_rows = set(idx.row() for idx in selected_indexes)
            other_rows = set(idx.row() for idx in other_selected)
            if len(this_rows) >= 2 and this_rows == other_rows:
                if view == self.frozen_view:
                    x_col, y_col = this_col, other_col
                else:
                    x_col, y_col = other_col, this_col
                min_row = min(this_rows)
                num_rows = len(this_rows)
                self._show_plot_menu(pos, view, x_col, y_col, min_row, num_rows, enabled=True)
                return

        # else: if total 2 cols, show disabled
        if len(total_cols) == 2:
            cols_list = sorted(list(total_cols))
            x_col, y_col = cols_list[0], cols_list[1]
            all_rows = set(idx.row() for idx in all_selected)
            if all_rows:
                min_row = min(all_rows)
                num_rows = max(all_rows) - min_row + 1
            else:
                min_row = 0
                num_rows = 0
            self._show_plot_menu(pos, view, x_col, y_col, min_row, num_rows, enabled=False)
        else:
            return

    def _show_plot_menu(self, pos, view, x_col, y_col, min_row, num_rows, enabled=True):
        x_name = self.model.headerData(x_col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole).replace('\n', ' ')
        y_name = self.model.headerData(y_col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole).replace('\n', ' ')
        menu = QMenu(self)
        act1 = QAction(f"绘制x/y图，x={x_name}，y={y_name}", menu)
        act1.triggered.connect(lambda: self._plot_xy_scatter(x_col, y_col, min_row, num_rows))
        act1.setEnabled(enabled)
        act2 = QAction(f"绘制x/y图，x={y_name}，y={x_name}", menu)
        act2.triggered.connect(lambda: self._plot_xy_scatter(y_col, x_col, min_row, num_rows))
        act2.setEnabled(enabled)
        menu.addAction(act1)
        menu.addAction(act2)
        menu.exec(view.mapToGlobal(pos))

    def _plot_xy_scatter(self, x_col_idx, y_col_idx, start_row, num_rows):
        """
        接收已按视觉顺序确定的逻辑列索引进行绘图。
        """
        try:
            # 直接使用正确的逻辑索引提取数据
            x_data_series = pd.to_numeric(self._df.iloc[start_row : start_row + num_rows, x_col_idx], errors='coerce')
            y_data_series = pd.to_numeric(self._df.iloc[start_row : start_row + num_rows, y_col_idx], errors='coerce')

            # 验证1：检查是否有非数值数据
            if x_data_series.isnull().any() or y_data_series.isnull().any():
                QMessageBox.warning(self, "绘图错误", "选中区域包含无法转换为数字的单元格。")
                return

            # # 验证2：检查X轴数据是否唯一
            # if x_data_series.nunique() < 2:
            #     QMessageBox.warning(self, "绘图错误", "X轴数据（左侧列）需要至少两个不同的数值才能绘图。")
            #     return
            
            # 获取清理后的列标题
            x_header = self.model.headerData(x_col_idx, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole).replace('\n', ' ')
            y_header = self.model.headerData(y_col_idx, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole).replace('\n', ' ')

            # 创建并显示绘图窗口
            plot_dialog = XYScatterPlotDialog(x_data_series.to_numpy(), y_data_series.to_numpy(), x_header, y_header, self)
            self.scatter_plot_windows.append(plot_dialog)
            plot_dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "未知错误", f"绘图时发生错误: {e}")


    def _connect_signals(self):
        if self.model:
            self.main_view.selectionModel().selectionChanged.connect(self._update_highlights_main)
            self.frozen_view.selectionModel().selectionChanged.connect(self._update_highlights_frozen)


    def save_geom(self):
        if self.parent():
            self.parent().data_table_geometry = self.saveGeometry()

    def load_geom(self):
        if self.parent() and self.parent().data_table_geometry is not None:
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

    def freeze_column(self, logical_col):
        var_name = self._df.columns[logical_col]
        if var_name not in self.frozen_columns:
            self.frozen_columns.append(var_name)
            non_frozen = [c for c in self._df.columns if c not in self.frozen_columns]
            self._df = self._df[self.frozen_columns + non_frozen]
            self.model = PandasTableModel(self._df, self.units)
            self.main_view.setModel(self.model)
            self.frozen_view.setModel(self.model)
            self._connect_signals()
            self._update_views()

    def unfreeze_column(self, logical_col):
        var_name = self._df.columns[logical_col]
        if var_name in self.frozen_columns:
            self.frozen_columns.remove(var_name)
            non_frozen = [c for c in self._df.columns if c not in self.frozen_columns]
            self._df = self._df[self.frozen_columns + non_frozen]
            self.model = PandasTableModel(self._df, self.units)
            self.main_view.setModel(self.model)
            self.frozen_view.setModel(self.model)
            self._connect_signals()
            self._update_views()

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

        selected = menu.exec(header.mapToGlobal(pos))
        if selected == act_delete:
            self._remove_column(logical_col)
        elif selected == act_freeze:
            if var_name in self.frozen_columns:
                self.unfreeze_column(logical_col)
            else:
                self.freeze_column(logical_col)


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
        if var_name in self.frozen_columns:
            self.frozen_columns.remove(var_name)
        
        # We need to drop by name as logical_col index can change
        self._df.drop(columns=[var_name], inplace=True)

        self.model = PandasTableModel(self._df, self.units)
        self.main_view.setModel(self.model)
        self.frozen_view.setModel(self.model)
        self._connect_signals()
        self._update_views()

    def update_data(self, loader):
        if self.model is None or self._df.empty:
            return
        scroll_pos = self.main_view.verticalScrollBar().value()
        frozen_cols = self.frozen_columns.copy()
        current_cols = list(self._df.columns)
        removed = []
        for col in current_cols:
            if col in loader.df.columns:
                self._df[col] = loader.df[col]
            else:
                self._df.drop(columns=[col], inplace=True)
                if col in self.frozen_columns:
                    self.frozen_columns.remove(col)
                removed.append(col)
        self.units = loader.units
        self.model = PandasTableModel(self._df, self.units)
        self.main_view.setModel(self.model)
        self.frozen_view.setModel(self.model)
        self._connect_signals()
        self.frozen_columns = [col for col in frozen_cols if col in self._df.columns]
        self._update_views()
        QTimer.singleShot(0, lambda: self.main_view.verticalScrollBar().setValue(scroll_pos))
        if removed:
            msg = f"以下变量已从数据中移除：{', '.join(removed)}"
            QMessageBox.information(self, "更新通知", msg)
        if self._df.empty:
            self.close()


class LayoutInputDialog(QDialog):
    def __init__(self, 
                 max_rows:int=4, 
                 max_cols:int=2, 
                 cur_rows:int=1, 
                 cur_cols:int=1,
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
        layout.addRow("刻度数量:", self.tick_count_input)

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

            # 设置范围
            if self.axis_type == "X":
                self.view_box.setXRange(min_val, max_val, padding=DEFAULT_PADDING_VAL)
            else:
                self.view_box.setYRange(min_val, max_val, padding=DEFAULT_PADDING_VAL)

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
class MyTableWidget(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["变量名", "单位"])

        # 字体 
        hdr = self.horizontalHeader()
        # 默认 3:1 的初始宽度 
        total = 255          # 首次拿不到 width 时给一个兜底
        self.setColumnWidth(0, int(total * 0.75))
        self.setColumnWidth(1, int(total * 0.25))


        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(False)  # 关闭自动拉伸最后一列
        self.verticalHeader().setVisible(False)  # 隐藏行号
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setSortingEnabled(False)  # 我们手动排序
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # 设置字体大小
        # font = QFont()
        # font.setPointSize(12)  # 调小字体大小
        # self.setFont(font) 

    # def resizeEvent(self, event):
    #         super().resizeEvent(event)
    #         total_width = self.viewport().width()
    #         self.setColumnWidth(0, int(total_width * 0.75))  # 变量名 3/4
    #         self.setColumnWidth(1, int(total_width * 0.25))  # 单位 1/4

       
    def startDrag(self, supportedActions):
        indexes = self.selectedIndexes()
        if not indexes:
            return
        row = indexes[0].row()
        item = self.item(row, 0)
        if item is None:
            return
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(item.text())
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)

    def mouseDoubleClickEvent(self, event):
        index = self.indexAt(event.pos())
        if not index.isValid():
            super().mouseDoubleClickEvent(event)
            return

        row = index.row()
        var_name = self.item(row, 0).text()
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
        self.clearContents()
        self.setRowCount(len(var_names))

        # 创建列表并排序: 先按validity降序, 然后按原顺序 (使用stable sort)
        items = list(zip(var_names, [units.get(v, '') for v in var_names], [validity.get(v, -1) for v in var_names]))
        # 为了保持相同validity的原顺序, 我们用enumerate添加index
        indexed_items = [(valid, idx, name, unit) for idx, (name, unit, valid) in enumerate(items)]  # idx asc for original order
        indexed_items.sort(key=lambda x: (-x[0], x[1]))  # valid desc (-valid), then original idx asc
        sorted_names = [name for valid, idx, name, unit in indexed_items]
        sorted_units = [unit for valid, idx, name, unit in indexed_items]
        sorted_valids = [valid for valid, idx, name, unit in indexed_items]

        for row, (name, unit, valid) in enumerate(zip(sorted_names, sorted_units, sorted_valids)):
            name_item = QTableWidgetItem(name)
            unit_item = QTableWidgetItem(unit)

            if valid == 1:
                brush = QBrush(QColor(0, 255, 0, 50))  # 半透明绿色
            elif valid == 0:
                brush = QBrush(QColor(255, 255, 0, 50))  # 半透明黄色
            elif valid == -1:
                brush = QBrush(QColor(255, 192, 203, 50))  # 半透明粉色
            else:
                brush = QBrush(Qt.GlobalColor.transparent)
            #print(f"channel: {name} validity is {valid}")
            name_item.setBackground(brush)
            unit_item.setBackground(brush)

            self.setItem(row, 0, name_item)
            self.setItem(row, 1, unit_item)


# 新增自定义 ViewBox 类
class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_x = None  # 记录右键点击时的 x 坐标
        self.plot_widget = None  # 将在 DraggableGraphicsLayoutWidget 中设置

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

        # 添加新 action: "Jump to Data" (检查是否已存在以避免重复)
        existing_texts = [act.text() for act in menu.actions()]
        if "Jump to Data" not in existing_texts:
            jump_act = QAction("Jump to Data", menu)
            jump_act.triggered.connect(self.trigger_jump_to_data)
            if menu.actions():
                menu.insertAction(menu.actions()[0], jump_act)
            else:
                menu.addAction(jump_act)

        return menu

    def trigger_jump_to_data(self):
        if self.plot_widget:
            self.plot_widget.jump_to_data_impl(self.context_x)

class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, units_dict, dataframe, time_channels_info={},synchronizer=None):
        super().__init__()
        self.factor = 1.0
        self.offset = 0.0
        self.original_index_x = None
        self.original_y = None
        self.mark_region = None
        self.setup_ui(units_dict, dataframe, time_channels_info, synchronizer)
        
    def setup_ui(self, units_dict, dataframe, time_channels_info={},synchronizer=None):
        """初始化UI组件和布局"""
        self.setAcceptDrops(True)
        self.units = units_dict
        self.data = dataframe
        self.time_channels_info = time_channels_info
        self.synchronizer = synchronizer
        self.curve = None
        #self.ci.layout.setContentsMargins(0, 0, 0, 5)
        
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
        layout = QGraphicsLinearLayout(Qt.Orientation.Horizontal)
        

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
        #self.label_left.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        proxy_left = QGraphicsProxyWidget()
        proxy_left.setWidget(self.label_left)

        # 右侧文本（使用代理窗口部件）
        self.label_right = QLabel("")
        self.label_right.setStyleSheet("""
            color: #000;
            background-color: transparent;
        """)
        self.label_right.setSizePolicy(QSizePolicy.Policy.Minimum,
                                       QSizePolicy.Policy.Preferred)
        #self.label_right.setAlignment(Qt.AlignmentFlag.AlignBottom| Qt.AlignmentFlag.AlignRight)
        proxy_right = QGraphicsProxyWidget()
        proxy_right.setWidget(self.label_right)
        
        # 添加文本到布局
        layout.addItem(proxy_left)
        layout.addItem(proxy_right)
        layout.setStretchFactor(proxy_left, 2)
        layout.setStretchFactor(proxy_right, 1)
        layout.setAlignment(proxy_left, Qt.AlignmentFlag.AlignBottom| Qt.AlignmentFlag.AlignLeft)
        layout.setAlignment(proxy_right, Qt.AlignmentFlag.AlignBottom| Qt.AlignmentFlag.AlignRight)

        #layout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        header.setLayout(layout)
        #header.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.addItem(header, row=0, col=0, colspan=2)

    def setup_plot_area(self):
        """配置绘图区域基本属性"""
        self.plot_item = self.addPlot(row=1, col=0, colspan=2, viewBox=CustomViewBox())
        self.view_box = self.plot_item.vb
        self.view_box.plot_widget = self  # 设置 plot_widget 以确保 trigger_jump_to_data 能调用 jump_to_data_impl
        
        # 移除 self._customize_plot_menu()，因为现在用 CustomViewBox 实现菜单定制
        
        self.view_box.setAutoVisible(x=False, y=True)  # 自动适应可视区域
        self.plot_item.setTitle(None)
        self.plot_item.hideButtons()
        self.plot_item.setClipToView(True)
        self.plot_item.setDownsampling(True)
        self.setBackground('w')

        pen = pg.mkPen('#f00',width=1)
        self.plot_item.getAxis('left').setGrid(255) 
        self.plot_item.getAxis('bottom').setGrid(255) 
        self.plot_item.showGrid(x=True, y=True, alpha=0.1)

# 基于点数修改曲线风格
        self.view_box.sigRangeChanged.connect(self.update_plot_style)
        
    def jump_to_data_impl(self, x):
        if not self.curve or not self.y_name:
            # 没有曲线，直接返回
            return

        main_window = self.window()
        if not hasattr(main_window, 'loader') or main_window.loader is None:
            return

        y_name = self.y_name

        # a. 打开/激活数值变量表，并添加系列（如果不存在）
        dlg = DataTableDialog.popup(y_name, main_window.loader.df[y_name], parent=main_window)

        # b. popup 已处理：如果已在（冻结或非冻结），不添加；否则添加到非冻结区域

        # c. 计算行索引（0-based）
        if self.factor == 0:
            return  # 避免除零

        index = (x - self.offset) / self.factor
        index = int(round(index)) - 1  # 转换为 0-based 行索引
        index = max(0, min(index, len(main_window.loader.df) - 1))  # 夹到有效范围

        # 获取模型和列索引
        model = dlg.model
        col_idx = dlg._df.columns.get_loc(y_name)  # 逻辑列索引

        # 确定使用哪个视图（冻结或主视图）
        if y_name in dlg.frozen_columns:
            view = dlg.frozen_view
        else:
            view = dlg.main_view

        # 获取视觉列索引（因为列可拖动）
        header = view.horizontalHeader()
        visual_col = header.visualIndex(col_idx)

        # 创建 QModelIndex
        qindex = model.index(index, col_idx)

        # 跳转并居中，使用 QTimer 确保在窗口显示后执行
        QTimer.singleShot(0, lambda: view.scrollTo(qindex, QAbstractItemView.ScrollHint.PositionAtCenter))

        # 选中该单元格
        QTimer.singleShot(0, lambda: view.selectionModel().select(qindex, QItemSelectionModel.SelectionFlag.ClearAndSelect))

    def auto_range(self):
        datalength = self.window().loader.datalength if hasattr(self.window(), 'loader') else 1
        index_x = np.arange(1, datalength + 1)

        special_limits = self.handle_single_point_limits(index_x * self.factor + self.offset, np.array([0] * datalength))  # 虚拟y
        if special_limits:
            min_x, max_x, _, _ = special_limits
        else:
            min_x = self.offset + self.factor * index_x.min()
            max_x = self.offset + self.factor * index_x.max()
        
        self.view_box.setXRange(min_x, max_x, padding=DEFAULT_PADDING_VAL)
        self.view_box.autoRange()

    def auto_y_in_x_range(self):
        vb=self.view_box
        vb.enableAutoRange(axis=vb.YAxis, enable=True)

    def update_left_header(self, left_text=None):
        """更新顶部文本内容"""
        if left_text is not None:
            self.label_left.setText(left_text)

    def update_right_header(self, right_text=None):
        """更新顶部文本内容"""
        if right_text is not None:
            self.label_right.setText(right_text)
            self.label_right.setAlignment(Qt.AlignmentFlag.AlignRight)


    def reset_plot(self,index_xMin,index_xMax):

        self.plot_item.setLimits(xMin=None, xMax=None)  # 解除X轴限制
        self.plot_item.setLimits(yMin=None, yMax=None)  # 解除Y轴限制
        
        xMin = self.offset + self.factor * index_xMin
        xMax = self.offset + self.factor * index_xMax
        
        if not (np.isnan(xMax) or np.isinf(xMax)):
            if xMax ==xMin:
                xMax = xMax + 0.5 * self.factor  # 基于factor扩展，避免零范围
                xMin = xMin - 0.5 * self.factor

            self.view_box.setXRange(xMin, xMax, padding=DEFAULT_PADDING_VAL)
            padding_xVal=0.1
            limits_xMin = xMin - padding_xVal * (xMax - xMin)
            limits_xMax = xMax + padding_xVal * (xMax - xMin)
            self.plot_item.setLimits(xMin=limits_xMin, xMax=limits_xMax)

        self.view_box.setYRange(0,1,padding=0) 
        self.vline.setBounds([None, None]) 

        self.xMin = xMin
        self.xMax = xMax
        # self.x_name = ''
        # self.x_format = ''
        self.y_name = ''
        self.y_format = ''
        #self.plot_item.update()
        self.plot_item.clearPlots() 
        self.axis_y.setLabel(text="")
        self.update_left_header("channel name")
        self.update_right_header("")



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
        
        # Y轴标签
        self.axis_y.setLabel(
            #text="YYYYYYYYYYY",
            color='black',
            angle=-90,
            **{'font-family': 'Arial', 'font-size': '12pt', 'font-weight': 'bold'}
        )
    
    def setup_interaction(self):
        """配置交互元素"""
        # 光标线
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 0, 0, 100), width=4) )
        self.vline.setZValue(100) 
        self.cursor_label = pg.TextItem("", anchor=(1, 1), color="red")
        self.plot_item.addItem(self.vline, ignoreBounds=True)
        self.plot_item.addItem(self.cursor_label, ignoreBounds=True)
        self.vline.setVisible(False)
        self.cursor_label.setVisible(False)
        
        # 信号连接
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        #self.vline.sigPositionChanged.connect(self.update_cursor_label)
        self.setAntialiasing(False)

    def handle_single_point_limits(self, x_values, y_values):
        if len(x_values) == 1:
            x = x_values[0]
            min_x = x - 0.5 * self.factor  # 基于factor扩展，避免零范围
            max_x = x + 0.5 * self.factor
            if len(y_values) == 1:
                y = y_values[0]
                min_y = y - 0.5 if y != 0 else -0.5
                max_y = y + 0.5 if y != 0 else 0.5
            else:
                min_y = np.nanmin(y_values)
                max_y = np.nanmax(y_values)
            return min_x, max_x, min_y, max_y
        else:
            return None  # 返回None表示不特殊处理
        
    def wheelEvent(self, ev):
        vb = self.plot_item.getViewBox()
        delta = ev.angleDelta().y()
        # 只在没有按下任何修饰键（Ctrl/Shift/Alt…）时才执行缩放
        if ev.modifiers() == Qt.KeyboardModifier.NoModifier:
            if delta != 0:
                # 获取鼠标位置
                mouse_pos = ev.position().toPoint()
                scene_pos = self.mapToScene(mouse_pos)
                view_pos = vb.mapSceneToView(scene_pos)
                mouse_x = view_pos.x()
                mouse_y = view_pos.y()

                factor = 0.8 if delta > 0 else 1.2
                vb.scaleBy((factor, 1), center=(mouse_x, mouse_y))
                ev.accept()  # 确保事件被处理
            else:
                super().wheelEvent(ev)
        else:
            # 有按键按下，交给父类默认处理（或自己写别的逻辑）
            super().wheelEvent(ev)
        
        #ev.accept()  # 确保事件被处理
    
    def mouse_moved(self, evt):
        """鼠标移动事件处理"""
        pos = evt[0]
        if not self.plot_item.sceneBoundingRect().contains(pos):
            return
        mousePoint = self.plot_item.vb.mapSceneToView(pos)
        if hasattr(self.window(), 'sync_crosshair'):
            self.window().sync_crosshair(mousePoint.x(), self)
            #print(f"mouse in pos {mousePoint.x()}")

    def sInt_to_fmtStr(self,value:int):
        td = pd.to_timedelta(pd.Series(value, dtype='float64'), unit='s')
        total = td.dt.total_seconds() % (24*3600)           # Series of float (秒)
        hh = (total // 3600).astype(int)
        mm = (total % 3600 // 60).astype(int)
        ss = total % 60
        return (hh.apply(lambda x: f"{x:02d}") + ':' +
                mm.apply(lambda x: f"{x:02d}") + ':' +
                ss.apply(lambda x: f"{x:05.2f}")).tolist()
    
    def dateInt_to_fmtStr(self,value:int):
        correct_dates = pd.to_datetime(pd.Series(value), unit='s').dt.strftime('%Y/%m/%d')
        return correct_dates.tolist()
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
        """更新光标标签位置和内容"""
        if len(self.plot_item.listDataItems()) == 0:
            #self.cursor_label.setText("")
            self.update_right_header("")
            return
        
        try:
            x = self.vline.value()           
            curve = self.plot_item.listDataItems()[0]
            x_data, y_data = curve.getData()
            if x_data is None or len(x_data) == 0:
                #self.cursor_label.setText("")
                self.update_right_header("")
                return
            x = np.clip(x, x_data.min(), x_data.max())
            idx = np.argmin(np.abs(x_data - x))
            y_val = y_data[idx]
            #(x_min, x_max), (y_min, y_max) = self.view_box.viewRange()
            x_str = self._significant_decimal_format_str(value=float(x),ref=self.factor)
            if self.y_format == 's':
                time_str=self.sInt_to_fmtStr(y_val)
                #self.update_right_header(f"x={x:.0f}, y={time_str}")
                self.update_right_header(f"x={x_str}, y={time_str}")
            elif self.y_format == 'date':
                date_str=self.dateInt_to_fmtStr(y_val)
                #self.update_right_header(f"x={x:.0f}, y={date_str}")
                self.update_right_header(f"x={x_str}, y={date_str}")
            else:
                #self.update_right_header(f"x={x:.0f}, y={y_val:.2f}")

                self.update_right_header(f"x={x_str}, y={y_val:.5g}")

        except Exception as e:
            print(f"Cursor update error: {e}")
            # self.cursor_label.setText("")
            self.update_right_header("")

    def toggle_cursor(self, show: bool):
        """切换光标显示状态"""
        self.vline.setVisible(show)
        self.cursor_label.setVisible(show)
        if show:
            self.update_cursor_label()
        else:
            self.update_right_header("")
            
    def clear_value_cache(self):
        #self._value_cache: dict[str, tuple] = {}
        pass
    def datetime_to_unix_seconds(self,series: pd.Series) -> pd.Series:
        if "ns" in str(series.dtype):
            return series.astype("int64") / 10**9
        elif "us" in str(series.dtype):
            return series.astype("int64") / 10**6
        elif "ms" in str(series.dtype):
            return series.astype("int64") / 10**3
        else:
            raise ValueError(f"Unsupported datetime dtype: {series.dtype}")
        
    def get_value_from_name(self,var_name)-> tuple | None:
        main_window = self.window()
        if var_name in main_window.value_cache:
            return main_window.value_cache[var_name]

        raw_values = self.data[var_name]
        dtype_kind = raw_values.dtype.kind
        if dtype_kind in "iuf":
            y_values = raw_values
            y_format = 'number'
            return y_values, y_format
        elif var_name in self.time_channels_info:
            fmt = self.time_channels_info[var_name]
            try:
                if "%H:%M:%S" in fmt:
                    #time
                    times = pd.to_datetime(raw_values, format=fmt, errors="coerce")
                    today = pd.Timestamp.today().normalize()
                    dt_values = today + (times.dt.hour.astype("timedelta64[h]") +
                        times.dt.minute.astype("timedelta64[m]") +
                        times.dt.second.astype("timedelta64[s]"))
                    
                    y_values = self.datetime_to_unix_seconds(dt_values)
                    y_format = 's'

                    # reverse method：
                    # y_sec = dt_values.astype("int64") // 10**6  
                    # kk = pd.to_datetime(y_sec, unit="s")
                else:
                    #date
                    dt_values = pd.to_datetime(raw_values,format=fmt, errors='coerce')
                    y_values = self.datetime_to_unix_seconds(dt_values)
                    y_format = 'date'
            except:
                # cannot parse the format
                return None,None

        else:
            # cannot get right info
            return None,None
        
        # finally
        main_window.value_cache[var_name] = (y_values, y_format)
        return y_values, y_format
    
    def update_time_correction(self, new_factor, new_offset):
        old_factor = self.factor
        old_offset = self.offset
        self.factor = new_factor
        self.offset = new_offset

        if self.original_index_x is not None:
            new_x = self.offset + self.factor * self.original_index_x
            self.curve.setData(new_x, self.original_y)

        datalength = len(self.original_index_x) if self.original_index_x is not None else (
            self.window().loader.datalength if hasattr(self.window(), 'loader') else 0)
        padding_xVal = 0.1
        index_min = 1 - padding_xVal * datalength
        index_max = datalength + padding_xVal * datalength
        limits_xMin = self.offset + self.factor * index_min
        limits_xMax = self.offset + self.factor * index_max
        self.plot_item.setLimits(xMin=limits_xMin, xMax=limits_xMax, minXRange=self.factor * 5)

        data_min_x = self.offset + self.factor * 1 if datalength > 0 else 0
        data_max_x = self.offset + self.factor * datalength if datalength > 0 else 1
        self.vline.setBounds([data_min_x, data_max_x])

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
        
        # 确保曲线数据更新后立即更新标记统计
        QTimer.singleShot(0, self.window().update_mark_stats)


# ---------------- 拖拽相关 ----------------
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
        var_name = event.mimeData().text()
        self.plot_variable(var_name)
        event.acceptProposedAction()
        self.window().update_mark_stats()

    def plot_variable(self, var_name: str) -> bool:
        """绘制变量到图表，返回是否成功"""
        if var_name not in self.data.columns:
            QMessageBox.warning(self, "错误", f"变量 {var_name} 不存在")
            return False
        
        y_values, y_format = self.get_value_from_name(var_name=var_name)
        
        if y_values is None:
            QMessageBox.warning(self, "错误", f"变量 {var_name} 没有有效数据")
            return False
        
        # 如果正常
        self.y_format = y_format
        self.y_name = var_name
        self.original_index_x = np.arange(1, len(y_values) + 1)
        self.original_y = y_values.to_numpy() if isinstance(y_values, pd.Series) else np.array(y_values)
        x_values = self.offset + self.factor * self.original_index_x

        self.plot_item.clearPlots()             
        _pen = pg.mkPen(color='blue', width=3)  # 初始粗线
        self.curve = self.plot_item.plot(x_values, self.original_y, pen=_pen, name=var_name)
        # 无初始symbol，等待update_plot_style
        QTimer.singleShot(0, lambda: self.update_plot_style(self.view_box, self.view_box.viewRange(), None))  # 修正: 传整个viewRange() 和 None for rect

        full_title = f"{var_name} ({self.units.get(var_name, '')})".strip()
        self.update_left_header(full_title)
        padding_xVal = 0.1
        padding_yVal = 0.5

        special_limits = self.handle_single_point_limits(x_values, self.original_y)
        if special_limits:
            min_x, max_x, min_y, max_y = special_limits
            # extra margin for x-axis limit : 50% * factor
            # limits_xMin = min_x - 0.5 * self.factor
            # limits_xMax = max_x + 0.5 * self.factor

        else:
            min_x = np.min(x_values)
            max_x = np.max(x_values)
            min_y = np.nanmin(self.original_y)
            max_y = np.nanmax(self.original_y)
        limits_xMin = min_x - padding_xVal * (max_x - min_x)
        limits_xMax = max_x + padding_xVal * (max_x - min_x)

        


        
        if np.isnan(min_y) or np.isnan(max_y) or min_y == max_y:
            y_center = min_y if not np.isnan(min_y) else 0
            y_range = 1.0 if y_center == 0 else abs(y_center) * 0.2
            self.plot_item.setLimits(yMin=y_center - y_range, yMax=y_center + y_range)
            self.view_box.setYRange(y_center - y_range, y_center + y_range, padding=0.05)
        else:
            self.plot_item.setLimits(
                yMin=min_y - padding_yVal * (max_y - min_y),
                yMax=max_y + padding_yVal * (max_y - min_y))
            self.view_box.setYRange(min_y, max_y, padding=0.05)
        
        self.plot_item.setLimits(xMin=limits_xMin, xMax=limits_xMax, minXRange=min(3,len(x_values))*self.factor)
        self.vline.setBounds([min_x, max_x])
        
        self.plot_item.update()
        if hasattr(self.window(), 'cursor_btn'):
            self.vline.setBounds([min_x, max_x])
            self.toggle_cursor(self.window().cursor_btn.isChecked())
        else:
            self.toggle_cursor(False)

        return True

    def clear_plot_item(self):
        #self.plot_item.setLimits(xMin=None, xMax=None)  # 解除X轴限制
        self.plot_item.setLimits(yMin=None, yMax=None)  # 解除Y轴限制

        self.view_box.setYRange(0,1,padding=DEFAULT_PADDING_VAL) 
        self.vline.setBounds([None, None]) 

        #self.plot_item.update()
        self.plot_item.clearPlots() 
        self.axis_y.setLabel(text="")
        self.y_name = ''
        self.y_format=''
        self.update_left_header("channel name")
        self.update_right_header("")
        self.curve = None
        self.original_index_x = None
        self.original_y = None

    # ---------------- 双击轴弹出对话框 ----------------
    def mouseDoubleClickEvent(self, event):
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            super().mouseDoubleClickEvent(event)
            return
        
        if event.button() == Qt.MouseButton.MiddleButton:
            self.clear_plot_item()
            self.window().update_mark_stats()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            y_axis_rect_scene = self.axis_y.mapToScene(self.axis_y.boundingRect()).boundingRect()
            x_axis_rect_scene = self.axis_x.mapToScene(self.axis_x.boundingRect()).boundingRect()

            if y_axis_rect_scene.contains(scene_pos):
                dialog = AxisDialog(self.axis_y, self.view_box, "Y", self)
                if dialog.exec():
                    self.plot_item.update()
                return 
            elif x_axis_rect_scene.contains(scene_pos):
                dialog = AxisDialog(self.axis_x, self.view_box, "X", self)
                if dialog.exec():
                    min_val, max_val = self.view_box.viewRange()[0]
                    for view in self.window().findChildren(DraggableGraphicsLayoutWidget):
                        #view.view_box.setXRange(min_val, max_val, padding=0.00)
                        #view.plot_item.setXRange(min_val, max_val, padding=0.00)
                        self.set_xrange_with_link_handling(xmin=min_val,xmax=max_val,padding=DEFAULT_PADDING_VAL)
                        view.plot_item.update()
                return
        return super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
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
        if self.rubberBand.isVisible() and event.button() == Qt.MouseButton.LeftButton:
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
        if self.mark_region:
            self.plot_item.removeItem(self.mark_region)
            self.mark_region = None

    def update_mark_region(self):
        if self.mark_region:
            old_min, old_max = self.mark_region.getRegion()
            # 更新基于新factor/offset，但由于x是scaled的，不需要额外缩放
            self.mark_region.setRegion([old_min, old_max])  # 实际不需要变，因为x已scale

    def get_mark_stats(self):
        if not self.curve or not self.mark_region:
            return None
        min_x, max_x = self.mark_region.getRegion()
        x_data, y_data = self.curve.getData()
        if x_data is None or len(x_data) == 0:
            return None
        idx_left = np.argmin(np.abs(x_data - min_x))
        idx_right = np.argmin(np.abs(x_data - max_x))
        x1 = x_data[idx_left]
        y1 = y_data[idx_left]
        x2 = x_data[idx_right]
        y2 = y_data[idx_right]
        dx = x2 - x1
        dy = y2 - y1
        slope = float('inf') if dx == 0 else dy / dx  # Handle zero division
        
        # 计算区域内 y 的统计
        mask = (x_data >= min_x) & (x_data <= max_x)
        y_region = y_data[mask]
        if len(y_region) == 0:
            y_avg = y_max = y_min = np.nan
        else:
            y_avg = np.nanmean(y_region)
            y_max = np.nanmax(y_region)
            y_min = np.nanmin(y_region)
        
        return (x1, x2, y1, y2, dx, dy, slope, self.label_left.text(), y_avg, y_max, y_min)
    def update_plot_style(self, view_box, range, rect=None):
        if not self.curve:
            return
        x_data, _ = self.curve.getData()
        if x_data is None:
            return
        
        x_min, x_max = range[0]
        # 计算可见点数（近似：二分查找或mask计数）
        visible_mask = (x_data >= x_min) & (x_data <= x_max)
        visible_points = np.sum(visible_mask)
        
        threshold = 100  # 可调阈值
        tolerance = 0.5
        if visible_points > threshold * (1 + tolerance):
            # 粗线，无symbol
            pen = pg.mkPen(color='blue', width=3)
            self.curve.setPen(pen)
            self.curve.setSymbol(None)  # 移除symbol

        elif visible_points < threshold * (1 - tolerance):
            # 细线 + symbol
            pen = pg.mkPen(color='blue', width=1)
            self.curve.setPen(pen)
            self.curve.setSymbol('s')  # 或's'如原有
            self.curve.setSymbolSize(3)
            self.curve.setSymbolPen('blue')
            self.curve.setSymbolBrush('blue')
        else:
            pass

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
        flags |= Qt.WindowType.CustomizeWindowHint
        flags &= ~Qt.WindowType.WindowCloseButtonHint
        self.setWindowFlags(flags)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Plot", "x1", "x2", "y1", "y2", "dx", "dy", "slope", "y_avg", "y_max", "y_min"])
        self.tree.setColumnWidth(0,200)
        #self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
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
                    item = QTreeWidgetItem(self.tree, [
                    f"Plot {idx+1} -> {stats[7]}",
                    f"{stats[0]:.2f}", f"{stats[1]:.2f}",
                    f"{stats[2]:.2f}", f"{stats[3]:.2f}",
                    f"{stats[4]:.2f}", f"{stats[5]:.2f}",
                    f"{stats[6]:.2f}" if not np.isinf(stats[6]) else "inf",
                    f"{stats[8]:.2f}", f"{stats[9]:.2f}", f"{stats[10]:.2f}"
                ])            
            else:
                has_no_curve = True
                sub_item = QTreeWidgetItem(self.no_curve_item, [f"Plot {idx+1}", "", "", "", "", "", "", "", "", "", ""])
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.defaultTitle = "数据快速查看器(PyQt6), Alpha版本"

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
        self.resize(1600, 900)
        self._factor_default  = 1
        self._offset_default = 0
        self.factor = self._factor_default
        self.offset = self._offset_default

        self.loaded_path = ''
        self.var_names = None
        self.units = None
        self.time_channels_infos = None
        self.data = None
        self.data_validity = None

        # 窗口几何信息
        self.data_table_geometry = None
        self.mark_stats_geometry = None
        self.time_correction_geometry = None

        # value cache
        self.value_cache = {}

        # ---------------- 中央控件 ----------------
        central = QWidget()
        self.setCentralWidget(central)

        # 总水平布局：左侧变量列表 + 右侧绘图区
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for precise alignment

        # ---------------- 左侧变量列表 ----------------
        left_widget = QWidget()
        left_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # 左侧大标题
        left_layout_title=QLabel("变量列表")
        font = left_layout_title.font()
        font.setBold(True)
        left_layout_title.setFont(font)
        #left_layout_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(left_layout_title)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("输入变量名关键词（空格分隔）")
        self.filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.filter_input)

        self.unit_filter_input = QLineEdit()
        self.unit_filter_input.setPlaceholderText("输入单位关键词（空格分隔）")
        self.unit_filter_input.setContentsMargins(60,0,0,0)

        self.unit_filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.unit_filter_input)

        # self.load_btn = QPushButton("导入数据文件")
        # self.load_btn.clicked.connect(self.load_btn_click)
        # left_layout.addWidget(self.load_btn)
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
        self.left_widget=left_widget
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

        # self.reload_btn = QPushButton("重新加载数据")
        # self.reload_btn.clicked.connect(self.reload_data)
        # top_bar.addWidget(self.reload_btn)

        self.clear_all_plots_btn = QPushButton("清除绘图")
        self.clear_all_plots_btn.clicked.connect(self.clear_all_plots)
        top_bar.addWidget(self.clear_all_plots_btn)

        self.help_btn = QPushButton("帮助")
        self.help_btn.clicked.connect(self.show_help)
        top_bar.addWidget(self.help_btn)

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
        self.plot_layout=QGridLayout()
        self.plot_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        self.plot_layout.setSpacing(0)  # No spacing
        root_layout.addLayout(self.plot_layout, 1)    # 1 表示可伸缩
        main_layout.addWidget(self.plot_widget, 4)

        # ---------------- 子图 ----------------
        self.plot_widgets = []

        # put default plots into the window
        self._plot_row_max_default = 4
        self._plot_col_max_default = 3
        # 延迟创建：self.create_subplots_matrix(self._plot_row_max_default,self._plot_col_max_default)

        # turn on/off the plots
        self._plot_row_current = 4
        self._plot_col_current = 1
        # 延迟设置：self.set_plots_visible(row_set=self._plot_row_current,col_set=self._plot_col_current)

        self.placeholder_label = QLabel("请导入 CSV 文件以查看数据", self.plot_widget)
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 24px; color: gray;")
        self.plot_layout.addWidget(self.placeholder_label, 0, 0)

        self.drop_overlay = DropOverlay(self.centralWidget())
        self.drop_overlay.lower()          # 初始在最下层
        self.drop_overlay.hide()

        # 全局拖拽过滤器
        QApplication.instance().installEventFilter(self)
   

        # ---------------- 命令行直接加载文件 ----------------
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            self.load_csv_file(file_path)

        # 标记区域相关
        self.saved_mark_range = None
        self.mark_stats_window = None
        # self._settings = QSettings("MyCompany", "MarkStatsWindow")
        #self.resize(900, 300)
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

        #print(f"actual window width = {self.width()}")
            
    def show_help(self):
        dlg = HelpDialog(self)
        dlg.exec()
    def load_btn_click(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV File (*.csv);;m File (*.mfile);;t00 File (*.t00);;all File (*.*)")
        if file_path:
            self.load_csv_file(file_path)

    def load_csv_file(self, file_path: str):
        if not file_path or not os.path.isfile(file_path):
            return
        self.raise_()  # Bring window to front
        self.activateWindow()  # Set focus
        try:
            self._load_file(file_path)
        except Exception as e:
            QMessageBox.critical(self, "加载错误", str(e))
        finally:
            if hasattr(self, 'loader') and self.loader:  # 如果加载成功
                self._post_load_actions(file_path)

    def set_button_status(self,status:bool):
        if status is not None:
            self.time_correction_btn.setEnabled(status)
            #self.reload_btn.setEnabled(status)
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

        self._load_file(self.loader.path, is_reload=True)

    def _load_file(self, file_path: str, is_reload: bool = False):
        file_ext = os.path.splitext(file_path)[1].lower()

        # load default
        delimiter_typ = ','
        descRows = 0
        hasunit = True

        # try to load config json files
        if os.path.isfile("config_dict.json"):
            try:
                config_dict=self.load_dict("config_dict.json")
                ext_dict=config_dict.get(file_ext[1:],{})
                delimiter_typ=ext_dict.get('sep')
                descRows = int(ext_dict.get('skiprows'))
                hasunit = bool(ext_dict.get('hasunit'))

            except Exception as e:     
                print(f"配置文件读取失败: {e}")
        else:
            if file_ext in ['.csv',]:
                delimiter_typ = ','
                descRows = 0
                hasunit = True
            elif file_ext in ['.mfile','.t00','.t01','t10','t11']:
                delimiter_typ = '\t'
                descRows = 2
                hasunit=True       
            elif file_ext in ['.txt',]:
                delimiter_typ = '\t'
                descRows = 0
                hasunit=True         
            else:
                QMessageBox.critical(self, "读取失败",f"无法读取后缀为:'{file_ext}'的文件")
                return

        _Threshold_Size_Mb=FILE_SIZE_LIMIT_BACKGROUND_LOADING 

        # < 5 MB 直接读
        file_size =os.path.getsize(file_path)
        if file_size < _Threshold_Size_Mb * 1024 * 1024:
            status = self._load_sync(file_path, descRows=descRows,sep=delimiter_typ,hasunit=hasunit)
            if status:
                self.set_button_status(True)
                self._post_load_actions(file_path)
        else:
            # 5 MB 以上走线程
            self._progress = QProgressDialog("正在读取数据...", "取消", 0, 100, self)
            self._progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            self._progress.setAutoClose(True)
            self._progress.setCancelButton(None)            # 不可取消
            self._progress.show()

            self._thread = DataLoadThread(file_path, descRows=descRows,sep=delimiter_typ,hasunit=hasunit)
            self._thread.progress.connect(self._progress.setValue)
            self._thread.finished.connect(lambda loader: self._on_load_done(loader, file_path))
            self._thread.error.connect(self._on_load_error)
            self._thread.start()

    def _post_load_actions(self, file_path: str):
        self.loaded_path = file_path
        self.setWindowTitle(f"{self.defaultTitle} ---- 数据文件: [{file_path}]")
        self.set_button_status(True)
        # if DataTableDialog._instance is not None:
        #     DataTableDialog._instance.close()   # 触发 closeEvent → 清空
        #     DataTableDialog._instance = None

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
        
    def _load_sync(self, 
                   file_path:str,
                   descRows:int =0,
                   sep:str = ',',
                   hasunit:bool=True):
        """小文件直接读"""
        loader = None
        status = False
        try:
            loader = FastDataLoader(file_path, descRows=descRows,sep=sep,hasunit=hasunit)
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

    def _on_load_done(self,loader, file_path: str):
        self._progress.close()
        self.loader=loader
        self._apply_loader()
        self._post_load_actions(file_path)

    def _on_load_error(self, msg):
        self._progress.close()
        QMessageBox.critical(self, "读取失败", msg)

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
        # self.value_cache = {}

        # self.factor = 1.0
        # self.offset = 0.0
        # for container in self.plot_widgets:
        #     widget = container.plot_widget
        #     widget.factor = 1.0
        #     widget.offset = 0.0

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
                DataTableDialog._instance.close()


        self.filter_variables() 
        if self.mark_region_btn.isChecked():
            self.update_mark_stats()

    def filter_variables(self):
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
        if dlg.exec() == QDialog.DialogCode.Accepted:
            r, c = dlg.values()
            self.set_plots_visible (r, c)
            self.update_mark_regions_on_layout_change()

    def open_time_correction_dialog(self):
        dialog = TimeCorrectionDialog(self.factor, self.offset, self)
        if dialog.window_geometry:
            dialog.restoreGeometry(dialog.window_geometry)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_factor, new_offset = dialog.values()
            if new_factor <= 0:
                QMessageBox.warning(self, "错误", "Factor 必须是正数")
                return
            old_factor = self.factor
            old_offset = self.offset
            self.factor = new_factor
            self.offset = new_offset

            # 获取当前视图范围（假设所有视图联动，使用第一个）
            if self.plot_widgets:
                curr_min, curr_max = self.plot_widgets[0].plot_widget.view_box.viewRange()[0]
            else:
                curr_min, curr_max = 0, 1

            # 更新所有图表的数据和限制，但不设置范围
            for container in self.plot_widgets:
                container.plot_widget.update_time_correction(new_factor, new_offset)

            # 计算新范围
            if old_factor != 0:
                index_min = (curr_min - old_offset) / old_factor
                index_max = (curr_max - old_offset) / old_factor
                new_min = new_offset + new_factor * index_min
                new_max = new_offset + new_factor * index_max
            else:
                # fallback
                datalength = self.loader.datalength if hasattr(self, 'loader') else 1
                new_min = new_offset + new_factor * 1
                new_max = new_offset + new_factor * datalength

            # 只设置第一个图表的 X 轴范围，其他图表通过 XLink 同步
            if self.plot_widgets:
                first_plot = self.plot_widgets[0].plot_widget
                first_plot.view_box.enableAutoRange(x=False)  # 禁用自动范围调整
                first_plot.view_box.setXRange(new_min, new_max, padding=0)  # 明确设置 padding=0

            # 更新标记统计
            self.update_mark_stats()

    def update_mark_regions_on_layout_change(self):
        if self.mark_region_btn.isChecked():
            # 移除旧的
            if self.plot_widgets[0] and self.plot_widgets[0].plot_widget.mark_region:
                self.saved_mark_range = self.plot_widgets[0].plot_widget.mark_region.getRegion()

            for container in self.plot_widgets:
                container.plot_widget.remove_mark_region()
            # 添加新的到可见plot
            view_min, view_max = self.plot_widgets[0].plot_widget.view_box.viewRange()[0]
            min_x, max_x = self.saved_mark_range if self.saved_mark_range else (view_min + (view_max - view_min) / 3, view_min + 2 * (view_max - view_min) / 3)
            for container in self.plot_widgets:
                if container.isVisible():
                    container.plot_widget.add_mark_region(min_x, max_x)
            self.update_mark_stats()

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QEvent.Type.DragEnter:
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                self.show_drop_overlay()
                if any(u.toLocalFile().lower().endswith(('.csv','.txt','.mfile','.t00','t01'))
                       for u in urls):                    
                    self.drop_overlay.adjust_text(file_type_supported=True)
                    event.acceptProposedAction()
                else:
                    self.drop_overlay.adjust_text(file_type_supported=False)
                    event.acceptProposedAction()
                    return True
        elif etype == QEvent.Type.DragLeave:
            self.hide_drop_overlay()
        elif etype == QEvent.Type.Drop:
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                for u in urls:
                    path = u.toLocalFile()
                    if path.lower().endswith(('.csv','.txt','.mfile','.t00','t01')):
                        self.hide_drop_overlay()
                        self.load_csv_file(path)
                        event.accept()
                        return True
        return super().eventFilter(obj, event)

    def show_drop_overlay(self):
        self.drop_overlay.setGeometry(self.centralWidget().rect())
        self.drop_overlay.raise_()
        self.drop_overlay.show()
        self.drop_overlay.activateWindow()

    def hide_drop_overlay(self):
        self.drop_overlay.hide()


    def reset_plots_after_loading(self,index_xMin,index_xMax):
        for container in self.plot_widgets:
             container.plot_widget.reset_plot(index_xMin, index_xMax)
             container.plot_widget.clear_value_cache()
        # for plot_widget in self.plot_widgets:
        #     plot_widget.reset_plot(xMin,xMax)
        self.saved_mark_range = None
        if self.mark_stats_window:
            self.mark_stats_window.hide()  # Hide instead of close
            self.mark_stats_window.tree.clear()  # Clear stats to prevent duplication
        if self.mark_region_btn.isChecked():
            self.mark_region_btn.setChecked(False)
            self.toggle_mark_region(False)

    # ---------------- 公用函数 ----------------
    def toggle_cursor_all(self, checked):
        for container in self.plot_widgets:
            widget=container.plot_widget
            widget.toggle_cursor(checked)
        self.cursor_btn.setText("隐藏光标" if checked else "显示光标")
        
    def sync_crosshair(self, x, sender_widget):
        if not self.cursor_btn.isChecked():
            return
        for container in self.plot_widgets:
            w = container.plot_widget
            w.vline.setVisible(True)
            w.vline.setPos(x)
            w.update_cursor_label()

    def clear_all_plots(self):
        for container in self.plot_widgets:
            widget=container.plot_widget
            widget.clear_plot_item()
        self.saved_mark_range = None
        self.update_mark_stats()

    def auto_range_all_plots(self):
        for container in self.plot_widgets:
            widget=container.plot_widget
            widget.auto_range()
            
    def auto_y_in_x_range(self):
        for container in self.plot_widgets:
            widget=container.plot_widget
            widget.auto_y_in_x_range()

    def create_subplots_matrix(self, m: int, n: int):
        # 先全部清掉
        for i in reversed(range(self.plot_layout.count())):
            w = self.plot_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.plot_widgets.clear()

        first_viewbox = None   # 用于 XLink

        for r in range(m):
            for c in range(n):
                plot_widget = DraggableGraphicsLayoutWidget(self.units, self.data, self.time_channels_infos)
                plot_widget.toggle_cursor(self.cursor_btn.isChecked())

                # XLink：让同一行的所有列都 link 到第一列
                if c == 0 and r == 0:
                    first_viewbox = plot_widget.view_box
                else:
                    plot_widget.view_box.setXLink(first_viewbox)

                # 用一个 QWidget 包一层，方便隐藏
                wrapper = QVBoxLayout()
                wrapper.setContentsMargins(QMargins(0, 0, 5, 5))
                wrapper.addWidget(plot_widget)

                container = QWidget()
                container.setLayout(wrapper)
                container.plot_widget = plot_widget   # 保留引用，方便后面找
                #container.setVisible(True)            # 默认全部显示

                self.plot_layout.addWidget(container, r, c)
                self.plot_widgets.append(container)   # 保存容器

        # 设置行列权重=1，确保均分
        for r in range(m):
            self.plot_layout.setRowStretch(r, 1)
        for c in range(n):
            self.plot_layout.setColumnStretch(c, 1)
        if self.mark_region_btn.isChecked():
            self.toggle_mark_region(True)

    def set_plots_visible(self, row_set: int = 1, col_set: int = 1):

        m, n = self._plot_row_max_default, self._plot_col_max_default

        for idx, container in enumerate(self.plot_widgets):
            r, c = divmod(idx, n)
            visible = r < row_set and c < col_set
            container.setVisible(visible)

            # 同时改 stretch，避免隐藏后仍占空间
            self.plot_layout.setRowStretch(r, 1 if visible else 0)
            self.plot_layout.setColumnStretch(c, 1 if visible else 0)

        # 把多余的行列 stretch 统一设为 0
        for r in range(row_set, m):
            self.plot_layout.setRowStretch(r, 0)
        for c in range(col_set, n):
            self.plot_layout.setColumnStretch(c, 0)

        self._plot_row_current = row_set
        self._plot_col_current = col_set
        self.update_mark_regions_on_layout_change()

    def replots_after_loading(self):
        # 收集所有 y_name (包括未显示的)
        all_y_names = [container.plot_widget.y_name for container in self.plot_widgets if container.plot_widget.y_name]
        if DataTableDialog._instance is not None:
            all_y_names.extend(DataTableDialog._instance._df.columns.tolist())

        unique_y_names = set(all_y_names)
        if not unique_y_names:
            self.reset_plots_after_loading(1, self.loader.datalength)
            return

        # 找到在新数据中存在的有效 y_name
        found = [y for y in unique_y_names if y in self.loader.var_names and self.loader.df_validity.get(y, -1) == 1]
        ratio = len(found) / len(unique_y_names)

        if ratio <= RATIO_RESET_PLOTS or len(found) < 1:
            self.reset_plots_after_loading(1, self.loader.datalength)
        else:
            self.value_cache = {}
            cleared = []
            for idx, container in enumerate(self.plot_widgets):
                widget = container.plot_widget
                y_name = widget.y_name
                # 更新 limits
                original_index_x = np.arange(1, self.loader.datalength + 1)
                min_x = widget.offset + widget.factor * np.min(original_index_x)
                max_x = widget.offset + widget.factor * np.max(original_index_x)
                limits_xMin = min_x - 0.1 * (max_x - min_x)
                limits_xMax = max_x + 0.1 * (max_x - min_x)
                widget.plot_item.setLimits(xMin=limits_xMin, xMax=limits_xMax, minXRange=widget.factor * 5)
                widget.vline.setBounds([min_x, max_x])
                if not y_name:
                    continue
                if y_name in self.loader.df.columns and self.loader.df_validity.get(y_name, -1) >=0 :
                    success = widget.plot_variable(y_name)
                    if not success:
                        widget.clear_plot_item()
                        cleared.append((idx + 1, "无效数据"))
                else:
                    widget.clear_plot_item()
                    reason = f"未找到变量:{y_name}" if y_name not in self.loader.df.columns else f"无效数据:{y_name}"
                    cleared.append((idx + 1, reason))

            # 恢复 xRange
            if self.plot_widgets:
                first_plot = self.plot_widgets[0].plot_widget
                curr_min, curr_max = first_plot.view_box.viewRange()[0]
                first_plot.view_box.setXRange(curr_min, curr_max, padding=DEFAULT_PADDING_VAL) # 明确设置 padding=0
                # for container in self.plot_widgets:
                #     widget = container.plot_widget
                #     widget.view_box.setXRange(curr_min, curr_max, padding=DEFAULT_PADDING_VAL)

            # 如果有清除，弹窗
            if cleared:
                msg = "以下图表被清除：\n"
                for plot_idx, reason in cleared:
                    msg += f"Plot {plot_idx}: {reason}\n"
                QMessageBox.information(self, "更新通知", msg)

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    # app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    # compile
    # mac: pyinstaller --noconsole --onefile --add-data "README.md:." test_pyqt6_v5.py
    # win: pyinstaller --noconsole --onefile --add-data "README.md;." test_pyqt6_v5.py