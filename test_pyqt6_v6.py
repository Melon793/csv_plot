from __future__ import annotations 
import sys
import os

# 屏蔽 macOS ICC 警告
os.environ["QT_LOGGING_RULES"] = (
    "qt6ct.debug=false; "      # 原来想关的 qt6ct 日志
    "qt.gui.icc=false"         # 关闭 ICC 解析相关日志
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData, QRectF, QMargins, QTimer,QSettings, QEvent, QMargins,Qt, QAbstractTableModel, QModelIndex,QModelIndex
from PyQt6.QtGui import QFont, QFontMetrics, QDrag, QPen, QColor,QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QProgressDialog,QGridLayout,QSpinBox,QMenu,
    QFileDialog, QPushButton, QAbstractItemView, QLabel, QLineEdit,QTableView,
    QMessageBox, QDialog, QFormLayout, QSizePolicy,QGraphicsLinearLayout,QGraphicsProxyWidget,QGraphicsWidget,QTableWidget,QTableWidgetItem,QHeaderView
)
import pyqtgraph as pg
#import numpy as np

from typing import Dict, Callable


# some change 
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
            total_bytes = os.path.getsize(self.file_path)#Path(self.file_path).stat().st_size

            def _progress_cb(bytes_read: int):
                if total_bytes > 0:
                    self.progress.emit(int(bytes_read / total_bytes * 100))

            # print("Calling FastDataLoader with _progress:", _progress_cb) 

            # 给 FastDataLoader 打补丁：加一个回调
            loader = FastDataLoader(
                self.file_path,
                # 其他参数照抄
                descRows=self.descRows,
                sep=self.sep,
                hasunit=self.hasunit,
                chunksize=10_000,          
                _progress= _progress_cb,
            )
            self.finished.emit(loader)
        except Exception as e:
            self.error.emit(str(e))

class FastDataLoader:
    # 脏数据清单
    _NA_VALUES = [
        "", "nan", "NaN", "NAN", "NULL", "null", "None",
        "Inf", "inf", "-inf", "-Inf", "1.#INF", "-1.#INF", "data err", '* *', '----',
    ]

    def __init__(
        self,
        csv_path: str ,
        *,
        max_rows_infer: int = 1000,
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
        import polars as pl
        import numpy as np
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
        self._var_names, self._units, self.encoding_used = self._load_header_units_v2(
            self._path, desc_rows=self.descRows, usecols=self.usecols, sep=self.sep,hasunit=self.hasunit,
        )

        # 推断 dtype
        
        sample = pl.read_csv(
            self._path,
            separator=self.sep,
            skip_rows=(2 + self.descRows) if self.hasunit else (1+self.descRows),
            n_rows=self.max_rows_infer,
            infer_schema_length=self.max_rows_infer,
            has_header=False,
            new_columns=self._var_names,
            encoding=self.encoding_used,
            columns=self.usecols,
            null_values=self._NA_VALUES,
        )

        dtype_map, parse_dates, date_formats,downcast_ratio = self._infer_schema(sample)

        self.date_formats = date_formats
        self.sample_mem_size = sample.estimated_size()

        print(f"the estimated downcast ratio is {downcast_ratio*100:2f} %, the compression ratio estimated {(0.6*downcast_ratio+1*(1-downcast_ratio))}")
        print(f"sample of {sample.shape[0]} lines has costed memory {self.sample_mem_size/(1024**2):2f}Mb")
        self.byte_per_line = ((0.5*downcast_ratio+1*(1-downcast_ratio))*self.sample_mem_size)/sample.height

        self.estimated_lines = int(self.file_size/(self.byte_per_line ))
        print(f"this file might have lines of {self.estimated_lines}")

        # 计算 chunk 大小
        if chunksize is None:
            chunksize = 10_000
        
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
        print(f"actual lines of data files is {self.row_count}")
        print(f"the actual file size: {(self.file_size/1024**2):2f}Mb, the mem size:{(self._df.estimated_size()/1024**2):2f}Mb")
        # 后处理
        if drop_empty:
            self._df = self._df.drop_nulls()
        if downcast_float:
            self._downcast_numeric()
        self._df_validity=self._check_df_validity()
        import gc
        gc.collect()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_field(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] == '"':
            s = s[1:-1].replace('""', '"')   # 处理 "" 转义
        return s
    
    @staticmethod
    def _load_header_units_v2(path, desc_rows=0, usecols=None, sep=',', hasunit=True):
        from pathlib import Path
        raw = Path(path).read_bytes()
        need = 2 if hasunit else 1
        for enc in ('utf-8', 'cp1252', 'gbk'):
            try:
                lines = raw.decode(enc).splitlines()[desc_rows:desc_rows + (2 if hasunit else 1)]
                if len(lines) < need:          
                    # 行数不足
                    raise ValueError
                rows = [[FastDataLoader._strip_field(f) for f in line.split(sep)] for line in lines]
                #rows = [line.split(sep) for line in lines]
                break
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            raise RuntimeError('无法以 utf8/cp1252/gbk 解码')

        names = [v if v is not None else '_' for v in rows[0]]
        names = FastDataLoader._make_unique(names) 

        units = dict(zip(names, [rows[1][i] if hasunit else '-' for i in range(len(names))]))
        return names, units, ('utf8' if enc == 'utf-8' else 'utf8-lossy')


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
        import polars as pl
        nrows = 2 if hasunit else 1
        encodings = ["utf8", "utf8-lossy"]
        for enc in encodings:
            try:
                df = pl.read_csv(
                    path,
                    separator=sep,
                    skip_rows=desc_rows,
                    n_rows=nrows,
                    has_header=False,
                    columns=usecols,
                    encoding=enc,
                )
                break
            except pl.ComputeError:
                continue
        else:
            raise RuntimeError("无法以任何可用编码读取文件")

        if df.height < nrows:
            raise ValueError("文件至少需要两行（变量名 + 单位）")

        var_names = df.row(0)
        var_names = [str(v) if v is not None else '' for v in var_names]
        var_names = FastDataLoader._make_unique(var_names) 
        if hasunit:
            unit_row = df.row(1)
            units = dict(zip(var_names, [str(u) if u is not None else '' for u in unit_row]))
        else:
            units = dict(zip(var_names, ['-'] * len(var_names)))
        return var_names, units, enc
    


    @staticmethod
    def _infer_schema(sample) -> tuple[dict[str, str], list[str], dict[str, str],float]:
        import polars as pl
        import numpy as np
        dtype_map: dict = {}
        parse_dates: list[str] = []
        date_formats: dict[str, str] = {}

        date_candidates = [
            "%Y/%m/%d","%H:%M:%S", "%H:%M:%S.%f",
            "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d","%d-%m-%Y", "%m-%d-%Y"
        ]
        float_cols = sample.select(pl.col(pl.Float64))
        downcast_ratio_est = float_cols.shape[1] / sample.shape[1] if sample.shape[1] > 0 else 0.000001
        numeric_types = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        for col in sample.columns:
            s = sample[col]
            if s.null_count() == s.len():
                dtype_map[col] = pl.Utf8
                continue
            date_matched = False
            if s.dtype == pl.String:
                for fmt in date_candidates:
                    parsed = s.str.strptime(pl.Datetime, fmt, strict=False)
                    if parsed.null_count() == 0:
                        parse_dates.append(col)
                        date_formats[col] = fmt
                        date_matched = True
                        break
            if date_matched:
                dtype_map[col] = pl.Utf8  # Keep as string for now, parse later if needed
            else:
                if s.dtype in numeric_types:
                    dtype_map[col] = pl.Float32
                else:
                    dtype_map[col] = pl.Utf8
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
    ):
        import polars as pl
        chunks: list[pl.DataFrame] = []
        # do not parse date
        if not self.do_parse_date:
            parse_dates=[]
        lazy_df = (
            pl.scan_csv(
                path,
                separator=sep,
                skip_rows=(2 + descRows) if hasunit else (1 + descRows),
                has_header=False,
                new_columns=self._var_names,
                schema_overrides=dtype_map,
                encoding=self.encoding_used,
                null_values=self._NA_VALUES,
            )
        )
        # Polars doesn't have direct chunksize for scan_csv, but we can slice
        total_rows = lazy_df.select(pl.len()).collect().item()
        for start in range(0, total_rows, chunksize):
            chunk = lazy_df.slice(start, chunksize).collect()
            if self._progress_cb:
                bytes_read = (start + chunk.height) * self.byte_per_line   # 粗略估算
                self._progress_cb(min(bytes_read, self.file_size))
            chunks.append(chunk)
        df = pl.concat(chunks)
        # Parse dates if needed
        if self.do_parse_date:
            for col in parse_dates:
                fmt = self.date_formats[col]
                df = df.with_columns(pl.col(col).str.strptime(pl.Datetime, fmt))
        return df

    def _downcast_numeric(self) -> None:
        import polars as pl
        import time 
        t_start = time.time()
        float_cols = [col for col, dtype in self._df.schema.items() if dtype == pl.Float64]
        for col in float_cols:
            self._df = self._df.with_columns([
                pl.col(col).cast(pl.Float32) for col in float_cols
            ])
        print(f"used time: {time.time()-t_start}")
            # self._df = self._df.with_columns(
            #     pl.col(col).cast(pl.Float32)
            # )
            

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------

    def _check_df_validity(self) -> Dict:
        #import time
        #t_start = time.time()
        validity : Dict = {}
        for col in self._df.columns:
            validity[col] = self._classify_column(self._df[col])
        #print(f"used time: {time.time()-t_start}")
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
    def _classify_column(s: pl.Series) -> int:
        """
        1: 全部可转数字，且 ≥2 个不同有效值
        0: 全部可转数字，且唯一有效值
        -1: 存在非数字 或 全部 NaN/空
        """
        import polars as pl
        # 1. 先转 float；失败或非数字会被置成 null
        numeric = s.cast(pl.Float64, strict=False)

        # 2. 去掉 null / NaN
        valid = numeric.drop_nulls().drop_nans()
        if valid.len() == 0:
            return -1

        # 3. 判断唯一值个数
        return 0 if valid.n_unique() == 1 else 1
    # def _classify_column(series) -> int:
    #     """
    #     1: 全部可转数字，且 ≥2 个不同有效值
    #     0: 全部可转数字，且唯一有效值
    #     -1: 存在非数字 或 全部 NaN
    #     """
    #     import polars as pl
    #     # 1) 先尝试整列转 float，失败直接 C
    #     try:
    #         numeric = series.cast(pl.Float64, strict=True)
    #     except pl.ComputeError:
    #         return (-1)

    #     # 2) 去掉 NaN 后看有效值
    #     valid = numeric.drop_nulls()
    #     if valid.len() == 0:          # 全 NaN
    #         return (-1)

    #     unique_vals = valid.unique()
    #     if unique_vals.len() == 1:
    #         return (0)
    #     return (1)
    
    @property
    def df(self):
        return self._df

    @property
    def units(self) -> dict[str, str]:
        return self._units
    
    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def datalength(self) -> int:
        return self._df.height

    @property
    def var_names(self) -> list[str]:
        return self._df.columns
    
    @property
    def row_count(self) -> int:
        return self._df.height
    
    @property
    def column_count(self) -> int:
        return len(self._df.columns)
    
    @property
    def time_channels_info(self) -> dict[str, str]:
        return self.date_formats
    
    @property
    def df_validity(self) -> Dict:
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
            background-color: rgba(68, 68, 68, 200);
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


class PolarsTableModel(QAbstractTableModel):
    """只读官方虚拟模型，支持千万行秒开"""
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self._df = df

    # 三个必须实现的纯虚函数
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.height

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self._df[:, index.column()][index.row()]
        return str(value) if value is not None else ""

    def headerData(self, section, orientation, role):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)           # 行号 1-based

    def removeColumns(self, column, count, parent=QModelIndex()):
        if column < 0 or column + count > self.columnCount():
            return False
        self.beginRemoveColumns(parent, column, column + count - 1)
        self._df = self._df.drop(self._df.columns[column:column + count])
        self.endRemoveColumns()
        return True
    
class DataTableDialog(QDialog):
    _instance = None
    _settings = QSettings("MyCompany", "DataTableDialog")

    @classmethod
    def popup(cls, var_name: str, data, parent=None):
        import polars as pl
        if cls._instance is None:
            cls._instance = cls(parent)

        dlg = cls._instance
        if dlg.has_column(var_name):
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return dlg

        dlg.add_series(var_name, data)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("变量数值表")

        # 1. 用官方 QTableView 代替 QTableWidget
        self.view = QTableView(self)
        self.view.setSortingEnabled(True)           # 需要排序可加 QSortFilterProxyModel
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)

        # 内部 DataFrame，所有列都存在这里
        import polars as pl
        self._df = pl.DataFrame()

        # 允许用户拖拽列标题，改变列顺序
        self.view.horizontalHeader().setSectionsMovable(True)
        self.view.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.view.horizontalHeader().customContextMenuRequested.connect(self._on_header_right_click)
        
        # 设置字体
        font = self.view.horizontalHeader().font()
        font.setBold(True)
        self.view.horizontalHeader().setFont(font)

        # 恢复上一次的位置/大小
        geom = self._settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
        else:
            self.resize(400, 600)

    def closeEvent(self, event):
        import polars as pl
        self._df = pl.DataFrame()          # 释放内存
        self.view.setModel(None)
        self.hide()
        event.accept()
        self._settings.setValue("geometry", self.saveGeometry())

    def has_column(self, var_name: str) -> bool:
        return var_name in self._df.columns

    def add_series(self, var_name: str, data):
        # ---- 追加列 ----
        self._df = self._df.with_columns(data.alias(var_name))

        # 用官方模型挂载（零拷贝，只保存 DataFrame 指针）
        model = PolarsTableModel(self._df)
        self.view.setModel(model)

        # ---- 自动向右伸展宽度（仅当前列） ----
        self._auto_resize_right()

    def _auto_resize_right(self):
        COL_WIDTH = 120
        MARGIN = 40
        left_x = self.x()
        needed_width = len(self._df.columns) * COL_WIDTH + MARGIN
        screen = QApplication.primaryScreen().availableGeometry()
        max_right = screen.right() - MARGIN
        new_width = max(300, min(needed_width, max_right - left_x))  # 最小宽 300
        self.resize(new_width, self.height())


    def _on_header_right_click(self, pos):
        header = self.view.horizontalHeader()
        col = header.logicalIndexAt(pos)
        if col < 0:      # 点在空白区
            return

        menu = QMenu(self)
        act = menu.addAction(f"删除列 “{self._df.columns[col]}”")
        if menu.exec(header.mapToGlobal(pos)) == act:
            self._remove_column(col)

    # 3. 真正执行删除
    def _remove_column(self, col):
        model = self.view.model()
        model.removeColumns(col, 1)          # 从模型/DF 中删除
        self._auto_resize_right()            # 窗口自动缩回

        
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
                self.view_box.setXRange(min_val, max_val, padding=0.00)
            else:
                self.view_box.setYRange(min_val, max_val, padding=0.00)

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

        # ---------------- 关键修改 1：字体 ----------------
        hdr = self.horizontalHeader()
        # ---------------- 默认 3:1 的初始宽度 ----------------
        total = 255          # 首次拿不到 width 时给一个兜底
        self.setColumnWidth(0, int(total * 0.75))
        self.setColumnWidth(1, int(total * 0.25))

        # font = QFont()
        # font.setPointSize(12)   # 想要多大就改多大
        # font.setBold(True)
        # hdr.setFont(font)

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

        DataTableDialog.popup(var_name, series, parent=main_window)

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

class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, units_dict, dataframe, time_channels_info={},synchronizer=None):
        super().__init__()
        self.setup_ui(units_dict, dataframe, time_channels_info, synchronizer)
        
    def setup_ui(self, units_dict, dataframe, time_channels_info={},synchronizer=None):
        """初始化UI组件和布局"""
        self.setAcceptDrops(True)
        self.units = units_dict
        self.data = dataframe
        self.time_channels_info = time_channels_info
        self.synchronizer = synchronizer
        self.curve = None
        self._value_cache: dict[str, tuple] = {}
        #self.ci.layout.setContentsMargins(0, 0, 0, 5)
        

        self.y_format = ''
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
        #self.label_right.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
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
        self.plot_item = self.addPlot(row=1, col=0, colspan=2)
        self.view_box = self.plot_item.vb
        self.view_box.setAutoVisible(True)  # 自动适应可视区域
        self.plot_item.setTitle(None)
        self.plot_item.hideButtons()
        self.plot_item.setClipToView(True)
        self.plot_item.setDownsampling(True)
        self.setBackground('w')

        pen = pg.mkPen('#f00',width=1)
        self.plot_item.getAxis('left').setGrid(255) 
        #self.plot_item.getAxis('left').gridPen=(pen) 
        self.plot_item.getAxis('bottom').setGrid(255) 
        #self.plot_item.getAxis('bottom').gridPen=(pen) 
        self.plot_item.showGrid(x=True, y=True, alpha=0.1)
        

    def update_left_header(self, left_text=None):
        """更新顶部文本内容"""
        if left_text is not None:
            self.label_left.setText(left_text)

    def auto_range(self):
        self.view_box.setXRange(self.xMin, self.xMax, padding=0.02)
        self.view_box.autoRange()

    def auto_y_in_x_range(self):
        vb=self.view_box
        vb.enableAutoRange(axis=vb.YAxis, enable=True)

    def update_right_header(self, right_text=None):
        """更新顶部文本内容"""
        if right_text is not None:
            self.label_right.setText(right_text)
            self.label_right.setAlignment(Qt.AlignmentFlag.AlignRight)


    def reset_plot(self,xMin,xMax):
        import numpy as np
        self.plot_item.setLimits(xMin=None, xMax=None)  # 解除X轴限制
        self.plot_item.setLimits(yMin=None, yMax=None)  # 解除Y轴限制
        if not (np.isnan(xMax) or np.isinf(xMax)):
            self.view_box.setXRange(xMin, xMax, padding=0.02)
            padding_xVal=0.1
            self.plot_item.setLimits(xMin=0-padding_xVal*(xMax-xMin), xMax=(padding_xVal+1)*(xMax-xMin))

        self.view_box.setYRange(0,1,padding=0) 
        self.vline.setBounds([None, None]) 

        self.xMin = xMin
        self.xMax = xMax

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
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 0, 0, 100), width=3) )
        self.vline.setZValue(100) 
        self.cursor_label = pg.TextItem("", anchor=(1, 1), color="red")
        self.plot_item.addItem(self.vline, ignoreBounds=True)
        self.plot_item.addItem(self.cursor_label, ignoreBounds=True)
        self.vline.setVisible(False)
        self.cursor_label.setVisible(False)
        
        # 信号连接
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        self.vline.sigPositionChanged.connect(self.update_cursor_label)
        self.setAntialiasing(True)
    
    def wheelEvent(self, ev):
        vb = self.plot_item.getViewBox()
        delta = ev.angleDelta().y()  # 获取垂直滚动增量
        if delta > 0:
            #print("向上滚动,放大")  # 正值表示向上
            vb.scaleBy((0.8, 1))  # 仅缩放x轴
        elif delta < 0:
            #print("向下滚动,缩小")  # 负值表示向下
            vb.scaleBy((1.2, 1))  # 仅缩放x轴
        else:
            super().wheelEvent(ev)
    
    def mouse_moved(self, evt):
        """鼠标移动事件处理"""
        pos = evt[0]
        if not self.plot_item.sceneBoundingRect().contains(pos):
            return
        mousePoint = self.plot_item.vb.mapSceneToView(pos)
        if hasattr(self.window(), 'sync_crosshair'):
            self.window().sync_crosshair(mousePoint.x(), self)
            #print(f"mouse in pos {mousePoint.x()}")

    def msInt_to_fmtStr(self,value:int):
        total_seconds = value / 1000
        hh = int(total_seconds // 3600)
        mm = int((total_seconds % 3600) // 60)
        ss = total_seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:06.3f}"
    
    def dateInt_to_fmtStr(self,value:int):
        import datetime
        base_date = datetime.date(1970, 1, 1)
        target_date = base_date + datetime.timedelta(days=int(value))
        return target_date.strftime('%Y-%m-%d')
    
    def update_cursor_label(self):
        """更新光标标签位置和内容"""
        if len(self.plot_item.listDataItems()) == 0:
            #self.cursor_label.setText("")
            self.update_right_header("")
            return
        
        try:
            import numpy as np
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
            if self.y_format == 'ms':
                time_str=self.msInt_to_fmtStr(y_val)
                self.update_right_header(f"x={x:.0f}, y={time_str}")
            elif self.y_format == 'date':
                date_str=self.dateInt_to_fmtStr(y_val)
                self.update_right_header(f"x={x:.0f}, y={date_str}")
            else:
                self.update_right_header(f"x={x:.0f}, y={y_val:.2f}")

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
        self._value_cache: dict[str, tuple] = {}

    def get_value_from_name(self,var_name)-> tuple | None:
        import polars as pl
        import numpy as np
        if var_name in self._value_cache:
            #y_values, self.y_format = self._value_cache[var_name]
            return self._value_cache[var_name]

        raw_values = self.data[var_name]
        dtype = raw_values.dtype
        numeric_types = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        if dtype in numeric_types:
            y_values = raw_values.to_numpy()
            y_format = 'number'

        elif var_name in self.time_channels_info:
            fmt = self.time_channels_info[var_name]
            try:
                if "%H:%M:%S" in fmt:
                    #time
                    datetime_values = raw_values.str.strptime(pl.Datetime, fmt, strict=False)
                    y_values = (
                        datetime_values.dt.hour() * float(3_600_000) +
                        datetime_values.dt.minute() * float(60_000) +
                        datetime_values.dt.second() * float(1_000) +
                        datetime_values.dt.millisecond()     
                    ).to_numpy()
                    y_format = 'ms'
                else:
                    #date
                    datetime_values = raw_values.str.strptime(pl.Date, fmt, strict=False)
                    # 假设日期格式为"YYYY-MM-DD"
                    result = (
                        raw_values.to_frame(name="dates")
                        .with_columns(
                            pl.col("dates").str.strptime(pl.Date, fmt),  # 转换为Date类型
                            (pl.col("dates").str.strptime(pl.Date, fmt) - pl.date(1970, 1, 1)).dt.total_days().alias("days_since_epoch")
                        )
                    )

                    # 提取天数差列作为numpy数组
                    y_values = result["days_since_epoch"].to_numpy().astype(int)

                    # y_values = raw_values.to_frame(name="dates").with_columns(
                    #     pl.col("dates").str.strptime(pl.Date, fmt)
                    # ).to_numpy()

                    #y_values = (datetime_values - pl.lit(pl.date(1970,1,1))).dt.total_days().to_numpy()
                    y_format = 'date'
            except:
                # cannot parse the format
                return None

        else:
            # cannot get right info
            return None
        
        # finally
        self._value_cache[var_name] = (y_values, y_format)
        return y_values, y_format
    
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
        import numpy as np
        var_name = event.mimeData().text()
        if var_name not in self.data.columns:
            QMessageBox.warning(self, "错误", f"变量 {var_name} 不存在")
            return
        
        y_values,y_format = self.get_value_from_name(var_name=var_name)

        if y_values is None:
            QMessageBox.warning(self, "错误", f"变量 {var_name} 没有有效数据")
            event.acceptProposedAction()
            return
        
        # 如果正常
        self.y_format=y_format
        x_values = np.arange(1,1+len(y_values))

        # if any(isinstance(item, pg.PlotDataItem) for item in self.plot_item.items):
        #     self.curve.setData(x_values,y_values)
        # else:
        self.plot_item.clearPlots()             
        _pen = pg.mkPen(color='blue', width=4)
        self.curve=self.plot_item.plot(x_values, y_values, pen=_pen, name=var_name)

        full_title = f"{var_name} ({self.units.get(var_name, '')})".strip()
        self.update_left_header(full_title)
        padding_xVal = 0.1
        padding_yVal = 0.5
        if np.nanmin(y_values) == np.nanmax(y_values):
            y_center = np.nanmin(y_values)
            y_range = 1.0 if y_center == 0 else abs(y_center) * 0.2
            
            # limit x/y range
            self.plot_item.setLimits(xMin=0-padding_xVal*len(x_values), xMax=(padding_xVal+1)*len(x_values), 
                minXRange=5,
                yMin=y_center - y_range,
                yMax=y_center + y_range) 
            self.view_box.setYRange(y_center - y_range, y_center + y_range, padding=0.05)       
        else:            
            # limit x/y range            
            self.plot_item.setLimits(xMin=0-padding_xVal*len(y_values), xMax=(padding_xVal+1)*len(y_values), 
                minXRange=5,
                yMin=np.nanmin(y_values)-padding_yVal*(np.nanmax(y_values)-np.nanmin(y_values)), 
                yMax=np.nanmax(y_values)+padding_yVal*(np.nanmax(y_values)-np.nanmin(y_values)))
            
            self.view_box.setYRange(np.nanmin(y_values), np.nanmax(y_values), padding=0.05)

        

        self.plot_item.update()
        if hasattr(self.window(), 'cursor_btn'):
            self.vline.setBounds([min(x_values), max(x_values)])
            self.toggle_cursor(self.window().cursor_btn.isChecked())
        else:
            self.toggle_cursor(False)

        event.acceptProposedAction()

    def clear_plot_item(self):
        self.plot_item.setLimits(xMin=None, xMax=None)  # 解除X轴限制
        self.plot_item.setLimits(yMin=None, yMax=None)  # 解除Y轴限制

        self.view_box.setYRange(0,1,padding=0) 
        self.vline.setBounds([None, None]) 

        #self.plot_item.update()
        self.plot_item.clearPlots() 
        self.axis_y.setLabel(text="")
        self.update_left_header("channel name")
        self.update_right_header("")

    # ---------------- 双击轴弹出对话框 ----------------
    def mouseDoubleClickEvent(self, event):
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            super().mouseDoubleClickEvent(event)
            return
        
        if event.button() == Qt.MouseButton.MiddleButton:
            self.clear_plot_item()

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
                        view.view_box.setXRange(min_val, max_val, padding=0.00)
                        view.plot_item.update()
                return
        return super().mouseDoubleClickEvent(event)



# ---------------- 主窗口 ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.defaultTitle = "数据快速查看器(PyQt6), Alpha版本"
        self.setWindowTitle(self.defaultTitle)
        self.resize(1600, 900)

        self.var_names = None
        self.units = None
        self.time_channels_infos = None
        self.data = None
        self.data_validity = None

        # ---------------- 中央控件 ----------------
        central = QWidget()
        self.setCentralWidget(central)

        # 总水平布局：左侧变量列表 + 右侧绘图区
        main_layout = QHBoxLayout(central)
        #main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for precise alignment

        # ---------------- 左侧变量列表 ----------------
        left_widget = QWidget()
        left_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        left_layout.addWidget(QLabel("变量列表"))

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("输入变量名关键词（空格分隔）")
        self.filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.filter_input)

        self.unit_filter_input = QLineEdit()
        self.unit_filter_input.setPlaceholderText("输入单位关键词（空格分隔）")
        self.unit_filter_input.setContentsMargins(60,0,0,0)

        self.unit_filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.unit_filter_input)

        self.load_btn = QPushButton("导入 CSV")
        self.load_btn.clicked.connect(self.load_btn_click)
        left_layout.addWidget(self.load_btn)

        self.list_widget = MyTableWidget()
        left_layout.addWidget(self.list_widget)

        main_layout.addWidget(left_widget, 0)

        # ---------------- 右侧绘图区 ----------------
        self.plot_widget = QWidget()
        root_layout = QVBoxLayout(self.plot_widget)

        root_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        root_layout.setSpacing(0)  # Remove spacing

        # 顶部按钮栏：弹簧 + 光标按钮（右对齐）
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 5, 0)
        top_bar.addStretch(1)




        
        self.auto_range_btn = QPushButton("自动缩放")
        self.auto_range_btn.clicked.connect(self.auto_range_all_plots)
        

        self.auto_y_btn = QPushButton("仅调节y轴")
        self.auto_y_btn.clicked.connect(self.auto_y_in_x_range)
        
        
        self.cursor_btn = QPushButton("显示光标")
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.clicked.connect(self.toggle_cursor_all)
        
        self.grid_layout_btn = QPushButton("修改布局")
        self.grid_layout_btn.clicked.connect(self.open_layout_dialog)

        self.clear_all_plots_btn = QPushButton("清除绘图")
        self.clear_all_plots_btn.clicked.connect(self.clear_all_plots)
        #self.cursor_btn.setFixedSize(100, 28)

        top_bar.addWidget(self.clear_all_plots_btn)
        top_bar.addWidget(self.grid_layout_btn)
        top_bar.addWidget(self.cursor_btn)
        top_bar.addWidget(self.auto_y_btn)
        top_bar.addWidget(self.auto_range_btn)
        

        root_layout.addLayout(top_bar)

        # 真正容纳子图的布局
        #self.plot_layout = QVBoxLayout()
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

    def open_layout_dialog(self):
        dlg = LayoutInputDialog(max_rows=self._plot_row_max_default, 
                                max_cols=self._plot_col_max_default, 
                                cur_rows=self._plot_row_current,
                                cur_cols=self._plot_col_current,
                                parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            r, c = dlg.values()
            self.set_plots_visible (r, c)

    def show_drop_overlay(self):
        self.drop_overlay.setGeometry(self.centralWidget().rect())
        self.drop_overlay.raise_()
        self.drop_overlay.show()
        self.drop_overlay.activateWindow()

    def hide_drop_overlay(self):
        self.drop_overlay.hide()


    def reset_plots_after_loading(self,xMin,xMax):
        for container in self.plot_widgets:
             container.plot_widget.reset_plot(xMin,xMax)
             container.plot_widget.clear_value_cache()
        # for plot_widget in self.plot_widgets:
        #     plot_widget.reset_plot(xMin,xMax)

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
                container.setVisible(True)            # 默认全部显示

                self.plot_layout.addWidget(container, r, c)
                self.plot_widgets.append(container)   # 保存容器

        # 设置行列权重=1，确保均分
        for r in range(m):
            self.plot_layout.setRowStretch(r, 1)
        for c in range(n):
            self.plot_layout.setColumnStretch(c, 1)

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
            
        
    def load_btn_click(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV File (*.csv);;m File (*.mfile);;t00 File (*.t00);;all File (*.*)")
        status = self.load_csv_file(file_path)
        if status:
            if DataTableDialog._instance is not None:
                DataTableDialog._instance.close()   # 触发 closeEvent → 清空
                DataTableDialog._instance = None
            self.setWindowTitle(f"{self.defaultTitle} ---- 数据文件: [{file_path}]")

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
        
    def load_csv_file(self,file_path):
        #file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV File (*.csv);;m File (*.mfile);;t00 File (*.t00);;all File (*.*)")
        
        # raise window
        self.raise_()  # Bring window to front
        self.activateWindow()  # Set focus

        status = False
        #data = pd.DataFrame()
        if not file_path:
            return status
        
        if not os.path.isfile(file_path):
            QMessageBox.critical(self, "读取失败",f"文件不存在: {file_path}")
            return status
        
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
            if file_ext in ['.csv','.txt']:
                delimiter_typ = ','
                descRows = 0
                hasunit = True
            elif file_ext in ['.mfile','.t00','.t01']:
                delimiter_typ = '\t'
                descRows = 2
                hasunit=True                    
            else:
                QMessageBox.critical(self, "读取失败",f"无法读取后缀为:'{file_ext}'的文件")
                return status
        
        _Threshold_Size_Mb=20 

        # < 20 MB 直接读
        file_size =os.path.getsize(file_path)
        if file_size < _Threshold_Size_Mb * 1024 * 1024:               
            status = self._load_sync(file_path, descRows=descRows,sep=delimiter_typ,hasunit=hasunit)
        else:
            # 20 MB 以上走线程
            self._progress = QProgressDialog("正在读取数据...", "取消", 0, 100, self)
            self._progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            self._progress.setAutoClose(True)
            self._progress.setCancelButton(None)            # 不可取消
            self._progress.show()

            self._thread = DataLoadThread(file_path, descRows=descRows,sep=delimiter_typ,hasunit=hasunit)
            self._thread.progress.connect(self._progress.setValue)
            self._thread.finished.connect(self._on_load_done)
            self._thread.error.connect(self._on_load_error)
            self._thread.start()

            status = True

        return status

    def _load_sync(self, 
                   file_path:str,
                   descRows:int =0,
                   sep:str = ',',
                   hasunit:bool=True):
        """小文件直接读"""
        try:
            loader = FastDataLoader(file_path, descRows=descRows,sep=sep,hasunit=hasunit)
            self._apply_loader(loader)
            return True
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return False

    def _on_load_done(self, loader):
        self._progress.close()
        self._apply_loader(loader)

    def _on_load_error(self, msg):
        self._progress.close()
        QMessageBox.critical(self, "读取失败", msg)

    def _apply_loader(self, loader):
        """把 loader 的内容同步到 UI"""
        self.loader = loader
        self.var_names = loader.var_names
        self.units = loader.units
        self.time_channels_infos = loader.time_channels_info
        self.data_validity = loader.df_validity
        self.list_widget.populate(self.var_names, self.units, self.data_validity)

        # 移除占位符
        self.placeholder_label.setParent(None)

        # 现在创建子图矩阵
        self.create_subplots_matrix(self._plot_row_max_default, self._plot_col_max_default)
        self.set_plots_visible(self._plot_row_current, self._plot_col_current)

        # 更新所有 plot_widgets 的数据
        for container in self.plot_widgets:
            widget = container.plot_widget
            widget.data = loader.df
            widget.units = loader.units
            widget.time_channels_info = loader.time_channels_info

        self.reset_plots_after_loading(xMin=0, xMax=loader.datalength)
        self.setWindowTitle(f"{self.defaultTitle} ---- 数据文件: [{loader.path}]")


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


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())