"""BLF / CAN 数据库相关 component 测试的公共替身。

``FileLoaderManager`` 的 BLF 分支（方案 §5）只依赖 MainWindow 上的一小撮
属性（按钮、banner、变量列表、loader、数据版本号），但**通用的** reload
机械会连带摸到布局管理器、绘图容器、游标同步、表格对话框等十几个对象。
本模块把两侧切开：

* :class:`BlfFlowHost` —— 真实 Qt 控件组成的最小宿主（banner/按钮必须是
  真控件，否则显隐断言没有意义）；
* :class:`BlfFlowManager` —— 只替换掉与 BLF 决策无关的通用 reload 步骤，
  BLF 自身的 A/B/C/F/G 路径一行不改地跑真代码；
* :class:`FakeLoadThread` —— 顶掉后台线程，让用例自己决定何时「扫完 /
  报错 / 被取消」，不必真等 python-can 读文件。

被 ``test_can_db_banner.py`` 与 ``test_blf_import_flow.py`` 共用。
"""

from __future__ import annotations

from contextlib import contextmanager

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMainWindow, QPushButton

from src.ui.file_loader_manager import FileLoaderManager
from src.ui.widgets.can_db_banner import CanDbBanner

#: 左栏按钮行里 ``set_button_status()`` 会逐个 setEnabled 的按钮
_STATUS_BUTTONS = (
    "time_correction_btn",
    "clear_all_plots_btn",
    "auto_range_btn",
    "auto_y_btn",
    "cursor_btn",
    "mark_region_btn",
    "grid_layout_btn",
)


class RecordingListWidget:
    """变量列表替身：只记录 populate 的调用参数（重选后必须刷新全靠它）。"""

    def __init__(self):
        self.calls: list[tuple[list, dict, object]] = []

    def populate(self, var_names, units, validity=None):
        self.calls.append((list(var_names), dict(units), validity))


class RecordingCursorSync:
    """游标同步替身：清曲线与重建曲线都只记录次数（取消恢复靠的就是后者）。"""

    def __init__(self):
        self.cleared = 0
        self.replots: list[dict] = []

    def clear_all_plots(self):
        self.cleared += 1

    def replots_after_loading(self, skip_pin_reset: bool = False):
        self.replots.append({"skip_pin_reset": skip_pin_reset})


class BlfFlowHost(QMainWindow):
    """``FileLoaderManager`` BLF 路径需要的最小宿主窗口。"""

    def __init__(self):
        super().__init__()
        self.loader = None
        self.load_btn = QPushButton("导入数据文件", self)
        self.reload_btn = QPushButton("重载", self)
        self.can_db_btn = QPushButton("CAN 数据库…", self)
        self.can_db_banner = CanDbBanner(self)
        for name in _STATUS_BUTTONS:
            setattr(self, name, QPushButton(name, self))
        self.list_widget = RecordingListWidget()
        self.cursor_sync_manager = RecordingCursorSync()
        self.plot_widgets = []
        self.var_names: list[str] = []
        self.units: dict = {}
        self.data_validity = None
        self.var_stats_cache = {"stale": 1}
        self.loaded_path = ""
        self._data_version = 0
        self._is_loading_new_data = False
        #: 加载反馈流水（main 侧挂钩 4：BLF 分派前 begin、失败/取消/F/错误端 end）
        self.feedback_calls: list = []

    # --- 加载反馈：真实现会动状态横幅与忙碌光标，最小宿主只记录 ---------------

    def begin_load_feedback(self, file_path: str):
        self.feedback_calls.append(("begin", file_path))

    def end_load_feedback(self, hint: str = "", level: str = "info"):
        self.feedback_calls.append(("end", hint, level))


class BlfFlowManager(FileLoaderManager):
    """通用 reload 步骤换成记录器的 ``FileLoaderManager``。

    刻意保留 BLF 分支的全部真实代码：``_load_file`` 的 blf 判定、
    ``_resolve_can_databases``、``_start_blf_load``、``_on_blf_*``、
    ``_update_can_db_ui``、``reselect_can_database``、``show_can_unmatched``。
    """

    def __init__(self, mw):
        super().__init__(mw)
        #: 供断言的调用流水
        self.events: list = []
        self._reload_in_progress = False

    # --- 与 BLF 决策无关的通用步骤：只记录 ---------------------------------

    def _begin_data_reload(self):
        self.events.append("begin")
        self.mw._is_loading_new_data = True
        self.mw._data_version += 1

    def _end_data_reload(self):
        self.events.append("end")
        self.mw._is_loading_new_data = False

    def _swap_loader(self, new_loader, is_reload: bool = False):
        self.events.append(("swap", new_loader, is_reload))
        self.mw.loader = new_loader
        self.mw.var_names = new_loader.var_names
        self.mw.units = new_loader.units
        self.mw.data_validity = new_loader.df_validity
        self.mw.list_widget.populate(
            self.mw.var_names, self.mw.units, self.mw.data_validity
        )
        self._update_can_db_ui()

    def _apply_loader(self, is_reload: bool = False):
        """「变量集变更后」的统一收敛入口：只记录被调用，不摸真绘图容器。

        真实现会下发 per-widget 单位/时间信息、重绘、刷表格与变量信息窗等
        十几个对象，与本宿主的最小控件集不兼容。BLF 路径只关心**它有没有被走**
        （重选数据库后这些下游必须一起跟着变），所以副作用只留可断言的那几项。
        """
        self.events.append(("apply", is_reload))
        loader = self.mw.loader
        self.mw.var_names = loader.var_names
        self.mw.units = loader.units
        self.mw.data_validity = loader.df_validity
        self.mw.list_widget.populate(
            self.mw.var_names, self.mw.units, self.mw.data_validity
        )
        self._update_can_db_ui()

    def set_button_status(self, status: bool):
        self.events.append(("buttons", status))

    def _post_load_actions(self, file_path: str, is_reload: bool = False):
        self.events.append(("post", file_path, is_reload))
        self.mw.loaded_path = file_path


class FakeLoadThread(QObject):
    """``DataLoadThread`` 替身：由用例显式触发完成 / 失败 / 取消。

    信号名与参数与真线程逐一对齐（``progress(int)`` / ``finished(object)``
    / ``error(str)`` / ``cancelled(str)``），否则接线写错这里第一时间暴。
    """

    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal(str)

    def __init__(self, file_path: str, parent=None, db_paths=None):
        super().__init__(parent)
        self.file_path = file_path
        self.db_paths = list(db_paths or [])
        self.scan_stats: dict = {}
        self.started = 0
        self.cancel_requests = 0
        self._cancelled = False
        # ``_start_blf_load`` 的线程保活逻辑靠 ``isRunning()`` 判断旧线程是否
        # 还需要留引用 —— 真 ``QThread`` 有这个接口，替身没有的话取消后重启
        # 会直接 AttributeError，被 ``load_csv_file`` 的兜底 except 吞成一个
        # 阻塞的 critical 框（offscreen 下表现为整条用例挂死）。
        self._running = False

    # --- 真线程的公开接口 ---------------------------------------------------

    def isRunning(self) -> bool:
        return self._running

    def start(self):
        self.started += 1
        self._running = True

    def request_cancel(self):
        self.cancel_requests += 1
        self._cancelled = True

    # --- 测试侧驱动 --------------------------------------------------------

    def emit_progress(self, value: int):
        self.progress.emit(value)

    def emit_done(self, loader):
        self._running = False
        self.finished.emit(loader)

    def emit_error(self, message: str):
        self._running = False
        self.error.emit(message)

    def emit_cancelled(self):
        self._running = False
        self.cancelled.emit(self.file_path)


@contextmanager
def blf_flow_host(qapp):
    """``(host, manager, threads)`` 三元组：BLF 导入链路的最小可运行宿主。

    ``threads`` 收集替身线程：每次 ``_start_blf_load`` 都会新增一个
    :class:`FakeLoadThread`，用例自己决定它何时回「扫完 / 报错 / 被取消」。

    这里是 contextmanager 而非 ``@pytest.fixture``：夹具统一在
    ``tests/component/conftest.py`` 声明（测试文件里「import 夹具 + 同名形参」
    会被 ruff 判成 F811），本模块只供实现、可被非 pytest 代码复用。
    """
    import src.ui.file_loader_manager as flm

    host = BlfFlowHost()
    manager = BlfFlowManager(host)
    manager._kept_alive = host  # 基类用弱引用持宿主，测试里得自己留一手
    threads: list[FakeLoadThread] = []

    def _factory(file_path, parent=None, db_paths=None, **kwargs):
        thread = FakeLoadThread(file_path, parent=parent, db_paths=db_paths)
        threads.append(thread)
        return thread

    original = flm.DataLoadThread
    flm.DataLoadThread = _factory
    try:
        yield host, manager, threads
    finally:
        flm.DataLoadThread = original
        manager._kept_alive = None
