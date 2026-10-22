"""加载失败出口的 UI 状态收口与「不得连带清掉旧数据」（全仓审查 P1-3 / P1-4）。

两条缺陷都是「失败一次就回不来」这一族：

* P1-3 —— ``_load_file`` 一进门就 ``begin_load_feedback()`` + ``set_button_status(False)``，
  而同步失败分支只恢复 ``load_btn`` / ``reload_btn``：忙等光标一直覆盖、状态栏
  永远停在「正在加载 … Ns」、6 个分析按钮永久灰掉，只有下一次**成功**加载才清得掉。
* P1-4 —— ``load_csv_file`` 的外层 ``except`` 无条件调 ``_release_old_data()``，
  而后者 docstring 写着「仅在新 loader 成功后调用」。用户正看着一屏曲线，去打开
  另一个文件失败（检测阶段文件不可读、BLF 起线程失败等），原有数据和曲线被一起
  清掉、loader 被 close()，且没有任何撤销入口。

失败出口的正确形态与后台线程的 ``_on_load_error``、BLF 的 ``_on_blf_error`` 同口径：
反馈收口 + 分析按钮按「当前是否还有有效数据」恢复 + 旧数据原样保留。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

from src.ui.file_loader_manager import FileLoaderManager
from tests.fixtures.data_factory import make_simple_rows, write_csv

#: ``set_button_status()`` 会逐个 setEnabled 的分析类按钮
_STATUS_BUTTONS = (
    "clear_all_plots_btn",
    "auto_range_btn",
    "auto_y_btn",
    "cursor_btn",
    "mark_region_btn",
    "grid_layout_btn",
)


class _FakeLoader:
    """旧数据替身：``_release_old_data`` 会 close() 它 —— 正是断言点。"""

    LOADER_TYPE = "csv"

    def __init__(self, path="old.csv", datalength=20):
        self.path = path
        self.datalength = datalength
        self.var_names = ["time", "speed"]
        self.units = {"time": "s", "speed": "km/h"}
        self.time_channels_info = {}
        self.df_validity = {}
        self.closed = False
        self.released = False

    def close(self):
        self.closed = True

    def release_memory(self):
        self.released = True


class _RecordingCursorSync:
    def __init__(self):
        self.reset_calls = 0

    def reset_all_pin_states(self):
        self.reset_calls += 1


class _FakeHost(QWidget):
    def __init__(self):
        super().__init__()
        self.load_btn = QPushButton(self)
        self.reload_btn = QPushButton(self)
        for name in _STATUS_BUTTONS:
            setattr(self, name, QPushButton(name, self))
        self.loader = None
        self.data = None
        self.var_names: list[str] = []
        self.units: dict = {}
        self.data_validity = {}
        self.var_stats_cache = {}
        self.value_cache = {}
        self._enum_text_maps = {}
        self.plot_widgets: list = []
        self.cursor_sync_manager = _RecordingCursorSync()
        self._is_loading_new_data = False
        self._is_being_destroyed = False
        self._data_version = 0
        self.loaded_path = ""
        #: ``begin/end_load_feedback`` 的调用流水
        self.feedback_calls: list = []

    def begin_load_feedback(self, file_path: str):
        self.feedback_calls.append(("begin", file_path))

    def end_load_feedback(self, hint: str = "", level: str = "info"):
        self.feedback_calls.append(("end", hint, level))


@pytest.fixture()
def host(qtbot):
    mw = _FakeHost()
    qtbot.addWidget(mw)
    mgr = FileLoaderManager(mw)
    return mw, mgr


@pytest.fixture()
def dialogs(monkeypatch):
    """模态框替身：offscreen 下真弹会挂死用例。"""
    records = {"critical": [], "warning": []}

    def make(kind, ret):
        def fake(parent, title, text, *args, **kwargs):
            records[kind].append((title, text))
            return ret

        return fake

    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(make("critical", QMessageBox.StandardButton.Ok)),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(make("warning", QMessageBox.StandardButton.Ok)),
    )
    return records


@pytest.fixture()
def csv_file(tmp_path):
    return write_csv(
        tmp_path / "sample.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(20),
    )


def _fake_fmt():
    return SimpleNamespace(sep=",", header_row=0, has_unit=True, encoding="utf-8")


def _disable_buttons(mw):
    """把界面摆成「加载中」的样子，便于断言失败出口有没有放平。"""
    mw.load_btn.setEnabled(False)
    for name in _STATUS_BUTTONS:
        getattr(mw, name).setEnabled(False)


def _status_buttons_enabled(mw):
    return [getattr(mw, name).isEnabled() for name in _STATUS_BUTTONS]


class TestSyncFailureRestoresUiState:
    def test_load_sync_false_ends_feedback_and_restores_buttons(
        self, host, dialogs, csv_file, monkeypatch
    ):
        """同步加载失败：忙等光标/秒表收口，分析按钮按「有没有数据」恢复。"""
        mw, mgr = host
        monkeypatch.setattr(
            "src.ui.file_loader_manager.FastDataLoader.auto_detect",
            staticmethod(lambda path: _fake_fmt()),
        )
        monkeypatch.setattr(mgr, "_load_sync", lambda *a, **k: False)
        _disable_buttons(mw)

        mgr.load_csv_file(str(csv_file))

        assert ("end", "加载失败", "error") in mw.feedback_calls, (
            f"失败出口必须收掉加载反馈，实际流水: {mw.feedback_calls}"
        )
        assert mw.load_btn.isEnabled() is True
        # 没有有效数据 → 分析按钮保持灰（点亮等于让用户去点必然再报错的入口）
        assert _status_buttons_enabled(mw) == [False] * len(_STATUS_BUTTONS)
        # _end_data_reload 已执行（兜底定时器就位）；双锁本身由它调度的
        # _post_reload_ui_refresh 在下一轮事件循环里放平，这里不重复断言
        assert mgr._safety_unlock_version == mw._data_version

    def test_failure_with_existing_data_reenables_analysis_buttons(
        self, host, dialogs, csv_file, monkeypatch
    ):
        """失败前已有有效数据：分析按钮必须恢复可用，用户还能继续看旧数据。"""
        mw, mgr = host
        old = _FakeLoader()
        mw.loader = old
        mw.data = object()
        monkeypatch.setattr(
            "src.ui.file_loader_manager.FastDataLoader.auto_detect",
            staticmethod(lambda path: _fake_fmt()),
        )
        monkeypatch.setattr(mgr, "_load_sync", lambda *a, **k: False)
        _disable_buttons(mw)

        mgr.load_csv_file(str(csv_file))

        assert _status_buttons_enabled(mw) == [True] * len(_STATUS_BUTTONS)
        assert ("end", "加载失败", "error") in mw.feedback_calls


class TestFailureKeepsLoadedData:
    def test_generic_failure_does_not_release_old_data(
        self, host, dialogs, csv_file, monkeypatch
    ):
        """P1-4 主诉：打开另一个文件失败，不得连带清掉当前数据。"""
        mw, mgr = host
        old = _FakeLoader()
        old_data = object()
        mw.loader = old
        mw.data = old_data
        mw.var_names = ["time", "speed"]

        def boom(path, *a, **k):
            raise RuntimeError("模拟 _load_file 内部失败")

        monkeypatch.setattr(mgr, "_load_file", boom)

        mgr.load_csv_file(str(csv_file))

        assert mw.loader is old, "失败不得释放旧 loader"
        assert old.closed is False, "失败不得 close() 旧 loader"
        assert old.released is False
        assert mw.data is old_data
        assert mw.var_names == ["time", "speed"]
        # 弹窗照旧要有，且失败出口把状态放平
        assert [t for t, _ in dialogs["critical"]] == ["加载错误"]
        assert ("end", "加载失败", "error") in mw.feedback_calls
        assert mw._is_loading_new_data is False
        assert mw.load_btn.isEnabled() is True

    def test_auto_detect_oserror_reports_unreadable_and_keeps_data(
        self, host, dialogs, csv_file, monkeypatch
    ):
        """检测阶段文件变得不可读：报「文件不可读」，旧数据与锁都不动。"""
        mw, mgr = host
        old = _FakeLoader()
        mw.loader = old
        mw.data = object()

        def unreadable(path):
            raise OSError(2, "No such file or directory")

        monkeypatch.setattr(
            "src.ui.file_loader_manager.FastDataLoader.auto_detect",
            staticmethod(unreadable),
        )
        _disable_buttons(mw)

        mgr.load_csv_file(str(csv_file))

        assert [t for t, _ in dialogs["critical"]] == ["文件不可读"]
        assert mw.loader is old
        assert old.closed is False
        assert mw._is_loading_new_data is False, "不得留下未释放的加载锁"
        # 这条路径压根没进 begin_load_feedback，不该冒出「加载失败」横幅
        assert mw.feedback_calls == []
        assert mw.load_btn.isEnabled() is True

    def test_memory_error_still_releases_old_data(
        self, host, dialogs, csv_file, monkeypatch
    ):
        """内存不足是唯一例外：不腾出内存连重试都做不了。"""
        mw, mgr = host
        old = _FakeLoader()
        mw.loader = old
        mw.data = object()

        def oom(path, *a, **k):
            raise MemoryError("无法分配内存")

        monkeypatch.setattr(mgr, "_load_file", oom)

        mgr.load_csv_file(str(csv_file))

        assert [t for t, _ in dialogs["critical"]] == ["内存不足"]
        assert old.closed is True
        assert mw.loader is None


class TestAsyncFailureRestoresUiState:
    def test_on_load_done_apply_failure_ends_feedback(
        self, host, dialogs, monkeypatch
    ):
        """后台加载完成但应用失败：与同步口径一致地收口。"""
        mw, mgr = host
        mw._progress = SimpleNamespace(close=lambda: None)
        monkeypatch.setattr(
            mgr,
            "_swap_loader",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("应用数据失败")),
        )
        post_actions = []
        monkeypatch.setattr(
            mgr, "_post_load_actions", lambda *a, **k: post_actions.append(1)
        )
        mgr._begin_data_reload()
        mgr._reload_in_progress = True
        mw.reload_btn.setEnabled(False)
        _disable_buttons(mw)

        mgr._on_load_done(object(), "/tmp/whatever.csv")

        assert ("end", "加载失败", "error") in mw.feedback_calls
        assert mw.load_btn.isEnabled() is True
        assert mw.reload_btn.isEnabled() is True
        assert mgr._reload_in_progress is False
        assert _status_buttons_enabled(mw) == [False] * len(_STATUS_BUTTONS)
        assert post_actions == [], "失败路径不得继续走加载后动作"
