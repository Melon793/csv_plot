"""worker 线程内不得发生自动分代回收。

背景（实测钉死，不是推测）：CPython 的分代堆是**进程级**的，自动收集由「哪个线程
先撞上分配阈值」触发，扫的是整代垃圾 —— 与触发线程是否引用这些对象**无关**。所以
worker 线程不需要持有任何指向 GUI 的引用边，一次落在它身上的收集就足以把主线程创建
的 QObject 包装器拿到 worker 上析构（Qt 对象只能在属主线程销毁）。

两种实测落点：
* worker 持 Qt 锁等 GIL、主线程在 Shiboken 延迟删除队列里持 GIL 等同一把 Qt 锁
  → 整窗永久挂起（进度条一直转）；
* 主线程定时器链表留下悬垂节点 → 段错误固定在 ``QTimerInfoList::activateTimers``。

A/B 重复计数（同一必崩配方各 5 轮）：默认 GC 5/5 崩，全程 ``gc.disable()`` 0/5 崩。
本文件钉的就是这条不变量：**收集不得在 worker 线程上发生**。

与 ``test_load_worker_no_full_gc.py`` 的分工：那边钉「显式 gc.collect()」（打桩可拦），
这边钉「自动分代收集」（只有 ``gc.callbacks`` 能观测）。
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from shiboken6 import isValid

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QWidget

from src.core.gc_guard import no_autogc
from src.data.loader import DataLoadThread
from src.ui.file_loader_manager import FileLoaderManager, _LoaderSignalRelay
from tests.fixtures.data_factory import make_simple_rows, write_csv
from tests.fixtures.waits import flush_deferred_deletes

# 2 万个自引用 dict：每个都是只能靠分代收集释放的环，远超 gen0 默认阈值 700。
# 刻意**不释放**它们 —— 这样窗口退出时 gen0 计数必然处于「待发」状态，是最坏情况。
CYCLE_CHURN = 20_000


@pytest.fixture()
def collection_threads():
    """记录每次收集（含自动分代）的执行线程 id。

    ``gc.callbacks`` 是自动收集唯一可观测的挂钩点；打桩 ``gc.collect`` 拦不到它。
    """
    idents: list[int] = []

    def _cb(phase, _info):
        if phase == "start":
            idents.append(threading.get_ident())

    gc.callbacks.append(_cb)
    try:
        yield idents
    finally:
        gc.callbacks.remove(_cb)


def _churn():
    """制造只能被分代收集释放的环垃圾，并把 gen0 计数顶到远超阈值。"""
    junk = []
    for i in range(CYCLE_CHURN):
        d = {"i": i}
        d["self"] = d
        junk.append(d)


def _make_csv(tmp_path, name: str = "worker_autogc.csv"):
    return write_csv(
        tmp_path / name,
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(80),
    )


# ---------------------------------------------------------------------------
# 加载线程
# ---------------------------------------------------------------------------


class _ProbeLoadThread(DataLoadThread):
    """在真实加载路径里探针 GC 状态；垃圾在 _load() 内造，即落在防护窗口里。"""

    worker_ident: int | None = None
    gc_enabled_in_window: bool | None = None
    counts_at_entry: tuple | None = None
    counts_at_exit: tuple | None = None

    def run(self):
        self.worker_ident = threading.get_ident()
        super().run()

    def _load(self):
        self.gc_enabled_in_window = gc.isenabled()
        self.counts_at_entry = gc.get_count()
        _churn()
        super()._load()
        self.counts_at_exit = gc.get_count()


def test_load_worker_runs_no_collection_on_worker_thread(
    qtbot, tmp_path, collection_threads
):
    """异步加载全程：自动 GC 关着，且没有任何一次收集落在 worker 线程上。"""
    enabled_before = gc.isenabled()
    thread = _ProbeLoadThread(
        str(_make_csv(tmp_path)), desc_rows=0, sep=",", has_unit=True
    )

    with qtbot.waitSignal(thread.finished, timeout=30_000) as blocker:
        thread.start()

    assert thread.wait(5_000), "worker 未在等待时间内结束"
    loader = blocker.args[0]
    assert len(loader.var_names) > 0, "加载必须真的跑成功，否则用例是空过"

    assert thread.gc_enabled_in_window is False, "加载窗口内自动 GC 未被关闭"
    assert thread.counts_at_exit[0] >= gc.get_threshold()[0], (
        "前提不成立：worker 攒的环垃圾不足以触发一次 gen0 收集，用例是空过"
    )
    # 关闭期间只有 gen0 计数在涨（gen1/gen2 计数只在下层收集完成后才递增）。
    # 这条保证即使防护漏了一次收集，也只能是 gen0，扫不到长期存活的 GUI 堆。
    assert thread.counts_at_exit[1:] == thread.counts_at_entry[1:], (
        f"窗口内 gen1/gen2 计数发生了变化：{thread.counts_at_entry} → "
        f"{thread.counts_at_exit}"
    )

    assert thread.worker_ident != threading.get_ident(), "前提：worker 不应是主线程"
    assert thread.worker_ident not in collection_threads, (
        f"worker 线程上发生了 {collection_threads.count(thread.worker_ident)} 次收集："
        "它会把主线程 QObject 包装器拿到 worker 析构 → 整窗挂起或 activateTimers 段错误"
    )
    assert gc.isenabled() == enabled_before, "窗口退出后必须把自动 GC 恢复原状"


# ---------------------------------------------------------------------------
# 统计线程
# ---------------------------------------------------------------------------


def test_var_info_worker_runs_no_collection_on_worker_thread(
    qapp, monkeypatch, collection_threads
):
    """统计线程逐条任务的窗口同样不得让收集落在 worker 上。

    本线程与对话框同寿，所以窗口按「单条任务」而不是整个 ``run()`` 划分；
    这里连 shutdown 路径一并计入，覆盖任务结束后回到 cond.wait 的那段。
    """
    from src.ui.dialogs import variable_info_dialog as vid_mod

    seen: dict = {}
    done = threading.Event()

    def _fake_compute_stats(loader, var_name, cancel_cb):
        seen["ident"] = threading.get_ident()
        seen["gc_enabled"] = gc.isenabled()
        seen["counts_at_entry"] = gc.get_count()
        _churn()
        seen["counts_at_exit"] = gc.get_count()
        done.set()
        # 抛出去让 worker 自己构造 VarStats，避免用例耦合其字段
        raise RuntimeError("测试桩")

    monkeypatch.setattr(vid_mod.var_info, "compute_stats", _fake_compute_stats)

    enabled_before = gc.isenabled()
    worker = vid_mod.VarInfoWorker(None)
    try:
        worker.submit([("speed", lambda: object(), 0)])
        assert done.wait(10.0), "统计任务未在等待时间内执行"
    finally:
        worker.shutdown(3000)
        assert not worker.isRunning(), "统计线程未退出，用例会留下野线程"

    assert seen["gc_enabled"] is False, "任务窗口内自动 GC 未被关闭"
    assert seen["counts_at_exit"][0] >= gc.get_threshold()[0], "前提不成立：垃圾不够"
    assert seen["counts_at_exit"][1:] == seen["counts_at_entry"][1:]

    assert seen["ident"] != threading.get_ident(), "前提：worker 不应是主线程"
    assert seen["ident"] not in collection_threads, (
        f"统计线程上发生了 {collection_threads.count(seen['ident'])} 次收集"
    )
    assert gc.isenabled() == enabled_before


# ---------------------------------------------------------------------------
# 窗口本身
# ---------------------------------------------------------------------------


def test_nested_windows_restore_only_at_outermost_exit():
    """重叠窗口：内层退出不得提前打开 GC，外层 worker 还在跑。"""
    assert gc.isenabled(), "前提：用例开始时自动 GC 应为开启"
    with no_autogc():
        assert not gc.isenabled()
        with no_autogc():
            assert not gc.isenabled()
        assert not gc.isenabled(), "内层退出就打开 GC，外层窗口形同虚设"
    assert gc.isenabled()


def test_window_restores_gc_after_exception():
    with pytest.raises(RuntimeError, match="boom"):
        with no_autogc():
            assert not gc.isenabled()
            raise RuntimeError("boom")
    assert gc.isenabled(), "异常路径没把自动 GC 打开，之后整个进程都不再回收"


def test_window_does_not_reenable_gc_caller_disabled():
    """调用方本就关着 GC（如压测脚本）时，窗口退出后不能替它打开。"""
    gc.disable()
    try:
        with no_autogc():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_window_exit_disarms_primed_gen0_collection():
    """窗口退出时必须拆掉自己攒下的那次「待发」收集。

    ``gc.disable()`` 只挡住窗口内的收集；gen0 计数照样在涨。若退出时计数已越过
    阈值，**下一个分配对象的线程**就会立刻触发收集 —— 而那一刻多半还站在 worker
    上，防护等于漏了一次（实测：worker 退出前重新 enable，200 个主线程 QObject
    全部在 worker 上被析构）。``freeze`` + ``unfreeze`` 清零三代计数且不执行任何
    析构，等于把这次收集顺延给主线程。
    """
    with no_autogc():
        _churn()
        assert gc.get_count()[0] >= gc.get_threshold()[0]
    assert gc.get_count()[0] < gc.get_threshold()[0], (
        "窗口退出时 gen0 仍处于待发状态：下一次分配就会在 worker 线程上触发收集"
    )
    assert gc.isenabled()


# ---------------------------------------------------------------------------
# BLF 预扫线程
# ---------------------------------------------------------------------------


def test_blf_prescan_thread_runs_no_collection_on_worker_thread(
    qapp, tmp_path, monkeypatch, collection_threads
):
    """BLF 预扫线程同样不得让自动收集落在自己身上。

    这条路径此前是裸窗口：``prescan_blf_frames`` 每 4k 帧 ``list(islice(...))`` +
    ``Counter`` 累加，分配量足够触发 gen0（实测每轮全量 8 次落在 ``blf-prescan``
    上）。线程是 daemon 且关窗只置取消位、不 join，漏掉的分配会跨用例继续。
    """
    from src.ui.dialogs import can_db_selector as cds_mod

    seen: dict = {}

    def _fake_prescan(_path, should_cancel=None, error_counts=None):
        seen["ident"] = threading.get_ident()
        seen["gc_enabled"] = gc.isenabled()
        seen["counts_at_entry"] = gc.get_count()
        _churn()
        seen["counts_at_exit"] = gc.get_count()
        return {}

    monkeypatch.setattr(cds_mod, "prescan_blf_frames", _fake_prescan)

    blf = tmp_path / "log.blf"
    blf.write_bytes(b"\x00" * 2048)  # 只要求「存在且低于预扫上限」，内容不解析

    enabled_before = gc.isenabled()
    dlg = cds_mod.CanDatabaseSelectorDialog(data_file=str(blf))
    try:
        thread = dlg._prescan_thread
        assert thread is not None, "前提：BLF 必须真的启动预扫线程"
        thread.join(10.0)
        assert not thread.is_alive(), "预扫线程未在等待时间内结束"
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()
        flush_deferred_deletes()

    assert seen, "预扫线程没有执行到预扫函数，用例是空过"
    assert seen["gc_enabled"] is False, "预扫窗口内自动 GC 未被关闭"
    assert seen["counts_at_exit"][0] >= gc.get_threshold()[0], (
        "前提不成立：预扫攒的环垃圾不足以触发一次 gen0 收集，用例是空过"
    )
    assert seen["counts_at_exit"][1:] == seen["counts_at_entry"][1:]

    assert seen["ident"] != threading.get_ident(), "前提：worker 不应是主线程"
    assert seen["ident"] not in collection_threads, (
        f"预扫线程上发生了 {collection_threads.count(seen['ident'])} 次收集："
        "它会把主线程 QObject 包装器拿到 worker 析构 → activateTimers 段错误"
    )
    assert gc.isenabled() == enabled_before, "窗口退出后必须把自动 GC 恢复原状"


# ---------------------------------------------------------------------------
# Qt 退出与信号 receiver 生命周期
# ---------------------------------------------------------------------------


def test_shutdown_waits_until_load_thread_really_stops(qapp):
    """有界 wait 超时后不能继续关窗：运行中的 QThread 被销毁会直接终止进程。"""
    from src.ui.main_window import MainWindow

    class _Thread:
        def __init__(self):
            self.running = True
            self.cancel_requests = 0
            self.waits = []

        def isRunning(self):
            return self.running

        def request_cancel(self):
            self.cancel_requests += 1

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) >= 3:
                self.running = False
                return True
            return False  # 超时只表示继续轮询，不能继续关窗

    class _Logger:
        def info(self, *_args, **_kwargs):
            pass

        def debug(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    class _Manager:
        def __init__(self):
            self.closed = []
            self.released = False

        def _close_load_progress(self):
            self.closed.append("load")

        def _close_blf_progress(self):
            self.closed.append("blf")

        def _release_pending_blf_loader(self):
            self.released = True

    thread = _Thread()
    manager = _Manager()
    host = type(
        "Host", (),
        {
            "_thread": thread,
            "file_loader_manager": manager,
            "_logger": _Logger(),
        },
    )()

    MainWindow._shutdown_data_load_thread(host)

    assert thread.cancel_requests == 1
    assert thread.waits == [50, 50, 50], "每次只等 50ms，超时后继续等并处理事件"
    assert thread.running is False
    assert manager.closed == ["load", "blf"]
    assert manager.released is True


def test_load_file_refuses_destroyed_window_and_releases_pending(qapp):
    """统一加载入口必须拦住关窗态，不能弹框或起线程。"""
    class _Pending:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    host = QWidget()
    host._is_being_destroyed = True
    host.loader = None
    manager = FileLoaderManager(host)
    pending = _Pending()
    manager._blf_pending_loader = pending
    manager._blf_force_dialog = True
    manager._blf_preset_cfg = object()

    manager._load_file("Demo.blf")

    assert manager._blf_pending_loader is None
    assert manager._blf_force_dialog is False
    assert manager._blf_preset_cfg is None
    assert pending.closed is True
    host.deleteLater()
    flush_deferred_deletes()


def test_shutdown_blocks_pending_blf_retry_from_reentering_load(qapp, monkeypatch):
    """关窗轮询放行重试定时器时，不得重新进入加载入口。"""
    from src.ui.main_window import MainWindow

    class _Thread:
        def __init__(self):
            self.running = True
            self.timeouts = []

        def isRunning(self):
            return self.running

        def request_cancel(self):
            pass

        def wait(self, timeout=None):
            self.timeouts.append(timeout)
            if len(self.timeouts) >= 2:
                self.running = False
                return True
            return False

    class _Logger:
        def info(self, *_args, **_kwargs):
            pass

        def debug(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    host = QWidget()
    host._is_being_destroyed = True
    host._progress = None
    host.loader = None
    host._logger = _Logger()
    thread = _Thread()
    host._thread = thread
    manager = FileLoaderManager(host)
    host.file_loader_manager = manager
    entered = []
    monkeypatch.setattr(manager, "load_csv_file", lambda path: entered.append(path))

    retry = QTimer(host)
    retry.setSingleShot(True)
    retry.timeout.connect(lambda: manager._restart_blf_import("Demo.blf"))
    retry.start(0)

    MainWindow._shutdown_data_load_thread(host)

    assert entered == [], "销毁态下 BLF 重试不得重新进入加载入口"
    assert thread.timeouts == [50, 50]
    host.deleteLater()
    flush_deferred_deletes()


def test_qt_widgets_exit_cleanly_in_subprocess():
    """pytest passed 不等于解释器退出成功；退出码与 stderr 都必须单独判定。"""
    code = """
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QMainWindow
from src.ui.dialogs.can_db_selector import CanDatabaseSelectorDialog

app = QApplication([])
host = QMainWindow()
child = CanDatabaseSelectorDialog(host)
standalone = CanDatabaseSelectorDialog()
for widget in (child, standalone, host):
    widget.close()
    widget.deleteLater()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
app.processEvents()
assert not QApplication.topLevelWidgets()
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Qt 子进程退出码 {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "shared QObject was deleted directly" not in result.stderr


class _LoaderSignalSender(QObject):
    finished = Signal(object)


def test_loader_signal_relay_disconnects_with_main_window(qapp):
    calls = []

    class Manager:
        def _on_load_done(self, loader, file_path, load_version, is_reload):
            calls.append((loader, file_path, load_version, is_reload))

    host = QWidget()
    manager = Manager()
    relay = _LoaderSignalRelay(host, manager, "Demo.csv", 7, False, blf=False)
    sender = _LoaderSignalSender()
    sender.finished.connect(relay.finished)

    host.close()
    host.deleteLater()
    flush_deferred_deletes()
    assert not isValid(relay), "主窗口销毁后 relay 必须随之销毁并断开信号"

    sender.finished.emit(object())
    assert calls == [], "receiver 已销毁后，迟到的线程信号不得再调用 manager"
    sender.deleteLater()
    flush_deferred_deletes()


def test_loader_signal_relay_releases_after_terminal_signal(qapp):
    calls = []

    class Manager:
        def _on_load_done(self, loader, file_path, load_version, is_reload):
            calls.append((loader, file_path, load_version, is_reload))

    host = QWidget()
    relay = _LoaderSignalRelay(host, Manager(), "Demo.csv", 7, True, blf=False)
    loader = object()
    relay.finished(loader)
    assert calls == [(loader, "Demo.csv", 7, True)]
    flush_deferred_deletes()
    assert not isValid(relay), "terminal 信号后 relay 不得在主窗口下累积"
    host.deleteLater()
    flush_deferred_deletes()


def test_loader_signal_relay_drops_done_while_window_closing(qapp):
    calls = []
    discarded = []

    class Manager:
        def _on_load_done(self, loader, file_path, load_version, is_reload):
            calls.append((loader, file_path, load_version, is_reload))

        def _discard_loader(self, loader):
            discarded.append(loader)

    host = QWidget()
    host._is_being_destroyed = True
    relay = _LoaderSignalRelay(host, Manager(), "Demo.csv", 7, False, blf=False)
    loader = object()
    relay.finished(loader)

    assert calls == [], "关窗等待期间处理事件，不得把迟到结果应用到将销毁的窗口"
    assert discarded == [loader], "迟到 loader 必须释放，不能随窗口销毁前留资源"
    flush_deferred_deletes()
    assert not isValid(relay)
    host.close()
    host.deleteLater()
    flush_deferred_deletes()
