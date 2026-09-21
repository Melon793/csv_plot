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
import threading

import pytest

from src.core.gc_guard import no_autogc
from src.data.loader import DataLoadThread
from tests.fixtures.data_factory import make_simple_rows, write_csv

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
