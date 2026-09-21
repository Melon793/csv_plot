"""异步加载 worker 内不得发生全量 GC。

背景：`DataLoadThread` 在 worker 线程构造 `FastDataLoader`。GC 在哪个线程执行，
就在哪个线程析构对象——worker 里全量回收会把主线程创建的 QObject 包装器拿到
worker 去销毁，worker 持 Qt 锁等 GIL、主线程持 GIL 等同一把 Qt 锁 → 整窗死锁
（ApplicationModal 加载框永不关闭）。

本用例只钉「显式 gc.collect()」：自动分代收集不走 Python 层的 gc.collect，
打桩拦不到，也正因如此这里不假装覆盖它。
"""

import gc
import threading

import pytest

from src.data.loader import DataLoadThread, FastDataLoader
from tests.fixtures.data_factory import make_simple_rows, write_csv


class _IdentRecordingThread(DataLoadThread):
    """run() 在 worker 线程执行，借此记录 worker 线程 id。"""

    worker_ident: int | None = None

    def run(self):
        self.worker_ident = threading.get_ident()
        super().run()


@pytest.fixture()
def collect_calls(monkeypatch):
    """记录每次 gc.collect 的调用线程 id（自动分代收集不经过此处）。"""
    idents: list[int] = []
    real_collect = gc.collect

    def spy(*args, **kwargs):
        idents.append(threading.get_ident())
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(gc, "collect", spy)
    return idents


def _make_csv(tmp_path, name: str = "worker_gc.csv"):
    return write_csv(
        tmp_path / name,
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(80),
    )


def test_worker_thread_load_does_no_full_gc(qtbot, tmp_path, collect_calls):
    """异步路径：加载成功，且 collect 一次都没落在 worker 线程上。"""
    csv = _make_csv(tmp_path)
    thread = _IdentRecordingThread(str(csv), desc_rows=0, sep=",", has_unit=True)

    with qtbot.waitSignal(thread.finished, timeout=30_000) as blocker:
        thread.start()

    assert thread.wait(5_000), "worker 未在等待时间内结束"
    worker_ident = thread.worker_ident
    assert worker_ident is not None, "run() 未在 worker 线程执行"
    assert worker_ident != threading.get_ident(), "前提：worker 与主线程不应是同一线程"

    loader = blocker.args[0]
    assert isinstance(loader, FastDataLoader), f"finished 未带 loader: {loader!r}"
    assert len(loader.var_names) > 0, "异步加载必须真的跑成功，否则用例是空过"

    worker_collects = [ident for ident in collect_calls if ident == worker_ident]
    assert worker_collects == [], (
        f"worker 线程内发生了 {len(worker_collects)} 次 gc.collect()：它会析构主线程的 "
        "QObject 包装器，与 Shiboken 主线程延迟删除互锁 → 整窗死锁"
    )


def test_sync_loader_keeps_full_gc(tmp_path, collect_calls):
    """主线程同步路径保持原行为，防止开关把内存回收一并关掉。"""
    main_ident = threading.get_ident()
    loader = FastDataLoader(str(_make_csv(tmp_path)), has_unit=True, sep=",")

    assert len(loader.var_names) > 0
    assert [i for i in collect_calls if i == main_ident], "同步路径应仍做全量 GC"


def test_allow_gc_false_suppresses_full_gc(tmp_path, collect_calls):
    """开关本身：allow_gc=False 时构造过程不产生任何 gc.collect。"""
    main_ident = threading.get_ident()
    loader = FastDataLoader(
        str(_make_csv(tmp_path)), has_unit=True, sep=",", allow_gc=False
    )

    assert len(loader.var_names) > 0
    assert [i for i in collect_calls if i == main_ident] == [], (
        "allow_gc=False 下不应有 gc.collect()"
    )
