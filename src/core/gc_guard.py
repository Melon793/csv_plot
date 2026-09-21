"""worker 线程内屏蔽自动分代回收。

CPython 的自动 GC 在「哪个线程先撞上分配阈值」就在哪个线程执行，回收出来的
垃圾也在该线程析构。worker 线程跑 pandas 会产生海量分配，一旦这次收集判定为
垃圾的引用环里含主线程创建的 QObject 包装器，Shiboken 就在 worker 上执行
~QObject —— 而 Qt 对象只能在属主线程销毁。实测两种落点：

* 主线程 Shiboken 延迟删除队列与 worker 锁序倒置 → 整窗永久挂起；
* 主线程定时器链表留下悬垂节点 → ``QTimerInfoList::activateTimers`` 段错误。

显式 ``gc.collect()`` 已由各调用点自行避开（``FastDataLoader(allow_gc=False)``），
这里堵的是打桩拦不到的自动分代路径。``gc.disable()`` 是进程级开关，窗口内任何
线程都不会再触发自动收集；代价是窗口内的引用环垃圾要等窗口结束后由主线程回收
——窗口按「一次加载 / 一条统计任务」划分，长度有界，且 numpy/pandas 的临时对象
绝大多数靠引用计数即时释放，实际增量很小。

因为开关是进程级的，用引用计数包住可能重叠的 worker 窗口，最后一个退出时才恢复。
"""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager

_lock = threading.Lock()
_depth = 0
_restore_enabled = True


@contextmanager
def no_autogc():
    """关闭自动分代回收，退出时按嵌套深度恢复。

    窗口内**不要**调用 ``gc.collect()``：显式回收不受 ``disable`` 影响，照样会在
    当前线程析构主线程的 QObject 包装器，等于把本模块要防的事又做一遍。
    """
    global _depth, _restore_enabled
    with _lock:
        if _depth == 0:
            # 调用方本就关着 GC（如压测脚本）时，退出后不能替它打开
            _restore_enabled = gc.isenabled()
            if _restore_enabled:
                gc.disable()
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                _rearm()


def _rearm():
    """恢复自动回收，并顺手拆掉窗口攒下的那次"待发"收集。调用方须持有 ``_lock``。

    关闭期间只有 gen0 计数在涨（gen1/gen2 计数只在下层收集完成后才递增），所以
    窗口退出时若 gen0 已越过阈值，**下一个分配对象的线程**就会立刻触发一次收集
    ——而那一刻多半还站在 worker 线程上，防护等于漏了一次。``freeze`` +
    ``unfreeze`` 把三代计数清零且不执行任何析构（实测 40 万对象约 1 ms），
    相当于把这次收集顺延给主线程。
    """
    if gc.get_count()[0] >= gc.get_threshold()[0]:
        gc.freeze()
        gc.unfreeze()
    if _restore_enabled:
        gc.enable()
