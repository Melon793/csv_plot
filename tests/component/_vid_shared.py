"""变量信息窗口组件测试的共享替身（拆分产物，勿直接收集）。

由 test_variable_info_dialog.py 按职责拆成三个文件后，被两处以上**调用**的
替身留在这里：假主窗口与等待 helper 三件套。夹具（env / page / mdf_env …）
走的是 tests/component/conftest.py —— 夹具只能靠 conftest 免 import 共享。

拆分只搬代码，未改任何断言。
"""



from PySide6.QtWidgets import (
    QMainWindow,
)

from tests.fixtures.waits import (
    wait_until,
)

# ---------------------------------------------------------------------------
# 测试替身与辅助
# ---------------------------------------------------------------------------


class FakeMainWindow(QMainWindow):
    """MainWindow 替身：只提供对话框实际读取的四个属性。

    对话框通过 ``hasattr(window, "loader")`` 判定宿主，因此 loader 属性
    必须存在（即使为 None）。
    """

    def __init__(self, loader=None):
        super().__init__()
        self.loader = loader
        self.var_stats_cache = {}
        self.var_info_geometry = None
        self._data_version = 0


def _wait_stats(dlg, *names, require_computed: bool = False, timeout_s: float = 8.0):
    """等到这些页的统计**回填到页面**（内容条件，不睡钟表）。

    为什么不再拿"worker 队列空"当判据：``VarInfoWorker._process_job`` 先
    ``_finish_current()``（清 ``_current``、推进进度信号）再 ``item_ready.emit()``，
    worker 侧"空闲"与主线程拿到结果之间天然存在窗口，旧实现的固定 pump 只是赌
    这个窗口比它短。这里等的是用例真正要断言的观测量。

    Args:
        require_computed: 再要求算成功（``computed=True``）。默认只要求"有终态"
            —— 非数值列/全空列同样会给出带 error 的终态，那是合法结果。
    """

    def ready(name: str) -> bool:
        page = dlg._pages.get(name)
        stats = None if page is None else page.stats
        if stats is None:
            return False
        return stats.computed if require_computed else True

    return wait_until(lambda: all(ready(n) for n in names), timeout_s)


def _wait_delivered(dlg, *names, timeout_s: float = 8.0):
    """等到这些变量的在途结果**已回到主线程并处理完**。

    判据取 ``_revalidating`` 去重标记：``_on_stats_ready`` 一进来就 discard，
    随后（同一个 slot 调用内）才写缓存/回填页面，而这个回调在主线程上执行期间
    测试的轮询插不进去，所以"标记已释放"等价于"这次结果已经落地"。
    与 ``_wait_stats`` 的分工：命中缓存的页显示的是现成的旧值，只能靠标记判断
    再验证是否已经回来。
    """
    return wait_until(lambda: all(n not in dlg._revalidating for n in names), timeout_s)


def _wait_worker_idle(dlg, timeout_s: float = 8.0):
    """等到 worker 队列排空且没有正在运行的任务。

    只在**没有页面可见结果**可等时使用（例：任务被取消，worker 只走跳过分支、
    不发 item_ready）。要等"结果已回填"请用 ``_wait_stats``。
    """
    return wait_until(
        lambda: dlg.worker.queue_size() == 0 and dlg.worker._current is None,
        timeout_s,
    )
