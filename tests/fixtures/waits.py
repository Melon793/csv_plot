"""测试等待与事件循环工具（收敛自 tests/ 里 12 份逐字重复的本地 helper）。

为什么上收：
- ``pump(ms)`` 固定钟表等待曾在 4 个文件里各写一份，改造时容易只改一处；
- ``flush_deferred_deletes`` 有 5 份变体（有的收 qapp、有的收 widget），语义一致；
- 等待改造（把"等表走字"换成条件等待/确定性排空）需要一个单点维护的位置。

**选型顺序（新增/改造用例请照此）**：
1. ``wait_until(条件)`` —— 有可观测条件时一律用它；
2. ``settle()`` / ``flush_deferred_deletes()`` —— 同步调用之后补事件处理，
   不睡钟表；
3. ``pump(ms)`` —— 等一个没有可观测条件的延迟回调时才用，且留 3× 余量。
"""

from __future__ import annotations

import time

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication


def pump(ms: int = 50) -> None:
    """固定钟表等待：驱动事件循环 ``ms`` 毫秒，让 QTimer.singleShot 的延后回调落地。

    历史实现，语义如字面——**不等条件，只等钟表**。改造用例时优先换成
    ``wait_until()``/``settle()``。
    """
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.002)


def settle(rounds: int = 3) -> None:
    """确定性排空事件队列：不睡钟表，只把**已在队列里的**事件处理干净。

    适合"同步调用之后补一次事件处理"的场景（几何生效、resize/scroll
    事件投递），替代 `pump(50)` 这类"等表走字"。多轮是为了让回调里再
    投递的事件也在本轮落地。
    """
    for _ in range(rounds):
        QApplication.processEvents()


def flush_deferred_deletes() -> None:
    """让 ``deleteLater()`` 真正落地。

    单用 ``deleteLater()`` 在测试里**不会生效**：Qt 把 DeferredDelete 事件
    推迟到"回到调用时的或更外层事件循环"才处理，而测试从不调 ``app.exec()``，
    因此 ``processEvents()`` 不会消化它（实测子对象仍存活）。必须显式
    ``sendPostedEvents(None, DeferredDelete)``。
    """
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def force_delete(widget) -> None:
    """立即销毁 widget 及其子对象的 C++ 部分，保留 Python 包装器。

    留下的包装器正对应线上"引用还在、C++ 对象已死"的形态，用于验证
    存活判定分支。
    """
    widget.deleteLater()
    flush_deferred_deletes()


def wait_until(predicate, timeout_s: float = 8.0, interval_s: float = 0.01) -> bool:
    """条件等待：泵事件 + 轮询直到 ``predicate()`` 为真。

    返回是否在超时前满足——调用方决定 ``assert ok, "..."`` 还是继续，
    避免把"失败原因"藏在 helper 里。
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(interval_s)
    return False
