"""变量信息窗口：单例、标签页生命周期与进度语义。

由 test_variable_info_dialog.py 拆分而来（只搬运，未改任何断言）：
单例累积与标签栏可见性、关闭取消在途任务、状态栏自愈、关窗等同「关闭全部」、
reload 钩子。
"""

import threading

import pytest

from PySide6.QtWidgets import (
    QMessageBox,
)
from shiboken6 import isValid

from tests.fixtures.waits import (
    force_delete,
    flush_deferred_deletes,
    settle,
    wait_until,
)
from src.data import var_info
from src.data.loader import FastDataLoader
from src.ui.dialogs import variable_info_dialog as vid_mod
from src.ui.dialogs.variable_info_dialog import (
    VariableInfoDialog,
)

from tests.component._vid_shared import FakeMainWindow, _wait_stats, _wait_delivered, _wait_worker_idle

def fake_msgbox(asked: list, answer):
    """返回一个替身 QMessageBox：记录 question 的调用并给出固定答案。

    刻意替换模块级名字而不是 patch Qt 类属性：PySide6 的静态方法挂在
    C++ 类型上，monkeypatch 未必生效，且会污染同进程内的其它测试。
    """

    class _Fake:
        StandardButton = QMessageBox.StandardButton

        @staticmethod
        def question(*args, **kwargs):
            asked.append(args)
            return answer

    return _Fake


# ---------------------------------------------------------------------------
# 单例与标签页
# ---------------------------------------------------------------------------


class TestSingletonAndTabs:
    def test_popup_returns_singleton_and_accumulates(self, env):
        first = VariableInfoDialog.popup(["speed"], parent=env.mw)
        second = VariableInfoDialog.popup(["load"], parent=env.mw)

        assert first is not None
        assert first is second, "popup 必须复用同一实例而非新建"
        assert VariableInfoDialog._instance is first
        assert first.tabs.count() == 2, "新变量应追加为标签页"
        assert set(first._pages) == {"speed", "load"}

    def test_popup_existing_variable_activates_without_duplicating(self, env):
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert dlg.tabs.count() == 2

        again = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert again is dlg
        assert dlg.tabs.count() == 2, "已存在的变量不得重复建页"
        assert dlg.tabs.currentIndex() == dlg.tabs.indexOf(dlg._pages["speed"])

    def test_popup_with_empty_names_returns_none(self, env):
        assert VariableInfoDialog.popup([], parent=env.mw) is None
        assert VariableInfoDialog.popup(None, parent=env.mw) is None
        assert VariableInfoDialog._instance is None

    def test_single_tab_hides_tab_bar(self, env):
        """只剩一个标签页时隐藏标签栏，视觉上退化为单变量窗口。"""
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert dlg.tabs.count() == 1
        assert dlg.tabs.tabBar().isVisibleTo(dlg.tabs) is False

    def test_multi_tab_shows_tab_bar(self, env):
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        dlg.add_variables(["load"])
        assert dlg.tabs.count() == 2
        assert dlg.tabs.tabBar().isVisibleTo(dlg.tabs) is True

    def test_tab_title_carries_unit(self, env):
        """标题带单位，便于区分同名不同单位的变量。"""
        dlg = VariableInfoDialog.popup(["speed", "note"], parent=env.mw)
        assert dlg.tabs.tabText(dlg.tabs.indexOf(dlg._pages["speed"])) == "speed [km/h]"
        # 单位为 "-" 时不追加方括号
        assert dlg.tabs.tabText(dlg.tabs.indexOf(dlg._pages["note"])) == "note"

    def test_popup_rebuilds_after_owner_destroyed(self, env):
        """回归防护：主窗口销毁后，单例必须能自愈而不是崩溃。

        对话框的 parent 是主窗口，主窗口销毁会连带销毁其 C++ 对象，
        但类属性 ``_instance`` 仍持有 Python 包装器。旧实现在 popup 里
        直接调 ``cls._instance._update_owner(parent)``，会抛
        ``RuntimeError: Internal C++ object already deleted``。

        销毁前先 shutdown_worker，对应真实时序：MainWindow.closeEvent →
        _shutdown_var_info_worker() → 窗口销毁。不停线程就销毁宿主会
        让 Qt 直接 abort（"QThread: Destroyed while thread is still
        running"）—— 这正是主窗口必须兜底终止线程的原因。
        """
        owner = FakeMainWindow(env.loader_a)
        first = VariableInfoDialog.popup(["speed"], parent=owner)
        assert VariableInfoDialog._instance is first

        first.shutdown_worker()
        force_delete(owner)
        assert VariableInfoDialog._live_instance() is None, (
            "已销毁的单例应被 _live_instance 识别并回收"
        )
        assert VariableInfoDialog._instance is None

        # 重新指定宿主后应能干净地重建，且不得沿用旧页面
        second = VariableInfoDialog.popup(["load"], parent=env.mw)
        assert second is not None
        assert second is not first
        assert second.tabs.count() == 1
        assert set(second._pages) == {"load"}

    def test_class_hooks_tolerate_destroyed_instance(self, env):
        """reload 钩子在主窗口已销毁时不得抛异常（它们在主流程里被调用）。"""
        owner = FakeMainWindow(env.loader_a)
        dlg = VariableInfoDialog.popup(["speed"], parent=owner)
        dlg.shutdown_worker()
        force_delete(owner)

        VariableInfoDialog.on_loader_released()
        VariableInfoDialog.refresh_after_reload(env.loader_a)
        VariableInfoDialog.reset_for_tests()
        assert VariableInfoDialog._instance is None

    def test_shutdown_worker_is_idempotent(self, env):
        """兜底入口必须可重复调用，且调完线程确实停了。

        背景约束（无法用断言表达，因为进程会直接 abort，pytest 收不到
        失败）：线程运行中销毁宿主会触发 "QThread: Destroyed while
        thread is still running"。因此主窗口的 closeEvent **必须**调
        shutdown_worker，真实保障见 MainWindow._shutdown_var_info_worker。
        本测试只验证该入口幂等且有效，不真的去销毁宿主。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed")

        dlg.shutdown_worker()
        dlg.shutdown_worker()
        assert dlg.worker.isRunning() is False

    def test_tabs_truncated_at_max_tabs(self, env, many_loader, monkeypatch):
        """超出 VAR_INFO_MAX_TABS 的部分必须被截断并提示。

        真实误操作场景：在变量列表里全选后右键（990 通道的 .dat 全选
        即 990 个标签页），每页都要建 QTreeWidget 并渲染十几个分组，
        不加限制会让 UI 直接卡死。
        """
        loader, names = many_loader
        monkeypatch.setattr(vid_mod, "VAR_INFO_MAX_TABS", 3)
        env.mw.loader = loader

        dlg = VariableInfoDialog.popup(names[:6], parent=env.mw)

        assert dlg.tabs.count() == 3, "应截断到上限"
        assert list(dlg._pages) == names[:3], "应保留提交顺序中靠前的"
        assert "上限 3" in dlg.status_label.text()
        assert "忽略 3 个" in dlg.status_label.text()

    def test_truncation_accounts_for_existing_tabs(self, env, many_loader, monkeypatch):
        """额度按「剩余容量」算：已存在的标签页不占新增额度。"""
        loader, names = many_loader
        monkeypatch.setattr(vid_mod, "VAR_INFO_MAX_TABS", 3)
        env.mw.loader = loader

        dlg = VariableInfoDialog.popup([names[0]], parent=env.mw)
        assert dlg.tabs.count() == 1

        # 已有 1 页 → 容量仅剩 2；本次提交里 names[0] 已存在不占额度，
        # 真正新增的是 names[1:4] 共 3 个，故应截掉 1 个
        dlg.add_variables(names[:4])
        settle()

        assert dlg.tabs.count() == 3
        assert set(dlg._pages) == set(names[:3])
        assert "忽略 1 个" in dlg.status_label.text()
        # 激活的是提交顺序中最后一个**成功处理**的页（names[2]）；
        # names[3] 被截断，不得影响激活目标
        assert dlg.tabs.currentIndex() == dlg.tabs.indexOf(dlg._pages[names[2]])

    def test_activation_follows_submission_order(self, env, many_loader):
        """锁定“激活最后提到的那个”语义：已存在页也参与。

        重构 add_variables 引入截断时很容易把循环拆成“先处理已存在、
        再处理新增”，那会让新增页总是赢，与提交顺序无关。
        """
        loader, names = many_loader
        env.mw.loader = loader
        v1, v2, v3 = names[0], names[1], names[2]

        dlg = VariableInfoDialog.popup([v1, v2], parent=env.mw)
        assert dlg.tabs.count() == 2

        # 提交顺序里 v1 在最后，已存在的 v1 应被激活
        dlg.add_variables([v3, v1])
        settle()

        assert dlg.tabs.count() == 3
        assert dlg.tabs.currentIndex() == dlg.tabs.indexOf(dlg._pages[v1])

    def test_duplicate_names_do_not_consume_quota(self, env, many_loader, monkeypatch):
        """同一批里的重复名字只建一页，不得白占额度。"""
        loader, names = many_loader
        monkeypatch.setattr(vid_mod, "VAR_INFO_MAX_TABS", 2)
        env.mw.loader = loader
        v1, v2 = names[0], names[1]

        dlg = VariableInfoDialog.popup([v1, v1, v2, v1], parent=env.mw)

        assert dlg.tabs.count() == 2
        assert set(dlg._pages) == {v1, v2}
        assert "上限" not in dlg.status_label.text(), "未超额时不应报截断"

    def test_notice_survives_async_progress(self, env, many_loader, monkeypatch):
        """回归防护：一次性提示不得被异步进度信号覆盖。

        实测缺陷：旧实现提示与进度共用 status_label，提交 2 个统计
        任务后，progress 信号在数十毫秒内就把“标签页已达上限”抹掉，
        用户根本看不到自己被截断了。现在二者分开保存并拼接显示。
        """
        loader, names = many_loader
        monkeypatch.setattr(vid_mod, "VAR_INFO_MAX_TABS", 2)
        env.mw.loader = loader

        dlg = VariableInfoDialog.popup(names[:5], parent=env.mw)
        # 等两页回填完成：进度信号由 worker 先于 item_ready 发出，结果
        # 已落到页面即说明含终态 (0,0) 在内的进度文案都已处理过
        assert _wait_stats(dlg, names[0], names[1])

        text = dlg.status_label.text()
        assert "上限 2" in text, f"提示被进度覆盖: {text!r}"
        assert "忽略 3 个" in text
        assert "统计中" not in text, "任务完成后进度文本应已清除"

    def test_notice_cleared_on_tab_switch(self, env, many_loader, monkeypatch):
        """用户切换标签页即表示已读，一次性提示应退场。"""
        loader, names = many_loader
        monkeypatch.setattr(vid_mod, "VAR_INFO_MAX_TABS", 2)
        env.mw.loader = loader

        dlg = VariableInfoDialog.popup(names[:5], parent=env.mw)
        assert "上限" in dlg.status_label.text()
        assert dlg.tabs.count() == 2

        dlg.tabs.setCurrentIndex(0)
        settle()

        assert "上限" not in dlg.status_label.text()

    def test_add_variables_without_loader_notifies(self, env):
        env.mw.loader = None
        # 必须持有 owner 的 Python 引用：否则临时对象立即被 GC，
        # 连带销毁子对话框的 C++ 对象，后续 deleteLater 会抛 RuntimeError
        owner = FakeMainWindow(None)
        dlg = VariableInfoDialog(owner)
        try:
            VariableInfoDialog._instance = dlg
            dlg.add_variables(["speed"])
            assert dlg.tabs.count() == 0
            assert "尚未加载" in dlg.status_label.text()
        finally:
            VariableInfoDialog._instance = None
            dlg.shutdown_worker()
            dlg.deleteLater()
            flush_deferred_deletes()


# ---------------------------------------------------------------------------
# 标签页关闭与线程生命周期
# ---------------------------------------------------------------------------


class TestTabCloseAndWorker:
    def test_close_tab_removes_page_and_cancels_job(self, env):
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert dlg.tabs.count() == 2

        cancelled: list = []
        original_cancel = dlg.worker.cancel
        dlg.worker.cancel = lambda name: (cancelled.append(name), original_cancel(name))[1]
        try:
            dlg._on_tab_close(dlg.tabs.indexOf(dlg._pages["speed"]))
        finally:
            dlg.worker.cancel = original_cancel

        assert cancelled == ["speed"], "关页必须先取消其后台任务"
        assert "speed" not in dlg._pages
        assert dlg.tabs.count() == 1
        assert dlg.tabs.tabBar().isVisibleTo(dlg.tabs) is False, "剩一页应隐藏标签栏"

    def test_close_last_tab_hides_dialog(self, env):
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert dlg.isVisible()

        dlg._on_tab_close(0)

        assert dlg.tabs.count() == 0
        assert dlg._pages == {}
        assert dlg.isVisible() is False, "无标签页时窗口应自动隐藏"
        # 单例本身保留，下次右键可立即复用
        assert VariableInfoDialog._instance is dlg

    def test_close_all_tabs_cancels_everything(self, env):
        dlg = VariableInfoDialog.popup(["speed", "load", "time"], parent=env.mw)
        dlg._close_all_tabs()

        assert dlg.tabs.count() == 0
        assert dlg._pages == {}
        assert dlg.worker.queue_size() == 0
        assert dlg.isVisible() is False

    def test_on_loader_released_drains_queue(self, env):
        """旧 loader 释放**之前**必须取消任务，否则会读一个即将 close 的句柄。"""
        dlg = VariableInfoDialog.popup(["speed", "load", "time"], parent=env.mw)
        VariableInfoDialog.on_loader_released()

        assert dlg.worker.queue_size() == 0
        # cancel_all 后只允许保留「正在运行那一条」的取消标记
        assert dlg.worker._cancel in (set(), {dlg.worker._current})

    def test_resubmit_after_cancel_all_is_not_skipped(self, env):
        """回归防护：残留的取消标记会让**下一次**同名 submit 被永久跳过。

        cancel_all 会把正在运行的那条重新加入 ``_cancel``，而 run() 只在
        「出队即跳过」分支里 discard；若任务是在 compute_stats 内部被取消
        的，标记就会残留。submit 必须负责清除它。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        dlg.worker.cancel_all()
        dlg.worker._cancel.add("speed")  # 强制制造残留标记
        assert _wait_worker_idle(dlg), "被取消的任务未被 worker 处理完"

        dlg.recompute("speed")
        assert _wait_delivered(dlg, "speed")

        stats = dlg._pages["speed"].stats
        assert stats is not None
        assert stats.computed is True, "残留取消标记不得让重算被跳过"
        assert stats.error != "已取消"

    def test_on_loader_released_without_instance_is_noop(self):
        VariableInfoDialog.reset_for_tests()
        assert VariableInfoDialog._instance is None
        VariableInfoDialog.on_loader_released()  # 不得抛异常
        VariableInfoDialog.refresh_after_reload(None)

    def test_shutdown_stops_thread(self, env):
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed")
        dlg.shutdown_worker()

        assert dlg.worker.isRunning() is False
        # shutdown 后 submit 必须被忽略，否则线程会被重新拉起
        dlg.worker.submit([("speed", lambda: env.loader_a, 0)])
        settle()
        assert dlg.worker.isRunning() is False
        assert dlg.worker.queue_size() == 0


# ---------------------------------------------------------------------------
# 进度分母的「本波」语义
# ---------------------------------------------------------------------------


class TestProgressWaveSemantics:
    """进度分母必须是「本波」语义，且取消后状态栏必须自愈。

    实测两个用户可见缺陷：

    1. 连续提交两批各 3 个变量，第二批状态栏显示「统计中 4/6…」—— 旧实现
       维护进程内单调递增的 ``_done_count``，分母也只增不减，于是第二批的
       分子分母都是两批之和，而用户眼里只有 3 个
    2. 关掉一个**还在排队**的标签页后，状态栏永久停在「统计中 2/3…」——
       被 cancel 摘除的任务永远不会走到 ``_advance_progress``，旧实现既不扣
       分母也不补发信号，队列排空后进度永远追不平

    前三条直接在 worker 层面验证：跳批与取消的时序需要精确控制（哪一条正在
    跑、哪一条还在队列里），走对话框路径会被 add_variables 的渲染耗时搞浑。
    用户可见面由最后一条覆盖。
    """

    @staticmethod
    def _worker_with_recorder():
        w = vid_mod.VarInfoWorker(None)
        seen: list = []
        # 普通 Python 可调用对象在 PySide6 里仍由「连接时所在线程」的一个内部
        # 接收者承接（实测：emit 线程 join 后 seen 仍为空，processEvents 之后
        # 才追加到主线程），因此记录永远不会落在 worker 线程内 —— 读 seen 前
        # 必须先泵事件循环，_drained 的终态判据正是为此存在
        w.progress.connect(lambda d, t: seen.append((d, t)))
        return w, seen

    @staticmethod
    def _drained(w, seen, timeout_s: float = 10.0) -> bool:
        """等到本波彻底排空**且终态信号已落地**。

        不能只看 ``_pending_total == 0``：它是在 emit **之前**被复位的，主线程
        可能在信号抵达前就判定排空，导致断言读到不完整的 seen。
        """
        return wait_until(
            lambda: bool(seen)
            and seen[-1] == (0, 0)
            and w.queue_size() == 0
            and w._current is None,
            timeout_s,
        )

    @staticmethod
    def _wait_current(w, name, timeout_s: float = 10.0) -> bool:
        return wait_until(lambda: w._current == name, timeout_s)

    def test_second_wave_denominator_restarts(self, qapp, monkeypatch):
        gate = threading.Event()
        monkeypatch.setattr(
            vid_mod.var_info,
            "compute_stats",
            lambda ld, name, sc=None: (
                gate.wait(10.0), var_info.VarStats(computed=True))[1],
        )
        w, seen = self._worker_with_recorder()
        try:
            # 只要 ref() 非 None 即可：compute_stats 已被替身接管，不碰 loader
            def ref():
                return object()

            w.submit([("a", ref, 0), ("b", ref, 0), ("c", ref, 0)])
            assert self._wait_current(w, "a"), "第一条应进入运行态"
            gate.set()
            assert self._drained(w, seen), f"第一波未排空: {seen}"
            first = list(seen)
            seen.clear()
            gate.clear()

            w.submit([("d", ref, 0), ("e", ref, 0), ("f", ref, 0)])
            assert self._wait_current(w, "d"), "第二批第一条应进入运行态"
            gate.set()
            assert self._drained(w, seen), f"第二波未排空: {seen}"

            assert first == [(1, 3), (2, 3), (0, 0)]
            assert seen == [(1, 3), (2, 3), (0, 0)], (
                f"第二批分母必须是本批的 3，而非累加的 6: {seen}"
            )
        finally:
            gate.set()
            w.shutdown()
            w.wait(3000)

    def test_cancel_queued_job_shrinks_denominator(self, qapp, monkeypatch):
        """取消还在排队的任务：分母同步缩减并当场补发信号。"""
        hold = threading.Event()

        def fake_stats(ld, name, sc=None):
            if name == "a":
                hold.wait(10.0)
            return var_info.VarStats(computed=True)

        monkeypatch.setattr(vid_mod.var_info, "compute_stats", fake_stats)
        w, seen = self._worker_with_recorder()
        try:
            def ref():
                return object()

            w.submit([("a", ref, 0), ("b", ref, 0), ("c", ref, 0)])
            assert self._wait_current(w, "a")
            assert w.queue_size() == 2

            w.cancel("b")
            assert w._pending_total == 2, "分母必须随队列同步缩减"
            assert seen == [(0, 2)], "摘除后应当场补发一次信号，否则文案不自愈"

            hold.set()
            assert self._drained(w, seen), f"未排空: {seen}"
            # 末了必须是终态 (0, 0)：旧实现的缺陷正是排空后再无信号发出
            assert seen == [(0, 2), (1, 2), (0, 0)]
        finally:
            hold.set()
            w.shutdown()
            w.wait(3000)

    def test_cancel_running_job_does_not_over_shrink(self, qapp, monkeypatch):
        """正在运行的那条不扣分母：它仍会走到 ``_finish_current`` 推进进度。

        取消只是让 compute_stats 在下一个分块边界提前返回，任务本身仍会结束
        并推进一次。替它扣分母会让已完成数算多一条（实测会显示
        「统计中 1/2…」而其实一条都没算完）。
        """
        hold = threading.Event()
        monkeypatch.setattr(
            vid_mod.var_info,
            "compute_stats",
            lambda ld, name, sc=None: (
                hold.wait(10.0), var_info.VarStats(computed=True))[1],
        )
        w, seen = self._worker_with_recorder()
        try:
            def ref():
                return object()

            w.submit([("a", ref, 0), ("b", ref, 0), ("c", ref, 0)])
            assert self._wait_current(w, "a")

            w.cancel("a")
            assert w._pending_total == 3, "运行中的那条不在扣减之列"
            assert seen == [], "既未摘除任务也未排空，不该发信号"

            hold.set()
            assert self._drained(w, seen), f"未排空: {seen}"
            assert seen == [(1, 3), (2, 3), (0, 0)]
        finally:
            hold.set()
            w.shutdown()
            w.wait(3000)

    def test_cancel_on_empty_queue_keeps_error_notice(self, qapp, monkeypatch):
        """队列本已为空时 cancel 不得补发 (0, 0)。

        ``recompute(force=True)`` 会在队列可能为空的时刻先 cancel 再 submit；
        此时补发终态信号会把 ``_on_stats_ready`` 刚写入状态欄的错误提示清掉。
        """
        monkeypatch.setattr(
            vid_mod.var_info,
            "compute_stats",
            lambda ld, name, sc=None: var_info.VarStats(computed=True),
        )
        w, seen = self._worker_with_recorder()
        try:
            assert w._pending_total == 0
            w.cancel("never_submitted")
            settle()
            assert seen == [], f"空队列上的 cancel 不应发信号: {seen}"
        finally:
            w.shutdown()
            w.wait(3000)

    def test_status_bar_self_heals_after_closing_queued_tab(
        self, env, many_loader, monkeypatch
    ):
        """用户可见面：关掉排队中的标签页后状态栏必须自愈。

        旧实现的缺陷是**永久滞留** —— 队列早已排空，状态栏却一直显示
        「统计中 2/3…」，直到关掉整个窗口。
        """
        loader, names = many_loader
        env.mw.loader = loader
        hold = threading.Event()
        v1, v2, v3 = names[0], names[1], names[2]

        def fake_stats(ld, name, sc=None):
            if name == v1:
                hold.wait(10.0)
            return var_info.VarStats(computed=True)

        monkeypatch.setattr(vid_mod.var_info, "compute_stats", fake_stats)
        dlg = VariableInfoDialog.popup([v1, v2, v3], parent=env.mw)
        try:
            # 等 v1 真的进入运行态，此时 v2 / v3 仍在队列里
            assert wait_until(lambda: dlg.worker._current == v1, 10.0), "v1 未进入运行态"
            assert dlg.worker.queue_size() == 2
            assert "统计中" in dlg._progress_text

            dlg._on_tab_close(dlg.tabs.indexOf(dlg._pages[v2]))
            assert dlg.worker._pending_total == 2

            hold.set()
            # v2 的页已被关闭：只有 v1 / v3 会有结果回到页面，等它们落地
            # 即说明本波（含 worker 最后发出的终态进度信号）已处理完
            assert _wait_stats(dlg, v1, v3)

            assert dlg.worker.queue_size() == 0
            assert dlg.worker._current is None
            assert "统计中" not in dlg.status_label.text(), (
                f"队列已空却仍显示进度: {dlg.status_label.text()!r}"
            )
        finally:
            hold.set()


# ---------------------------------------------------------------------------
# 关窗语义（等同「关闭全部」）
# ---------------------------------------------------------------------------


class TestCloseSemantics:
    """关窗必须清空所有标签页 —— 用户实测反馈的语义反转。

    旧实现只 hide()，于是“关窗 → 再右键打开另一个变量”会变成 2 个 tab，
    甚至加载新数据后旧变量名仍挂在标签上。
    """

    YES = QMessageBox.StandardButton.Yes
    NO = QMessageBox.StandardButton.No

    def test_close_single_tab_clears_without_prompt(self, env, monkeypatch):
        """单个标签页直接清空：关了再开就是它自己，没有误操作损失。"""
        asked: list = []
        monkeypatch.setattr(vid_mod, "QMessageBox", fake_msgbox(asked, self.YES))
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed")

        dlg.close()
        flush_deferred_deletes()

        assert dlg.tabs.count() == 0, "关窗必须清空标签页"
        assert dlg._pages == {}
        assert dlg.isVisible() is False
        assert VariableInfoDialog._instance is dlg, "closeEvent 不得销毁单例"
        assert env.mw.var_info_geometry is not None, "关闭时须保存几何信息"
        assert asked == [], "单个标签页不该弹确认"

    def test_close_multi_tab_prompts_and_clears_on_yes(self, env, monkeypatch):
        asked: list = []
        monkeypatch.setattr(vid_mod, "QMessageBox", fake_msgbox(asked, self.YES))
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", "load")

        dlg.close()
        flush_deferred_deletes()

        assert len(asked) == 1, "多标签必须弹确认"
        assert "2 个变量标签页" in asked[0][2], "确认文案须报出标签页数量"
        assert dlg.tabs.count() == 0
        assert dlg._pages == {}

    def test_close_multi_tab_cancel_keeps_everything(self, env, monkeypatch):
        """用户选“否”时必须完全维持原状，包括窗口可见性。"""
        monkeypatch.setattr(vid_mod, "QMessageBox", fake_msgbox([], self.NO))
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", "load")

        dlg.close()
        settle()

        assert dlg.isVisible() is True, "取消后窗口不得被隐藏"
        assert dlg.tabs.count() == 2
        assert set(dlg._pages) == {"speed", "load"}
        assert dlg._pages["speed"].snapshot is not None

    def test_no_prompt_during_app_shutdown(self, env, monkeypatch):
        """退出流程中不得弹模态框，否则会把关闭流程卡住。

        实测：主窗口 close() 不会给子对话框发 closeEvent，但
        closeAllWindows() 会，而那时 QApplication.closingDown() 仍为 False
        —— 所以守卫必须是 shutdown_worker 置的显式标志。
        """
        asked: list = []
        monkeypatch.setattr(vid_mod, "QMessageBox", fake_msgbox(asked, self.YES))
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", "load")

        dlg.shutdown_worker()  # 主窗口 closeEvent 走的就是这条
        dlg.close()
        flush_deferred_deletes()

        assert asked == [], "退出流程中不得弹确认框"
        assert dlg.tabs.count() == 0, "但标签页仍须清空"

    def test_reopen_after_close_starts_empty(self, env):
        """直接回归用户报告的现象。"""
        first = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(first, "speed")
        first.close()
        flush_deferred_deletes()

        second = VariableInfoDialog.popup(["load"], parent=env.mw)
        settle()

        assert second is first, "仍复用同一单例"
        assert second.tabs.count() == 1, "旧标签页必须已被清空"
        assert set(second._pages) == {"load"}

    def test_close_destroys_pages_not_just_detaches(self, env):
        """回归防护：实测 Qt 6.11.1 下 tabs.clear() 只摘标签、不销毁页面。

        关窗即清空后这条路径从“偶尔点按钮”变成“每次关窗”，页面若仍
        挂在 QStackedWidget 下就会逐次累积 QTreeWidget。
        """
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", "load")
        pages = [dlg._pages["speed"], dlg._pages["load"]]

        dlg._close_all_tabs()
        flush_deferred_deletes()

        assert dlg.tabs.count() == 0
        assert [isValid(p) for p in pages] == [False, False], "页面必须真正销毁"


# ---------------------------------------------------------------------------
# reload 钩子
# ---------------------------------------------------------------------------


class TestReloadHooks:
    def test_refresh_rerenders_surviving_variable(self, env):
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed", require_computed=True)
        assert dlg._pages["speed"].stats.min == pytest.approx(10.0)

        loader_b = FastDataLoader(str(env.path_b), has_unit=True, sep=",")
        try:
            env.mw.loader = loader_b
            env.mw._data_version += 1
            # 不手动清缓存：重建本身就应作废旧统计
            VariableInfoDialog.refresh_after_reload(loader_b)
            assert _wait_stats(dlg, "speed", require_computed=True)

            page = dlg._pages["speed"]
            assert page.is_stale is False
            assert page.snapshot.generation == 1, "快照须按新 generation 重建"
            assert page.stats is not None
            assert page.stats.min == pytest.approx(1.0), "应反映新数据"
            assert page.stats.max == pytest.approx(5.0)
            assert page.stats.generation == 1
        finally:
            loader_b.release_memory()

    def test_refresh_removes_missing_variable_tab(self, env):
        """新数据中不存在的变量：摘掉标签页并告知，而不是留个“(失效)”。

        留一个点不动也统计不出的旧变量名，只会让人以为换了文件后窗口
        没跟上（用户实测反馈）。但也不能静默消失，否则“我刚才明明开了
        2 个”变成无从解释，因此必须在状态栏报出数量。
        """
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", "load", require_computed=True)
        assert env.mw.var_stats_cache.get("load") is not None

        loader_b = FastDataLoader(str(env.path_b), has_unit=True, sep=",")
        try:
            env.mw.loader = loader_b
            env.mw._data_version += 1
            VariableInfoDialog.refresh_after_reload(loader_b)
            assert _wait_stats(dlg, "speed", require_computed=True)

            assert "load" not in dlg._pages, "失效变量须被摘除"
            assert dlg.tabs.count() == 1
            assert dlg._pages["speed"].is_stale is False, "仍存在的须原地重建"
            assert "已移除 1 个" in dlg.status_label.text(), "摘除必须告知用户"
            assert env.mw.var_stats_cache.get("load") is None, "旧统计须作废"
        finally:
            loader_b.release_memory()

    def test_refresh_invalidates_cache_for_all_pages(self, env):
        """旧统计一律作废：键相同但新数据里的数值可能完全不同。

        这是缓存失效的第一道保险（第二道是每条自带的 generation 令牌）。
        被摘除的页永远不会拿到新统计，因此其缓存条目必须被彻底移除，
        而不能留下一条旧数据的 min/max 供人误读。
        """
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", "load", require_computed=True)
        assert set(env.mw.var_stats_cache) >= {"speed", "load"}

        loader_b = FastDataLoader(str(env.path_b), has_unit=True, sep=",")
        try:
            env.mw.loader = loader_b
            env.mw._data_version += 1
            VariableInfoDialog.refresh_after_reload(loader_b)
            assert _wait_stats(dlg, "speed", require_computed=True)

            assert "load" not in env.mw.var_stats_cache, (
                "失效变量的旧统计必须被移除，不得残留"
            )
            assert env.mw.var_stats_cache["speed"].generation == 1
        finally:
            loader_b.release_memory()
