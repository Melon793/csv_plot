"""变量信息窗口（VariableInfoDialog）的组件测试。

覆盖交互与生命周期语义，而非 var_info 的数据正确性（那部分见
tests/unit/data/test_var_info.py）：

* 全局单例 + 标签页累积、标签栏可见性随标签数切换
* 元数据同步渲染（打开即有内容）与统计异步回填
* 缓存写入 / 命中 / generation 失效 / FIFO 淘汰
* 标签页关闭取消任务、reload 钩子（on_loader_released / refresh_after_reload）
* Markdown 导出尊重标签页顺序、closeEvent 只隐藏不销毁

统一使用合成 CSV（3 行）而非 data/ 下的真实大文件，保证测试秒级完成且
不依赖仓库数据。
"""

import time

import pytest

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from src.data import var_info
from src.data.loader import FastDataLoader
from src.ui.dialogs import variable_info_dialog as vid_mod
from src.ui.dialogs.variable_info_dialog import VariableInfoDialog
from tests.fixtures.data_factory import write_csv


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


def pump(ms: int = 30) -> None:
    """驱动事件循环，让跨线程的 queued 信号得以投递。"""
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.002)


def force_delete(widget) -> None:
    """立即销毁 widget 及其子对象的 C++ 部分。

    单用 ``deleteLater()`` 在测试里**不会生效**：Qt 把 DeferredDelete
    事件推迟到“回到调用时的或更外层事件循环”才处理，而测试从不调
    ``app.exec()``，因此 ``processEvents()`` 不会消化它（实测子对象仍存活）。
    必须显式 ``sendPostedEvents(None, DeferredDelete)``。
    """
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    pump(20)


def wait_idle(dlg, timeout_s: float = 10.0) -> bool:
    """等到 worker 队列排空且无正在运行的任务，再多泵一轮让信号落地。"""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if dlg.worker.queue_size() == 0 and dlg.worker._current is None:
            pump(60)
            if dlg.worker.queue_size() == 0 and dlg.worker._current is None:
                return True
        time.sleep(0.005)
    return False


@pytest.fixture()
def env(qapp, tmp_path):
    """每个测试一套独立环境：合成 CSV + 替身主窗口 + 单例复位。

    复位必须放在 yield **之前**也执行一次：单例是类级状态，若上一个测试
    异常退出未清理，本测试会继承到脏页面。
    """
    VariableInfoDialog.reset_for_tests()
    pump(30)

    path_a = write_csv(
        tmp_path / "a.csv",
        header=["time", "speed", "load", "note"],
        units=["s", "km/h", "%", "-"],
        rows=[
            ["0.0", "10.0", "20.0", "aa"],
            ["0.1", "20.0", "30.0", "bb"],
            ["0.2", "30.0", "40.0", "cc"],
        ],
    )
    # reload 用：刻意缺 load / note，用于验证"变量不在新数据中"的失效标注
    path_b = write_csv(
        tmp_path / "b.csv",
        header=["time", "speed"],
        units=["s", "km/h"],
        rows=[["0.0", "1.0"], ["0.1", "5.0"]],
    )

    loader_a = FastDataLoader(str(path_a), has_unit=True, sep=",")
    mw = FakeMainWindow(loader_a)

    class Env:
        pass

    e = Env()
    e.mw = mw
    e.loader_a = loader_a
    e.path_b = path_b
    e.tmp_path = tmp_path

    yield e

    VariableInfoDialog.reset_for_tests()
    pump(50)
    mw.loader = None
    for attr in ("var_stats_cache",):
        setattr(mw, attr, {})
    loader_a.release_memory()
    mw.deleteLater()
    pump(30)


@pytest.fixture()
def many_loader(qapp, tmp_path):
    """含 v1..v8 八列的合成 CSV，用于验证标签页上限截断。

    用真实 loader 而非测试替身：截断逻辑依赖 build_snapshot 与
    compute_stats 的真实返回，替身反而会把真正要测的集成面遮掉。
    """
    names = [f"v{i}" for i in range(1, 9)]
    path = write_csv(
        tmp_path / "many.csv",
        header=["time"] + names,
        units=["s"] + ["-"] * len(names),
        rows=[["0.0"] + [str(i + 1.0) for i in range(len(names))],
              ["0.1"] + [str(i + 2.0) for i in range(len(names))]],
    )
    loader = FastDataLoader(str(path), has_unit=True, sep=",")
    yield loader, names
    loader.release_memory()


@pytest.fixture()
def submit_spy(env):
    """返回一个函数：给当前单例的 worker.submit 装上记录器。

    不能在 popup **之前**装：弹窗本身就要靠 submit 提交首批任务。
    也不能靠 queue_size()==0 反推“未提交”：任务可能已经跑完，
    队列自然为空。
    """
    calls: list = []

    def install(dlg):
        original = dlg.worker.submit

        def spy(jobs):
            calls.extend(jobs)
            return original(jobs)

        dlg.worker.submit = spy
        return original

    yield install, calls


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
        assert wait_idle(dlg)

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
        pump(30)

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
        pump(30)

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
        # 充分等待：确保所有进度信号都已抵达并处理完毕
        assert wait_idle(dlg)
        pump(100)

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
        pump(20)

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
            pump(30)


# ---------------------------------------------------------------------------
# 元数据同步渲染与统计异步回填
# ---------------------------------------------------------------------------


class TestRendering:
    def test_metadata_available_immediately(self, env):
        """快照零磁盘 I/O，故打开即有内容，不需要"加载中"占位。"""
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        page = dlg._pages["speed"]

        assert page.snapshot is not None
        # 页面置顶的「统计特征」+ CSV 固定三分组（基本/列/文件信息）
        assert page.tree.topLevelItemCount() == 4
        assert page.tree.topLevelItem(0).text(0) == "统计特征"
        assert page.tree.topLevelItem(1).text(0) == "基本信息"
        assert page.is_stale is False

    def test_stats_backfilled_by_worker(self, env):
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert wait_idle(dlg), "统计任务未在超时内完成"

        page = dlg._pages["speed"]
        assert page.stats is not None
        assert page.stats.computed is True
        assert page.stats.min == pytest.approx(10.0)
        assert page.stats.max == pytest.approx(30.0)
        assert page.stats.mean == pytest.approx(20.0)

    def test_stats_written_to_main_window_cache(self, env):
        """回归防护（曾出现的 P0）：缓存起始为空 dict，而空 dict 是 falsy。

        旧实现用 ``if not cache: return`` 判定"无处可存"，把"缓存尚空"
        一并误判，导致统计结果**永远**写不进缓存。
        """
        assert env.mw.var_stats_cache == {}

        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert wait_idle(dlg)

        assert "speed" in env.mw.var_stats_cache, "空缓存字典也必须能写入"
        assert env.mw.var_stats_cache["speed"].computed is True

    def test_non_numeric_column_gets_terminal_message(self, env):
        """非数值列不会被提交统计，必须给出终态说明。

        否则「统计特征」会永远停在"计算中…"，让用户误以为后台仍在算。
        """
        dlg = VariableInfoDialog.popup(["note"], parent=env.mw)
        wait_idle(dlg)
        page = dlg._pages["note"]

        assert page.snapshot.is_numeric is False
        assert page.stats is not None, "非数值列也须有终态，不能停在计算中"
        assert page.stats.computed is False
        assert "非数值" in page.stats.error
        assert page.btn_refresh.isEnabled() is False
        assert dlg.worker.queue_size() == 0

    def test_all_empty_column_gets_friendly_message(self, env):
        """全空列不能说"非数值列（object）"（设计文档 §12.2）。

        这列本质上是数值数据只是没值。报"类型不对"会让用户以为
        自己的列被误判，而且与页面上"NaN 数量"那行对不上。
        """
        path = write_csv(
            env.tmp_path / "empty.csv",
            header=["time", "v"],
            units=["s", "-"],
            rows=[["0.0", ""], ["0.1", ""]],
        )
        env.mw.loader = FastDataLoader(str(path), has_unit=True, sep=",")

        dlg = VariableInfoDialog.popup(["v"], parent=env.mw)
        wait_idle(dlg)
        page = dlg._pages["v"]

        assert page.snapshot.all_empty is True
        assert "全部为空值" in page.stats.error
        assert "非数值" not in page.stats.error, "不得再说“类型不对”"
        # 仍然不可统计：刷新按钮必须继续禁用，否则提交了注定失败的任务
        assert page.btn_refresh.isEnabled() is False
        assert dlg.worker.queue_size() == 0
        # 树里的「列信息」与统计区必须口径一致
        assert "全部为空值" in page.to_markdown()

    def test_unknown_variable_renders_stale(self, env):
        """变量不在当前数据中：保留标签页但显式标注，不静默丢弃。"""
        dlg = VariableInfoDialog.popup(["no_such_var"], parent=env.mw)
        page = dlg._pages["no_such_var"]

        assert page.snapshot is None
        assert page.is_stale is True
        assert dlg.tabs.tabText(0) == "no_such_var (失效)"
        assert page.tree.topLevelItemCount() == 1
        assert page.btn_refresh.isEnabled() is False


# ---------------------------------------------------------------------------
# 缓存语义
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_hit_skips_resubmit(self, env, submit_spy):
        """命中缓存时不得再提交任务：这是缓存存在的唯一意义。"""
        install, calls = submit_spy
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert wait_idle(dlg)
        assert env.mw.var_stats_cache["speed"].computed is True

        # 先等首批统计落地，再装监听器：只关心“重新打开时是否又提交”
        original = install(dlg)
        try:
            dlg._on_tab_close(0)
            assert "speed" not in dlg._pages
            calls.clear()

            dlg.add_variables(["speed"])
            pump(30)

            assert calls == [], f"命中缓存却重新提交了任务: {calls}"
            page = dlg._pages["speed"]
            assert page.stats is not None
            assert page.stats.cached is True, "UI 须标注结果来自缓存"
            assert page.stats.computed is True
        finally:
            dlg.worker.submit = original

    def test_lookup_cache_rejects_stale_generation(self):
        """generation 是失效的第二道保险：即使忘记清空缓存也不会采用旧值。"""
        stale = var_info.VarStats(min=1.0, computed=True, generation=7)
        cache = {"speed": stale}

        assert VariableInfoDialog._lookup_cache(cache, "speed", 7) is stale
        assert VariableInfoDialog._lookup_cache(cache, "speed", 8) is None
        assert "speed" not in cache, "陈旧条目应被顺手剔除，避免长期占位"

    def test_lookup_cache_rejects_pending_entry(self):
        """既未算完也无错误的占位条目不算命中，否则会永久卡住该变量。"""
        cache = {"speed": var_info.VarStats(generation=0)}
        assert VariableInfoDialog._lookup_cache(cache, "speed", 0) is None

    def test_lookup_cache_handles_none(self):
        """缓存不可用时传 None（而非空 dict），不得抛异常。"""
        assert VariableInfoDialog._lookup_cache(None, "speed", 0) is None
        assert VariableInfoDialog._lookup_cache({}, "speed", 0) is None

    def test_stale_stats_result_is_discarded(self, env):
        """reload 期间完成的陈旧结果：既不写缓存也不回填页面。"""
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert wait_idle(dlg)
        page = dlg._pages["speed"]
        good_min = page.stats.min
        env.mw.var_stats_cache.clear()

        # 模拟：任务在 reload 后才完成，携带的是旧 generation
        env.mw._data_version = 5
        dlg._on_stats_ready(
            "speed", var_info.VarStats(min=999.0, computed=True, generation=0)
        )

        assert "speed" not in env.mw.var_stats_cache, "陈旧结果不得写入缓存"
        assert page.stats.min == good_min, "陈旧结果不得覆盖已回填的页面"

    def test_store_cache_fifo_eviction(self, env, monkeypatch):
        """超上限时淘汰最早写入者（dict 自 3.7 起保序，故首键即最旧）。"""
        monkeypatch.setattr(vid_mod, "VAR_INFO_STATS_CACHE_MAX", 2)
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        # 先停掉后台线程：否则 worker 回填 "speed" 会与下面的 FIFO 断言竞争
        dlg.shutdown_worker()
        wait_idle(dlg)
        env.mw.var_stats_cache.clear()

        for name in ("v1", "v2", "v3"):
            dlg._store_cache(name, var_info.VarStats(min=0.0, computed=True))

        assert len(env.mw.var_stats_cache) == 2
        assert list(env.mw.var_stats_cache) == ["v2", "v3"], "应淘汰最早写入的 v1"

    def test_store_cache_noop_without_main_window(self, qapp, monkeypatch):
        """宿主已销毁时静默放弃写缓存，不得抛异常（统计仍会回填 UI）。"""
        monkeypatch.setattr(
            QApplication, "activeWindow", staticmethod(lambda: None)
        )
        dlg = VariableInfoDialog(None)
        dlg._owner_ref = None
        try:
            assert dlg._get_main_window() is None
            assert dlg._cache() is None
            dlg._store_cache("speed", var_info.VarStats(computed=True))
        finally:
            dlg.shutdown_worker()
            dlg.deleteLater()
            pump(30)


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
        assert wait_idle(dlg)

        dlg.recompute("speed")
        assert wait_idle(dlg)

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
        wait_idle(dlg)
        dlg.shutdown_worker()

        assert dlg.worker.isRunning() is False
        # shutdown 后 submit 必须被忽略，否则线程会被重新拉起
        dlg.worker.submit([("speed", lambda: env.loader_a, 0)])
        pump(30)
        assert dlg.worker.isRunning() is False
        assert dlg.worker.queue_size() == 0


# ---------------------------------------------------------------------------
# reload 钩子
# ---------------------------------------------------------------------------


class TestReloadHooks:
    def test_refresh_rerenders_surviving_variable(self, env):
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert wait_idle(dlg)
        assert dlg._pages["speed"].stats.min == pytest.approx(10.0)

        loader_b = FastDataLoader(str(env.path_b), has_unit=True, sep=",")
        try:
            env.mw.loader = loader_b
            env.mw._data_version += 1
            # 不手动清缓存：重建本身就应作废旧统计
            VariableInfoDialog.refresh_after_reload(loader_b)
            assert wait_idle(dlg)

            page = dlg._pages["speed"]
            assert page.is_stale is False
            assert page.snapshot.generation == 1, "快照须按新 generation 重建"
            assert page.stats is not None
            assert page.stats.min == pytest.approx(1.0), "应反映新数据"
            assert page.stats.max == pytest.approx(5.0)
            assert page.stats.generation == 1
        finally:
            loader_b.release_memory()

    def test_refresh_marks_missing_variable_stale(self, env):
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert wait_idle(dlg)
        assert env.mw.var_stats_cache.get("load") is not None

        loader_b = FastDataLoader(str(env.path_b), has_unit=True, sep=",")
        try:
            env.mw.loader = loader_b
            env.mw._data_version += 1
            VariableInfoDialog.refresh_after_reload(loader_b)
            wait_idle(dlg)

            load_page = dlg._pages["load"]
            assert load_page.is_stale is True
            assert load_page.snapshot is None
            assert load_page.stats is None
            assert "失效" in dlg.tabs.tabText(dlg.tabs.indexOf(load_page))
            # 标签页保留，不静默移除（用户 reload 的往往是同一批测量）
            assert dlg.tabs.count() == 2
            assert dlg._pages["speed"].is_stale is False
        finally:
            loader_b.release_memory()

    def test_refresh_invalidates_cache_for_all_pages(self, env):
        """旧统计一律作废：键相同但新数据里的数值可能完全不同。

        这是缓存失效的第一道保险（第二道是每条自带的 generation 令牌）。
        失效页永远不会拿到新统计，因此其缓存条目必须被彻底移除，
        而不能留下一条旧数据的 min/max 供人误读。
        """
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert wait_idle(dlg)
        assert set(env.mw.var_stats_cache) >= {"speed", "load"}

        loader_b = FastDataLoader(str(env.path_b), has_unit=True, sep=",")
        try:
            env.mw.loader = loader_b
            env.mw._data_version += 1
            VariableInfoDialog.refresh_after_reload(loader_b)
            assert wait_idle(dlg)

            assert "load" not in env.mw.var_stats_cache, (
                "失效变量的旧统计必须被移除，不得残留"
            )
            assert env.mw.var_stats_cache["speed"].generation == 1
        finally:
            loader_b.release_memory()


# ---------------------------------------------------------------------------
# Markdown 导出与几何
# ---------------------------------------------------------------------------


class TestExportAndGeometry:
    def test_copy_all_writes_markdown_to_clipboard(self, env):
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert wait_idle(dlg)

        dlg._on_copy_all()
        text = QApplication.clipboard().text()

        assert "speed" in text and "load" in text
        assert "\n---\n" in text, "多个变量须以水平分隔线拼接"
        assert "已复制 2 个变量" in dlg.status_label.text()

    def test_copy_all_respects_tab_order(self, env):
        """用户可拖动调整标签顺序，导出必须尊重当前顺序而非插入顺序。"""
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert wait_idle(dlg)

        names_before = [p.var_name for p in dlg._ordered_pages()]
        assert names_before == ["speed", "load"]

        dlg.tabs.tabBar().moveTab(0, 1)
        names_after = [p.var_name for p in dlg._ordered_pages()]
        assert names_after == ["load", "speed"]

        dlg._on_copy_all()
        text = QApplication.clipboard().text()
        assert text.index("load") < text.index("speed"), "导出顺序应与标签一致"

    def test_copy_all_without_snapshot_notifies(self, env):
        """全部页面都失效时不得抛异常，并给出明确提示。"""
        clipboard = QApplication.clipboard()
        clipboard.setText("")

        dlg = VariableInfoDialog.popup(["no_such_var"], parent=env.mw)
        dlg._on_copy_all()

        assert "没有可复制" in dlg.status_label.text()
        assert clipboard.text() == "", "无内容时不得污染剪贴板"

    def test_close_event_hides_but_preserves_tabs(self, env):
        """关闭只隐藏：标签页内容保留，用户再次右键时立即可见。"""
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert wait_idle(dlg)

        dlg.close()
        pump(30)

        assert dlg.isVisible() is False
        assert VariableInfoDialog._instance is dlg, "closeEvent 不得销毁单例"
        assert dlg.tabs.count() == 1, "标签页应保留"
        assert dlg._pages["speed"].snapshot is not None, "内容不得被清空"
        assert env.mw.var_info_geometry is not None, "关闭时须保存几何信息"

    def test_geometry_roundtrip(self, env):
        """save_geom → load_geom 应把窗口尺寸恢复回去。

        两个平台约束必须避开，否则断言的是约束而非几何逻辑：
        offscreen 默认屏幕仅 800x600（超屏尺寸会被夹住），且布局的
        minimumSizeHint 会把过小的 resize 顶回去（实测 400 宽被顶到 489）。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        dlg.resize(700, 500)
        pump(20)
        saved = (dlg.width(), dlg.height())
        assert saved == (700, 500), "前置条件：尺寸已生效"
        dlg.save_geom()
        assert env.mw.var_info_geometry is not None

        dlg.resize(600, 450)
        pump(20)
        assert (dlg.width(), dlg.height()) != saved, "前置条件：尺寸确实被改小了"

        dlg.load_geom()
        pump(20)
        assert (dlg.width(), dlg.height()) == saved

    def test_reset_for_tests_clears_singleton(self, env):
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        wait_idle(dlg)

        VariableInfoDialog.reset_for_tests()
        pump(50)

        assert VariableInfoDialog._instance is None
        assert dlg.worker.isRunning() is False, "线程必须停止，否则析构时崩溃"
