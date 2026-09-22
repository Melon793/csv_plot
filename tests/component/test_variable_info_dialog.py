"""变量信息窗口（VariableInfoDialog）的组件测试。

覆盖交互与生命周期语义，而非 var_info 的数据正确性（那部分见
tests/unit/data/test_var_info.py）：

* 全局单例 + 标签页累积、标签栏可见性随标签数切换
* 元数据同步渲染（打开即有内容）与统计异步回填
* 缓存写入 / 命中 / generation 失效 / FIFO 淘汰
* 标签页关闭取消任务、reload 钩子（on_loader_released / refresh_after_reload）
* Markdown 导出尊重标签页顺序
* 值列行尾悬停复制按钮（几何/可见性/命中区/与选区互不干扰、提示只报
  键名不顶宽窗口）
* 行高与列宽调优的落地守卫（行高由委托 sizeHint 给出且得容下按钮、
  「属性」列下限走 minimumSectionSize 且管着所有列）
* 关窗等同「关闭全部」（含多标签确认与页面真正销毁）
* 分组默认全展开、「属性」列可拖动且同窗口共享
* MDF 归属信息块的树内位置、推断依据行与 Markdown 导出

统一使用合成 CSV（3 行）与 asammdf 现场生成的合成 MDF（12 点）而非 data/
下的真实大文件，保证测试秒级完成且不依赖仓库数据。
"""

import sys
import threading
import time

import pytest

from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QMainWindow,
    QMessageBox,
    QTreeWidgetItem,
)
from shiboken6 import isValid

from tests.fixtures.waits import (
    force_delete,
    flush_deferred_deletes,
    settle,
    wait_until,
)
from src.core.config import (
    VAR_INFO_COL0_MIN_WIDTH,
    VAR_INFO_COPY_BTN_MARGIN,
    VAR_INFO_COPY_BTN_SIZE,
    VAR_INFO_ROW_HEIGHT,
)
from src.data import mdf_attribution as mda
from src.data import var_info
from src.data.loader import FastDataLoader
from src.data.mdf_lazy_loader import MDFLazyLoader
from src.ui.dialogs import variable_info_dialog as vid_mod
from src.ui.dialogs.variable_info_dialog import (
    CopyFieldDelegate,
    VariableInfoDialog,
    copy_button_rect,
)
from tests.fixtures.data_factory import (
    ATTRIBUTION_CHANNEL_COMMENT,
    ATTRIBUTION_CHANNEL_NAME,
    ATTRIBUTION_DEVICE,
    ATTRIBUTION_ECU,
    ATTRIBUTION_HIERARCHY_NAME,
    ATTRIBUTION_MESSAGE_GROUP,
    ATTRIBUTION_SIGNAL_NAME,
    write_csv,
    write_mdf,
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


@pytest.fixture()
def env(qapp, tmp_path):
    """每个测试一套独立环境：合成 CSV + 替身主窗口 + 单例复位。

    复位必须放在 yield **之前**也执行一次：单例是类级状态，若上一个测试
    异常退出未清理，本测试会继承到脏页面。
    """
    VariableInfoDialog.reset_for_tests()
    # reset_for_tests 已 shutdown_worker（join 线程）并把窗口 deleteLater：
    # 这里只需让删除落地，并排空它留下的投递事件
    flush_deferred_deletes()

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
    flush_deferred_deletes()
    mw.loader = None
    for attr in ("var_stats_cache",):
        setattr(mw, attr, {})
    loader_a.release_memory()
    mw.deleteLater()
    flush_deferred_deletes()


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
def mdf_env(qapp, tmp_path):
    """一套带归属信息的合成 MDF4 环境（不复用 ``env``）。

    不复用是因为 ``env`` 走 CSV 路径、永远不可能是归属块的正面样本；
    两者各守一侧：本夹具验「MDF 该有的东西有」，``env`` 验「CSV 不应
    有的东西没有」。

    变量名直接用 ``ATTRIBUTION_*`` 常量而不是字面量：改名时只需改工厂，
    否则测试会因变量不存在而降级成“错误页也能通过断言”的空转。
    """
    VariableInfoDialog.reset_for_tests()
    flush_deferred_deletes()

    # 12 点与 unit 测试的 attr4_loader 一致；点数不影响归属提取（全部取自
    # 加载期已解析的块结构），只影响统计回填的数值
    path = write_mdf(
        tmp_path / "attr.mf4", version="4.10", n=12, with_attribution=True
    )
    loader = MDFLazyLoader(str(path))
    mw = FakeMainWindow(loader)

    class Env:
        pass

    e = Env()
    e.mw = mw
    e.loader = loader

    yield e

    VariableInfoDialog.reset_for_tests()
    flush_deferred_deletes()
    mw.loader = None
    loader.close()
    mw.deleteLater()
    flush_deferred_deletes()


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
        assert _wait_stats(dlg, "speed", require_computed=True), (
            "统计任务未在超时内回填"
        )

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
        assert _wait_stats(dlg, "speed")

        assert "speed" in env.mw.var_stats_cache, "空缓存字典也必须能写入"
        assert env.mw.var_stats_cache["speed"].computed is True

    def test_non_numeric_column_gets_terminal_message(self, env):
        """非数值列不会被提交统计，必须给出终态说明。

        否则「统计特征」会永远停在"计算中…"，让用户误以为后台仍在算。
        """
        dlg = VariableInfoDialog.popup(["note"], parent=env.mw)
        assert _wait_stats(dlg, "note"), "非数值列也须拿到终态"
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
        assert _wait_stats(dlg, "v")
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
    def test_cache_hit_revalidates_current_page_once(self, env, submit_spy):
        """方案 E：命中缓存的页面成为当前可见页时，恰好提交一次静默再验证。

        这是"可见即再验证"的核心语义：缓存负责瞬时出数，可见页随后必然
        拿到本会话现算结果。回填完成后页面标记已验证，再次右键同名变量
        （页已存在、直接激活）不得重复提交 —— 每 tab 每会话至多一次。
        """
        install, calls = submit_spy
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed", require_computed=True)
        assert env.mw.var_stats_cache["speed"].computed is True

        original = install(dlg)
        try:
            dlg._on_tab_close(0)
            assert "speed" not in dlg._pages
            calls.clear()

            dlg.add_variables(["speed"])
            submitted = [j[0] for j in calls]
            assert submitted == ["speed"], (
                f"命中缓存且成为当前页，应恰好提交一次再验证: {submitted}"
            )
            assert _wait_delivered(dlg, "speed")

            page = dlg._pages["speed"]
            assert page.stats is not None
            assert page.stats.computed is True
            assert page.stats.from_cache is False, "回填后应是现算结果"
            assert page.validated_this_session is True
            assert dlg.worker.queue_size() == 0

            # 已验证的页面不得重复触发
            calls.clear()
            dlg.add_variables(["speed"])
            settle()
            assert calls == [], f"已验证的页面不得重复再验证: {calls}"
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

    def test_lookup_cache_rejects_cancelled_entry(self):
        """缺-5 纵深防御：cancelled 条目不是有效结果，拒绝采用并顺手剔除。

        正常路径下 _store_cache 已拒绝写入；此处兜住历史版本残留与
        异常路径（例如绕过 store 直接塞进缓存的取消结果）。
        """
        cache = {
            "speed": var_info.VarStats(
                error="已取消", cancelled=True, generation=0
            )
        }
        assert VariableInfoDialog._lookup_cache(cache, "speed", 0) is None
        assert "speed" not in cache, "毒条目应被顺手剔除，避免长期占位"

    def test_lookup_cache_handles_none(self):
        """缓存不可用时传 None（而非空 dict），不得抛异常。"""
        assert VariableInfoDialog._lookup_cache(None, "speed", 0) is None
        assert VariableInfoDialog._lookup_cache({}, "speed", 0) is None

    def test_stale_stats_result_is_discarded(self, env):
        """reload 期间完成的陈旧结果：既不写缓存也不回填页面。"""
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed", require_computed=True)
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
        # shutdown 已 join 线程，此后不会再有新结果投递，只需排空队列
        settle()
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
            flush_deferred_deletes()


# ---------------------------------------------------------------------------
# 可见即再验证（方案 E）
# ---------------------------------------------------------------------------


class TestVisibleRevalidation:
    """缓存负责瞬时出数；当前可见页随后必然拿到本会话现算结果。

    触发点共三个：切 tab（currentChanged）、首开/复用窗口当前页
    （add_variables 末尾显式兜底 —— 单标签或索引未变时不发信号）、
    关窗后再开（走"新建页 + 命中"，由前两条覆盖）。
    """

    def test_first_open_with_cache_revalidates(self, env, submit_spy):
        """首开窗口即命中缓存（无任何切换动作）也须提交一次再验证。

        预置一条与真实数据不符的陈旧缓存，验证可见页的数值最终被
        现算结果覆盖 —— 这正是 E 的存在意义：兜住"缓存值与数据不一致"
        的一切未来路径（缺-1）。
        """
        env.mw.var_stats_cache["speed"] = var_info.VarStats(
            min=1.0, max=2.0, mean=1.5, std=0.5, computed=True, generation=0
        )
        install, calls = submit_spy
        # 再验证发生在 popup → add_variables 内部，事后安装监听会错过提交
        # 时机：必须先构建单例、装好监听，再走 popup 的复用路径
        dlg = VariableInfoDialog(env.mw)
        VariableInfoDialog._instance = dlg
        install(dlg)
        assert VariableInfoDialog.popup(["speed"], parent=env.mw) is dlg

        submitted = [j[0] for j in calls]
        assert submitted == ["speed"], f"首开命中缓存应提交一次再验证: {submitted}"
        assert _wait_delivered(dlg, "speed")

        page = dlg._pages["speed"]
        assert page.stats.min == pytest.approx(10.0), "陈旧缓存值须被现算覆盖"
        assert page.stats.from_cache is False
        assert page.validated_this_session is True

    def test_revalidate_error_keeps_cached_values(self, env, monkeypatch):
        """R2：再验证返回错误/取消时，页面保留原有正确值，只在状态栏提示。

        迟到的失败结果（取消 / loader 关闭 / 读盘异常）不得把用户正在
        看的正确数字覆盖成"已取消"。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed", require_computed=True)
        page = dlg._pages["speed"]
        good_min = page.stats.min
        assert good_min == pytest.approx(10.0)

        def broken(loader, var_name, should_cancel=None):
            return var_info.VarStats(error="读取失败: 模拟故障")

        monkeypatch.setattr(vid_mod.var_info, "compute_stats", broken)
        # 模拟缓存命中态：页面显示算完的值但本会话未验证
        page.validated_this_session = False
        dlg._revalidate_page_if_needed(page)
        assert "speed" in dlg._revalidating
        assert _wait_delivered(dlg, "speed")

        assert page.stats.computed is True, "失败结果不得覆盖页面上的正确值"
        assert page.stats.min == good_min
        assert "模拟故障" in dlg._progress_text, "失败原因须在状态栏提示"
        # 一次机会已用完：不得反复重试（每 tab 每会话至多一次）
        assert page.validated_this_session is True
        assert "speed" not in dlg._revalidating

    def test_revalidate_stale_result_discards_and_clears(self, env):
        """R1：再验证在途时发生 reload → 陈旧结果被丢弃且去重标记被释放。

        _on_stats_ready 的 generation 不匹配分支是直接 return，不释放
        标记的话该变量在本会话内将永远无法再次触发再验证。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed", require_computed=True)
        page = dlg._pages["speed"]
        good_min = page.stats.min

        # 模拟在途再验证 + reload 后版本递增才完成的迟到结果
        dlg._revalidating.add("speed")
        dlg._on_stats_ready(
            "speed", var_info.VarStats(min=999.0, computed=True, generation=-1)
        )

        assert page.stats.min == good_min, "陈旧结果不得覆盖已回填的页面"
        assert "speed" not in dlg._revalidating, "丢弃路径也须释放去重标记"

    def test_revalidate_dedup_while_inflight(self, env, submit_spy):
        """R1 去重：同一变量的再验证在途时，再次触发不得重复提交。"""
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed", require_computed=True)
        page = dlg._pages["speed"]
        install, calls = submit_spy
        original = install(dlg)
        try:
            dlg._revalidating.add("speed")  # 模拟在途
            page.validated_this_session = False
            dlg._revalidate_page_if_needed(page)
            dlg._revalidate_page_if_needed(page)
            assert calls == [], "在途去重失效，重复提交了任务"
        finally:
            dlg.worker.submit = original
            dlg._revalidating.discard("speed")

    def test_rapid_tab_switch_no_duplicate_revalidate(self, env, submit_spy):
        """快速连切多标签：同一变量至多提交一次再验证。

        去重集合（在途）与 validated_this_session（已完成）双保险，
        分别挡住"结果未到"与"结果已到"两个窗口期的重复提交。
        """
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", require_computed=True)
        speed_page = dlg._pages["speed"]
        speed_page.validated_this_session = False  # 模拟缓存命中态

        install, calls = submit_spy
        original = install(dlg)
        try:
            load_idx = dlg.tabs.indexOf(dlg._pages["load"])
            speed_idx = dlg.tabs.indexOf(speed_page)
            for _ in range(3):
                dlg.tabs.setCurrentIndex(load_idx)
                dlg.tabs.setCurrentIndex(speed_idx)
            submitted = [j[0] for j in calls]
            assert submitted.count("speed") <= 1, f"重复提交: {submitted}"
        finally:
            dlg.worker.submit = original

    def test_copy_all_markdown_has_no_cached_marker(self, env):
        """D′：缓存命中态下导出的 Markdown 不得含「（缓存）」字样。

        旧实现的后缀经 snapshot_to_markdown 复用进入导出，把数值字段
        污染成 `3.5（缓存）`。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed")
        # 关闭再打开：此刻页面显示的是缓存副本（from_cache=True）
        dlg._close_all_tabs()
        dlg.add_variables(["speed"])

        md = "\n".join(p.to_markdown() for p in dlg._ordered_pages())
        assert "（缓存）" not in md

    def test_cancelled_result_not_cached_and_recomputes(self, env, monkeypatch):
        """缺-5：在途任务被取消后，「已取消」不得写入缓存；重开该变量
        必须正常提交重算，而不是把「已取消」当终态展示。"""
        release = threading.Event()
        real_compute = vid_mod.var_info.compute_stats

        def blocking_compute(loader, var_name, should_cancel=None):
            while not release.is_set():
                if should_cancel is not None and should_cancel(var_name):
                    return var_info.VarStats(error="已取消", cancelled=True)
                time.sleep(0.005)
            return real_compute(loader, var_name, should_cancel)

        monkeypatch.setattr(vid_mod.var_info, "compute_stats", blocking_compute)
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)

        assert wait_until(lambda: dlg.worker._current == "speed", 5.0), (
            "worker 未开始执行阻塞任务"
        )

        reached = []
        dlg.worker.item_ready.connect(lambda name, _st: reached.append(name))
        dlg._on_tab_close(0)  # 取消在途任务
        # 取消结果靠跨线程信号回到主线程（普通可调用对象是队列投递，
        # 由这里的 processEvents 送达）：item_ready 是 _process_job 的最后
        # 一步，它到达即说明这条任务在 worker 侧已彻底跑完
        assert wait_until(lambda: "speed" in reached, 5.0), "取消结果未回到主线程"
        release.set()
        assert "speed" not in env.mw.var_stats_cache, "「已取消」不得写入缓存"

        monkeypatch.undo()  # 恢复真实计算
        dlg.add_variables(["speed"])
        assert _wait_stats(dlg, "speed", require_computed=True)
        page = dlg._pages["speed"]
        assert page.stats is not None
        assert page.stats.computed is True, "重开后必须自动重算，而非展示「已取消」"
        assert page.stats.error != "已取消"

    def test_close_all_and_loader_released_clear_revalidating(self, env):
        """R1：关闭全部与 loader 释放两个整批清理点都必须释放去重标记。

        _close_all_tabs 绕过 _on_tab_close 自行循环摘页，且 cancel_all
        对排队未启动的任务静默丢弃（不 emit）—— 不清理的话，"关闭全部
        后再打开"将永远不再触发再验证且无任何报错。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed")

        dlg._revalidating.add("speed")
        dlg._close_all_tabs()
        assert dlg._revalidating == set(), "_close_all_tabs 后标记必须清空"

        dlg._revalidating.add("speed")
        VariableInfoDialog.on_loader_released()
        assert dlg._revalidating == set(), "on_loader_released 后标记必须清空"


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
# 树的展现：默认展开与列宽
# ---------------------------------------------------------------------------


class TestTreePresentation:
    def test_all_sections_expanded_by_default(self, env):
        """CSV 路径的顶层分组固定为 4 个：统计特征 + 基本信息 / 列信息 / 文件信息。

        写死 4 而不是 ``>= 2``：全展开之所以可接受，前提就是分组数已经降
        下来（MDF 路径实测 8 → 4，总行数 57 → 41）。若将来又长出几个
        分组，该重新评估“全展开会不会把关键信息挤出可视区”，而不
        是让这条测试静默地继续通过。

        “将来”已到了：MDF 路径现在多一个「归属信息」块（实测 5 个顶层
        分组、行数 41 → 48），但仍在“全部展开不挤出关键信息”的区间内
        （统计特征置顶且不受影响），因此判定维持不变。该侧的正面断言
        见 ``TestMdfAttributionPresentation``；本测试用 CSV，4 这个数字
        同时兼职“归属块不得外溢到 CSV”，不得改成 ``>= 4``。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed")
        tree = dlg._pages["speed"].tree

        assert tree.topLevelItemCount() == 4
        expanded = []
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            if item.childCount():
                assert item.isExpanded(), f"分组「{item.text(0)}」默认必须展开"
                expanded.append(item.text(0))
        assert expanded, "至少得有一个带子项的分组，否则本测试是空转的"

    def test_column0_is_user_resizable(self, env):
        """ResizeToContents 的定义就是“用户拖不动”，必须换成 Interactive。"""
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed")
        tree = dlg._pages["speed"].tree

        assert tree.header().sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
        before = tree.columnWidth(0)
        tree.setColumnWidth(0, before + 40)
        assert tree.columnWidth(0) != before, "Interactive 下用户必须能改宽度"

    def test_stats_backfill_does_not_reset_column_width(self, env):
        """异步回填走 _fill_stats_rows 而非 render，不得顶掉用户调的宽度。

        这正是旧实现用 ResizeToContents 时用户看到的“列宽自行跳动”。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        tree = dlg._pages["speed"].tree
        tree.setColumnWidth(0, tree.columnWidth(0) + 40)
        target = tree.columnWidth(0)

        assert _wait_stats(dlg, "speed")  # 统计回填（含树重建）在此期间完成
        assert tree.columnWidth(0) == target

    def test_manual_width_shared_across_tabs(self, env):
        """不同步的话，切标签页会看到不同列宽，视觉上像“设置没生效”。"""
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", "load")
        a = dlg._pages["speed"].tree
        b = dlg._pages["load"].tree

        a.setColumnWidth(0, a.columnWidth(0) + 40)

        assert b.columnWidth(0) == a.columnWidth(0)

    def test_new_tab_inherits_shared_width(self, env):
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed")
        a = dlg._pages["speed"].tree
        a.setColumnWidth(0, a.columnWidth(0) + 40)
        expected = a.columnWidth(0)

        dlg.add_variables(["load"])
        settle()

        assert dlg._pages["load"].tree.columnWidth(0) == expected


# ---------------------------------------------------------------------------
# MDF「归属信息」块的展现
# ---------------------------------------------------------------------------


def _titles(tree) -> list[str]:
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


def _rows(item) -> dict:
    return {
        item.child(i).text(0): item.child(i).text(1)
        for i in range(item.childCount())
    }


def _group(tree, title: str):
    for i in range(tree.topLevelItemCount()):
        if tree.topLevelItem(i).text(0) == title:
            return tree.topLevelItem(i)
    raise AssertionError(f"分组「{title}」不存在：{_titles(tree)}")


class TestMdfAttributionPresentation:
    def test_attribution_section_follows_basic_section(self, mdf_env):
        """归属块在树里的位置与内容：紧跟「基本信息」且默认展开。

        写死完整顶层标题序列而不是 ``"归属信息" in titles``：归属块若被
        插到文件信息之后，用户就需手动滚动才能看到“这变量来自哪个 ECU”，
        而功能价值正好在“一眼定位”。
        """
        dlg = VariableInfoDialog.popup([ATTRIBUTION_SIGNAL_NAME], parent=mdf_env.mw)
        assert _wait_stats(dlg, ATTRIBUTION_SIGNAL_NAME)
        tree = dlg._pages[ATTRIBUTION_SIGNAL_NAME].tree

        assert _titles(tree) == [
            "统计特征",
            "基本信息",
            "归属信息",
            "转换规则 (CCBLOCK)",
            "文件信息 (HDBLOCK)",
        ]

        group = _group(tree, "归属信息")
        assert group.isExpanded(), "归属块同样必须默认展开"
        rows = _rows(group)
        assert rows[mda.LABEL_DEVICE] == ATTRIBUTION_DEVICE
        assert rows[mda.LABEL_ECU] == ATTRIBUTION_ECU
        assert rows[mda.LABEL_BUS_TYPE] == "CAN"
        assert rows[mda.LABEL_SOURCE_TYPE] == "ECU"
        assert rows[mda.LABEL_GROUP] == ATTRIBUTION_MESSAGE_GROUP
        # 首行必须是设备：它是“这数据哪来的”的第一答案
        assert group.child(0).text(0) == mda.LABEL_DEVICE
        assert group.child(0).toolTip(1) == ATTRIBUTION_DEVICE

    def test_inferred_function_shows_basis_and_hint(self, mdf_env):
        """函数名是推断出来的，就必须带上依据与提示行。只看到
        “所属函数：RBArithmeticElement”会让用户把它当成 A2L 定义的事实，
        而 MDF 里根本没有函数层级块。
        """
        dlg = VariableInfoDialog.popup([ATTRIBUTION_CHANNEL_NAME], parent=mdf_env.mw)
        assert _wait_stats(dlg, ATTRIBUTION_CHANNEL_NAME)
        group = _group(dlg._pages[ATTRIBUTION_CHANNEL_NAME].tree, "归属信息")

        rows = _rows(group)
        assert rows[mda.LABEL_FUNCTION] == ATTRIBUTION_CHANNEL_COMMENT
        assert rows[mda.LABEL_FUNCTION_BASIS] == mda.BASIS_AUX_TEXT
        assert rows[mda.LABEL_HINT] == mda.HINT_INFERRED
        # 行序：函数 → 依据 → 提示，不得把依据行排到函数行后面很远
        labels = [group.child(i).text(0) for i in range(group.childCount())]
        assert labels.index(mda.LABEL_FUNCTION) + 1 == labels.index(
            mda.LABEL_FUNCTION_BASIS
        )

    def test_hierarchy_channel_is_not_flagged_as_inferred(self, mdf_env):
        """通道名层级（``Fkt/Var\\Device``）是硬信息，不需提示行。

        提示行只在走注释推断时出现；若变成常驻，用户会连硬信息一并不信。
        """
        name = ATTRIBUTION_HIERARCHY_NAME
        dlg = VariableInfoDialog.popup([name], parent=mdf_env.mw)
        assert _wait_stats(dlg, name)
        rows = _rows(_group(dlg._pages[name].tree, "归属信息"))

        assert rows[mda.LABEL_FUNCTION] == "EpmCaS_phiSegOfs_CA"
        assert rows[mda.LABEL_FUNCTION_BASIS] == mda.BASIS_NAME
        assert mda.LABEL_HINT not in rows

    def test_long_value_does_not_widen_layout_and_stays_full(self, mdf_env):
        """数据层不截断的前提：长值不撑宽布局，且全值可从 tooltip / 复制拿到。

        曾经的 120 字符截断以“撑宽值列”为由，但值列是 ``Stretch`` 的、只吃
        剩余空间，属性列又只按短标签自适应：实测 3000 字符的归属值下
        窗口宽与「属性」列宽均不变。本测试把那个结论固定下来，同时守住
        tooltip 与行尾复制按钮给全值 —— 三者合起来才是“不截断就
        不丢信息”成立的依据。

        量的是「属性」列而不是「值」列：值列宽 = 视口宽 - 属性列宽，而视口
        宽会随垂直滚动条的有无在 712/730 间跳（实测），拿它做基准会误报。
        """
        dlg = VariableInfoDialog.popup([ATTRIBUTION_SIGNAL_NAME], parent=mdf_env.mw)
        assert _wait_stats(dlg, ATTRIBUTION_SIGNAL_NAME)
        page = dlg._pages[ATTRIBUTION_SIGNAL_NAME]
        tree = page.tree
        col0_before = tree.columnWidth(0)
        win_before = dlg.width()

        long_value = "X" * 3000
        snap = page.snapshot
        snap.sections["归属信息"] = [(mda.LABEL_DEVICE, long_value)]
        page.render(snap, page.stats)
        settle()

        group = _group(tree, "归属信息")
        assert tree.columnWidth(0) == col0_before, "「属性」列不得被归属长值撑宽"
        assert dlg.width() == win_before, "窗口宽度不得随内容变化"
        assert group.child(0).toolTip(1) == long_value

        page._on_copy_field(group.child(0))
        assert QApplication.clipboard().text() == long_value, "复制必须拿到全值"

    def test_attribution_rows_reach_markdown_export(self, mdf_env):
        """归属行必须随 Markdown 一起出口，否则报告里丢失数据溯源。

        同时反向守住长值不得以“…”形式混入导出：那会把不完整的
        信息当成权威结果递到工具外部。
        """
        dlg = VariableInfoDialog.popup([ATTRIBUTION_SIGNAL_NAME], parent=mdf_env.mw)
        assert _wait_stats(
            dlg, ATTRIBUTION_SIGNAL_NAME, require_computed=True
        )
        page = dlg._pages[ATTRIBUTION_SIGNAL_NAME]
        # 未回填时导出会写“计算中…”，那条省略号不是截断，会干扰下面的反向断言
        assert page.stats is not None and page.stats.computed

        md = page.to_markdown()
        assert "## 归属信息" in md
        assert f"| {mda.LABEL_DEVICE} | {ATTRIBUTION_DEVICE} |" in md
        assert "…" not in md


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


# ---------------------------------------------------------------------------
# Markdown 导出与几何
# ---------------------------------------------------------------------------


class TestExportAndGeometry:
    def test_copy_all_writes_markdown_to_clipboard(self, env):
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", "load")

        dlg._on_copy_all()
        text = QApplication.clipboard().text()

        assert "speed" in text and "load" in text
        assert "\n---\n" in text, "多个变量须以水平分隔线拼接"
        assert "已复制 2 个变量" in dlg.status_label.text()

    def test_copy_all_respects_tab_order(self, env):
        """用户可拖动调整标签顺序，导出必须尊重当前顺序而非插入顺序。"""
        dlg = VariableInfoDialog.popup(["speed", "load"], parent=env.mw)
        assert _wait_stats(dlg, "speed", "load")

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

    def test_geometry_roundtrip(self, env):
        """save_geom → load_geom 应把窗口尺寸恢复回去。

        两个平台约束必须避开，否则断言的是约束而非几何逻辑：
        offscreen 默认屏幕仅 800x600（超屏尺寸会被夹住），且布局的
        minimumSizeHint 会把过小的 resize 顶回去（实测 400 宽被顶到 489）。

        load_geom 对可见窗口是 no-op（restoreGeometry 非幂等，开着时
        当前几何即最新状态），因此恢复断言需先 hide 走不可见路径。
        """
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        dlg.resize(700, 500)
        settle()
        saved = (dlg.width(), dlg.height())
        assert saved == (700, 500), "前置条件：尺寸已生效"
        dlg.save_geom()
        assert env.mw.var_info_geometry is not None

        dlg.resize(600, 450)
        settle()
        assert (dlg.width(), dlg.height()) != saved, "前置条件：尺寸确实被改小了"

        # 可见状态下 load_geom 被守卫拦截，不应重置当前几何
        dlg.load_geom()
        settle()
        assert (dlg.width(), dlg.height()) != saved, "可见窗口 load_geom 应 no-op"

        dlg.hide()
        dlg.load_geom()
        settle()
        assert (dlg.width(), dlg.height()) == saved

    def test_reset_for_tests_clears_singleton(self, env):
        dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
        assert _wait_stats(dlg, "speed")

        VariableInfoDialog.reset_for_tests()
        # 这里刻意只 settle()：reset_for_tests 对窗口做过 deleteLater，
        # 若用 flush_deferred_deletes() 会把 C++ 窗口真删掉，下面就没法
        # 读 worker.isRunning() 了。销毁路径本身由夹具的 flush 覆盖
        settle()

        assert VariableInfoDialog._instance is None
        assert dlg.worker.isRunning() is False, "线程必须停止，否则析构时崩溃"


# ---------------------------------------------------------------------------
# 值列行尾悬停复制按钮
# ---------------------------------------------------------------------------

_SENTINEL = "SENTINEL-MUST-NOT-BE-CLOBBERED"


def _find_row(tree, key: str):
    """按「属性」列文本定位行（递归遍历，分组标题同样可命中）。"""

    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            if child.text(0) == key:
                return child
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(tree.invisibleRootItem())


def _value_cell_rect(tree, item) -> QRect:
    """某个 item 的「值」列单元格矩形（视口坐标），并卡住“行可见”前提。

    visualItemRect 返回横跨整行的矩形，拿不到分列边界；视图自己算单元格
    用的是 columnViewportPosition + columnWidth，此处必须同口径，否则
    测的是另一块地方。

    末尾的断言是全部调用点的公共前置：行在视口外时 visualItemRect 仍会
    给出一套落在其它行上的坐标，拿它去点击就是静默误测（行高调大后可视
    行数变少，此风险从“测不出”变成“很容易撞上”）。需要时先 scrollToItem。
    """
    row_rect = tree.visualItemRect(item)
    assert 0 <= row_rect.top() < tree.viewport().height(), (
        f"目标行不在视口内，点击会落到别的行：{item.text(0)} row={row_rect} "
        f"视口高={tree.viewport().height()}"
    )
    return QRect(
        tree.columnViewportPosition(1),
        row_rect.top(),
        tree.columnWidth(1),
        row_rect.height(),
    )


def _viewport_image(tree) -> QImage:
    img = QImage(tree.viewport().size(), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.white)
    tree.viewport().render(img)
    return img


def _region_diff(a: QImage, b: QImage, rect: QRect) -> int:
    """rect 内颜色不同的像素数：用于区分“图标只落在热区”与“整行重绘”。"""
    count = 0
    for y in range(max(rect.top(), 0), min(rect.bottom(), a.height() - 1) + 1):
        for x in range(max(rect.left(), 0), min(rect.right(), a.width() - 1) + 1):
            if a.pixelColor(x, y) != b.pixelColor(x, y):
                count += 1
    return count


@pytest.fixture()
def qt_exceptions():
    """捕获 Qt 回调（委托 paint / 事件出口）里抛出的 Python 异常。

    PySide6 对虚函数重写里的异常只走 sys.excepthook、不向调用方上抛，
    offscreen 下更是看不见 stderr —— 不挂钩就会把崩溃类缺陷测成“通过”。
    """
    caught: list = []
    original = sys.excepthook

    def hook(exc_type, exc, tb):
        caught.append(exc)

    sys.excepthook = hook
    yield caught
    sys.excepthook = original


@pytest.fixture()
def page(env):
    """当前可见的单标签页（统计已回填），并卡住几何前置条件。

    只测当前可见页：QStackedWidget 下非当前页没参与布局，单元格矩形全是
    零尺寸，拿它测坐标等于空转。
    """
    dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
    assert _wait_stats(dlg, "speed")
    # offscreen 默认屏幕仅 800x600，超屏尺寸会被夹住，故取一个安全尺寸
    dlg.resize(700, 520)
    settle()

    pg = dlg._pages["speed"]
    reserved = VAR_INFO_COPY_BTN_SIZE + 2 * VAR_INFO_COPY_BTN_MARGIN
    # 前置条件而非功能断言：列宽不够时按钮热区会贴到单元格左边界外，
    # 后面的坐标就没有意义
    assert pg.tree.columnWidth(1) > 4 * reserved, f"值列过窄：{pg.tree.columnWidth(1)}"
    assert pg.tree.viewport().size().isValid()
    return dlg, pg


class TestFieldCopyButton:
    def test_tree_uses_copy_delegate(self, page):
        """接线前提：页面用的必须是带委托的子类，不是裸 QTreeWidget。"""
        _, pg = page
        assert isinstance(pg.tree, vid_mod.VarInfoTree)
        assert isinstance(pg.tree.itemDelegate(), CopyFieldDelegate)

    def test_copy_button_rect_geometry(self):
        """纯几何：正方形、右端对齐留 margin、垂直居中、不超出单元格。"""
        cell = QRect(120, 40, 600, 24)
        btn = copy_button_rect(cell)

        assert btn.width() == btn.height() == VAR_INFO_COPY_BTN_SIZE
        assert cell.right() - btn.right() == VAR_INFO_COPY_BTN_MARGIN
        assert btn.top() >= cell.top() and btn.bottom() <= cell.bottom()
        # 与文本区的分界：整条预留宽度内不允许被文本占用
        assert btn.left() == cell.right() - VAR_INFO_COPY_BTN_MARGIN - VAR_INFO_COPY_BTN_SIZE + 1

    def test_copy_button_rect_follows_runtime_constant(self, monkeypatch):
        """默认尺寸必须在调用时读全局，不得焊死在 def 上。

        踩过的坑：默认参数直接写 VAR_INFO_COPY_BTN_SIZE 时，它在 def 执行
        时就绑定了 config 的值，而 paint 里的文本区收缩读的是运行时全局
        —— 运行时改常量会让两处错位，“不重启预览尺寸”直接失效。
        本用例在旧写法下必然失败（拿到的还是 config 里的旧边长）。
        """
        monkeypatch.setattr(vid_mod, "VAR_INFO_COPY_BTN_SIZE", 30)
        monkeypatch.setattr(vid_mod, "VAR_INFO_COPY_BTN_MARGIN", 7)
        cell = QRect(0, 0, 600, 40)

        btn = copy_button_rect(cell)

        assert btn.width() == btn.height() == 30, "边长没跟运行时常量走"
        assert cell.right() - btn.right() == 7
        assert btn.top() == cell.center().y() - 15

    def test_is_copyable_rules(self, page):
        """值列有内容才给按钮；标题行、占位行、属性列一律不给。"""
        _, pg = page
        tree = pg.tree

        normal = _find_row(tree, "变量名")
        assert CopyFieldDelegate.is_copyable(tree.indexFromItem(normal, 1)) is True
        # 同一行的属性列不画按钮（否则满屏两个把手）
        assert CopyFieldDelegate.is_copyable(tree.indexFromItem(normal, 0)) is False

        header = _find_row(tree, "基本信息")
        assert header.parent() is None, "分组标题必须是顶层项"
        assert CopyFieldDelegate.is_copyable(tree.indexFromItem(header, 1)) is False

        dash = QTreeWidgetItem(tree, ["占位", "-"])
        assert CopyFieldDelegate.is_copyable(tree.indexFromItem(dash, 1)) is False

    def test_click_on_button_copies_value(self, page, qt_exceptions):
        """端到端：点击热区→剪贴板得到值列原文，状态栏回显键名。

        反向验证：未接线时（删掉 copy_requested 的 connect）本用例
        必然失败 —— 剪贴板停在哨兵值上。
        """
        dlg, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QApplication.clipboard().setText(_SENTINEL)

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        assert QApplication.clipboard().text() == item.text(1) == "speed"
        assert "变量名" in dlg.status_label.text()
        # 提示只报键名，不得把值也回显出去
        assert "speed" not in dlg.status_label.text()
        assert qt_exceptions == []

    def test_release_outside_button_does_not_copy(self, page, qt_exceptions):
        """按下在热区内、释放拖到热区外 → 取消（按按钮的通用语义）。"""
        _, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QApplication.clipboard().setText(_SENTINEL)

        QTest.mousePress(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        QTest.mouseRelease(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(btn.left() - VAR_INFO_COPY_BTN_MARGIN, btn.center().y()),
        )
        settle()

        assert QApplication.clipboard().text() == _SENTINEL
        assert qt_exceptions == []

    def test_press_on_button_leaves_selection_untouched(self, page):
        """复制是“读”操作，不得顺带把用户的选区弄乱。"""
        _, pg = page
        tree = pg.tree
        tree.clearSelection()
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        assert tree.selectionModel().selectedRows() == []

    def test_press_outside_button_still_selects_row(self, page):
        """热区外必须完全保持原有行为（补位验证上一条不是“整个事件被吞”）。"""
        _, pg = page
        tree = pg.tree
        tree.clearSelection()
        item = _find_row(tree, "变量名")
        cell = _value_cell_rect(tree, item)
        btn = copy_button_rect(cell)

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(cell.left() + 6, cell.center().y()),
        )
        settle()

        assert item in tree.selectedItems()
        assert btn.contains(QPoint(cell.left() + 6, cell.center().y())) is False

    def test_icon_only_drawn_on_hover_and_never_on_text(self, page, qt_exceptions):
        """悬停才出现图标，且只出现在热区 —— 文本区一个像素都不能变。"""
        _, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        cell = _value_cell_rect(tree, item)
        btn = copy_button_rect(cell)
        text_area = QRect(cell.left(), cell.top(), btn.left() - cell.left(), cell.height())

        before = _viewport_image(tree)
        QTest.mouseMove(tree.viewport(), btn.center())
        settle()
        hovered = _viewport_image(tree)

        assert _region_diff(before, hovered, btn) > 0, "悬停后热区内应画出图标"
        assert _region_diff(before, hovered, text_area) == 0, "预留宽度恒定，文本区不得因悬停而变"

        # 直接递一个 Leave 事件：offscreen 下合成“移出控件”的鼠标位不可靠，
        # 而被测的是我们自己的 leaveEvent 处理，不必依赖平台光标跟踪
        tree.leaveEvent(QEvent(QEvent.Type.Leave))
        settle()
        left = _viewport_image(tree)

        assert _region_diff(left, before, btn) == 0, "离开后图标必须消失"
        assert qt_exceptions == []

    def test_hover_state_survives_tree_rebuild(self, page, env, qt_exceptions):
        """悬停中重建整棵树 → 不崩，且新行仍可悬停复制。

        这就是悬停态用 QPersistentModelIndex 的全部理由：render() 里的
        tree.clear() 会重置模型，旧索引必须自动失效而不是被继续取用。
        """
        _, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QTest.mouseMove(tree.viewport(), btn.center())
        settle()

        snapshot = var_info.build_snapshot(env.loader_a, "speed", 0)
        old_stats = pg.stats
        pg.render(snapshot, old_stats)
        settle()
        assert qt_exceptions == [], "重建后旧悬停索引被取用会在此报异常"

        new_item = _find_row(tree, "变量名")
        new_btn = copy_button_rect(_value_cell_rect(tree, new_item))
        QApplication.clipboard().setText(_SENTINEL)
        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, new_btn.center(),
        )
        settle()
        assert QApplication.clipboard().text() == "speed"
        assert qt_exceptions == []

    def test_stats_backfilled_rows_are_copyable(self, page):
        """异步回填走 takeChildren 重建子行，新行同样要点得着。"""
        _, pg = page
        tree = pg.tree
        item = _find_row(tree, "最大值")
        assert item is not None, "前置：统计已回填"
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QApplication.clipboard().setText(_SENTINEL)

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        assert QApplication.clipboard().text() == item.text(1) == "30"

    def test_long_value_shows_key_only_and_never_widens_window(self, page):
        """长文件路径：剪贴板给全量，提示只报键名，且窗口不得被提示顶宽。

        后半段是回归闸：QLabel 的 sizeHint 会经布局抬高整个窗口的
        minimumWidth，实测复制一条长路径后窗口 700 → 769 且拖不回
        去；靠 status_label 的 Ignored 宽度策略消除。只断言提示文本
        变短是拦不住这个的 —— 真正的因是宽度策略。
        """
        dlg, pg = page
        tree = pg.tree
        long_value = "/Volumes/data/" + "measurement_" * 30 + ".mf4"
        item = QTreeWidgetItem(tree, ["文件路径", long_value])
        # 必须先滚入视口：行高变大后可视行数变少，追加到树末尾的行很
        # 可能已经在视口外，而 visualItemRect 仍会给出一套落在其它行上的
        # 坐标，不滚上来就是点错行的误测
        tree.scrollToItem(item)
        settle()
        btn = copy_button_rect(_value_cell_rect(tree, item))
        width_before = dlg.width()

        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        assert QApplication.clipboard().text() == long_value
        notice = dlg.status_label.text()
        assert notice == "已复制「文件路径」"
        assert long_value not in notice
        assert dlg.width() == width_before, "提示不得把窗口顶宽"
        assert dlg.minimumWidth() <= width_before, "minimumWidth 被抬高就缩不回去"

    def test_copy_then_tab_close_leaves_no_crash(self, page, qt_exceptions):
        """复制过的页被关闭：委托与持久索引随页销毁，不得有残留回调。

        必须真把 DeferredDelete 投递完（理由见 force_delete 的注释），否则
        “没崩”只是因为页还没死，测不出任何东西。
        """
        dlg, pg = page
        tree = pg.tree
        item = _find_row(tree, "变量名")
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QTest.mouseMove(tree.viewport(), btn.center())
        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()

        dlg._on_tab_close(0)
        flush_deferred_deletes()

        assert dlg._pages == {}
        assert isValid(tree) is False, "页面必须真被销毁，否则本用例是空转的"
        assert qt_exceptions == []


class TestRowHeightAndColumnWidth:
    """行高与列宽配置项的落地守卫（断言值全部取自 config，不写死）。

    这两个尺寸都不是独立可调的：行高受复制按钮边长托底，列宽下限
    实际作用于所有列。调优时改坏任一约束，靠本类兜住。
    """

    @pytest.mark.skipif(VAR_INFO_ROW_HEIGHT <= 0, reason="=0 表示沿用系统默认，无固定值可断言")
    def test_row_height_applies_to_every_row(self, page):
        """委托给的高度要真成为行高，且分组行与数据行同值。

        uniformRowHeights 开着时逐行给不同高度会被拉平成最大值，本用例
        同时把这个前提测出来：一旦有人关掉该开关去追求不等高，这里先失败。
        """
        _, pg = page
        tree = pg.tree
        group = tree.topLevelItem(0)
        assert tree.visualItemRect(group).height() == VAR_INFO_ROW_HEIGHT
        assert tree.visualItemRect(group.child(0)).height() == VAR_INFO_ROW_HEIGHT

    def test_row_height_fits_copy_button(self, page):
        """行高不得小于复制按钮，否则图标会溢出骑到相邻行上。

        绘制不裁剪到单元格，所以后果不是“被裁掉”而是“遮挡别的内容”
        （实测 40 px 按钮配 18 px 行高：上下共溢出 22 px、横跨约 3 行）。

        这条是常量间的耦合关系：把 VAR_INFO_ROW_HEIGHT 调小到低于
        VAR_INFO_COPY_BTN_SIZE 时，必须在这里报错而不是默默花屏。约
        束方向是“行高→按钮”：改大按钮不会把行高顶高。
        """
        _, pg = page
        tree = pg.tree
        cell = _value_cell_rect(tree, _find_row(tree, "变量名"))
        btn = copy_button_rect(cell)
        assert btn.top() >= cell.top() and btn.bottom() <= cell.bottom(), (
            f"行高 {cell.height()} 容不下 {VAR_INFO_COPY_BTN_SIZE} px 按钮"
        )

    def test_col0_min_width_clamps_narrow_drag(self, page):
        """往窄拖的下限就是该常量；名字里的 col0 不等于只管第 0 列。"""
        _, pg = page
        tree = pg.tree
        assert tree.header().minimumSectionSize() == VAR_INFO_COL0_MIN_WIDTH
        tree.setColumnWidth(0, 1)
        assert tree.columnWidth(0) == VAR_INFO_COL0_MIN_WIDTH

    def test_col0_default_width_not_clamped_by_minimum(self, page):
        """调小下限不会把默认列宽一起拖窄：默认值由内容算出，高于下限。"""
        _, pg = page
        tree = pg.tree
        assert tree.columnWidth(0) > VAR_INFO_COL0_MIN_WIDTH


class _FakeAction:
    """QAction 替身：只记录 triggered 的接线，可手动 fire。"""

    def __init__(self, text: str):
        self.text = text
        self._slots = []

    @property
    def triggered(self):
        return self

    def connect(self, slot):
        self._slots.append(slot)

    def fire(self):
        for slot in self._slots:
            slot()


class _FakeMenu:
    """QMenu 替身：记录每次弹出的菜单与其中的项，不真画出来。

    offscreen 下 ``exec()`` 会阻塞等用户，而菜单内容本身（有哪几项、每项
    复制出什么）才是被测对象，因此直接拦掉渲染。``created`` 同时用于
    断言“不该弹的时候不弹”。
    """

    def __init__(self, parent=None):
        self._items = []
        _FakeMenu.created.append(self)

    def addAction(self, text):  # noqa: N802 - 跟 Qt 同名
        action = _FakeAction(text)
        self._items.append(action)
        return action

    def exec(self, pos):  # noqa: N802 - 跟 Qt 同名
        return None

    @property
    def items(self):
        return list(self._items)


@pytest.fixture()
def fake_menu(monkeypatch):
    _FakeMenu.created = []
    monkeypatch.setattr(vid_mod, "QMenu", _FakeMenu)
    yield _FakeMenu


class TestFilePathCopy:
    r"""「文件路径」行的写法风格 / 引号 / 右键菜单（跨平台复制）。

    背景（用户实测报障）：Windows 上拖拽进主窗口的网盘文件，路径形如
    ``//fileserver/team-share/… TC01_ EU7 Calibration/…1am=0.8_Map.csv``
    —— 正斜杠 UNC + 空格 + ``=``，粘回 Windows 被 Shell 当 URL 交给浏览器
    （实测跳 Edge），只有 ``\\host\share`` 加引号才能直接用。本组用例盯的就是
    “复制出去能不能用”，以及风格开关是否真的在调用时生效。
    """

    #: 报障串的结构复刻（保留空格、``=`` 与两层前导斜杠；主机名/项目号/客户名
    #: 均为虚构标识符，请勿改回真实内网路径）
    RAW = (
        "//fileserver/team-share/PRJ-0000-00_Demo_ENG01_TC01_ EU7 Calibration"
        "/20260101_DEMO_ENG01_1am=0.8_Map.csv"
    )
    WIN = RAW.replace("/", "\\")

    def _add_path_row(self, pg, value=RAW):
        """手工注入一行「文件路径」并滚入视口（同长路径用例的做法）。"""
        tree = pg.tree
        item = QTreeWidgetItem(tree, [var_info.ROW_KEY_FILE_PATH, value])
        tree.scrollToItem(item)
        settle()
        return item

    def _click_copy_button(self, pg, item):
        tree = pg.tree
        QApplication.clipboard().setText(_SENTINEL)
        btn = copy_button_rect(_value_cell_rect(tree, item))
        QTest.mouseClick(
            tree.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, btn.center(),
        )
        settle()
        return QApplication.clipboard().text()

    def test_row_key_is_the_literal_ui_and_data_agree_on(self):
        """键名是 UI 查行的依据，被改字面就会静默失效（菜单与引号全不生效）。"""
        assert var_info.ROW_KEY_FILE_PATH == "文件路径"

    def test_button_follows_windows_style_from_config(self, page, monkeypatch):
        r"""PATH_COPY_STYLE=windows → 反斜杠 + UNC 两层前导保留 + 自动加引号。

        反向验证两处：旧写法（直接 ``setText(item.text(1))``）会原样给出
        正斜杠串 —— 就是报障里跳浏览器的那一条；而把风格常量绑在 def
        默认参数上时，monkeypatch 不生效，拿到的仍是 native。
        """
        dlg, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_STYLE", "windows")
        item = self._add_path_row(pg)

        copied = self._click_copy_button(pg, item)

        assert copied == '"' + self.WIN + '"'
        # 拆写避免转义看错：前导必为两个反斜杠（UNC），其后各段单反斜杠
        unc_prefix = '"' + "\\\\" + "fileserver" + "\\"
        assert copied.startswith(unc_prefix)
        assert dlg.status_label.text() == "已复制「文件路径」"

    def test_default_native_style_adds_quotes_but_keeps_slashes(self, page):
        """默认 native：macOS 上保持正斜杠，但含空格/``=`` 必须已被引号保护。"""
        _, pg = page
        item = self._add_path_row(pg)

        assert self._click_copy_button(pg, item) == '"' + self.RAW + '"'

    def test_quote_never_config_disables_wrapping(self, page, monkeypatch):
        """PATH_COPY_QUOTE=never → 粘进 Excel 单元格时不带引号。"""
        _, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_QUOTE", "never")
        item = self._add_path_row(pg)

        assert self._click_copy_button(pg, item) == self.RAW

    def test_quote_always_wraps_a_clean_path(self, page, monkeypatch):
        """PATH_COPY_QUOTE=always → 无特殊字符也包引号（对齐 Explorer）。"""
        _, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_STYLE", "windows")
        monkeypatch.setattr(vid_mod, "PATH_COPY_QUOTE", "always")
        clean = r"D:\Messung\x.dat"
        item = self._add_path_row(pg, clean)

        assert self._click_copy_button(pg, item) == '"' + clean + '"'

    def test_other_rows_are_copied_verbatim(self, page):
        """只有路径行参与转换：其余行必须原样取走（多一层处理反而不可预期）。"""
        _, pg = page
        item = _find_row(pg.tree, "变量名")

        assert self._click_copy_button(pg, item) == "speed"

    def test_context_menu_offers_three_styles(self, page, fake_menu, monkeypatch):
        """右键路径行 → 三项菜单，逐项得到各自写法（默认项跟配置走）。"""
        _, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_QUOTE", "never")
        item = self._add_path_row(pg)
        pos = _value_cell_rect(pg.tree, item).center()

        pg._on_tree_context_menu(pos)

        assert len(fake_menu.created) == 1
        actions = fake_menu.created[0].items
        assert len(actions) == 3
        assert [a.text for a in actions] == [
            "复制为 当前平台",
            "复制为 Windows \\host\\share",
            "复制为 POSIX //host/share",
        ]
        QApplication.clipboard().setText(_SENTINEL)
        actions[2].fire()
        assert QApplication.clipboard().text() == self.RAW
        actions[1].fire()
        assert QApplication.clipboard().text() == self.WIN

    def test_context_menu_skipped_for_other_rows_and_placeholder(
        self, page, fake_menu
    ):
        """非路径行不弹菜单；路径行的值为占位符 ``-`` 时也不弹。"""
        _, pg = page
        other = _find_row(pg.tree, "变量名")
        pg._on_tree_context_menu(_value_cell_rect(pg.tree, other).center())
        assert fake_menu.created == [], "非路径行不得弹菜单"

        dash = self._add_path_row(pg, "-")
        pg._on_tree_context_menu(_value_cell_rect(pg.tree, dash).center())
        assert fake_menu.created == [], "空路径无内容可复制，不得弹菜单"

    def test_ctrl_c_uses_the_same_formatting(self, page, monkeypatch):
        """Ctrl+C 与行尾按钮口径一致，否则会得到两种写法。"""
        _, pg = page
        monkeypatch.setattr(vid_mod, "PATH_COPY_STYLE", "windows")
        item = self._add_path_row(pg)
        pg.tree.clearSelection()
        item.setSelected(True)

        pg._on_copy_selection()

        text = QApplication.clipboard().text()
        assert text.startswith(var_info.ROW_KEY_FILE_PATH + "\t")
        assert text.split("\t")[1] == '"' + self.WIN + '"'

    def test_placeholder_path_is_not_promoted_to_a_fake_path(self, page):
        r"""值为占位符 ``-`` 时，三条复制出口一律原样给出 ``-``。

        反向验证：缺守卫时 ``display_path("-")`` 会把它当相对路径按当前工作
        目录绝对化，复制出 ``<cwd>/-`` —— 一条看着能用、实际打不开的假路径，
        而状态栏照样提示「已复制」，用户无从察觉。右键菜单此前已在调用点
        守了这条，本用例盯的是守卫进唯一出口后三条路径口径一致。
        """
        _, pg = page
        dash = self._add_path_row(pg, "-")

        # 唯一出口本身：三种风格一律不得改写占位符，空串同样原样返回
        for style in (None, "windows", "posix"):
            assert pg._format_path("-", style) == "-"
        assert pg._format_path("", None) == ""

        # Ctrl+C：此前唯一可达的漏口（行尾按钮被 is_copyable 拦在绘制层）
        pg.tree.clearSelection()
        dash.setSelected(True)
        QApplication.clipboard().setText(_SENTINEL)

        pg._on_copy_selection()

        assert QApplication.clipboard().text() == var_info.ROW_KEY_FILE_PATH + "\t-"

        # 行尾按钮的回调直调：is_copyable 只挡绘制，挡不住未来新增的 emit 方
        QApplication.clipboard().setText(_SENTINEL)
        pg._on_copy_field(dash)
        assert QApplication.clipboard().text() == "-"

    def test_tree_has_custom_context_menu_policy(self, page):
        """接线闸：菜单策略没改成 CustomContextMenu 时槽函数永不被调。"""
        _, pg = page
        assert pg.tree.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
