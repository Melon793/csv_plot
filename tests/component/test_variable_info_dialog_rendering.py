"""变量信息窗口：内容渲染、缓存与可见性去重。

由 test_variable_info_dialog.py 拆分而来（只搬运，未改任何断言）：
元数据同步渲染、缓存命中/失效/淘汰、可见页重校验去重、树呈现与「归属信息」
块、Markdown 导出。
"""

import threading
import time

import pytest

from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
)

from tests.fixtures.waits import (
    flush_deferred_deletes,
    settle,
    wait_until,
)
from src.data import mdf_attribution as mda
from src.data import var_info
from src.data.loader import FastDataLoader
from src.ui.dialogs import variable_info_dialog as vid_mod
from src.ui.dialogs.variable_info_dialog import (
    VariableInfoDialog,
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
)

from tests.component._vid_shared import _wait_stats, _wait_delivered

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
