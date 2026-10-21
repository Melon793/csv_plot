"""CanDbBanner / 「CAN 数据库…」按钮的组件测试（方案 §5.3 + §5.5 + §5.6）。

覆盖的是**状态呈现与联动**，不是解码数值（那在 ``test_blf_loader.py``）：

* 三态（正常 / 截断 / 有未匹配）的配色、文案与按钮组合
* 未匹配数是「常驻横幅 + 链接」而不是模态框（§5.1 的非阻塞原则）
* 长文件名 elide、``hide_banner`` 只隐不删（R6：避免 splitter 抖动）
* ``_update_can_db_ui`` 按 ``LOADER_TYPE`` 同步 banner 三态与 CAN 段显隐
  （段文案与 banner 同源：段写摘要、banner 两行全文进 tooltip）
* ``show_can_unmatched`` 打开详情表并把「重选」回传
* ``reselect_can_database`` 重绑后清统计缓存、刷新变量列表、**不重扫报文**
"""

from __future__ import annotations

import os

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QWidget

from src.core.config import (
    BLF_BANNER_HEIGHT,
    BLF_BANNER_NAME_ELLIDE_AT,
    BLF_BANNER_STYLE_NORMAL,
    BLF_BANNER_STYLE_TRUNCATED,
    BLF_BANNER_STYLE_UNMATCHED,
    BLF_MAX_MESSAGES,
    BLF_MAX_PAYLOAD_BYTES,
)
from src.data.blf_loader import BLFLazyLoader
from src.ui.dialogs.can_db_selector import CanDatabaseSelectorDialog
from src.ui.widgets.can_db_banner import CanDbBanner, _ellide
from tests.fixtures.data_factory import write_blf_pair, write_dbc


def _frame(can_id=0x1A0, name="EngineData", sig="Speed", factor=1.0):
    return {
        "name": name,
        "can_id": can_id,
        "size": 8,
        "cycle_ms": 20,
        "sender": "ECU",
        "signals": (
            {
                "name": sig,
                "start": 0,
                "size": 16,
                "order": "intel",
                "factor": factor,
                "offset": 0.0,
                "minimum": 0.0,
                "maximum": 1000.0,
                "unit": "rpm",
                "sweep": 10,
            },
        ),
    }


@pytest.fixture()
def banner(qapp):
    host = QMainWindow()
    b = CanDbBanner(host)
    b._kept_alive = host  # 防止宿主先被 GC，Qt 侧对象随之销毁
    yield b
    b._kept_alive = None


def _label(banner) -> str:
    return banner.info_label.text()


# ---------------------------------------------------------------------------
# 三态本体（§5.3 表格）
# ---------------------------------------------------------------------------

def test_a_fresh_banner_is_explicitly_hidden(qapp):
    """默认隐藏：非 BLF 数据源连开机都不该看到它。"""
    b = CanDbBanner()
    assert b.isHidden() is True
    assert b.state == ""
    assert b.height() == BLF_BANNER_HEIGHT
    assert not _label(b)


def test_state_one_is_green_and_lists_every_loaded_database(banner):
    banner.show_normal(
        db_count=2, frame_count=148, signal_count=2143,
        db_names=["PowerTrain.dbc", "Chassis.arxml"],
    )
    assert banner.isHidden() is False
    assert banner.state == "normal"
    text = _label(banner)
    assert "已加载 2 个数据库" in text
    assert "148 报文 / 2143 信号" in text
    assert "PowerTrain.dbc" in text and "Chassis.arxml" in text
    assert BLF_BANNER_STYLE_NORMAL[0] in banner.styleSheet()
    assert BLF_BANNER_STYLE_NORMAL[1] in banner.info_label.styleSheet()
    # 全命中时不给「查看」入口，但「重选…」始终在（§5.6）
    assert banner.view_btn.isHidden() is True
    assert banner.reselect_btn.isHidden() is False


def test_state_two_is_amber_and_quotes_both_the_limit_and_the_scanned_count(banner):
    banner.show_truncated(limit=BLF_MAX_MESSAGES, scanned=30_000_000,
                          db_count=2, db_names=["a.dbc", "b.arxml"])
    assert banner.state == "truncated"
    text = _label(banner)
    assert f"报文数超过 {BLF_MAX_MESSAGES:,}" in text
    assert "仅扫描前 30,000,000 条" in text
    assert "已用 2 个数据库" in text
    assert BLF_BANNER_STYLE_TRUNCATED[0] in banner.styleSheet()
    # 截断与「未匹配」是两回事，别把「查看」按钮顺手带出来
    assert banner.view_btn.isHidden() is True


def test_state_two_can_quote_the_payload_byte_limit_instead(banner):
    """字节上限触发时不能再谎称「报文数超过 N 条」。"""
    banner.show_truncated(
        limit=BLF_MAX_PAYLOAD_BYTES, scanned=8_000_000, db_count=1,
        db_names=["a.dbc"], reason="payload",
    )
    assert banner.state == "truncated"
    text = _label(banner)
    assert f"报文数据超过 {BLF_MAX_PAYLOAD_BYTES // (1024 * 1024)} MB" in text
    assert "报文数超过" not in text
    assert "仅扫描前 8,000,000 条" in text


def test_state_three_turns_the_count_into_a_link_and_a_view_button(banner):
    banner.show_unmatched(unmatched_count=37, db_count=2, frame_count=148,
                          signal_count=2143, db_names=["a.dbc"])
    assert banner.state == "unmatched"
    text = _label(banner)
    assert 'href="unmatched"' in text
    assert "37 个报文 ID 未在数据库中定义" in text
    assert "（未生成信号）" in text
    assert BLF_BANNER_STYLE_UNMATCHED[0] in banner.styleSheet()
    assert banner.view_btn.isHidden() is False
    # 富文本链接要能点：默认的文本报互模式会把它当选词操作
    flags = banner.info_label.textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextBrowserInteraction
    assert banner.info_label.openExternalLinks() is False


def test_clicking_the_link_opens_the_same_detail_as_the_button(banner):
    seen = []
    banner.view_unmatched_requested.connect(lambda: seen.append("link"))
    banner.show_unmatched(unmatched_count=2, db_count=1, frame_count=5,
                          signal_count=7, db_names=["a.dbc"])
    banner._on_link_activated("unmatched")
    assert seen == ["link"]

    banner.view_btn.click()
    assert seen == ["link", "link"], "「查看」按钮与链接必须走同一个出口"

    banner.show_normal(db_count=1, frame_count=5, signal_count=7)
    banner._on_link_activated("unmatched")
    assert seen == ["link", "link"], "状态①没有未匹配，链接不该再发信号"


def test_the_reselect_button_carries_the_no_rescan_promise(banner):
    seen = []
    banner.reselect_requested.connect(lambda: seen.append(True))
    assert "无需重新扫描报文" in banner.reselect_btn.toolTip()
    banner.reselect_btn.click()
    assert seen == [True]


def test_long_database_names_are_elided_but_keep_their_extension(banner):
    long_name = "PowerTrain_" + "x" * 60 + ".dbc"
    banner.show_normal(db_count=1, frame_count=2, signal_count=3,
                       db_names=[long_name])
    shown = _label(banner)
    assert long_name not in shown
    assert len(_ellide(long_name)) == BLF_BANNER_NAME_ELLIDE_AT
    # 用户靠后缀判断类型，省略中段而不是尾段正是为此
    assert shown.endswith(".dbc")
    assert _ellide("short.dbc") == "short.dbc"


def test_database_names_are_escaped_in_every_state(banner):
    """A8：info_label 是 RichText，文件名里的 & / < 不能当标记吞掉。"""
    name = "A&B<2>.dbc"
    expected = "A&amp;B&lt;2&gt;.dbc"
    banner.show_normal(db_count=1, frame_count=2, signal_count=3,
                       db_names=[name])
    assert expected in _label(banner)
    banner.show_truncated(limit=BLF_MAX_MESSAGES, scanned=1, db_count=1,
                          db_names=[name])
    assert expected in _label(banner)
    banner.show_unmatched(unmatched_count=2, db_count=1, frame_count=2,
                          signal_count=3, db_names=[name])
    assert expected in _label(banner)
    assert "A&B<2>" not in _label(banner)


def test_a_second_database_list_replaces_rather_than_appends(banner):
    banner.show_normal(db_count=2, frame_count=1, signal_count=1,
                       db_names=["a.dbc", "b.dbc"])
    assert "b.dbc" in _label(banner)
    banner.show_normal(db_count=1, frame_count=1, signal_count=1,
                       db_names=["c.dbc"])
    text = _label(banner)
    assert "c.dbc" in text and "b.dbc" not in text


def test_hiding_leaves_the_widget_in_place_for_the_splitter(banner):
    """§5.3 / R6：只 setVisible(False)，绝不从布局里摘掉控件。"""
    parent = banner.parent()
    banner.show_truncated(limit=10, scanned=10, db_count=1, db_names=["a.dbc"])
    banner.hide_banner()
    assert banner.isHidden() is True
    assert banner.state == ""
    assert banner.parent() is parent
    assert banner.height() == BLF_BANNER_HEIGHT


# ---------------------------------------------------------------------------
# 与数据源联动：_update_can_db_ui
# ---------------------------------------------------------------------------

@pytest.fixture()
def blf_pair(tmp_path):
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,))
    return blf, dbc


def test_a_non_blf_source_hides_the_banner_and_the_segment(blf_flow):
    host, manager, _threads = blf_flow

    class _CsvLike:
        LOADER_TYPE = "csv"

    host.loader = _CsvLike()
    manager._update_can_db_ui()
    assert host.can_db_banner.isHidden() is True
    assert host._can_segment.isHidden() is True
    assert host._can_segment.text() == ""


def test_no_loader_at_all_is_not_a_crash(blf_flow):
    host, manager, _threads = blf_flow
    host.loader = None
    manager._update_can_db_ui()
    assert host.can_db_banner.isHidden() is True


def test_a_blf_with_unmatched_ids_lands_on_the_orange_state(blf_flow, blf_pair):
    host, manager, _threads = blf_flow
    blf, dbc = blf_pair
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        manager._update_can_db_ui()
        assert loader.unmatched_ids, "样本必须留一个库里没有的 ID"
        assert host.can_db_banner.state == "unmatched"
        assert host.can_db_banner.isHidden() is False
        n = len(loader.unmatched_ids)
        assert host._can_segment.isHidden() is False, "§6.3：段与 banner 同步显隐"
        assert f"{n} 个 ID 未匹配" in host._can_segment.text()
        # tooltip 是 banner 两行全文（含库名清单）的稳定出口
        assert host._can_segment.toolTip() == host.can_db_banner.info_label.text()
        assert f"{n} 个报文 ID 未在数据库中定义" in _label(host.can_db_banner)
        assert os.path.basename(str(dbc)) in _label(host.can_db_banner)
    finally:
        loader.close()


def test_a_matching_database_flips_the_banner_back_to_green(blf_flow, tmp_path):
    host, manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,), with_unmatched=False)
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        manager._update_can_db_ui()
        assert host.can_db_banner.state == "normal"
        # 状态①段文案：库数 · 信号数（数字与 banner 同一只 database 出的）
        assert host._can_segment.isHidden() is False
        assert host._can_segment.text() == (
            f"CAN：1 库 · {loader.database.signal_count} 信号"
        )
    finally:
        loader.close()


def test_the_truncated_state_wins_over_unmatched(blf_flow, tmp_path):
    """两个坏消息同时发生时，先说更严重的那个（§5.3 的优先级）。"""
    host, manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,))
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        loader.truncated = True
        host.loader = loader
        manager._update_can_db_ui()
        assert host.can_db_banner.state == "truncated"
        assert f"{BLF_MAX_MESSAGES:,}" in _label(host.can_db_banner)
        # 状态②段文案带扫描量（千分位），不报库数/信号数 —— 那两个数已不完整
        assert "仅前" in host._can_segment.text()
        assert f"{loader.total_messages:,}" in host._can_segment.text()
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# §5.5 未匹配详情
# ---------------------------------------------------------------------------

@pytest.fixture()
def unmatched_dialogs(monkeypatch):
    """截获 ``CanUnmatchedDialog`` 的构造，返回 ``(构造记录, 待返回实例)``。

    真对话框在 ``show_can_unmatched`` 里是函数内 import，所以直接替换模块属性
    即可；``exec()`` 在 offscreen 下绝不能真弹。
    """
    from src.ui.dialogs import can_unmatched_dialog as mod

    class _Built(list):
        """构造记录列表（list 本体挂不住属性，又需要 ``== []`` / 下标可用）。"""

        answers: dict = {}

    built = _Built()
    answers = {}

    class _Stub:
        def __init__(self, unmatched, *, total_messages=0, parent=None, limit=None):
            self.unmatched = unmatched
            self.total_messages = total_messages
            self.parent_widget = parent
            self.limit = limit
            self.reselect_requested = answers.get("reselect", False)
            built.append(self)

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(mod, "CanUnmatchedDialog", _Stub)
    built.answers = answers
    return built


def test_the_view_entry_opens_the_detail_with_the_loaders_own_numbers(
    blf_flow, blf_pair, unmatched_dialogs
):
    host, manager, _threads = blf_flow
    blf, dbc = blf_pair
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        manager.show_can_unmatched()
        assert len(unmatched_dialogs) == 1
        dlg = unmatched_dialogs[0]
        assert dlg.unmatched == list(loader.unmatched_ids)
        assert dlg.total_messages == loader.total_messages
        assert dlg.parent_widget is host
    finally:
        loader.close()


def test_there_is_nothing_to_show_when_every_id_matched(blf_flow, tmp_path,
                                                        unmatched_dialogs):
    host, manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,), with_unmatched=False)
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        manager.show_can_unmatched()
        assert unmatched_dialogs == [], "没有未匹配就不该有对话框可关"
    finally:
        loader.close()


def test_requesting_reselect_from_the_detail_hands_off_to_the_selector(
    blf_flow, blf_pair, unmatched_dialogs, monkeypatch
):
    host, manager, _threads = blf_flow
    blf, dbc = blf_pair
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        unmatched_dialogs.answers["reselect"] = True
        called = []
        monkeypatch.setattr(
            manager, "reselect_can_database",
            lambda: called.append(True),
        )
        manager.show_can_unmatched()
        assert called == [True]
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# §5.5 未匹配详情表（真对话框的渲染层）
# ---------------------------------------------------------------------------

def test_the_detail_table_prints_the_frame_format_the_data_layer_reported(qapp):
    """11 位与 29 位 ID 的**取值会重叠**：按 ``can_id > 0x7FF`` 猜帧格式，
    一个 0x123 的扩展帧就会被标成标准帧，用户照着提示去补库就补不上。
    """
    from src.ui.dialogs.can_unmatched_dialog import CanUnmatchedDialog

    rows = [(0, 0x123, False, 5), (1, 0x123, True, 9), (0, 0x7FF, False, 1)]
    dlg = CanUnmatchedDialog(rows, total_messages=15)
    cells = {dlg.table.item(r, 1).text() for r in range(dlg.table.rowCount())}
    assert cells == {"0x123", "0x123E", "0x7FF"}
    # 排序按帧数降位（第三列不再是计数）
    assert dlg.table.item(0, 2).text() == "9"
    assert dlg.table.item(0, 0).text() == "1"
    assert dlg.table.item(0, 3).text() == "60.00%"


def test_reselecting_hands_the_refresh_back_to_the_single_convergence_entry(
    blf_flow, blf_pair, accepted_selector
):
    """§5.6 的换库必须走 ``_apply_loader``：历史上手写精简版漏了 per-widget
    单位下发、数值表刷新、变量编辑器/信息窗刷新四项（表现是「换了库但表里
    还是老单位」）。这里钉「交回唯一入口」这个结构，而不是钉它内部的副作用。
    """
    host, manager, _threads = blf_flow
    blf, dbc = blf_pair
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        accepted_selector["paths"] = [str(dbc)]
        manager.reselect_can_database()
        assert ("apply", True) in manager.events, manager.events
        assert not any(e[0] == "swap" for e in manager.events if isinstance(e, tuple)), \
            "换库不能走 _swap_loader —— 它会释放正在用的 loader"
    finally:
        loader.close()


def test_reselecting_holds_a_wait_cursor_only_while_the_database_is_parsed(
    blf_flow, blf_pair, accepted_selector, monkeypatch
):
    """M3 回归：``bind_databases`` 仍在本线程同步解析 DBC/ARXML。

    没有进度框，至少要有等待光标；且无论解析成败都必须复原，不能把整个
    应用留在忙碌态。
    """
    host, manager, _threads = blf_flow
    blf, dbc = blf_pair
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        accepted_selector["paths"] = [str(dbc)]
        seen = []
        original = loader.bind_databases

        def spy(paths, **kwargs):
            seen.append(QApplication.overrideCursor() is not None)
            return original(paths, **kwargs)

        monkeypatch.setattr(loader, "bind_databases", spy)
        manager.reselect_can_database()
        assert seen == [True], "解析期间应挂着等待光标"
        assert QApplication.overrideCursor() is None, "返回后必须复原"
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# §5.6 重选数据库
# ---------------------------------------------------------------------------

@pytest.fixture()
def accepted_selector(monkeypatch):
    """让选择对话框直接 Accepted，并按 ``paths`` 模拟用户的增删。"""
    chosen = {"paths": []}

    def fake_exec(self, *args, **kwargs):
        pick = chosen["paths"] or self.get_selected_paths()
        known = {r.path for r in self._rows}
        # 等价于用户在框里点了「添加文件…」、又移除了其余行
        self.add_paths([p for p in pick if p not in known])
        for row in list(self._rows):
            if row.path not in pick:
                self._rows.remove(row)
        self._rebuild_table()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CanDatabaseSelectorDialog, "exec", fake_exec)
    return chosen


def test_reselecting_rebinds_without_rescanning_and_refreshes_the_list(
    blf_flow, tmp_path, accepted_selector
):
    host, manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,))
    alt = write_dbc(tmp_path / "alt.dbc", [_frame(can_id=0x1A0, factor=2.0)])
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        scanned_before = loader.total_messages
        buckets_before = len(loader._buckets)
        accepted_selector["paths"] = [str(alt)]

        manager.reselect_can_database()

        assert loader.db_paths == [str(alt)]
        # 「不重扫」是 banner 敢做成轻量交互的前提（§1 关键设计）：
        # 报文桶总数与帧总数都是扫描产物，完全不变
        assert loader.total_messages == scanned_before
        assert len(loader._buckets) == buckets_before
        # 而 ``group_count`` 只数「生成了变量的桶」，换库后必然跟着变
        assert loader.group_count < buckets_before
        assert host.var_names == loader.var_names
        assert host.units == loader.units
        assert host.list_widget.calls[-1][0] == loader.var_names
        assert host.var_stats_cache == {}, "同名变量换库后数值全变，缓存必须作废"
        assert host.cursor_sync_manager.cleared == 1, "旧曲线引用的是作废名字"
        assert host.can_db_banner.isHidden() is False
        # alt.dbc 只定义了 5 个报文桶里的 1 个，换库后 banner 必须跟着改口：
        # 断成 "normal" 就等于「换了库但状态条还显示老库的覆盖率」。
        assert host.can_db_banner.state == "unmatched"
        assert str(len(loader.unmatched_ids)) in host.can_db_banner.info_label.text()
        assert host.can_db_banner.view_btn.isHidden() is False
    finally:
        loader.close()


def test_a_rebind_that_yields_no_signal_rolls_back_and_keeps_the_data(
    blf_flow, tmp_path, accepted_selector, monkeypatch
):
    """换到一个完全不搭的库：要说清「没信号」，并回滚到原库与曲线。

    旧实现先重绑、再 warning、最后照样清曲线并应用空变量集 —— 一次误选就
    把用户画好的整张画布换成空列表，且没有撤销入口。
    """
    import src.ui.file_loader_manager as flm

    host, manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,))
    unrelated = write_dbc(tmp_path / "other.dbc",
                          [_frame(can_id=0x4C0, name="Unrelated", sig="X")])
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    warnings = []
    monkeypatch.setattr(
        flm.QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    try:
        host.loader = loader
        before = list(loader.var_names)
        host.var_names = list(before)
        before_banner = host.can_db_banner.state
        accepted_selector["paths"] = [str(unrelated)]
        manager.reselect_can_database()
        assert warnings, "0 信号必须有反馈，否则用户以为程序没响应"
        assert "已保留原数据库" in warnings[0][2]
        assert loader.db_paths == [str(dbc)], "误选必须回滚，不能写进 db_paths"
        assert loader.var_names == before, "旧库的元数据必须原样恢复"
        assert host.var_names == before
        assert host.list_widget.calls == [], "没有换库成功就不该重建变量列表"
        assert host.cursor_sync_manager.cleared == 0, "旧曲线不能被清掉"
        assert host.can_db_banner.state == before_banner
    finally:
        loader.close()


def test_cancelling_the_reselect_dialog_changes_nothing(blf_flow, blf_pair,
                                                        monkeypatch):
    host, manager, _threads = blf_flow
    blf, dbc = blf_pair
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    monkeypatch.setattr(
        CanDatabaseSelectorDialog, "exec",
        lambda self, *a, **k: QDialog.DialogCode.Rejected,
    )
    try:
        host.loader = loader
        before = list(loader.var_names)
        manager.reselect_can_database()
        assert loader.var_names == before
        assert host.list_widget.calls == []
        assert host.var_stats_cache, "没换库就不该清缓存"
    finally:
        loader.close()


def test_the_banner_lives_in_the_drawer_and_survives_state_changes(blf_flow):
    """R6 改锚点（§6.4）：banner 不再是左栏过滤框之上的一条，而是抽屉的固定头部行。

    约束不变：一开机就建好、归属稳定、切换三态不增删控件、也不改自身固定高。
    「点外面就关」的是抽屉（Popup）；收着时 ``_update_can_db_ui`` 照常刷三态
    —— ``isHidden()`` 只看显式隐藏，不受父层收起影响。
    """
    host, _manager, _threads = blf_flow
    banner = host.can_db_banner
    drawer = host._can_db_drawer
    # 归属错了的两种形态（浮在桌面的彩条 / 挂在 mw 但没入布局）都靠这一条钉住
    assert banner.window() is drawer, "banner 必须从构造起就是抽屉子控件"
    assert drawer.isHidden() is True, "开机只有一扇隐身抽屉，不许桌面上多一层窗"
    assert banner.height() == BLF_BANNER_HEIGHT

    children_before = len(drawer.findChildren(QWidget))
    banner.show_normal(db_count=1, frame_count=1, signal_count=1, db_names=["a.dbc"])
    banner.show_unmatched(
        unmatched_count=2, db_count=1, frame_count=1, signal_count=1,
        db_names=["a.dbc"],
    )
    banner.hide_banner()
    assert len(drawer.findChildren(QWidget)) == children_before, "显隐不得增删控件"
    assert banner.height() == BLF_BANNER_HEIGHT
