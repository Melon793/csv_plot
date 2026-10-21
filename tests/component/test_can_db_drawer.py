"""CanDbDrawer 组件测试（§6.4）：库清单表、body 一次成型、动作转发。

三条约束各钉一处：

* **banner 归属** —— 从构造起就是抽屉的子控件（「先无 parent、开时 setParent」
  会在桌面上浮一条 44px 彩条，而旧断言的 ``isHidden()`` 看不见这种形态）；
* **body 只建一次** —— ``open_for()`` 重复调用后 banner 仍是活对象（基类
  ``set_body`` 会对旧 body 子控件 ``deleteLater``，重建一次就是死包装器）；
* **动作转发** —— 页脚「重选数据库…」与 banner 的两个入口都先把抽屉收掉，
  再把动作交给 ``file_loader_manager``（Popup 不能压在模态框上）。
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication

from src.core.config import BLF_DB_COLOR_ERROR, BLF_DB_COLOR_OK, BLF_DB_COLOR_WARN
from src.data.blf_loader import BLFLazyLoader
from src.data.can_database import DbFileReport
from src.ui.widgets.status_drawer import _can_db_status
from tests.fixtures.data_factory import write_blf_pair


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


# ---------------------------------------------------------------------------
# _can_db_status：三态词与配色（镜像选库框）
# ---------------------------------------------------------------------------

def test_the_status_word_mirrors_the_selector_and_colors_by_severity():
    assert _can_db_status(
        DbFileReport(path="a.dbc", ok=True, frame_count=3, signal_count=9)
    ) == ("✔ 已使用", BLF_DB_COLOR_OK)
    assert _can_db_status(
        DbFileReport(path="a.dbc", ok=True, frame_count=0)
    ) == ("⚠ 无 CAN 报文", BLF_DB_COLOR_WARN)
    text, color = _can_db_status(
        DbFileReport(path="a.dbc", ok=False, error="第一行\n第二行细节")
    )
    assert text == "✖ 解析失败: 第一行" and color == BLF_DB_COLOR_ERROR


# ---------------------------------------------------------------------------
# 开机形态与 body 复用
# ---------------------------------------------------------------------------

def test_the_eager_drawer_stays_hidden_until_first_open(blf_flow):
    host, _manager, _threads = blf_flow
    drawer = host._can_db_drawer
    assert drawer.isHidden() is True
    # 还没贴过任何状态栏：_anchor_bar 只由 open_above() 赋值
    assert drawer._anchor_bar is None
    assert host.can_db_banner.window() is drawer


def test_open_for_three_times_never_rebuilds_the_body(blf_flow, tmp_path):
    host, manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,))
    drawer = host._can_db_drawer
    banner = host.can_db_banner
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        host.resize(900, 640)
        host.show()
        QApplication.processEvents()
        for _ in range(3):
            assert drawer.open_for(host) is True
        # 三态刷新仍能落到 banner 上：旧实现（每次 set_body）在这一步
        # RuntimeError —— 第二次开抽屉起 banner 就是死包装器
        manager._update_can_db_ui()
        assert banner.isHidden() is False
        assert banner.window() is drawer
        assert drawer.table.rowCount() == len(loader.database.reports)
    finally:
        host.close()
        loader.close()


def test_the_table_mirrors_reports_including_the_failed_file(blf_flow, tmp_path):
    host, _manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,))
    # 坏文件必须真的解析不了：canmatrix 对垃圾 .dbc 是宽容的（读成 0 报文
    # 的"成功"），垃圾 .arxml 才会抛 XMLSyntaxError 落进 report.error
    broken = tmp_path / "broken.arxml"
    broken.write_text("NOT XML AT ALL\n", encoding="utf-8")
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc), str(broken)])
    try:
        host.loader = loader
        reports = loader.database.reports
        assert [r.ok for r in reports] == [True, False], "样本必须先好后坏"
        host._can_db_drawer._refresh(host)
        table = host._can_db_drawer.table
        assert table.rowCount() == 2
        assert table.item(0, 1).text() == os.path.basename(str(dbc))
        assert table.item(0, 1).toolTip() == str(dbc), "全路径留 tooltip"
        assert table.item(0, 2).text() == "DBC"
        assert table.item(0, 3).text() == (
            f"{reports[0].frame_count} / {reports[0].signal_count}"
        )
        assert table.item(0, 4).text() == "", "无备注时备注列为空"
        assert table.item(0, 5).text() == "✔ 已使用"
        assert table.item(1, 1).text() == os.path.basename(str(broken))
        assert table.item(1, 2).text() == "ARXML"
        assert table.item(1, 5).text().startswith("✖ 解析失败")
        assert table.item(1, 5).toolTip(), "失败原因要能悬停看到"
    finally:
        loader.close()
    # 行数回落也走同一张表（重新打开时 _refresh 再次跑）
    host.loader = None
    host._can_db_drawer._refresh(host)
    assert host._can_db_drawer.table.rowCount() == 0


def test_the_note_column_mirrors_the_user_remark(blf_flow, tmp_path):
    """抽屉的备注列是只读回显：写入口在选库对话框，这里只验证命中与 tooltip。"""
    from src.core.settings import AppSettings

    host, _manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,))
    AppSettings().set_blf_db_note(str(dbc), "动力总成主库")
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        host.loader = loader
        host._can_db_drawer._refresh(host)
        table = host._can_db_drawer.table
        assert table.item(0, 4).text() == "动力总成主库"
        assert table.item(0, 4).toolTip() == "动力总成主库", "长备注悬停可读全文"
        assert table.item(0, 5).text() == "✔ 已使用"
    finally:
        loader.close()
        AppSettings().set_blf_db_note(str(dbc), "")


# ---------------------------------------------------------------------------
# 动作转发：先收抽屉，再交给 manager
# ---------------------------------------------------------------------------

def test_view_unmatched_closes_the_drawer_first(blf_flow):
    host, manager, _threads = blf_flow
    manager.show_can_unmatched = lambda: host.drawer_events.append("show_unmatched")
    drawer = host._can_db_drawer
    drawer.show()
    QApplication.processEvents()
    drawer._on_view_unmatched()
    assert host.drawer_events == ["close_drawers", "show_unmatched"]
    assert drawer.isHidden() is True, "Popup 不许压在模态详情表上"


def test_both_reselect_entrances_close_the_drawer_first(blf_flow):
    host, manager, _threads = blf_flow
    manager.reselect_can_database = lambda: host.drawer_events.append("reselect")
    drawer = host._can_db_drawer
    drawer.show()
    drawer.reselect_btn.click()
    assert host.drawer_events == ["close_drawers", "reselect"]
    assert drawer.isHidden() is True
    # banner 头的「重选…」是同一落点
    host.drawer_events.clear()
    drawer.show()
    host.can_db_banner.reselect_btn.click()
    assert host.drawer_events == ["close_drawers", "reselect"]
