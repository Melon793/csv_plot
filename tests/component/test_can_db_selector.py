"""CanDatabaseSelectorDialog 的组件测试（方案 §5.2 + §8 的 component 层）。

覆盖的是**交互态与优先级语义**，不是 canmatrix 能不能解析某个文件（那部分
见 ``tests/unit/data/test_can_database.py``）：

* 空列表 / 全失败时「导入」禁用，且提示语区分「还没加文件」与「加了但没用」
* 坏文件标红**但不阻断**好文件（§5.2 的容错承诺）
* 列表顺序即优先级：↑/↓ 调序真的会换掉 first-match-wins 的赢家
* 拖入非 ``.dbc`` / ``.arxml`` 被拒
* 记忆命中但文件缺失（§5.1 的 C 路径）：标红 + 顶部黄条
* 「导入」后勾选记忆则写 ``BLF_LAST_DB_PATHS``（不勾则清空）；「取消」不写
* 后台预扫（不允许静默导入后的匹配度增强）：命中列回填 + 未编辑时按命中排序
* §5.2 的尺寸与列宽规格（项目规范：UI 容器需显式设置尺寸）

统一用合成 DBC：``canmatrix`` 的 DBC 导入器是逐行正则，垃圾文本会「解析成功
但 0 报文」，因此 ⚠ 与 ✖ 两种失败态分别用「空 dbc」与「坏 arxml」造。
"""

from __future__ import annotations

import os

import pytest

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QMainWindow,
)

from src.core.config import (
    BLF_DB_COL_WIDTHS,
    BLF_DB_COLOR_ERROR,
    BLF_DB_COLOR_OK,
    BLF_DB_COLOR_WARN,
    BLF_DB_DIALOG_MIN_SIZE,
    BLF_DB_DIALOG_SIZE,
)
from src.core.settings import AppSettings
from src.data.can_database import CanDatabaseSet
from src.ui.dialogs.can_db_selector import CanDatabaseSelectorDialog
from src.ui.file_loader_manager import FileLoaderManager
from tests.fixtures.data_factory import (
    CAN_FRAMES,
    write_arxml,
    write_blf_pair,
    write_dbc,
)

# 保活容器：offscreen 下手工构造的 QMimeData 不受 Qt 事件系统接管，局部变量
# 被 GC 后事件里访问 mimeData 会 SIGSEGV（tests/README.md 陷阱 #2）
_keep_alive: list = []


def _frame(can_id=0x1A0, name="EngineData", sig="Speed", factor=1.0, unit="rpm"):
    """单报文单信号的最小帧定义：只保留能影响状态列/优先级的字段。"""
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
                "unit": unit,
                "sweep": 10,
            },
        ),
    }


@pytest.fixture()
def dbc_a(tmp_path):
    return write_dbc(tmp_path / "A.dbc", [_frame(factor=1.0)])


@pytest.fixture()
def dbc_b(tmp_path):
    """与 ``dbc_a`` **同一 CAN ID**、不同 factor —— 优先级场景的最小样本。"""
    return write_dbc(tmp_path / "B.dbc", [_frame(factor=2.5)])


def _cell(dialog, row, col) -> str:
    item = dialog.table.item(row, col)
    return "" if item is None else item.text()


def _status_color(dialog, row) -> str:
    return dialog.table.item(row, 6).foreground().color().name()


def _import_btn(dialog):
    return dialog.button_box.button(QDialogButtonBox.StandardButton.Open)


# ---------------------------------------------------------------------------
# 规格（§5.2）
# ---------------------------------------------------------------------------

def test_the_dialog_keeps_the_specified_sizes(qapp, dbc_a):
    """尺寸/列宽是 §5.2 明文规定的，改了就必须在这里同步改。"""
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a)])
    assert dlg.minimumSize().width() == BLF_DB_DIALOG_MIN_SIZE[0]
    assert dlg.minimumSize().height() == BLF_DB_DIALOG_MIN_SIZE[1]
    assert dlg.size().width() >= BLF_DB_DIALOG_SIZE[0]
    assert dlg.size().height() >= BLF_DB_DIALOG_SIZE[1]
    assert dlg.table.columnWidth(0) == BLF_DB_COL_WIDTHS["priority"]
    assert dlg.table.columnWidth(2) == BLF_DB_COL_WIDTHS["type"]
    assert dlg.table.columnWidth(3) == BLF_DB_COL_WIDTHS["counts"]
    assert dlg.table.columnWidth(4) == BLF_DB_COL_WIDTHS["hit"]
    assert dlg.table.columnWidth(5) == BLF_DB_COL_WIDTHS["note"]
    assert dlg.table.columnWidth(6) == BLF_DB_COL_WIDTHS["status"]
    # 文件名列必须是唯一的弹性列，否则长文件名会把状态列挤出可视区
    assert (
        dlg.table.horizontalHeader().sectionResizeMode(1)
        == QHeaderView.ResizeMode.Stretch
    )
    assert dlg.table.editTriggers() == (
        dlg.table.EditTrigger.DoubleClicked | dlg.table.EditTrigger.EditKeyPressed
    ), "只有备注列可编辑（flags 摘除，见 test_the_note_column_is_the_only_editable_one）"
    assert dlg.table.selectionBehavior() == dlg.table.SelectionBehavior.SelectRows
    assert dlg.table.selectionMode() == dlg.table.SelectionMode.SingleSelection
    assert dlg.table.alternatingRowColors() is True
    assert dlg.acceptDrops() is True, "§5.2：把 dbc/arxml 直拖进对话框"


def test_the_header_explains_why_a_database_is_needed(qapp, tmp_path):
    blf = tmp_path / "log.blf"
    blf.write_bytes(b"\x00" * 2048)
    dlg = CanDatabaseSelectorDialog(data_file=str(blf))
    titles = [w for w in dlg.children() if isinstance(w, QLabel)]
    assert any("BLF 只记录原始报文" in w.text() for w in titles)
    assert "log.blf" in dlg.subtitle.text()


# ---------------------------------------------------------------------------
# 「导入」按钮的可用性（§5.2 空列表 / 全失败）
# ---------------------------------------------------------------------------

def test_an_empty_list_disables_import_with_a_tell_whole_hint(qapp):
    dlg = CanDatabaseSelectorDialog()
    assert _import_btn(dlg).isEnabled() is False
    assert _import_btn(dlg).toolTip() == "请先添加 DBC / ARXML 文件"
    # 没选中行时三个行操作按钮都该是灰的，点了才知道没反应很挫败
    assert dlg.remove_btn.isEnabled() is False
    assert dlg.up_btn.isEnabled() is False
    assert dlg.down_btn.isEnabled() is False


def test_a_usable_dbc_lights_up_import(qapp, dbc_a):
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a)])
    assert len(dlg.rows) == 1
    assert _import_btn(dlg).isEnabled() is True
    assert _import_btn(dlg).toolTip() == ""
    assert dlg.get_selected_paths() == [str(dbc_a)]
    assert _cell(dlg, 0, 3) == "1 / 1"
    assert _cell(dlg, 0, 6) == "✔ 已加载"
    assert _status_color(dlg, 0) == BLF_DB_COLOR_OK


def test_only_unreadable_files_leaves_import_disabled_with_a_different_hint(qapp,
                                                                           tmp_path):
    junk = tmp_path / "broken.arxml"
    junk.write_text("这不是 XML\n", encoding="utf-8")
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(junk)])
    assert _import_btn(dlg).isEnabled() is False
    # 与「还没加文件」区分的另一句：加了但一个都没读出来
    assert _import_btn(dlg).toolTip() == "至少选择一个可解析的数据库文件"
    assert _cell(dlg, 0, 6).startswith("✖ 解析失败")
    assert _status_color(dlg, 0) == BLF_DB_COLOR_ERROR
    # 完整异常放 tooltip，状态列只放第一行（否则整列被堆栈撑爆）
    assert "XMLSyntaxError" in dlg.table.item(0, 6).toolTip()


def test_accepting_with_no_usable_file_warns_and_keeps_the_dialog_open(qapp,
                                                                       tmp_path,
                                                                       monkeypatch):
    import src.ui.dialogs.can_db_selector as mod

    asked = []
    monkeypatch.setattr(
        mod.QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: asked.append(a)),
    )
    junk = tmp_path / "broken.arxml"
    junk.write_text("这不是 XML\n", encoding="utf-8")
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(junk)])
    dlg._on_accept()
    assert asked, "全失败时点导入必须有说明，否则用户以为程序卡住"
    assert "无法解析" in asked[0][2]
    assert dlg.result() != QDialog.DialogCode.Accepted


# ---------------------------------------------------------------------------
# 容错：坏文件不阻断好文件
# ---------------------------------------------------------------------------

def test_a_broken_file_sits_next_to_a_good_one_without_blocking_it(qapp,
                                                                   tmp_path,
                                                                   dbc_a):
    junk = tmp_path / "broken.arxml"
    junk.write_text("这不是 XML\n", encoding="utf-8")
    dlg = CanDatabaseSelectorDialog()
    assert dlg.add_paths([str(junk), str(dbc_a)]) == 2
    assert _status_color(dlg, 0) == BLF_DB_COLOR_ERROR
    assert _cell(dlg, 1, 6) == "✔ 已加载"
    # 坏行被跳过后仍可直接导入：这是「单文件失败不中断」的 UI 侧兑现
    assert _import_btn(dlg).isEnabled() is True
    assert dlg.get_selected_paths() == [str(dbc_a)]
    assert dlg.all_paths() == [str(junk), str(dbc_a)]


def test_a_dbc_that_parses_to_no_frames_warns_instead_of_failing(qapp, tmp_path):
    """⚠ 与 ✖ 的分界必须守住：0 报文是「库是空的」，不是「库读坏了」。"""
    empty = tmp_path / "empty.dbc"
    empty.write_text('VERSION ""\n\nNS_ :\n\nBS_:\n\nBU_:\n\n', encoding="utf-8")
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(empty)])
    assert _cell(dlg, 0, 6) == "⚠ 无 CAN 报文"
    assert _status_color(dlg, 0) == BLF_DB_COLOR_WARN
    assert _cell(dlg, 0, 3) == "0 / 0"
    # 「无 CAN 报文」的库对解码毫无贡献，不能让它点亮导入按钮
    assert _import_btn(dlg).isEnabled() is False
    assert dlg.get_selected_paths() == []


# ---------------------------------------------------------------------------
# 优先级 = 列表顺序（§2「数据库匹配」+ §5.2）
# ---------------------------------------------------------------------------

def test_reordering_the_list_changes_which_database_wins(qapp, dbc_a, dbc_b):
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a), str(dbc_b)])
    assert dlg.get_selected_paths() == [str(dbc_a), str(dbc_b)]
    assert _cell(dlg, 0, 0) == "1" and _cell(dlg, 1, 0) == "2"

    dlg.table.selectRow(1)
    assert dlg.up_btn.isEnabled() is True
    dlg.up_btn.click()
    assert dlg.get_selected_paths() == [str(dbc_b), str(dbc_a)]
    assert _cell(dlg, 0, 0) == "1" and _cell(dlg, 1, 0) == "2"

    # 顺序不是显示把戏：同一个 ID 的赢家真的会跟着换
    def winner_factor(paths):
        frame = CanDatabaseSet(paths).find_frame(0, 0x1A0)
        return frame.signals[0].factor

    assert winner_factor([str(dbc_a), str(dbc_b)]) == pytest.approx(1.0)
    assert winner_factor(dlg.get_selected_paths()) == pytest.approx(2.5)

    # ↓ 能原样换回去
    dlg.table.selectRow(0)
    dlg.down_btn.click()
    assert dlg.get_selected_paths() == [str(dbc_a), str(dbc_b)]


def test_the_edge_rows_only_offer_the_outward_move_button(qapp, dbc_a, dbc_b):
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a), str(dbc_b)])
    dlg.table.selectRow(0)
    assert dlg.up_btn.isEnabled() is False, "首行没有上方，按钮该灰而不是点了没反应"
    assert dlg.down_btn.isEnabled() is True
    dlg.table.selectRow(1)
    assert dlg.up_btn.isEnabled() is True
    assert dlg.down_btn.isEnabled() is False


def test_remove_drops_the_selected_row(qapp, dbc_a, dbc_b):
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a), str(dbc_b)])
    dlg.table.selectRow(0)
    dlg.remove_btn.click()
    assert dlg.get_selected_paths() == [str(dbc_b)]
    # 删一行后焦点自然落到新的首行，「移除」仍可用（刻意不强制取消选中）
    dlg.remove_btn.click()
    assert dlg.rows == []
    assert dlg.get_selected_paths() == []
    assert _import_btn(dlg).isEnabled() is False
    assert dlg.remove_btn.isEnabled() is False, "真没行了，按钮才该回灰"


# ---------------------------------------------------------------------------
# 拖拽（§5.2「支持把 .dbc/.arxml 拖入列表」）
# ---------------------------------------------------------------------------

def _url_event(cls, paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
    _keep_alive.append(mime)
    # QDropEvent 吃 QPointF（局部坐标），QDragEnterEvent 只接受 QPoint
    # （窗口坐标）—— 两个构造器签名不兼容，不能共用一个 pos
    pos = QPoint(10, 10) if cls is QDragEnterEvent else QPointF(10.0, 10.0)
    return cls(pos, Qt.DropAction.CopyAction, mime,
               Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


def test_dragging_in_a_mixed_bundle_keeps_only_the_databases(qapp, dbc_a, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    arxml = write_arxml(tmp_path / "sys.arxml")
    dlg = CanDatabaseSelectorDialog()

    enter = _url_event(QDragEnterEvent, [str(csv), str(dbc_a)])
    dlg.dragEnterEvent(enter)
    assert enter.isAccepted(), "有可收的文件就必须给拖放反馈"

    dlg.dropEvent(_url_event(QDropEvent, [str(csv), str(dbc_a), str(arxml)]))
    added = [os.path.basename(r.path) for r in dlg.rows]
    assert added == ["A.dbc", os.path.basename(str(arxml))]
    assert dlg.table.item(1, 2).text() == "ARXML", "类型列要跟后缀一致"


def test_dragging_only_rejected_files_does_not_accept_the_drop(qapp, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    dlg = CanDatabaseSelectorDialog()
    enter = _url_event(QDragEnterEvent, [str(csv)])
    dlg.dragEnterEvent(enter)
    assert not enter.isAccepted(), "全是不收的后缀还显示可放置，是纯误导"
    before = len(dlg.rows)
    dlg.dropEvent(_url_event(QDropEvent, [str(csv)]))
    assert len(dlg.rows) == before


def test_duplicate_paths_are_not_added_twice(qapp, dbc_a):
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a)])
    assert dlg.add_paths([str(dbc_a)]) == 0
    assert len(dlg.rows) == 1


# ---------------------------------------------------------------------------
# C 路径：记忆命中但有文件缺失（§5.1）
# ---------------------------------------------------------------------------

def test_missing_remembered_files_are_listed_in_the_notice(qapp, dbc_a, tmp_path):
    gone = tmp_path / "old_fota.dbc"
    dlg = CanDatabaseSelectorDialog(
        initial_paths=[str(gone), str(dbc_a)], missing_paths=[str(gone)]
    )
    assert dlg.notice.isVisibleTo(dlg) is True
    assert "old_fota.dbc" in dlg.notice.text()
    assert _cell(dlg, 0, 6) == "✖ 文件不存在"
    assert _cell(dlg, 0, 3) == "— / —", "缺失行不该拿 0 冒充「读到了空库」"
    # 缺失行被跳过，但剩下的仍可用 → 不必逼用户重来
    assert dlg.get_selected_paths() == [str(dbc_a)]
    assert _import_btn(dlg).isEnabled() is True


def test_missing_file_names_are_escaped_in_the_notice(qapp, tmp_path):
    """A8：提示条是 RichText，文件名里的 & / < 不能当标记吞掉。"""
    weird = tmp_path / "A&B<2>.dbc"
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(weird)])
    try:
        text = dlg.notice.text()
        assert "A&amp;B&lt;2&gt;.dbc" in text
        assert "A&B<2>" not in text
    finally:
        dlg.deleteLater()


# ---------------------------------------------------------------------------
# 记忆写入（§5.2「导入成功后写，取消不写」）
# ---------------------------------------------------------------------------

@pytest.fixture()
def flow_host(qapp):
    """最小宿主：只为跑 :meth:`FileLoaderManager._ask_can_databases`。

    该方法的实际依赖只有 ``mw``（当对话框 parent）与 AppSettings，其余
    加载机械一概不碰 —— 完整链路在 ``test_blf_import_flow.py`` 里覆盖。
    """
    mw = QMainWindow()
    mgr = FileLoaderManager(mw)
    # 基类持的是弱引用：测试里 mw 没有其它拥有者，不留一手会在用例
    # 跑一半时被 GC 抽走，报「MainWindow has been garbage collected」
    mgr._kept_alive = mw
    yield mgr
    mgr._kept_alive = None


@pytest.fixture()
def fresh_blf_settings(app_settings):
    """把 BLF 记忆键擦干净。

    ``app_settings`` 只重置单例，不删已写入的值：QSettings 是文件级的，
    上一个用例写的路径会漏到下一个用例里，“取消不写”就会被旧值伪造成绩。
    """
    app_settings.set_blf_last_db_paths([])
    return app_settings


def _auto_answer(monkeypatch, *, accept: bool, prepare=None) -> None:
    """把选择框的 ``exec()`` 换成固定答案（offscreen 下绝不能真弹模态）。"""
    code = (
        QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected
    )

    def fake_exec(self, *args, **kwargs):
        if prepare is not None:
            prepare(self)
        return code

    monkeypatch.setattr(CanDatabaseSelectorDialog, "exec", fake_exec)


def test_a_confirmed_selection_is_remembered_when_the_box_is_checked(
    flow_host, monkeypatch, dbc_a, fresh_blf_settings
):
    _auto_answer(monkeypatch, accept=True)
    paths = flow_host._ask_can_databases("log.blf", initial=[str(dbc_a)])
    assert paths == [str(dbc_a)]
    assert AppSettings().get_blf_last_db_paths() == [str(dbc_a)]


def test_unchecking_remember_clears_the_memory_instead_of_writing_it(
    flow_host, monkeypatch, dbc_a, fresh_blf_settings
):
    """复选框语义：勾选 =「记住此列表，下次自动预选」；不勾 = 不记住，
    连旧记忆一起清掉，否则旧列表会违背用户意图地在下次弹窗里出现。"""
    fresh_blf_settings.set_blf_last_db_paths([str(dbc_a)])
    _auto_answer(
        monkeypatch,
        accept=True,
        prepare=lambda dialog: dialog.remember_check.setChecked(False),
    )
    assert flow_host._ask_can_databases("log.blf", initial=[str(dbc_a)])
    assert AppSettings().get_blf_last_db_paths() == []


def test_cancelling_the_dialog_writes_nothing(flow_host, monkeypatch,
                                              dbc_a, fresh_blf_settings):
    _auto_answer(monkeypatch, accept=False)
    assert flow_host._ask_can_databases("log.blf", initial=[str(dbc_a)]) is None
    assert AppSettings().get_blf_last_db_paths() == []


# ---------------------------------------------------------------------------
# 用户备注：仅备注列可编辑、随改随存、打开回显、跟随行移动
# ---------------------------------------------------------------------------

def test_the_note_column_is_the_only_editable_one(app_settings, dbc_a):
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a)])
    assert dlg.table.item(0, 5).flags() & Qt.ItemFlag.ItemIsEditable
    for col in (0, 1, 2, 3, 4, 6):
        assert not (
            dlg.table.item(0, col).flags() & Qt.ItemFlag.ItemIsEditable
        ), f"第 {col} 列不该可编辑（备注列是唯一的输入点）"


def test_note_edits_persist_immediately(app_settings, dbc_a):
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a)])
    dlg.table.item(0, 5).setText("  动力总成库  ")
    # 随改随存（不等「导入」：备注是文件的属性，与本次导入是否发生无关），
    # 且落盘前去空白
    assert app_settings.get_blf_db_note(str(dbc_a)) == "动力总成库"
    assert dlg.rows[0].note == "动力总成库"
    assert _cell(dlg, 0, 5) == "动力总成库", "item 文本要与落盘值一致"


def test_a_stored_note_is_echoed_and_can_be_cleared(app_settings, dbc_a):
    app_settings.set_blf_db_note(str(dbc_a), "主库")
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a)])
    assert _cell(dlg, 0, 5) == "主库", "打开对话框要回显已存备注"
    dlg.table.item(0, 5).setText("  ")
    assert app_settings.get_blf_db_note(str(dbc_a)) == ""
    assert _cell(dlg, 0, 5) == "", "清空即删除条目"


def test_notes_follow_the_row_when_reordering(app_settings, dbc_a, dbc_b):
    app_settings.set_blf_db_note(str(dbc_a), "A 库")
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a), str(dbc_b)])
    dlg.table.selectRow(0)
    dlg.down_btn.click()
    assert [r.path for r in dlg.rows] == [str(dbc_b), str(dbc_a)]
    assert [r.note for r in dlg.rows] == ["", "A 库"], "备注必须跟着行走"


# ---------------------------------------------------------------------------
# 后台预扫：命中列与自动排序（不允许静默导入后的匹配度增强）
# ---------------------------------------------------------------------------

def _pump_until(dialog, cond, timeout_ms: int = 5000) -> None:
    from PySide6.QtCore import QCoreApplication
    import time

    end = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if cond():
            return
        time.sleep(0.005)
    pytest.fail("预扫没有在超时时间内完成")


def test_prescan_fills_the_hit_column_and_sorts_by_match(qapp, tmp_path):
    """预扫完成后命中列回填、未编辑的列表自动把最匹配的库顶上去。"""
    blf, dbc_full = write_blf_pair(tmp_path, n=24, buses=(0,))
    partial_dbc = write_dbc(tmp_path / "partial.dbc", CAN_FRAMES[:1])
    dlg = CanDatabaseSelectorDialog(
        data_file=str(blf), initial_paths=[str(partial_dbc), str(dbc_full)]
    )
    try:
        assert dlg._prescan_state == "running"
        _pump_until(dlg, lambda: dlg._prescan_state in ("done", "failed"))
        assert dlg._prescan_state == "done"
        assert [r.path for r in dlg.rows] == [str(dbc_full), str(partial_dbc)]
        hits = [r.hit_frames for r in dlg.rows]
        assert hits[0] > hits[1] >= 0, "全量库的命中帧数必须高于只含单帧的库"
        assert _cell(dlg, 0, 4).endswith("%")
        assert _cell(dlg, 1, 4).endswith("%")
        assert "已按命中帧数" in dlg.notice.text()
        assert dlg.get_selected_paths() == [str(dbc_full), str(partial_dbc)]
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_user_edited_order_is_never_resorted(qapp, tmp_path):
    """用户动过行序就不许自动排序：尊重手工顺序，命中列照常回填。"""
    blf, dbc_full = write_blf_pair(tmp_path, n=24, buses=(0,))
    partial_dbc = write_dbc(tmp_path / "partial.dbc", CAN_FRAMES[:1])
    dlg = CanDatabaseSelectorDialog(
        data_file=str(blf), initial_paths=[str(partial_dbc), str(dbc_full)]
    )
    try:
        # 挪下去再挪回来：视觉顺序复原，但 _user_edited 已置位
        dlg.table.selectRow(0)
        dlg.down_btn.click()
        dlg.table.selectRow(1)
        dlg.up_btn.click()
        assert [r.path for r in dlg.rows] == [str(partial_dbc), str(dbc_full)]
        _pump_until(dlg, lambda: dlg._prescan_state in ("done", "failed"))
        assert dlg._prescan_state == "done"
        assert [r.path for r in dlg.rows] == [str(partial_dbc), str(dbc_full)], \
            "手工调好的顺序不能被自动排序打回去"
        assert _cell(dlg, 0, 4).endswith("%"), "命中列照常回填"
        assert dlg.notice.isVisibleTo(dlg) is False, "没排序就别出排序提示"
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_hit_column_degrades_gracefully_without_a_blf(qapp, dbc_a):
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc_a)])
    assert dlg._prescan_state == "off", "data_file 不是 BLF 就不启动预扫"
    assert _cell(dlg, 0, 4) == "—"


def test_a_blf_at_the_prescan_cap_skips_it_without_hiding_other_notices(
        qapp, tmp_path, monkeypatch, dbc_a):
    """大文件不预扫（第二次完整 I/O）：命中列退化为「—」，且黄条叠加不覆盖。"""
    blf, _ = write_blf_pair(tmp_path, n=8, buses=(0,))
    monkeypatch.setattr(
        "src.ui.dialogs.can_db_selector.BLF_PRESCAN_MAX_BYTES",
        os.path.getsize(blf),  # 恰好触到上限：>= 即跳过
    )
    missing = tmp_path / "gone.dbc"
    dlg = CanDatabaseSelectorDialog(
        data_file=str(blf), initial_paths=[str(missing), str(dbc_a)]
    )
    try:
        assert dlg._prescan_state == "off"
        assert dlg._prescan_thread is None, "跳过时不许留下后台线程"
        assert dlg._prescan_timer is None
        for r in range(dlg.table.rowCount()):
            assert _cell(dlg, r, 4) == "—", "没有预扫就不许显示假命中率"
        text = dlg.notice.text()
        assert "不存在" in text, "缺文件的告警不能被跳过说明顶掉"
        assert "跳过" in text and "128 MB" in text
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_a_blf_below_the_prescan_cap_still_prescans(
        qapp, tmp_path, monkeypatch, dbc_a):
    """阈值只用体积判定：差 1 字节就照常预扫。"""
    blf, dbc_full = write_blf_pair(tmp_path, n=8, buses=(0,))
    monkeypatch.setattr(
        "src.ui.dialogs.can_db_selector.BLF_PRESCAN_MAX_BYTES",
        os.path.getsize(blf) + 1,
    )
    dlg = CanDatabaseSelectorDialog(
        data_file=str(blf), initial_paths=[str(dbc_a), str(dbc_full)]
    )
    try:
        assert dlg._prescan_state == "running"
        _pump_until(dlg, lambda: dlg._prescan_state in ("done", "failed"))
        assert dlg._prescan_state == "done"
        assert "跳过" not in dlg.notice.text()
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()
