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
from shiboken6 import isValid

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
from src.data.can_bindings import CanBindings
from src.data.can_database import CanDatabaseSet
from src.ui.dialogs.can_db_selector import CanDatabaseSelectorDialog
from src.ui.file_loader_manager import FileLoaderManager
from tests.fixtures.data_factory import (
    CAN_FRAMES,
    write_arxml,
    write_blf_pair,
    write_dbc,
)
from tests.fixtures.waits import flush_deferred_deletes

# 保活容器：offscreen 下手工构造的 QMimeData 不受 Qt 事件系统接管，局部变量
# 被 GC 后事件里访问 mimeData 会 SIGSEGV（tests/README.md 陷阱 #2）
_keep_alive: list = []


class _CanDialogFactory:
    """本文件专用的对话框生命周期跟踪器。

    只登记通过本工厂创建的对象；收尾顺序固定为：停止预扫 → 有界等待线程
    → close → deleteLater → 排空 DeferredDelete。禁止扫描/关闭其它顶层窗口。
    """

    def __init__(self):
        self.created: list[CanDatabaseSelectorDialog] = []
        self._cleaned = False

    def __call__(self, *args, **kwargs):
        dialog = CanDatabaseSelectorDialog(*args, **kwargs)
        self.created.append(dialog)
        return dialog

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        alive_threads: list[str] = []
        try:
            for dialog in reversed(self.created):
                if not isValid(dialog):
                    continue
                dialog._shutdown_prescan()
                thread = getattr(dialog, "_prescan_thread", None)
                if thread is not None and thread.is_alive():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        alive_threads.append(thread.name)
                dialog.close()
                dialog.deleteLater()
        finally:
            # pytest 不跑 app.exec()，只 deleteLater 不会真正销毁 C++ 对象。
            flush_deferred_deletes()
        if alive_threads:
            pytest.fail(f"CAN 预扫线程未在 5 秒内退出: {alive_threads}")


@pytest.fixture()
def can_dialog_factory(qapp):
    factory = _CanDialogFactory()
    try:
        yield factory
    finally:
        factory.cleanup()


def test_can_dialog_factory_tracks_and_cleans_created_objects(can_dialog_factory):
    dialog = can_dialog_factory()
    assert can_dialog_factory.created == [dialog]
    can_dialog_factory.cleanup()
    assert not isValid(dialog)


def test_can_dialog_factory_cleans_an_already_closed_object(can_dialog_factory):
    dialog = can_dialog_factory()
    dialog.close()
    can_dialog_factory.cleanup()
    assert not isValid(dialog)


def test_can_dialog_factory_cleans_after_exception(can_dialog_factory):
    dialog = can_dialog_factory()
    try:
        raise RuntimeError("测试异常路径")
    except RuntimeError:
        pass
    finally:
        can_dialog_factory.cleanup()
    assert not isValid(dialog)


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

def test_the_dialog_keeps_the_specified_sizes(can_dialog_factory, qapp, dbc_a):
    """尺寸/列宽是 §5.2 明文规定的，改了就必须在这里同步改。"""
    dlg = can_dialog_factory(initial_paths=[str(dbc_a)])
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


def test_the_header_explains_why_a_database_is_needed(can_dialog_factory, qapp, tmp_path):
    blf = tmp_path / "log.blf"
    blf.write_bytes(b"\x00" * 2048)
    dlg = can_dialog_factory(data_file=str(blf))
    titles = [w for w in dlg.children() if isinstance(w, QLabel)]
    assert any("BLF 只记录原始报文" in w.text() for w in titles)
    assert "log.blf" in dlg.subtitle.text()


# ---------------------------------------------------------------------------
# 「导入」按钮的可用性（§5.2 空列表 / 全失败）
# ---------------------------------------------------------------------------

def test_an_empty_list_disables_import_with_a_tell_whole_hint(can_dialog_factory, qapp):
    dlg = can_dialog_factory()
    assert _import_btn(dlg).isEnabled() is False
    assert _import_btn(dlg).toolTip() == "请先添加 DBC / ARXML 文件"
    # 没选中行时三个行操作按钮都该是灰的，点了才知道没反应很挫败
    assert dlg.remove_btn.isEnabled() is False
    assert dlg.up_btn.isEnabled() is False
    assert dlg.down_btn.isEnabled() is False


def test_a_usable_dbc_lights_up_import(can_dialog_factory, qapp, dbc_a):
    dlg = can_dialog_factory(initial_paths=[str(dbc_a)])
    assert len(dlg.rows) == 1
    assert _import_btn(dlg).isEnabled() is True
    assert _import_btn(dlg).toolTip() == ""
    assert dlg.get_selected_paths() == [str(dbc_a)]
    assert _cell(dlg, 0, 3) == "1 / 1"
    assert _cell(dlg, 0, 6) == "✔ 已加载"
    assert _status_color(dlg, 0) == BLF_DB_COLOR_OK


def test_only_unreadable_files_leaves_import_disabled_with_a_different_hint(can_dialog_factory, qapp,
                                                                           tmp_path):
    junk = tmp_path / "broken.arxml"
    junk.write_text("这不是 XML\n", encoding="utf-8")
    dlg = can_dialog_factory(initial_paths=[str(junk)])
    assert _import_btn(dlg).isEnabled() is False
    # 与「还没加文件」区分的另一句：加了但一个都没读出来
    assert _import_btn(dlg).toolTip() == "至少选择一个可解析的数据库文件"
    assert _cell(dlg, 0, 6).startswith("✖ 解析失败")
    assert _status_color(dlg, 0) == BLF_DB_COLOR_ERROR
    # 完整异常放 tooltip，状态列只放第一行（否则整列被堆栈撑爆）
    assert "XMLSyntaxError" in dlg.table.item(0, 6).toolTip()


def test_accepting_with_no_usable_file_warns_and_keeps_the_dialog_open(can_dialog_factory, qapp,
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
    dlg = can_dialog_factory(initial_paths=[str(junk)])
    dlg._on_accept()
    assert asked, "全失败时点导入必须有说明，否则用户以为程序卡住"
    assert "无法解析" in asked[0][2]
    assert dlg.result() != QDialog.DialogCode.Accepted


# ---------------------------------------------------------------------------
# 容错：坏文件不阻断好文件
# ---------------------------------------------------------------------------

def test_a_broken_file_sits_next_to_a_good_one_without_blocking_it(can_dialog_factory, qapp,
                                                                   tmp_path,
                                                                   dbc_a):
    junk = tmp_path / "broken.arxml"
    junk.write_text("这不是 XML\n", encoding="utf-8")
    dlg = can_dialog_factory()
    assert dlg.add_paths([str(junk), str(dbc_a)]) == 2
    assert _status_color(dlg, 0) == BLF_DB_COLOR_ERROR
    assert _cell(dlg, 1, 6) == "✔ 已加载"
    # 坏行被跳过后仍可直接导入：这是「单文件失败不中断」的 UI 侧兑现
    assert _import_btn(dlg).isEnabled() is True
    assert dlg.get_selected_paths() == [str(dbc_a)]
    assert dlg.all_paths() == [str(junk), str(dbc_a)]


def test_a_dbc_that_parses_to_no_frames_warns_instead_of_failing(can_dialog_factory, qapp, tmp_path):
    """⚠ 与 ✖ 的分界必须守住：0 报文是「库是空的」，不是「库读坏了」。"""
    empty = tmp_path / "empty.dbc"
    empty.write_text('VERSION ""\n\nNS_ :\n\nBS_:\n\nBU_:\n\n', encoding="utf-8")
    dlg = can_dialog_factory(initial_paths=[str(empty)])
    assert _cell(dlg, 0, 6) == "⚠ 无 CAN 报文"
    assert _status_color(dlg, 0) == BLF_DB_COLOR_WARN
    assert _cell(dlg, 0, 3) == "0 / 0"
    # 「无 CAN 报文」的库对解码毫无贡献，不能让它点亮导入按钮
    assert _import_btn(dlg).isEnabled() is False
    assert dlg.get_selected_paths() == []


# ---------------------------------------------------------------------------
# 优先级 = 列表顺序（§2「数据库匹配」+ §5.2）
# ---------------------------------------------------------------------------

def test_reordering_the_list_changes_which_database_wins(can_dialog_factory, qapp, dbc_a, dbc_b):
    dlg = can_dialog_factory(initial_paths=[str(dbc_a), str(dbc_b)])
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


def test_the_edge_rows_only_offer_the_outward_move_button(can_dialog_factory, qapp, dbc_a, dbc_b):
    dlg = can_dialog_factory(initial_paths=[str(dbc_a), str(dbc_b)])
    dlg.table.selectRow(0)
    assert dlg.up_btn.isEnabled() is False, "首行没有上方，按钮该灰而不是点了没反应"
    assert dlg.down_btn.isEnabled() is True
    dlg.table.selectRow(1)
    assert dlg.up_btn.isEnabled() is True
    assert dlg.down_btn.isEnabled() is False


def test_remove_drops_the_selected_row(can_dialog_factory, qapp, dbc_a, dbc_b):
    dlg = can_dialog_factory(initial_paths=[str(dbc_a), str(dbc_b)])
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


def test_dragging_in_a_mixed_bundle_keeps_only_the_databases(can_dialog_factory, qapp, dbc_a, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    arxml = write_arxml(tmp_path / "sys.arxml")
    dlg = can_dialog_factory()

    enter = _url_event(QDragEnterEvent, [str(csv), str(dbc_a)])
    dlg.dragEnterEvent(enter)
    assert enter.isAccepted(), "有可收的文件就必须给拖放反馈"

    dlg.dropEvent(_url_event(QDropEvent, [str(csv), str(dbc_a), str(arxml)]))
    added = [os.path.basename(r.path) for r in dlg.rows]
    assert added == ["A.dbc", os.path.basename(str(arxml))]
    assert dlg.table.item(1, 2).text() == "ARXML", "类型列要跟后缀一致"


def test_dragging_only_rejected_files_does_not_accept_the_drop(can_dialog_factory, qapp, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    dlg = can_dialog_factory()
    enter = _url_event(QDragEnterEvent, [str(csv)])
    dlg.dragEnterEvent(enter)
    assert not enter.isAccepted(), "全是不收的后缀还显示可放置，是纯误导"
    before = len(dlg.rows)
    dlg.dropEvent(_url_event(QDropEvent, [str(csv)]))
    assert len(dlg.rows) == before


def test_duplicate_paths_are_not_added_twice(can_dialog_factory, qapp, dbc_a):
    dlg = can_dialog_factory(initial_paths=[str(dbc_a)])
    assert dlg.add_paths([str(dbc_a)]) == 0
    assert len(dlg.rows) == 1


# ---------------------------------------------------------------------------
# C 路径：记忆命中但有文件缺失（§5.1）
# ---------------------------------------------------------------------------

def test_missing_remembered_files_are_listed_in_the_notice(can_dialog_factory, qapp, dbc_a, tmp_path):
    gone = tmp_path / "old_fota.dbc"
    dlg = can_dialog_factory(
        initial_paths=[str(gone), str(dbc_a)], missing_paths=[str(gone)]
    )
    assert dlg.notice.isVisibleTo(dlg) is True
    assert "old_fota.dbc" in dlg.notice.text()
    assert _cell(dlg, 0, 6) == "✖ 文件不存在"
    assert _cell(dlg, 0, 3) == "— / —", "缺失行不该拿 0 冒充「读到了空库」"
    # 缺失行被跳过，但剩下的仍可用 → 不必逼用户重来
    assert dlg.get_selected_paths() == [str(dbc_a)]
    assert _import_btn(dlg).isEnabled() is True


def test_missing_file_names_are_escaped_in_the_notice(can_dialog_factory, qapp, tmp_path):
    """A8：提示条是 RichText，文件名里的 & / < 不能当标记吞掉。"""
    weird = tmp_path / "A&B<2>.dbc"
    dlg = can_dialog_factory(initial_paths=[str(weird)])
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
    try:
        yield mgr
    finally:
        mgr._kept_alive = None
        mw.close()
        mw.deleteLater()
        flush_deferred_deletes()


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
    cfg = flow_host._ask_can_databases(
        "log.blf", initial_cfg=CanBindings.from_paths([str(dbc_a)])
    )
    assert cfg is not None and cfg.pool_list == [str(dbc_a)]
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
    assert flow_host._ask_can_databases(
        "log.blf", initial_cfg=CanBindings.from_paths([str(dbc_a)])
    )
    assert AppSettings().get_blf_last_db_paths() == []


def test_cancelling_the_dialog_writes_nothing(flow_host, monkeypatch,
                                              dbc_a, fresh_blf_settings):
    _auto_answer(monkeypatch, accept=False)
    assert flow_host._ask_can_databases(
        "log.blf", initial_cfg=CanBindings.from_paths([str(dbc_a)])
    ) is None
    assert AppSettings().get_blf_last_db_paths() == []


def test_repeated_cancellations_do_not_leave_hidden_dialogs(
    flow_host, monkeypatch, dbc_a, fresh_blf_settings
):
    _auto_answer(monkeypatch, accept=False)
    for _ in range(5):
        assert flow_host._ask_can_databases(
            "log.blf", initial_cfg=CanBindings.from_paths([str(dbc_a)])
        ) is None
    # deleteLater 已安排但尚未排空时仍算 parent 的 child，不能误判为泄漏。
    assert len(flow_host.mw.findChildren(CanDatabaseSelectorDialog)) == 5
    flush_deferred_deletes()
    assert flow_host.mw.findChildren(CanDatabaseSelectorDialog) == []


# ---------------------------------------------------------------------------
# 用户备注：仅备注列可编辑、随改随存、打开回显、跟随行移动
# ---------------------------------------------------------------------------

def test_the_note_column_is_the_only_editable_one(can_dialog_factory, app_settings, dbc_a):
    dlg = can_dialog_factory(initial_paths=[str(dbc_a)])
    assert dlg.table.item(0, 5).flags() & Qt.ItemFlag.ItemIsEditable
    for col in (0, 1, 2, 3, 4, 6):
        assert not (
            dlg.table.item(0, col).flags() & Qt.ItemFlag.ItemIsEditable
        ), f"第 {col} 列不该可编辑（备注列是唯一的输入点）"


def test_note_edits_persist_immediately(can_dialog_factory, app_settings, dbc_a):
    dlg = can_dialog_factory(initial_paths=[str(dbc_a)])
    dlg.table.item(0, 5).setText("  动力总成库  ")
    # 随改随存（不等「导入」：备注是文件的属性，与本次导入是否发生无关），
    # 且落盘前去空白
    assert app_settings.get_blf_db_note(str(dbc_a)) == "动力总成库"
    assert dlg.rows[0].note == "动力总成库"
    assert _cell(dlg, 0, 5) == "动力总成库", "item 文本要与落盘值一致"


def test_a_stored_note_is_echoed_and_can_be_cleared(can_dialog_factory, app_settings, dbc_a):
    app_settings.set_blf_db_note(str(dbc_a), "主库")
    dlg = can_dialog_factory(initial_paths=[str(dbc_a)])
    assert _cell(dlg, 0, 5) == "主库", "打开对话框要回显已存备注"
    dlg.table.item(0, 5).setText("  ")
    assert app_settings.get_blf_db_note(str(dbc_a)) == ""
    assert _cell(dlg, 0, 5) == "", "清空即删除条目"


def test_notes_follow_the_row_when_reordering(can_dialog_factory, app_settings, dbc_a, dbc_b):
    app_settings.set_blf_db_note(str(dbc_a), "A 库")
    dlg = can_dialog_factory(initial_paths=[str(dbc_a), str(dbc_b)])
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


def test_prescan_fills_the_hit_column_and_sorts_by_match(can_dialog_factory, qapp, tmp_path):
    """预扫完成后命中列回填、未编辑的列表自动把最匹配的库顶上去。"""
    blf, dbc_full = write_blf_pair(tmp_path, n=24, buses=(0,))
    partial_dbc = write_dbc(tmp_path / "partial.dbc", CAN_FRAMES[:1])
    dlg = can_dialog_factory(
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


def test_user_edited_order_is_never_resorted(can_dialog_factory, qapp, tmp_path):
    """用户动过行序就不许自动排序：尊重手工顺序，命中列照常回填。"""
    blf, dbc_full = write_blf_pair(tmp_path, n=24, buses=(0,))
    partial_dbc = write_dbc(tmp_path / "partial.dbc", CAN_FRAMES[:1])
    dlg = can_dialog_factory(
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


def test_hit_column_degrades_gracefully_without_a_blf(can_dialog_factory, qapp, dbc_a):
    dlg = can_dialog_factory(initial_paths=[str(dbc_a)])
    assert dlg._prescan_state == "off", "data_file 不是 BLF 就不启动预扫"
    assert _cell(dlg, 0, 4) == "—"


def test_a_blf_at_the_prescan_cap_skips_it_without_hiding_other_notices(
        can_dialog_factory, qapp, tmp_path, monkeypatch, dbc_a):
    """大文件不预扫（第二次完整 I/O）：命中列退化为「—」，且黄条叠加不覆盖。"""
    blf, _ = write_blf_pair(tmp_path, n=8, buses=(0,))
    monkeypatch.setattr(
        "src.ui.dialogs.can_db_selector.BLF_PRESCAN_MAX_BYTES",
        os.path.getsize(blf),  # 恰好触到上限：>= 即跳过
    )
    missing = tmp_path / "gone.dbc"
    dlg = can_dialog_factory(
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
        can_dialog_factory, qapp, tmp_path, monkeypatch, dbc_a):
    """阈值只用体积判定：差 1 字节就照常预扫。"""
    blf, dbc_full = write_blf_pair(tmp_path, n=8, buses=(0,))
    monkeypatch.setattr(
        "src.ui.dialogs.can_db_selector.BLF_PRESCAN_MAX_BYTES",
        os.path.getsize(blf) + 1,
    )
    dlg = can_dialog_factory(
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


# ---------------------------------------------------------------------------
# 通道绑定（D1–D7）：数据来源三档、绑定与命中、拖拽入口、配置保存载入
# ---------------------------------------------------------------------------

def _bind_rows(dialog) -> list[tuple[int, str]]:
    return [
        (
            dialog.bind_table.item(r, 0).data(Qt.ItemDataRole.UserRole),
            dialog.bind_table.item(r, 0).text(),
        )
        for r in range(dialog.bind_table.rowCount())
    ]


def test_channel_rows_come_from_prescan_with_binding_summary(can_dialog_factory, qapp, tmp_path):
    blf, dbc = write_blf_pair(tmp_path, n=24, buses=(0, 1))
    dlg = can_dialog_factory(data_file=str(blf), initial_paths=[str(dbc)])
    try:
        _pump_until(dlg, lambda: dlg._prescan_state in ("done", "failed"))
        assert dlg._prescan_state == "done"
        assert [c for c, _t in _bind_rows(dlg)] == [0, 1]
        assert [t for _c, t in _bind_rows(dlg)] == ["CAN1", "CAN2"]
        summary = dlg.bind_summary.text()
        assert "按当前绑定可解出" in summary
        # 0x1A0 被两条总线同时收到 → 未绑定时必须提示跨通道同 ID
        assert "跨通道同 ID" in summary
        assert "帧" in dlg.bind_table.item(0, 3).text()
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_binding_a_subset_changes_hits_and_result_config(can_dialog_factory, qapp, tmp_path):
    import re

    blf, dbc_full = write_blf_pair(tmp_path, n=24, buses=(0, 1))
    partial = write_dbc(tmp_path / "partial.dbc", CAN_FRAMES[:1])  # 只有 0x1A0
    dlg = can_dialog_factory(
        data_file=str(blf), initial_paths=[str(dbc_full), str(partial)]
    )
    try:
        _pump_until(dlg, lambda: dlg._prescan_state in ("done", "failed"))

        dlg._set_channel_binding(1, [str(partial)])

        assert dlg._bindings[1] == (str(partial),)
        text = dlg.bind_table.item(1, 3).text()
        ok, total = [
            int(x.replace(",", "")) for x in re.findall(r"([\d,]+)", text)[:2]
        ]
        assert ok < total, "只含一个 ID 的库不可能解出整条通道"
        cfg = dlg.get_result_config()
        assert cfg.by_channel == {1: (str(partial),)}
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_dragging_a_pool_row_onto_a_channel_binds_it(can_dialog_factory, qapp, tmp_path):
    """drop 回调 = 绑定唯一入口之一，与勾选面板共用 _set_channel_binding。"""
    blf, dbc_full = write_blf_pair(tmp_path, n=16, buses=(0, 1))
    partial = write_dbc(tmp_path / "partial.dbc", CAN_FRAMES[:1])
    dlg = can_dialog_factory(
        data_file=str(blf), initial_paths=[str(dbc_full), str(partial)]
    )
    try:
        _pump_until(dlg, lambda: dlg._prescan_state in ("done", "failed"))
        dlg.show()
        from PySide6.QtCore import QCoreApplication

        QCoreApplication.processEvents()
        mime = QMimeData()
        mime.setData("application/x-canview-db", str(partial).encode("utf-8"))
        _keep_alive.append(mime)  # offscreen 下防 GC 段错误（见文件头注释）
        rect = dlg.bind_table.visualItemRect(dlg.bind_table.item(0, 0))
        assert rect.height() > 0, f"通道行未布局: {rect}"
        event = QDropEvent(
            QPointF(rect.center()),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        dlg.bind_table.dropEvent(event)

        assert dlg._bindings.get(0) == (str(partial),)
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_selecting_the_whole_pool_means_unbound(can_dialog_factory, qapp, tmp_path):
    blf, dbc_full = write_blf_pair(tmp_path, n=16, buses=(0,))
    partial = write_dbc(tmp_path / "partial.dbc", CAN_FRAMES[:1])
    dlg = can_dialog_factory(
        data_file=str(blf), initial_paths=[str(dbc_full), str(partial)]
    )
    try:
        dlg._set_channel_binding(0, [str(dbc_full)])
        assert 0 in dlg._bindings

        dlg._set_channel_binding(0, [str(dbc_full), str(partial)])
        assert 0 not in dlg._bindings, "全选 = 未绑定（用全局池）"
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_known_channels_fill_rows_when_prescan_is_unavailable(can_dialog_factory, qapp, tmp_path):
    _blf, dbc = write_blf_pair(tmp_path, n=8, buses=(0,))
    dlg = can_dialog_factory(
        initial_paths=[str(dbc)], known_channels=[(0, 20), (1, 5)]
    )
    try:
        assert dlg._prescan_state == "off"
        assert [c for c, _t in _bind_rows(dlg)] == [0, 1]
        assert "—（导入后可知）" in dlg.bind_table.item(0, 3).text()
        assert "预扫不可用" in dlg.bind_summary.text()
    finally:
        dlg.deleteLater()


def test_manual_channel_row_can_be_added_and_removed(can_dialog_factory, qapp, tmp_path, monkeypatch):
    import src.ui.dialogs.can_db_selector as cds

    _blf, dbc = write_blf_pair(tmp_path, n=8, buses=(0,))
    dlg = can_dialog_factory(initial_paths=[str(dbc)])
    try:
        monkeypatch.setattr(
            cds.QInputDialog, "getInt",
            staticmethod(lambda *a, **k: (3, True)),
        )
        dlg._on_add_channel()
        assert (2, "CAN3") in _bind_rows(dlg)

        dlg._on_remove_channel(2)
        assert all(c != 2 for c, _t in _bind_rows(dlg))
    finally:
        dlg.deleteLater()


def test_save_and_load_config_roundtrip(can_dialog_factory, qapp, tmp_path, monkeypatch):
    import src.ui.dialogs.can_db_selector as cds

    blf, dbc_full = write_blf_pair(tmp_path, n=16, buses=(0, 1))
    partial = write_dbc(tmp_path / "partial.dbc", CAN_FRAMES[:1])
    target = tmp_path / "x.canview.yaml"
    dlg = can_dialog_factory(
        data_file=str(blf), initial_paths=[str(dbc_full), str(partial)]
    )
    try:
        dlg._set_channel_binding(1, [str(partial)])
        dlg._aliases[1] = "车身CAN"
        monkeypatch.setattr(
            cds.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "")),
        )
        dlg._on_save_config()
        assert target.exists()
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()

    dlg2 = can_dialog_factory(initial_paths=[])
    try:
        monkeypatch.setattr(
            cds.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (str(target), "")),
        )
        dlg2._on_load_config()
        cfg = dlg2.get_result_config()
        assert cfg.pool == (str(dbc_full), str(partial))
        assert cfg.by_channel == {1: (str(partial),)}
        assert cfg.alias == {1: "车身CAN"}
    finally:
        dlg2._shutdown_prescan()
        dlg2.deleteLater()


# ---------------------------------------------------------------------------
# 合并前评审回归（P0/P1/P2）
# ---------------------------------------------------------------------------

def test_single_channel_group_starts_collapsed_and_multi_expands(can_dialog_factory, qapp, tmp_path):
    """D1：多通道默认展开；单通道默认折叠（评审 P1）。"""
    blf1, dbc1 = write_blf_pair(tmp_path / "one", n=8, buses=(0,))
    dlg1 = can_dialog_factory(
        data_file=str(blf1), initial_paths=[str(dbc1)]
    )
    try:
        _pump_until(dlg1, lambda: dlg1._prescan_state in ("done", "failed"))
        assert dlg1.bind_box.isChecked() is False
        assert dlg1.bind_content.isHidden() is True
    finally:
        dlg1._shutdown_prescan()
        dlg1.deleteLater()

    blf2, dbc2 = write_blf_pair(tmp_path / "two", n=8, buses=(0, 1))
    dlg2 = can_dialog_factory(
        data_file=str(blf2), initial_paths=[str(dbc2)]
    )
    try:
        _pump_until(dlg2, lambda: dlg2._prescan_state in ("done", "failed"))
        assert dlg2.bind_box.isChecked() is True
    finally:
        dlg2._shutdown_prescan()
        dlg2.deleteLater()


def test_dropping_a_dbc_file_onto_the_pool_table_still_adds_it(can_dialog_factory, qapp, tmp_path):
    """评审 P1：库表接受自定义 MIME 后，外部文件拖入必须显式转发，不能靠冒泡。"""
    blf, dbc = write_blf_pair(tmp_path, n=8, buses=(0,))
    extra = write_dbc(tmp_path / "extra.dbc", CAN_FRAMES[:1])
    dlg = can_dialog_factory(
        data_file=str(blf), initial_paths=[str(dbc)]
    )
    try:
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(extra))])
        _keep_alive.append(mime)

        enter = QDragEnterEvent(
            QPoint(5, 5), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        dlg.table.dragEnterEvent(enter)
        assert enter.isAccepted(), "表格必须接受外部文件拖入"

        event = QDropEvent(
            QPointF(5, 5), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        dlg.table.dropEvent(event)
        assert str(extra) in [r.path for r in dlg.rows]
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_removing_a_bound_library_warns_and_falls_back_with_notice(can_dialog_factory, qapp, tmp_path):
    """评审 P1：绑定被移除后不能静默放宽，必须提示。"""
    blf, dbc_full = write_blf_pair(tmp_path, n=16, buses=(0, 1))
    partial = write_dbc(tmp_path / "partial.dbc", CAN_FRAMES[:1])
    dlg = can_dialog_factory(
        data_file=str(blf), initial_paths=[str(dbc_full), str(partial)]
    )
    try:
        _pump_until(dlg, lambda: dlg._prescan_state in ("done", "failed"))
        dlg._set_channel_binding(1, [str(partial)])
        assert dlg._bindings.get(1) == (str(partial),)

        dlg.table.selectRow(1)   # partial 在第二行
        dlg._on_remove()

        assert 1 not in dlg._bindings
        assert "改为使用全局库" in dlg.notice.text()
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_dropping_a_file_onto_the_channel_table_adds_it_to_the_pool(can_dialog_factory, qapp, tmp_path):
    """通道表也显式转发外部文件，不依赖 ignore 冒泡。"""
    blf, dbc = write_blf_pair(tmp_path, n=8, buses=(0,))
    extra = write_dbc(tmp_path / "extra.dbc", CAN_FRAMES[:1])
    dlg = can_dialog_factory(
        data_file=str(blf), initial_paths=[str(dbc)]
    )
    try:
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(extra))])
        _keep_alive.append(mime)
        event = QDropEvent(
            QPointF(5, 5), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        dlg.bind_table.dropEvent(event)
        assert str(extra) in [r.path for r in dlg.rows]
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_a_manual_expand_survives_channel_table_rebuilds(can_dialog_factory, qapp, tmp_path):
    blf, dbc = write_blf_pair(tmp_path, n=8, buses=(0,))
    dlg = can_dialog_factory(
        data_file=str(blf), initial_paths=[str(dbc)]
    )
    try:
        _pump_until(dlg, lambda: dlg._prescan_state in ("done", "failed"))
        assert dlg.bind_box.isChecked() is False

        dlg.bind_box.setChecked(True)
        dlg._rebuild_channel_table()

        assert dlg.bind_box.isChecked() is True
        assert dlg.bind_content.isHidden() is False
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_load_config_reorders_existing_pool_to_yaml_priority(can_dialog_factory, qapp, tmp_path, monkeypatch):
    """P1-3（第 5 轮）：载入配置必须把已有池重排成 YAML 顺序。

    YAML 有序池 = 分享者的 first-match 优先级；add_paths 只追加不重排，
    旧序保留会让同 ID 多库取错赢家（静默解出错误物理值）。YAML 未包含的
    行保持相对顺序垫底（非破坏性）；显式载入置 _user_edited 防预扫重排。
    """
    import yaml

    import src.ui.dialogs.can_db_selector as cds
    from src.data.can_bindings import CanBindings

    a = str(write_dbc(tmp_path / "a.dbc"))
    b = str(write_dbc(tmp_path / "b.dbc"))
    c = str(write_dbc(tmp_path / "c.dbc"))
    target = tmp_path / "shared.canview.yaml"
    sharer = CanBindings.from_paths([b, a], {}, {})
    target.write_text(
        yaml.safe_dump(sharer.to_yaml_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    dlg = can_dialog_factory(initial_paths=[a, b, c])
    try:
        monkeypatch.setattr(
            cds.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a_, **k_: (str(target), "")),
        )
        dlg._on_load_config()
        assert [r.path for r in dlg._rows] == [b, a, c], "池序必须按 YAML 优先级"
        assert dlg._user_edited is True, "显式载入后预扫自动排序不得覆盖"
        assert dlg.get_result_config().pool == (b, a, c)
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_load_config_appends_new_paths_then_reorders(can_dialog_factory, qapp, tmp_path, monkeypatch):
    """P1-3 混合场景：YAML 含当前行以外的新库 → 先追加再整体按 YAML 序重排。"""
    import yaml

    import src.ui.dialogs.can_db_selector as cds
    from src.data.can_bindings import CanBindings

    a = str(write_dbc(tmp_path / "a.dbc"))
    b = str(write_dbc(tmp_path / "b.dbc"))
    target = tmp_path / "shared.canview.yaml"
    sharer = CanBindings.from_paths([b, a], {}, {})
    target.write_text(
        yaml.safe_dump(sharer.to_yaml_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    dlg = can_dialog_factory(initial_paths=[a])
    try:
        monkeypatch.setattr(
            cds.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a_, **k_: (str(target), "")),
        )
        dlg._on_load_config()
        assert [r.path for r in dlg._rows] == [b, a]
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()


def test_load_config_duplicate_pool_paths_keep_first_index(can_dialog_factory, qapp, tmp_path, monkeypatch):
    """复审 NB-2：YAML 池含重复路径时按首次出现索引排序（first-match 语义）。"""
    import yaml

    import src.ui.dialogs.can_db_selector as cds
    from src.data.can_bindings import CanBindings

    a = str(write_dbc(tmp_path / "a.dbc"))
    b = str(write_dbc(tmp_path / "b.dbc"))
    target = tmp_path / "dup.canview.yaml"
    sharer = CanBindings.from_paths([b, a, b], {}, {})
    target.write_text(
        yaml.safe_dump(sharer.to_yaml_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    dlg = can_dialog_factory(initial_paths=[a, b])
    try:
        monkeypatch.setattr(
            cds.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a_, **k_: (str(target), "")),
        )
        dlg._on_load_config()
        assert [r.path for r in dlg._rows] == [b, a]
    finally:
        dlg._shutdown_prescan()
        dlg.deleteLater()
