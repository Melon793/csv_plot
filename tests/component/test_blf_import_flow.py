"""BLF 导入流程状态机的组件测试（方案 §5.1 的 A–G 七条路径 + §5.4 进度）。

这里管的是**决策与恢复**，不是解码（``test_blf_loader.py``）也不是控件外观
（``test_can_db_selector.py`` / ``test_can_db_banner.py``）：

* B（每次新导入）：**一律弹窗**（不允许静默解析），上次记忆的库列表只作为
  预选项；导入成功后按复选框写/清记忆
* C：记忆命中但文件缺失 → 弹同一个框并给出黄条
* F：扫完 0 个信号 → warning 后**直接重开对话框**（回到 B），不进空数据态
* G：取消 → 静默恢复原数据与按钮，不弹错误框，迟到的扫描结果被丢弃
* 重载：始终复用 ``loader.db_paths``，绝不主动弹窗；库没了才走 C
* BLF 一律走后台线程（文件再小也不同步），进度文案分「扫描中 / 生成列表」

后台线程被 :class:`FakeLoadThread` 顶掉，用例自己决定何时回「扫完 / 报错 /
被取消」，因此整组用例秒级完成，且不依赖 python-can 真读大文件。
"""

from __future__ import annotations

import gc
import os
import threading

import pytest

from PySide6.QtWidgets import QDialog, QPushButton, QProgressDialog

from src.core.config import (
    BLF_TEXT_NO_SIGNAL,
    BLF_TEXT_TRUNCATED,
    BLF_TEXT_TRUNCATED_PAYLOAD,
)
from src.core.settings import AppSettings
from src.data.blf_loader import BLFLazyLoader
from src.data.loader import DataLoadThread
from src.ui.dialogs.can_db_selector import CanDatabaseSelectorDialog
from tests.fixtures.data_factory import write_blf_pair, write_dbc


def _pump(ms: int = 60) -> None:
    from PySide6.QtCore import QCoreApplication
    import time

    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.002)


@pytest.fixture()
def remembered(app_settings):
    """擦干净 BLF 记忆键，返回一个「写入记忆」的小助手。"""

    def _set(paths):
        app_settings.set_blf_last_db_paths([])
        app_settings._settings.sync()
        if paths is not None:
            app_settings.set_blf_last_db_paths(paths)
        app_settings._settings.sync()
        return app_settings

    _set(None)
    return _set


@pytest.fixture()
def dialogs(monkeypatch):
    """截获选择对话框：记录每次打开，并按 ``answers`` 决定结果。

    ``answers`` 是一次性脚本（列表.pop(0)），缺省一律 Accepted —— F 回环这种
    「同一个入口被走两次」的场景必须能分别设定两次的答案。
    """
    state = {"opened": [], "answers": [], "default_paths": []}

    def fake_exec(self, *args, **kwargs):
        state["opened"].append(self)
        if state["answers"]:
            answer = state["answers"].pop(0)
        else:
            answer = (QDialog.DialogCode.Accepted, None)
        code, prepare = answer
        if prepare is not None:
            prepare(self)
        else:
            pick = state["default_paths"] or self.get_selected_paths()
            known = {r.path for r in self._rows}
            self.add_paths([p for p in pick if p not in known])
            for row in list(self._rows):
                if row.path not in pick:
                    self._rows.remove(row)
            self._rebuild_table()
        return code

    monkeypatch.setattr(CanDatabaseSelectorDialog, "exec", fake_exec)
    return state


def _reject_on_sight(monkeypatch) -> list:
    """守卫「不该弹框」的场景（重载 / 收口路径）：真弹了框就记下。"""
    boom = []

    def fake_exec(self, *args, **kwargs):
        boom.append(self)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(CanDatabaseSelectorDialog, "exec", fake_exec)
    return boom


def _make_loader(blf, dbc):
    return BLFLazyLoader(str(blf), db_paths=[str(dbc)])


def _cancel_btn(dialog: QProgressDialog) -> QPushButton:
    """取进度框的取消按钮。

    PySide6 没把 ``QProgressDialog.cancelButton()`` 绑出来（C++ 有），只能从
    子控件里拿 —— 扫描进度框除了这一个取消按钮没别的 QPushButton。
    拿不到就当场失败，正好等价于「§5.4 要求的取消入口不见了」。
    """
    btns = [b for b in dialog.findChildren(QPushButton) if b.text()]
    assert len(btns) == 1, f"取消入口缺失或不止一个: {[b.text() for b in btns]}"
    return btns[0]


# ---------------------------------------------------------------------------
# B：每次新导入都弹窗，记忆只作预选
# ---------------------------------------------------------------------------

def test_a_new_import_always_asks_but_preselects_the_remembered_set(
    blf_flow, tmp_path, remembered, dialogs
):
    """不允许静默导入：有记忆也必须弹窗确认，扫描在用户点「导入」后才启动。"""
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    manager.load_csv_file(str(blf))

    assert len(dialogs["opened"]) == 1, "不允许静默导入：有记忆也要弹窗确认"
    dlg = dialogs["opened"][0]
    assert [r.path for r in dlg.rows] == [str(dbc)], "记忆列表要预选进弹窗"
    assert len(threads) == 1
    assert threads[0].db_paths == [str(dbc)]
    assert threads[0].started == 1, "BLF 必须真的把后台线程跑起来"

    loader = _make_loader(blf, dbc)
    threads[0].emit_done(loader)
    assert host.loader is loader
    assert host.can_db_banner.isHidden() is False
    assert host.can_db_banner.state == "unmatched"


def test_the_dialog_flow_still_forces_the_background_thread_for_a_tiny_file(
    blf_flow, tmp_path, remembered, dialogs
):
    """§5.4「BLF 一律异步」：2 KB 的样本也不该走同步分支。"""
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=4, buses=(0,))
    remembered([str(dbc)])
    manager._load_sync = lambda *a, **k: pytest.fail("BLF 不该走同步加载")

    manager.load_csv_file(str(blf))
    assert len(threads) == 1
    assert threads[0].started == 1
    assert manager._blf_progress is not None


# ---------------------------------------------------------------------------
# 进度（§5.4）
# ---------------------------------------------------------------------------

def test_the_progress_dialog_is_quantitative_and_cancellable(blf_flow, tmp_path,
                                                             remembered,
                                                             dialogs):
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    manager.load_csv_file(str(blf))
    dialog = manager._blf_progress
    assert isinstance(dialog, QProgressDialog)
    assert dialog.minimumDuration() == 0, "扫描可达分钟级，不该有 3 秒才出现的延迟"
    assert dialog.maximum() == 100, "必须可量化（tell()/file_size），不是摆动条"
    assert _cancel_btn(dialog).text(), "§5.4 的硬要求：必须有取消按钮"
    assert os.path.basename(str(blf)) in dialog.labelText()

    threads[0].scan_stats = {"messages": 12_345}
    threads[0].emit_progress(50)
    assert "正在扫描 BLF 报文… 50%" in dialog.labelText()
    assert "12,345 帧" in dialog.labelText()
    assert dialog.value() == 50

    threads[0].emit_progress(95)
    assert dialog.labelText() == "正在生成信号列表…"

    threads[0].emit_progress(150)  # 越界值必须被夹住，不然进度条画飞
    assert dialog.value() == 100


def test_progress_updates_arrive_even_without_a_dialog(blf_flow, tmp_path):
    """进度回调不能假设框还在 —— close/cancel 与后台线程本来就有竞态。"""
    _host, manager, _threads = blf_flow
    manager._blf_progress = None
    manager._on_blf_progress(42)  # 不抛异常即为通过


def test_closing_the_progress_dialog_is_not_a_cancel(blf_flow, tmp_path,
                                                     remembered, dialogs):
    """回归守卫：Qt 的 ``QProgressDialog.close()`` 也会 emit ``canceled()``。

    不先收线再关闭的话，每份成功扫完的 BLF 都会在 ``_on_blf_done`` 里
    被自己触发的取消标记推掉，表现为「导入完但变量列表是空的」。
    """
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    manager.load_csv_file(str(blf))
    dialog = manager._blf_progress
    manager._close_blf_progress()

    assert dialog.isVisible() is False
    assert manager._blf_cancelled_version is None, "关闭框不是取消"
    loader = _make_loader(blf, dbc)
    threads[0].emit_done(loader)
    assert host.loader is loader


# ---------------------------------------------------------------------------
# B：首次导入弹窗；取消即中止
# ---------------------------------------------------------------------------

def test_path_b_first_import_asks_once_and_remembers_the_choice(
    blf_flow, tmp_path, remembered, dialogs, app_settings
):
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered(None)  # 无记忆
    dialogs["default_paths"] = [str(dbc)]

    manager.load_csv_file(str(blf))

    assert len(dialogs["opened"]) == 1
    dlg = dialogs["opened"][0]
    # 副标题必须报出正在导入的是哪个数据文件（§5.2 第二行）
    assert blf.name in dlg.subtitle.text()
    assert threads[0].db_paths == [str(dbc)]
    assert AppSettings().get_blf_last_db_paths() == [str(dbc)]


def test_path_g_cancelling_the_dialog_aborts_before_any_scan(
    blf_flow, tmp_path, remembered, dialogs
):
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered(None)
    dialogs["answers"] = [(QDialog.DialogCode.Rejected, None)]
    host.load_btn.setEnabled(False)

    manager.load_csv_file(str(blf))

    assert threads == [], "没定数据库就不该开始扫描"
    assert manager._blf_progress is None
    assert host.loader is None, "§5.1：取消后回到导入前状态，不进空数据态"
    assert host.can_db_banner.isHidden() is True
    assert host.load_btn.isEnabled() is True, "按钮必须恢复，否则像卡死"
    assert host.reload_btn.isEnabled() is True
    # 取消不写记忆（§5.2）
    assert AppSettings().get_blf_last_db_paths() == []


def test_cancelling_mid_scan_is_silent_and_drops_the_late_result(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])
    boxes = []
    monkeypatch.setattr(
        flm.QMessageBox, "critical",
        staticmethod(lambda *a, **k: boxes.append(a)),
    )
    monkeypatch.setattr(
        flm.QMessageBox, "warning",
        staticmethod(lambda *a, **k: boxes.append(a)),
    )

    manager.load_csv_file(str(blf))
    thread = threads[0]
    late_loader = _make_loader(blf, dbc)

    thread.emit_progress(30)
    assert manager._blf_progress is not None
    _cancel_btn(manager._blf_progress).click()

    assert manager._blf_progress is None, "取消后进度框必须立刻收掉"
    assert thread.cancel_requests >= 1, "取消必须转达给后台线程"
    assert boxes == [], "§5.1 的 G：静默恢复，不弹错误框"
    assert host.load_btn.isEnabled() is True
    assert host.reload_btn.isEnabled() is True
    assert manager._blf_cancelled_version == manager._blf_load_version, \
        "取消标记必须带上本轮代次"

    thread.emit_cancelled()
    assert boxes == []

    # 取消与完成撞在一起：以取消为准，扫好的 loader 要被丢掉并关闭
    thread.emit_done(late_loader)
    assert host.loader is None
    assert late_loader._closed is True


def test_a_second_cancel_click_does_not_replay_the_restore(
    blf_flow, tmp_path, remembered, dialogs
):
    """取消按钮会被连点：恢复逻辑必须幂等。"""
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])
    manager.load_csv_file(str(blf))
    thread = threads[0]
    dialog = manager._blf_progress
    _cancel_btn(dialog).click()
    _cancel_btn(dialog).click()

    assert manager._blf_progress is None
    assert thread.started == 1
    # 恢复只该跑一次：canceled 同时接了「直达线程」与「_on_blf_cancel」两根
    # 线，后者自带幂等守卫，不会二次结束 reload / 二次恢复按钮
    assert manager.events.count("end") == 1, manager.events
    assert host.load_btn.isEnabled() is True


def test_cancelling_repaints_the_curves_that_were_there_before(
    blf_flow, tmp_path, remembered, dialogs
):
    """§5.1 的 G：「恢复原数据」含曲线。

    ``_begin_data_reload`` 已经把 item 从 scene 摘走，而取消路径不释放旧 loader
    —— widget.curves 里的名字/颜色/可见性都还在，能原地重建。不重建的话，用户
    点一下取消就白白丢掉整张画布，只能重新拖变量。
    """
    host, manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])
    loader = _make_loader(blf, dbc)
    host.loader = loader

    manager.load_csv_file(str(blf))
    assert host.cursor_sync_manager.replots == [], "没取消就不该重绘"
    _cancel_btn(manager._blf_progress).click()

    assert host.cursor_sync_manager.replots == [{"skip_pin_reset": True}]
    assert host.loader is loader, "取消不许把旧数据源换掉"
    # A2：取消路径必须把加载期禁用的 6 个分析按钮还回来，否则取消一次就永久灰掉
    assert ("buttons", True) in manager.events, manager.events


def test_cancelling_a_first_import_does_not_try_to_repaint_an_empty_canvas(
    blf_flow, tmp_path, remembered, dialogs
):
    """没有任何数据源时不去重绘：重建全靠 loader.var_names，拿不到就直返。"""
    host, manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    manager.load_csv_file(str(blf))
    _cancel_btn(manager._blf_progress).click()

    assert host.cursor_sync_manager.replots == []
    assert host.loader is None


def test_path_f_gives_the_canvas_back_too(blf_flow, tmp_path, remembered,
                                          dialogs, monkeypatch):
    """F（扫完 0 个信号）同样没让导入成立，没理由把用户原有的画布留空。"""
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    unrelated = write_dbc(
        tmp_path / "other.dbc",
        [{
            "name": "Unrelated", "can_id": 0x4C0, "size": 8, "cycle_ms": 20,
            "sender": "X",
            "signals": ({
                "name": "Nothing", "start": 0, "size": 8, "order": "intel",
                "factor": 1.0, "offset": 0.0, "minimum": 0.0,
                "maximum": 255.0, "unit": "", "sweep": 5,
            },),
        }],
    )
    remembered([str(unrelated)])
    monkeypatch.setattr(
        flm.QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )
    loader = _make_loader(blf, dbc)
    host.loader = loader

    manager.load_csv_file(str(blf))
    threads[0].emit_done(_make_loader(blf, unrelated))
    _pump(150)

    assert host.cursor_sync_manager.replots == [{"skip_pin_reset": True}]
    assert host.loader is loader, "F 后的数据源仍是导入前那一份"
    assert ("buttons", True) in manager.events, manager.events
    assert len(dialogs["opened"]) == 2, "重开选择框的回环不能被曲线恢复抢跑"


def test_a_late_cancel_from_the_previous_scan_does_not_abort_the_new_one(
    blf_flow, tmp_path, remembered, dialogs
):
    """取消标记必须按**代次**记：一个全局布尔会被旧线程迟到的信号误杀新导入。

    进度框的 canceled 是同步落地，而旧 ``DataLoadThread`` 的 cancelled 要在后来
    的事件循环里才到 —— 中间用户已经可以重跑一轮了。
    """
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    manager.load_csv_file(str(blf))
    old = threads[0]
    old.emit_progress(30)
    _cancel_btn(manager._blf_progress).click()
    assert manager._blf_cancelled_version == manager._blf_load_version

    manager.load_csv_file(str(blf))
    new = threads[1]
    assert manager._blf_cancelled_version is None, "新一轮开始时上一轮的标记要被擦掉"
    old.emit_cancelled()

    assert manager._blf_progress is not None, "旧一轮的取消不能收新一轮的线"
    assert new.cancel_requests == 0, "更不能拿它去转达取消"
    assert manager.events.count("end") == 1, manager.events

    loader = _make_loader(blf, dbc)
    new.emit_done(loader)
    assert host.loader is loader, "新一轮必须正常落地"


def test_a_late_error_from_the_previous_scan_does_not_abort_the_new_one(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """A1：旧一轮迟到的 error 不能收新一轮的线。

    取消路径的代次闸只覆盖 cancelled；error 若漏了闸，会关掉新一轮的进度框、
    弹「读取失败」并提前解锁 UI，用户可再发第三轮 → 两条扫描并发。
    """
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])
    boxes = []
    monkeypatch.setattr(
        flm.QMessageBox, "critical",
        staticmethod(lambda *a, **k: boxes.append(a)),
    )

    manager.load_csv_file(str(blf))
    old = threads[0]
    _cancel_btn(manager._blf_progress).click()

    manager.load_csv_file(str(blf))
    new = threads[1]
    dialog = manager._blf_progress
    assert dialog is not None

    old.emit_error("旧一轮扫描失败（迟到）")

    assert boxes == [], "过期错误不应弹框"
    assert manager._blf_progress is dialog, "新一轮的进度框与取消入口必须还在"
    assert host._is_loading_new_data is True, "不能被提前解锁"
    assert new.cancel_requests == 0

    loader = _make_loader(blf, dbc)
    new.emit_done(loader)
    assert host.loader is loader, "新一轮仍必须正常落地"


def test_a_late_done_from_the_previous_scan_does_not_close_the_new_progress(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """A3：迟到的 done 会在代次检查前关掉进度框。

    A 取消 → B 开始后，A 的 ``finished`` 若先 ``_close_blf_progress()`` 再判代次，
    B 的进度框与取消按钮就消失了；B 若在大文件上扫几分钟，用户只能强等。
    """
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    manager.load_csv_file(str(blf))
    _cancel_btn(manager._blf_progress).click()

    manager.load_csv_file(str(blf))
    dialog = manager._blf_progress
    assert dialog is not None

    late_loader = _make_loader(blf, dbc)
    threads[0].emit_done(late_loader)

    assert late_loader._closed is True, "过期结果仍必须释放"
    assert manager._blf_progress is dialog, "B 的进度框不能被 A 的迟到 done 收掉"
    assert dialog.isVisible() is True
    assert host._is_loading_new_data is True

    loader = _make_loader(blf, dbc)
    threads[1].emit_done(loader)
    assert host.loader is loader


def test_a_late_progress_from_the_previous_scan_does_not_touch_the_new_dialog(
    blf_flow, tmp_path, remembered, dialogs
):
    """A5：旧线程迟到的进度不许刷新一轮的框。

    百分比属于旧扫描，``scan_stats`` 更会被读成新一轮线程的数字 —— 不按代次
    闸住，用户会看到进度条从 10% 跳回 80%，帧数还是别的文件的。
    """
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    manager.load_csv_file(str(blf))
    old = threads[0]
    _cancel_btn(manager._blf_progress).click()

    manager.load_csv_file(str(blf))
    new = threads[1]
    dialog = manager._blf_progress
    new.scan_stats = {"messages": 7}
    new.emit_progress(10)
    assert dialog.value() == 10

    old.scan_stats = {"messages": 999_999}
    old.emit_progress(80)

    assert dialog.value() == 10, "旧轮的百分比不能跳新一轮的进度条"
    assert "80%" not in dialog.labelText()
    assert "999,999" not in dialog.labelText()


def test_a_failed_import_restores_the_analysis_buttons(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """A2：错误路径与既有异步错误路径同口径，不能失败一次就永久灰掉。"""
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])
    old_loader = _make_loader(blf, dbc)
    host.loader = old_loader
    boxes = []
    monkeypatch.setattr(
        flm.QMessageBox, "critical",
        staticmethod(lambda *a, **k: boxes.append(a)),
    )

    manager.load_csv_file(str(blf))
    threads[0].emit_error("BLF 读取失败")

    assert boxes, "真实错误仍要告知用户"
    assert host.loader is old_loader, "失败不该换掉旧数据源"
    assert ("buttons", True) in manager.events, manager.events
    assert host._is_loading_new_data is False


def test_a_failed_import_repaints_the_curves_that_were_there_before(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """A6：失败与取消同口径 —— 旧 loader 还在，画布就该还原。"""
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])
    old_loader = _make_loader(blf, dbc)
    host.loader = old_loader
    monkeypatch.setattr(
        flm.QMessageBox, "critical", staticmethod(lambda *a, **k: None)
    )

    manager.load_csv_file(str(blf))
    assert host.cursor_sync_manager.replots == [], "没失败就不该重绘"
    threads[0].emit_error("BLF 读取失败")

    assert host.cursor_sync_manager.replots == [{"skip_pin_reset": True}]
    assert host.loader is old_loader, "失败路径不换数据源，曲线必须回到原样"


def test_a_still_running_previous_thread_is_kept_alive(blf_flow, tmp_path,
                                                       remembered, dialogs):
    """B-1（SIGSEGV/SIGABRT 级别）：覆盖 ``mw._thread`` 前必须保住还在跑的旧线程。

    PySide6 里运行中的 QThread 一丢最后一条强引用就当场析构 C++ 对象：
    ``QThread: Destroyed while thread is still running`` → ``std::terminate``
    → SIGABRT（实测退出码 134，复现脚本 ``tmp/probe_qthread_drop.py``）。
    取消路径刻意不 ``wait()``，而取消后点「重载」（无对话框、按钮已恢复）就会
    在这一行覆盖旧线程，所以先把还在跑的旧线程挂进保活列表。
    """
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    manager.load_csv_file(str(blf))
    old = threads[0]
    _cancel_btn(manager._blf_progress).click()
    assert old.isRunning() is True, "取消只发请求，线程在下个分块边界才真退出"

    manager.load_csv_file(str(blf))
    assert old in manager._blf_thread_refs, "运行中的旧线程必须留在保活列表里"
    assert host._thread is threads[1]

    # 跑完就该放掉，否则反复导入会让列表只增不减
    threads[1].emit_done(_make_loader(blf, dbc))
    old.emit_cancelled()
    assert old.isRunning() is False
    manager.load_csv_file(str(blf))
    assert old not in manager._blf_thread_refs


class _HeldRealThread(DataLoadThread):
    """真 ``QThread``，只是 ``run()`` 卡在事件上等用例放行。

    B-1 的崩溃只在「线程真的还在跑」时发生：替身是 QObject，永远触发不了
    Qt 的 ``Destroyed while thread is still running``，所以这条守卫必须用真线程。
    ``request_cancel`` 直接继承：它只置标记，正好是真实世界的取消语义。
    """

    def __init__(self, file_path: str, parent=None, db_paths=None):
        super().__init__(file_path, parent=parent, db_paths=db_paths)
        self._gate = threading.Event()
        self.entered = threading.Event()

    def run(self):
        self.entered.set()
        self._gate.wait(10.0)

    def release(self) -> None:
        self._gate.set()


def test_a_cancelled_but_still_running_thread_survives_a_real_reload(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """B-1 的确定性版本：用真 QThread 把「覆盖时旧线程仍在跑」固定下来。

    测试侧刻意**不**留旧线程的强引用（唯一引用只能是 manager 的保活列表），
    否则无论修复在不在都能“活”，测不到东西。
    """
    import src.ui.file_loader_manager as flm

    host, manager, _threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    made: list[_HeldRealThread] = []

    def _factory(file_path, parent=None, db_paths=None, **kwargs):
        thread = _HeldRealThread(file_path, parent=parent, db_paths=db_paths)
        made.append(thread)
        return thread

    monkeypatch.setattr(flm, "DataLoadThread", _factory)
    old_ref: DataLoadThread | None = None
    try:
        manager.load_csv_file(str(blf))
        old = made[0]
        assert old.entered.wait(2.0), "run() 没进来 = 线程根本没跑起来"
        _cancel_btn(manager._blf_progress).click()
        assert old.isRunning() is True, "request_cancel 只置标记，不会让线程立刻退出"

        manager.load_csv_file(str(blf))       # 覆盖 mw._thread（B-1 的死法就在这行）
        old_ref = old
        del made[0], old
        gc.collect()                          # 没有保活列表的话，这里就是高危现场

        assert len(manager._blf_thread_refs) == 1, "旧线程必须被保活列表接住"
        assert old_ref.isRunning() is True, "gc 后线程还活着（没被 Qt 析构）"
        assert old_ref is not host._thread
    finally:
        if old_ref is not None:
            old_ref.release()
            made.append(old_ref)
        for thread in made:
            thread.release()
        for thread in made:
            # 收尾不留真线程：否则解释器退出时同样会 abort（整串用例一起挂）
            assert thread.wait(5000), "线程未能收尾"


# ---------------------------------------------------------------------------
# C：记忆命中但有文件缺失
# ---------------------------------------------------------------------------

def test_path_c_a_missing_remembered_file_reopens_the_same_dialog_with_a_notice(
    blf_flow, tmp_path, remembered, dialogs
):
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    gone = str(tmp_path / "old_fota.dbc")
    remembered([gone, str(dbc)])
    dialogs["default_paths"] = [str(dbc)]

    manager.load_csv_file(str(blf))

    assert len(dialogs["opened"]) == 1, "缺文件不能静默降级成「只用还能读的那个」"
    dlg = dialogs["opened"][0]
    assert dlg.notice.isVisibleTo(dlg) is True
    assert "old_fota.dbc" in dlg.notice.text()
    assert threads[0].db_paths == [str(dbc)]


def test_an_expired_memory_falls_back_to_a_plain_first_import(
    blf_flow, tmp_path, remembered, dialogs
):
    """记忆里的库全没了：退化成 B（黄条也就没意义了，直接让用户重选）。"""
    host, manager, threads = blf_flow
    blf, _dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(tmp_path / "gone1.dbc"), str(tmp_path / "gone2.dbc")])
    alt_dir = tmp_path / "alt"
    alt_dir.mkdir(exist_ok=True)
    _blf2, dbc2 = write_blf_pair(alt_dir, n=16, buses=(0,), basename="alt_can")
    dialogs["default_paths"] = [str(dbc2)]

    manager.load_csv_file(str(blf))

    assert len(threads) == 1
    assert threads[0].db_paths == [str(dbc2)]


# ---------------------------------------------------------------------------
# F：0 信号 → warning 后重开对话框
# ---------------------------------------------------------------------------

def test_path_f_warns_then_loops_straight_back_into_the_dialog(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, _dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    unrelated = write_dbc(
        tmp_path / "other.dbc",
        [{
            "name": "Unrelated", "can_id": 0x4C0, "size": 8, "cycle_ms": 20,
            "sender": "X",
            "signals": ({
                "name": "Nothing", "start": 0, "size": 8, "order": "intel",
                "factor": 1.0, "offset": 0.0, "minimum": 0.0,
                "maximum": 255.0, "unit": "", "sweep": 5,
            },),
        }],
    )
    remembered([str(unrelated)])
    warnings = []
    monkeypatch.setattr(
        flm.QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )

    manager.load_csv_file(str(blf))
    assert len(threads) == 1 and len(dialogs["opened"]) == 1, \
        "不允许静默导入：弹窗确认后才扫描"
    empty_loader = _make_loader(blf, unrelated)
    assert empty_loader.var_names == [], "样本本身要先成立：库与报文完全不通"
    threads[0].emit_done(empty_loader)

    assert warnings and BLF_TEXT_NO_SIGNAL in warnings[0][2]
    assert host.loader is None, "F 不能把空数据态留给用户"

    # 直接重开对话框（回到 B），而不是等用户再点一次导入
    _pump(150)
    assert len(dialogs["opened"]) == 2, "§5.1 的 F：warning 后自动重开选择框"
    assert dialogs["opened"][-1].rows, "重开的框要把上一轮的库预填进去（可就地改）"

    # 答案与上一份完全相同时必须收口：报文没变、库也没变，再走一轮必然还是 0 信号，
    # 而重绑已是毫秒级，没有「重扫天然限速」这道隐性保险了。
    assert len(threads) == 1, "同一份错库不得再起第二轮扫描（快速空转 = 连环弹窗）"
    assert host.loader is None, "收口后仍是「回到导入前」，不是空数据态"
    assert manager._blf_pending_loader is empty_loader, "扫好的桶要留着给下一次换库"
    assert empty_loader._closed is False, "桶还没用完，不能提前 close"


def test_cancelling_the_f_restart_dialog_releases_the_stashed_buckets(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """A7：F 重开框被取消 → 上一轮暂存的报文桶立刻释放。

    只有「答了同一份错库」的收口才值得留着桶等下次换库；用户按取消就是明确
    放弃本轮导入，再攥着一份几百 MB 的扫描结果只会白白占内存。
    """
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, _dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    unrelated = write_dbc(
        tmp_path / "other.dbc",
        [{
            "name": "Unrelated", "can_id": 0x4C0, "size": 8, "cycle_ms": 20,
            "sender": "X",
            "signals": ({
                "name": "Nothing", "start": 0, "size": 8, "order": "intel",
                "factor": 1.0, "offset": 0.0, "minimum": 0.0,
                "maximum": 255.0, "unit": "", "sweep": 5,
            },),
        }],
    )
    remembered([str(unrelated)])
    monkeypatch.setattr(
        flm.QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )
    dialogs["answers"] = [
        (QDialog.DialogCode.Accepted, None),
        (QDialog.DialogCode.Rejected, None),
    ]

    manager.load_csv_file(str(blf))
    empty_loader = _make_loader(blf, unrelated)
    threads[0].emit_done(empty_loader)
    _pump(150)

    assert len(dialogs["opened"]) == 2, "F 仍应自动重开选择框"
    assert manager._blf_pending_loader is None, "取消后不许再攥着报文桶"
    assert empty_loader._closed is True, "释放必须落到 close()，不能只丢引用"
    assert host.loader is None, "取消后回到导入前状态"
    assert host.load_btn.isEnabled() is True, "取消后总得让用户再点一次导入"


def test_path_f_rebinds_the_same_buckets_without_rescanning(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """F 回环里换了库 → 复用上一轮扫好的报文桶，只重绑数据库（§5.6 同一条承诺）。

    这是 P1 的行为契约：``_on_blf_done`` 的 0 信号分支过去会 ``close()`` 掉整轮
    扫描成果，于是每一次重试都把整个文件重扫一遍（实测 655 万帧 9.16 s/轮 vs
    重绑 3.7 ms）。现在必须证明「不再起线程、loader 身份不变、帧数不变、信号出来了」。
    """
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    unrelated = write_dbc(
        tmp_path / "other.dbc",
        [{
            "name": "Unrelated", "can_id": 0x4C0, "size": 8, "cycle_ms": 20,
            "sender": "X",
            "signals": ({
                "name": "Nothing", "start": 0, "size": 8, "order": "intel",
                "factor": 1.0, "offset": 0.0, "minimum": 0.0,
                "maximum": 255.0, "unit": "", "sweep": 5,
            },),
        }],
    )
    remembered([str(unrelated)])
    monkeypatch.setattr(
        flm.QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )

    manager.load_csv_file(str(blf))
    empty_loader = _make_loader(blf, unrelated)
    threads[0].emit_done(empty_loader)
    frames_before = empty_loader.total_messages
    buckets_before = len(empty_loader._buckets)

    # 第二次打开时改成「能对上的那份库」，等价于用户在重开的框里换掉选择
    dialogs["default_paths"] = [str(dbc)]
    _pump(150)

    assert len(dialogs["opened"]) == 2, "第一次导入 + F 重开各一次"
    assert len(threads) == 1, "换库只重绑，不该再起一条扫描线程"
    assert host.loader is empty_loader, "装的必须还是同一份 loader（桶没重建）"
    assert empty_loader.db_paths == [str(dbc)]
    assert empty_loader.total_messages == frames_before
    assert len(empty_loader._buckets) == buckets_before
    assert empty_loader.var_names, "换成配对的库之后就该有信号了"
    assert host.cursor_sync_manager.replots == [], "重绑成功的一轮不走「恢复原曲线」"


def test_a_given_up_restart_clears_the_force_dialog_flags(blf_flow, tmp_path,
                                                         remembered, dialogs,
                                                         monkeypatch):
    """F 回环被放弃时两个标志必须擦干净。

    ``_restart_blf_import`` 是一条纯 ``QTimer.singleShot`` 延迟链，而 ``mw``
    是弱引用：重试期间窗口可能被回收。``_blf_force_dialog`` 只对那一轮导入有意义，
    残留会让之后**任意**一次 BLF 导入都被强制弹窗（表现是「明明记住了数据库却还弹」）。
    """
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, _dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    unrelated = write_dbc(
        tmp_path / "other.dbc",
        [{
            "name": "Unrelated", "can_id": 0x4C0, "size": 8, "cycle_ms": 20,
            "sender": "X",
            "signals": ({
                "name": "Nothing", "start": 0, "size": 8, "order": "intel",
                "factor": 1.0, "offset": 0.0, "minimum": 0.0,
                "maximum": 255.0, "unit": "", "sweep": 5,
            },),
        }],
    )
    remembered([str(unrelated)])
    monkeypatch.setattr(
        flm.QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )

    manager.load_csv_file(str(blf))
    threads[0].emit_done(_make_loader(blf, unrelated))
    host._is_loading_new_data = True      # 模拟上一轮始终收不了尾
    _pump(1400)                           # 50 次 × 20ms 后必须放弃

    assert manager._blf_force_dialog is False
    assert manager._blf_preset_db_paths is None
    assert len(dialogs["opened"]) == 1, "放弃的那几轮不该偷偷再弹框（首弹来自本次导入）"

    # 下一次普通导入回到「弹窗预选」的正常流程
    host._is_loading_new_data = False
    manager.load_csv_file(str(blf))
    assert len(threads) == 2
    assert len(dialogs["opened"]) == 2, "每次新导入都该弹窗预选"


# ---------------------------------------------------------------------------
# 其它反馈：错误 / 截断 / 过期结果
# ---------------------------------------------------------------------------

def test_a_background_failure_shows_one_critical_box_and_restores(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])
    boxes = []
    monkeypatch.setattr(
        flm.QMessageBox, "critical",
        staticmethod(lambda *a, **k: boxes.append(a)),
    )

    manager.load_csv_file(str(blf))
    threads[0].emit_error("BLF 文件未包含任何 CAN 报文")

    assert boxes and "BLF 文件未包含任何 CAN 报文" in boxes[0][2]
    assert manager._blf_progress is None
    assert host.loader is None
    assert host.load_btn.isEnabled() is True


def test_a_truncated_scan_warns_but_still_loads_the_scanned_part(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """§5.8：截断要说明白，但不能因此把已经扫好的数据丢掉。"""
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])
    warnings = []
    monkeypatch.setattr(
        flm.QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )

    manager.load_csv_file(str(blf))
    loader = _make_loader(blf, dbc)
    loader.truncated = True
    threads[0].emit_done(loader)

    assert warnings and BLF_TEXT_TRUNCATED in warnings[0][2]
    assert host.loader is loader
    assert host.can_db_banner.state == "truncated"


def test_a_payload_truncated_scan_says_the_byte_limit_not_the_message_count(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """内存上限有两种触发方式，提示与状态条必须跟着改口。"""
    import src.ui.file_loader_manager as flm

    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])
    warnings = []
    monkeypatch.setattr(
        flm.QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    monkeypatch.setattr(flm, "BLF_MAX_PAYLOAD_BYTES", 512 * 1024 * 1024)

    manager.load_csv_file(str(blf))
    loader = _make_loader(blf, dbc)
    loader.truncated = True
    loader.truncated_reason = "payload"
    threads[0].emit_done(loader)

    assert warnings and BLF_TEXT_TRUNCATED_PAYLOAD in warnings[0][2]
    assert BLF_TEXT_TRUNCATED not in warnings[0][2], "别再说成条数上限"
    assert host.loader is loader
    assert host.can_db_banner.state == "truncated"
    assert "512 MB" in host._can_segment.text()


def test_a_scan_result_from_a_stale_generation_is_discarded(blf_flow, tmp_path,
                                                            remembered,
                                                            dialogs):
    """数据版本号是防污染的最后一道闸：过期结果绝不能换进 UI。"""
    host, manager, threads = blf_flow
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    remembered([str(dbc)])

    manager.load_csv_file(str(blf))
    stale_version = host._data_version
    host._data_version = stale_version + 1  # 模拟期间又发起了新一轮加载
    loader = _make_loader(blf, dbc)
    threads[0].emit_done(loader)

    assert host.loader is None
    assert loader._closed is True


# ---------------------------------------------------------------------------
# 重载：绝不主动弹窗
# ---------------------------------------------------------------------------

def test_a_reload_reuses_the_loaders_databases_without_asking(blf_flow, tmp_path,
                                                             remembered,
                                                             monkeypatch):
    host, manager, threads = blf_flow
    boom = _reject_on_sight(monkeypatch)
    blf, dbc = write_blf_pair(tmp_path, n=16, buses=(0,))
    loader = _make_loader(blf, dbc)
    host.loader = loader

    manager._load_file(str(blf), is_reload=True)

    assert boom == [], "§5.1：重载永远复用上次 db_paths"
    assert threads[0].db_paths == [str(dbc)]
    loader.close()


def test_a_reload_whose_database_disappeared_falls_back_to_the_dialog(
    blf_flow, tmp_path, remembered, dialogs, monkeypatch
):
    """唯一允许重载弹窗的情况：库真读不到了（走 C，且要说清缺了谁）。"""
    host, manager, threads = blf_flow
    src_dir = tmp_path / "keep"
    blf, dbc = write_blf_pair(src_dir, n=16, buses=(0,))
    loader = _make_loader(blf, dbc)
    host.loader = loader
    gone = tmp_path / "gone.dbc"
    loader.db_paths = [str(gone)]
    dialogs["default_paths"] = [str(dbc)]

    manager._load_file(str(blf), is_reload=True)

    assert len(dialogs["opened"]) == 1
    assert "gone.dbc" in dialogs["opened"][0].notice.text()
    assert threads[0].db_paths == [str(dbc)]
    loader.close()


def test_a_reload_of_a_non_blf_source_is_untouched(blf_flow, monkeypatch):
    """回归护栏：给 BLF 加的分派不能改变 CSV/MDF 的路径。"""
    host, manager, threads = blf_flow

    class _Csv:
        LOADER_TYPE = "csv"
        path = "x.csv"
        db_paths = None

    host.loader = _Csv()
    called = []
    monkeypatch.setattr(manager, "_start_blf_load",
                        lambda *a, **k: called.append(a))
    asked = []
    monkeypatch.setattr(manager, "_ask_can_databases",
                        lambda *a, **k: asked.append(a) or [])

    assert manager._is_blf_source() is False
    assert manager._extract_file_extension("x.blf") == ".blf"
    assert manager._extract_file_extension("X.MF4") == ".mf4"
    assert called == [] and asked == []

    manager._update_can_db_ui()
    assert host.can_db_banner.isHidden() is True
    assert host._can_segment.isHidden() is True
