"""身份匹配 e2e：三播报路径 remap>0 文案 + reload 行为不变回归（C3 / D5）。

真实 MainWindow + 真实 MDF 后台加载 + 合成组序对调夹具（press_si）。
remap=0 的既有精确文案断言在 test_status_bar.py，保持绿。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from src.data.mdf_lazy_loader import MDFLazyLoader
from tests.fixtures.data_factory import write_mdf

TIMEOUT = 15_000


@pytest.fixture()
def mdf_files(tmp_path):
    a = write_mdf(
        tmp_path / "syn_a.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True, press_si=True,
    )
    b = write_mdf(
        tmp_path / "syn_b.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True, press_si=True,
        dup_group_first=True,
    )
    return a, b


@pytest.fixture()
def mdf_window(main_window, qapp, qtbot, dialog_stubs, mdf_files):
    """真实主窗口 + MDF 加载助手；auto-save 默认关闭并清档，避免跨用例串档。"""
    mw = main_window
    asm = mw.plot_config_manager.auto_save_manager
    asm.set_auto_save_enabled(False)
    for f in (asm._auto_save_file, asm._auto_save_backup_file):
        f.unlink(missing_ok=True)

    def load(path):
        old = mw.loader
        dialog_stubs["open_file"] = (str(path), "MDF Files (*.mf4 *.mdf *.dat)")
        qtbot.mouseClick(mw.load_btn, Qt.MouseButton.LeftButton)

        def ready():
            if dialog_stubs["critical"]:
                raise RuntimeError(f"MDF 加载失败: {dialog_stubs['critical']}")
            return (
                mw.loader is not None
                and mw.loader is not old
                and isinstance(mw.loader, MDFLazyLoader)
                and len(mw.loader.var_names) > 0
                and not mw._is_loading_new_data
                and len(mw.plot_widgets) > 0
            )

        qtbot.waitUntil(ready, timeout=TIMEOUT)
        qapp.processEvents()

    yield mw, load
    asm.set_auto_save_enabled(False)
    for f in (asm._auto_save_file, asm._auto_save_backup_file):
        f.unlink(missing_ok=True)


def _pw(mw):
    return mw.current_page.plot_widgets[0].plot_widget


def _export_template_dict(mw):
    assert _pw(mw).plot_variable("Press_G0")
    cfg = mw.plot_config_manager.export_current_config(mw)
    assert "Press_G0" in cfg.pages[0].plots[0].curve_identities
    return cfg.to_dict()


def test_normal_apply_announces_remap(
    mdf_window, qapp, mdf_files, monkeypatch
):
    mw, load = mdf_window
    a, b = mdf_files
    load(a)
    cfg_dict = _export_template_dict(mw)
    load(b)
    qapp.processEvents()

    import src.core.template_identity as identity_engine

    real_resolve = identity_engine.resolve_identity_snapshot
    calls = []

    def counted_resolve(config, loader):
        calls.append(config)
        return real_resolve(config, loader)

    monkeypatch.setattr(identity_engine, "resolve_identity_snapshot", counted_resolve)
    mw._check_and_apply_template(SimpleNamespace(config=cfg_dict), "tid00001", "身份模板")
    qapp.processEvents()

    assert len(calls) == 1, "匹配、套用和播报必须复用同一 resolved 快照"
    assert mw._message_text == "已套用模板[身份模板] · 匹配 100%（自动重映射 1 条）"
    assert mw._message_level == "info"
    assert "Press_G1" in _pw(mw).curves
    assert "Press_G0" not in _pw(mw).curves


def test_forced_apply_announces_remap(mdf_window, qapp, qtbot, mdf_files, monkeypatch):
    mw, load = mdf_window
    a, b = mdf_files
    load(a)
    cfg_dict = _export_template_dict(mw)
    # 追加 3 个两边都不存在的变量 → 匹配 25% < 30% → 走低匹配弹窗
    cfg_dict["pages"][0]["plots"][0]["curves"].extend(["ghost1", "ghost2", "ghost3"])
    cfg_dict["plots"][0]["curves"].extend(["ghost1", "ghost2", "ghost3"])
    load(b)
    qapp.processEvents()

    def fake_exec(box):
        next(btn for btn in box.buttons() if btn.text() == "仍然加载").click()
        return 1

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    mw._check_and_apply_template(SimpleNamespace(config=cfg_dict), "tid00002", "身份模板")
    qapp.processEvents()

    assert mw._message_text == (
        "已强制套用模板[身份模板] · 匹配 25%（3 个变量缺失）（自动重映射 1 条）"
    )
    assert mw._message_level == "warn"
    assert "Press_G1" in _pw(mw).curves


def test_auto_restore_announces_remap(mdf_window, qapp, mdf_files):
    """MN-4：_announce_auto_restore 显式 resolve 后算 ratio，同规格播报。"""
    mw, load = mdf_window
    a, b = mdf_files
    load(a)
    cfg_dict = _export_template_dict(mw)
    load(b)
    qapp.processEvents()

    from src.core.plot_config import PlotSessionConfig
    cfg = PlotSessionConfig.from_dict(cfg_dict)
    mw.file_loader_manager._announce_auto_restore(cfg, list(mw.loader.var_names))
    assert mw._message_text == "已自动恢复上次布局 · 匹配 100%（自动重映射 1 条）"
    assert mw._message_level == "info"


def test_auto_restore_full_flow_remaps(mdf_window, qapp, qtbot, mdf_files):
    """加载收尾全链路：auto-save 放行判据与播报同口径（重映射计入 matched）。"""
    mw, load = mdf_window
    a, b = mdf_files
    load(a)
    cfg_dict = _export_template_dict(mw)
    asm = mw.plot_config_manager.auto_save_manager
    asm.set_auto_save_enabled(True)
    from src.core.plot_config import PlotSessionConfig
    asm.auto_save(PlotSessionConfig.from_dict(cfg_dict))
    # 清掉全部曲线，让加载 B 时走 auto-save 兜底（replots 无曲线可恢复）
    mw.cursor_sync_manager.clear_all_plots()
    qapp.processEvents()

    load(b)
    qtbot.waitUntil(lambda: "自动恢复" in mw._message_text, timeout=TIMEOUT)
    assert mw._message_text == "已自动恢复上次布局 · 匹配 100%（自动重映射 1 条）"
    qtbot.waitUntil(lambda: "Press_G1" in _pw(mw).curves, timeout=TIMEOUT)


def test_should_apply_auto_save_counts_remap(mdf_window, mdf_files):
    """放行判据：漂移曲线经 resolve 计入 matched， ratio 1.0 ≥ 0.6。"""
    mw, load = mdf_window
    a, b = mdf_files
    load(a)
    cfg_dict = _export_template_dict(mw)
    asm = mw.plot_config_manager.auto_save_manager
    asm.set_auto_save_enabled(True)
    from src.core.plot_config import PlotSessionConfig
    asm.auto_save(PlotSessionConfig.from_dict(cfg_dict))
    load(b)

    should, reason = asm.should_apply_auto_save(list(mw.loader.var_names), mw.loader)
    assert should is True
    assert "100%" in reason
    # 旧口径（不传 loader）：Press_G0 在 B 中同名仍在（BL-1 风险本身），
    # ratio 同为 100%——差别不在放行而在绑定正确性：传 loader 时套用
    # 画的是 Press_G1（bar），见 test_auto_restore_full_flow_remaps。
    should_old, reason_old = asm.should_apply_auto_save(list(mw.loader.var_names))
    assert should_old is True
    assert "100%" in reason_old


def test_reload_behavior_unchanged(mdf_window, qapp, qtbot, mdf_files):
    """reload 路径本轮明确排除身份重映射：按旧名重建，行为与现状一致。"""
    mw, load = mdf_window
    a, _b = mdf_files
    load(a)
    assert _pw(mw).plot_variable("Press_G0")
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=TIMEOUT)
    version_before = mw._data_version

    mw.file_loader_manager.reload_data()
    qtbot.waitUntil(lambda: mw._data_version > version_before, timeout=TIMEOUT)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=TIMEOUT)
    qapp.processEvents()

    assert list(_pw(mw).curves) == ["Press_G0"]
    assert "自动重映射" not in mw._message_text


# ---------------- M-1 方案乙：fail-closed 三播报路径 ----------------

@pytest.fixture()
def nosi_files(tmp_path):
    """无 SI 双组对调夹具（真实 MDF3 主流态）+ Speed 对照通道。"""
    a = write_mdf(
        tmp_path / "syn_a.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True,
        with_time_named_data=True,
    )
    b = write_mdf(
        tmp_path / "syn_b.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True,
        with_time_named_data=True, dup_group_first=True,
    )
    return a, b


@pytest.fixture()
def mdf_window_nosi(main_window, qapp, qtbot, dialog_stubs, nosi_files):
    mw = main_window
    asm = mw.plot_config_manager.auto_save_manager
    asm.set_auto_save_enabled(False)
    for f in (asm._auto_save_file, asm._auto_save_backup_file):
        f.unlink(missing_ok=True)

    def load(path):
        old = mw.loader
        dialog_stubs["open_file"] = (str(path), "MDF Files (*.mf4 *.mdf *.dat)")
        qtbot.mouseClick(mw.load_btn, Qt.MouseButton.LeftButton)

        def ready():
            if dialog_stubs["critical"]:
                raise RuntimeError(f"MDF 加载失败: {dialog_stubs['critical']}")
            return (
                mw.loader is not None
                and mw.loader is not old
                and isinstance(mw.loader, MDFLazyLoader)
                and len(mw.loader.var_names) > 0
                and not mw._is_loading_new_data
                and len(mw.plot_widgets) > 0
            )

        qtbot.waitUntil(ready, timeout=TIMEOUT)
        qapp.processEvents()

    yield mw, load
    asm.set_auto_save_enabled(False)
    for f in (asm._auto_save_file, asm._auto_save_backup_file):
        f.unlink(missing_ok=True)


def _export_nosi_template(mw, curves):
    for name in curves:
        assert _pw(mw).plot_variable(name)
    cfg = mw.plot_config_manager.export_current_config(mw)
    assert "Press_G0" in cfg.pages[0].plots[0].curve_identities
    return cfg.to_dict()


def test_fail_closed_normal_path_announces(mdf_window_nosi, qapp, nosi_files):
    """无 SI 单位矛盾降占位：ratio 不虚高（50%）+ 显式披露条数，warn。"""
    mw, load = mdf_window_nosi
    a, b = nosi_files
    load(a)
    cfg_dict = _export_nosi_template(mw, ["Press_G0", "Speed"])
    load(b)
    qapp.processEvents()

    mw._check_and_apply_template(SimpleNamespace(config=cfg_dict), "tid00003", "身份模板")
    qapp.processEvents()

    assert mw._message_text == (
        "已套用模板[身份模板] · 匹配 50%（1 个变量缺失）（1 条身份矛盾降占位）"
    )
    assert mw._message_level == "warn"
    assert "Speed" in _pw(mw).curves
    assert "Press_G0" not in _pw(mw).curves          # 不绘制
    assert "Press_G0" in _pw(mw).invalid_curves      # 灰色占位


def test_fail_closed_forced_path_announces(mdf_window_nosi, qapp, nosi_files, monkeypatch):
    mw, load = mdf_window_nosi
    a, b = nosi_files
    load(a)
    cfg_dict = _export_nosi_template(mw, ["Press_G0"])
    load(b)
    qapp.processEvents()

    def fake_exec(box):
        next(btn for btn in box.buttons() if btn.text() == "仍然加载").click()
        return 1

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    mw._check_and_apply_template(SimpleNamespace(config=cfg_dict), "tid00004", "身份模板")
    qapp.processEvents()

    assert mw._message_text == (
        "已强制套用模板[身份模板] · 匹配 0%（1 个变量缺失）（1 条身份矛盾降占位）"
    )
    assert mw._message_level == "warn"
    assert "Press_G0" not in _pw(mw).curves
    assert "Press_G0" in _pw(mw).invalid_curves


def test_fail_closed_auto_restore_announces(mdf_window_nosi, qapp, nosi_files):
    mw, load = mdf_window_nosi
    a, b = nosi_files
    load(a)
    cfg_dict = _export_nosi_template(mw, ["Press_G0"])
    load(b)
    qapp.processEvents()

    from src.core.plot_config import PlotSessionConfig
    cfg = PlotSessionConfig.from_dict(cfg_dict)
    mw.file_loader_manager._announce_auto_restore(cfg, list(mw.loader.var_names))
    assert mw._message_text == (
        "已自动恢复上次布局 · 匹配 0%（1 个变量缺失）（1 条身份矛盾降占位）"
    )
    assert mw._message_level == "warn"


def test_fail_closed_auto_save_rejects(mdf_window_nosi, nosi_files):
    """should_apply_auto_save 同口径：全降占位 → ratio 0 → 拒放。"""
    mw, load = mdf_window_nosi
    a, b = nosi_files
    load(a)
    cfg_dict = _export_nosi_template(mw, ["Press_G0"])
    asm = mw.plot_config_manager.auto_save_manager
    asm.set_auto_save_enabled(True)
    from src.core.plot_config import PlotSessionConfig
    asm.auto_save(PlotSessionConfig.from_dict(cfg_dict))
    load(b)

    should, reason = asm.should_apply_auto_save(list(mw.loader.var_names), mw.loader)
    assert should is False
    assert reason == "No matching variables"