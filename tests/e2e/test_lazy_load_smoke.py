"""P6 惰性加载接线 e2e 冒烟（offscreen 主窗口，§6.1）。

覆盖计划 P6 验收的自动化部分：
- 异步入口 → create_loader → 转换 → ParquetLazyLoader（df 恒为 None）
- 重载 = 重新转换一遍，已绘曲线存活（D3/D12）
- LAZY_CONVERT_ENABLED=false → 回到今天的 FastDataLoader（回滚验证）
- 同步入口永不转换（D16）
- 转换失败 → 删临时目录 → 回退内存 loader + 状态栏播报、不弹窗（D10/Q6）

CI 口径说明：真实阈值 50MB 的大文件不进 CI（§6.4 耗时护栏），本文件把
异步分流阈值与 LAZY_CONVERT_MIN_MB 双双打到 0，用 KB 级合成 CSV 验证
**路由与开关语义**；>50MB 真实文件的耗时/内存验收走 §6.3 手工冒烟与
tmp/ 下的实测脚本（报告用文件A/文件B 占位）。
"""

from __future__ import annotations

import os

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent

from src.core.settings import AppSettings
from src.data.loader import FastDataLoader
from src.data.parquet_converter import ParquetConversionError
from src.data.parquet_lazy_loader import ParquetLazyLoader
from src.data.temp_cache_dir import TempCacheDir
from src.ui.drag_drop import build_var_mimedata
from tests.fixtures.data_factory import make_simple_rows, write_csv

# 保活容器：offscreen 下手工构造的 QMimeData 需防 GC（README 陷阱 #2）
_keep_alive: list = []


@pytest.fixture()
def small_csv(tmp_path):
    return write_csv(
        tmp_path / "lazy_smoke.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(50),
    )


@pytest.fixture()
def force_async_lazy(monkeypatch):
    """把异步分流阈值与转换阈值都打到 0：小文件也走「异步 + 转换」。"""
    monkeypatch.setattr(
        "src.ui.file_loader_manager.FILE_SIZE_LIMIT_BACKGROUND_LOADING", 0
    )
    monkeypatch.setattr(AppSettings, "get_lazy_convert_min_mb", lambda self: 0)


def _load_via_button(mw, qtbot, qapp, dialog_stubs, path):
    dialog_stubs["open_file"] = (str(path), "CSV/TXT Files (*.csv)")
    qtbot.mouseClick(mw.load_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: mw.loader is not None, timeout=15000)
    qapp.processEvents()  # 让 singleShot(0) 的 _post_load_actions 落地


def _drop_plot(mw, qapp, var_name):
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata([var_name])
    _keep_alive.append(mime)
    ev = QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.dropEvent(ev)
    qapp.processEvents()
    return pw


def _pid_dirs() -> set[str]:
    base = TempCacheDir.base_dir()
    prefix = f"{os.getpid()}_"
    return {p.name for p in base.iterdir() if p.name.startswith(prefix)}


# ---------- 惰性路径：加载 → 绘图 → 重载 → 曲线存活 ----------

def test_lazy_load_reload_curves_survive(
    main_window, qtbot, qapp, dialog_stubs, small_csv, force_async_lazy
):
    mw = main_window
    _load_via_button(mw, qtbot, qapp, dialog_stubs, small_csv)

    assert isinstance(mw.loader, ParquetLazyLoader), "阈值满足时应走惰性转换路径"
    assert mw.loader.df is None  # D6 硬性契约
    assert mw.data is None  # 影子引用自然断裂（§2.4 #20）
    assert not dialog_stubs["critical"]

    pw = _drop_plot(mw, qapp, "speed")
    assert "speed" in pw.curves
    first_loader = mw.loader

    # 重载 = 重新转换一遍（D4：不做跨会话/指纹复用）。
    # _is_loading_new_data 由 _on_load_done 之后的链式 singleShot 清除
    # （_restore_cursor_state_after_reload → _post_reload_ui_refresh），
    # 必须等锁放开再点，否则 reload_data 静默忽略本次请求
    qtbot.waitUntil(
        lambda: mw.reload_btn.isEnabled() and not mw._is_loading_new_data,
        timeout=15000,
    )
    qtbot.mouseClick(mw.reload_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: mw.loader is not first_loader and mw.loader is not None,
        timeout=15000,
    )
    qapp.processEvents()

    assert isinstance(mw.loader, ParquetLazyLoader)
    assert "speed" in pw.curves, "重载后已绘曲线必须存活（validity=1，D12）"
    # 旧 loader 的临时目录已随 _release_old_data → close() 删除
    assert first_loader._cache_dir._cleaned is True


# ---------- 开关关闭 = 回到今天的行为（回滚验证） ----------

def test_switch_off_returns_memory_loader(
    main_window, qtbot, qapp, dialog_stubs, small_csv, force_async_lazy, monkeypatch
):
    monkeypatch.setattr(AppSettings, "is_lazy_convert_enabled", lambda self: False)
    mw = main_window
    _load_via_button(mw, qtbot, qapp, dialog_stubs, small_csv)

    assert isinstance(mw.loader, FastDataLoader)
    assert not isinstance(mw.loader, ParquetLazyLoader)
    assert mw.loader.df is not None
    assert mw.data is not None


# ---------- 同步入口永不转换（D16） ----------

def test_sync_entry_never_converts(
    main_window, qtbot, qapp, dialog_stubs, small_csv, monkeypatch
):
    """不打异步阈值补丁：KB 级文件走同步 _load_sync，即使 min_mb=0 也不转换。

    转换是 CPU 重活，落在 GUI 线程就是整窗冻结——该组合必须在设计上不可达。
    """
    monkeypatch.setattr(AppSettings, "get_lazy_convert_min_mb", lambda self: 0)
    mw = main_window
    _load_via_button(mw, qtbot, qapp, dialog_stubs, small_csv)

    assert isinstance(mw.loader, FastDataLoader)
    assert mw.loader.df is not None


# ---------- D10：转换失败 → 回退 + 播报，不弹窗 ----------

def test_conversion_failure_falls_back_with_status_announce(
    main_window, qtbot, qapp, dialog_stubs, small_csv, force_async_lazy, monkeypatch
):
    def boom(*args, **kwargs):
        raise ParquetConversionError("合成失败（测试注入）")

    monkeypatch.setattr("src.data.parquet_converter.convert_to_parquet", boom)
    mw = main_window
    before = _pid_dirs()
    _load_via_button(mw, qtbot, qapp, dialog_stubs, small_csv)

    # 回退到内存 loader，且加载成功（用户视角：文件照常打开）
    assert isinstance(mw.loader, FastDataLoader)
    assert mw.loader.df is not None
    assert not dialog_stubs["critical"], "D10：回退不得弹错误窗"
    assert not dialog_stubs["warning"], "D10/Q6：播报走状态栏，不弹窗"

    # 状态栏播报（唯一文案出口 _lazy_fallback_message）
    qtbot.waitUntil(lambda: "内存模式" in mw._message_text, timeout=5000)
    assert "转换失败" in mw._message_text

    # 失败的转换临时目录已删（不得泄漏本进程的新目录）
    assert _pid_dirs() == before
