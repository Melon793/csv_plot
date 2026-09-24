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

    # --- 护栏 1：断言强化（RCA §7 / §8 护栏 1） ---
    # 等收尾链完成（_restore_cursor_state_after_reload → _post_reload_ui_refresh）
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=15000)
    assert mw._is_loading_new_data is False, "重载收尾后全局锁必须释放"
    assert pw._is_updating_data is False, "重载收尾后 widget 锁必须释放"

    ci = pw.curves["speed"]
    assert ci.curve is not None, "CurveInfo.curve 不应为 None"
    assert ci.curve.scene() is not None, "曲线必须仍在 QGraphicsScene 中"

    # Y 轴范围应覆盖曲线数据域
    y_range = pw.view_box.viewRange()[1]
    y_data = ci.curve.getData()[1]
    if y_data is not None and len(y_data) > 0:
        import numpy as np
        assert y_range[0] <= float(np.nanmin(y_data)) + 1e-6, (
            f"Y 轴下界 {y_range[0]} 未覆盖数据最小值 {np.nanmin(y_data)}"
        )
        assert y_range[1] >= float(np.nanmax(y_data)) - 1e-6, (
            f"Y 轴上界 {y_range[1]} 未覆盖数据最大值 {np.nanmax(y_data)}"
        )


# ---------- 护栏 2：事故直接回归（调度失败 → 兜底不销毁曲线） ----------

def test_reload_finish_schedule_failure_keeps_curves(
    main_window, qtbot, qapp, dialog_stubs, small_csv, force_async_lazy, monkeypatch
):
    """模拟 Nuitka exe 里 _post_reload_ui_refresh 调度失败的场景。

    验证：3 秒兜底 _force_unlock_all 只解锁 + 重建失效曲线，绝不销毁好曲线。
    （RCA §8 护栏 2；tmp/rca-lazy-reload-curve-vanish.md）
    """
    mw = main_window
    _load_via_button(mw, qtbot, qapp, dialog_stubs, small_csv)
    pw = _drop_plot(mw, qapp, "speed")
    assert "speed" in pw.curves

    # 等首次加载收尾完成
    qtbot.waitUntil(
        lambda: mw.reload_btn.isEnabled() and not mw._is_loading_new_data,
        timeout=15000,
    )

    # --- 注入：拦截 _schedule_delayed，只阻止 _post_reload_ui_refresh 的调度 ---
    from src.ui.file_loader_manager import FileLoaderManager

    blocked_slots: list = []
    original_schedule = FileLoaderManager._schedule_delayed

    def patched_schedule(self, msec, slot):
        slot_name = getattr(slot, "__name__", "") or repr(slot)
        if "_post_reload_ui_refresh" in slot_name:
            blocked_slots.append(slot_name)
            return None  # 不调度 → 等价于打包形态的 TypeError
        return original_schedule(self, msec, slot)

    monkeypatch.setattr(FileLoaderManager, "_schedule_delayed", patched_schedule)

    # --- 触发重载 ---
    first_loader = mw.loader
    qtbot.mouseClick(mw.reload_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: mw.loader is not first_loader and mw.loader is not None,
        timeout=15000,
    )
    qapp.processEvents()

    # 断言事故现场成立：收尾被阻止、锁仍为 True
    assert blocked_slots, "_post_reload_ui_refresh 的调度应被拦截"
    assert mw._is_loading_new_data is True, "收尾未执行 → 全局锁应仍为 True"

    # --- 制造"曲线停在半空中"（scene=None），模拟兜底前的最坏状态 ---
    ci = pw.curves["speed"]
    assert ci.curve is not None
    pw.plot_item.removeItem(ci.curve)
    assert ci.curve.scene() is None, "removeItem 后曲线应脱离场景"

    # --- 调用兜底（等价于 3 秒定时器到点） ---
    # 直接调 _safety_force_unlock 而不真等 3 秒：等待不增加覆盖面，
    # 定时器到点 → _safety_force_unlock → _force_unlock_all 是确定性链路。
    mw.file_loader_manager._safety_force_unlock()
    qapp.processEvents()

    # --- 断言：锁释放 + 曲线恢复 ---
    assert mw._is_loading_new_data is False, "兜底后全局锁必须释放"
    assert pw._is_updating_data is False, "兜底后 widget 锁必须释放"
    assert mw.reload_btn.isEnabled(), "兜底后 reload 按钮必须可用"

    ci_after = pw.curves["speed"]
    assert ci_after.curve is not None, "兜底后 CurveInfo.curve 不应为 None"
    assert ci_after.curve.scene() is not None, (
        "兜底必须重建失效曲线（scene=None → _recreate_curve），绝不能销毁好曲线"
    )


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
