"""启动优化回归：后台导入、分阶段让帧、静态 splash 与惰性重依赖。"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QWidget

import csv_plot
from src.ui.main_window import MainWindow
from src.ui.splash_screen import SplashScreen


@pytest.fixture()
def splash(qapp, qtbot):
    sp = SplashScreen()
    qtbot.addWidget(sp)
    yield sp
    sp.close()


@pytest.fixture()
def restore_quit_policy(qapp):
    original = qapp.quitOnLastWindowClosed()
    yield qapp
    qapp.setQuitOnLastWindowClosed(original)


def test_preload_runs_off_ui_thread_and_window_builds_on_ui_thread(
    splash, restore_quit_policy, qtbot
):
    ui_thread = threading.get_ident()
    preload_threads: list[int] = []
    factory_threads: list[int] = []
    fake_window = QWidget()
    qtbot.addWidget(fake_window)

    def preload():
        preload_threads.append(threading.get_ident())
        time.sleep(0.03)

    def factory():
        factory_threads.append(threading.get_ident())
        return fake_window

    window, error = csv_plot._run_splash_startup(
        restore_quit_policy, splash, 0, factory, preload=preload
    )

    assert error == ""
    assert window is fake_window
    assert len(preload_threads) == 1
    assert preload_threads[0] != ui_thread
    assert factory_threads == [ui_thread]


def test_preload_exception_does_not_hang(splash, restore_quit_policy):
    def boom():
        raise RuntimeError("preload boom")

    window, error = csv_plot._run_splash_startup(
        restore_quit_policy,
        splash,
        0,
        lambda: QWidget(),
        preload=boom,
    )

    assert window is None
    assert "RuntimeError: preload boom" in error


def test_apply_appearance_uses_qapplication_font_and_icon(qapp, monkeypatch):
    monkeypatch.setattr(
        "src.core.font_cache.get_windows_chinese_font_cached", lambda: ""
    )
    csv_plot._apply_appearance(qapp, None)
    assert qapp.styleSheet()


def test_splash_reuses_static_layer_without_font_enumeration(
    splash, qapp, qtbot, monkeypatch
):
    def forbidden_font_enumeration():
        raise AssertionError("splash must not enumerate system fonts")

    monkeypatch.setattr(
        "src.core.font_cache.installed_font_families", forbidden_font_enumeration
    )
    splash.show()
    qtbot.wait(25)
    layer = splash._static_layer
    assert layer is not None and not layer.isNull()

    splash.elapsed = 500
    splash.update()
    qapp.processEvents()
    assert splash._static_layer is layer


def test_main_window_yields_after_every_construction_phase(qapp, qtbot, monkeypatch):
    phases = (
        "_init_basic_config",
        "_load_window_context",
        "_init_data_state",
        "_init_central_widget",
        "_init_left_panel",
        "_init_right_panel",
        "_init_status_bar",
        "_init_managers",
        "_handle_cli_args",
    )
    for phase in phases:
        monkeypatch.setattr(MainWindow, phase, lambda self: None)
    yielded: list[int] = []

    window = MainWindow(startup_yield=lambda: yielded.append(1))
    qtbot.addWidget(window)

    assert len(yielded) == len(phases)


def test_real_construction_defers_zero_delay_callbacks(qapp, qtbot, monkeypatch):
    from src.ui.layout_manager import LayoutManager

    early_activation: list[bool] = []
    early_splitter: list[bool] = []
    post_activation = MainWindow._post_activation
    ensure_splitter = LayoutManager._ensure_splitter_ready

    def checked_activation(self, *args):
        if self._startup_constructing:
            early_activation.append(True)
        return post_activation(self, *args)

    def checked_splitter(self):
        if self.mw._startup_constructing:
            early_splitter.append(True)
        return ensure_splitter(self)

    monkeypatch.setattr(MainWindow, "_post_activation", checked_activation)
    monkeypatch.setattr(LayoutManager, "_ensure_splitter_ready", checked_splitter)
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])

    window = MainWindow(
        startup_yield=lambda: qapp.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )
    )
    qtbot.addWidget(window)
    qtbot.wait(20)

    assert early_activation == []
    assert early_splitter == []
    assert window._startup_constructing is False
    assert window.layout_manager is not None


def test_core_package_exports_remain_lazy_and_compatible():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; import src.core.startup_metrics; "
        "assert 'src.core.settings' not in sys.modules; "
        "assert 'src.core.auto_save_manager' not in sys.modules; "
        "assert 'PySide6.QtCore' not in sys.modules; "
        "from src.core import AppSettings, ConfigKey, PlotConfig; "
        "assert AppSettings is not None; assert ConfigKey is not None; "
        "assert PlotConfig is not None"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_splash_rebuilds_static_layer_after_dpr_change(
    splash, qapp, monkeypatch
):
    monkeypatch.setattr(SplashScreen, "devicePixelRatioF", lambda self: 1.0)
    splash.show()
    qapp.processEvents()
    first = splash._static_layer
    assert first is not None and first.devicePixelRatio() == 1.0

    monkeypatch.setattr(SplashScreen, "devicePixelRatioF", lambda self: 2.0)
    splash.update()
    qapp.processEvents()

    assert splash._static_layer is not first
    assert splash._static_layer.devicePixelRatio() == 2.0


def test_main_window_import_defers_pandas_and_data_loaders():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; import src.ui.main_window; "
        "assert 'pandas' not in sys.modules; "
        "assert 'src.ui.table_dialog' not in sys.modules; "
        "assert 'src.ui.plot_variable_editor' not in sys.modules; "
        "assert 'src.data.loader' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_data_package_exports_remain_lazy_and_compatible():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; import src.data; "
        "assert 'src.data.loader' not in sys.modules; "
        "from src.data import FastDataLoader, MDFLazyLoader, VALID; "
        "assert FastDataLoader.LOADER_TYPE == 'csv'; "
        "assert MDFLazyLoader.LOADER_TYPE == 'mdf'; "
        "assert VALID is not None"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
