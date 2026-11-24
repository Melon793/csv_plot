"""恢复链路正式契约回归：R4/R6/R10 与非事务部分应用边界。

只使用合成变量名与测试替身；不依赖 tmp 探针或真实数据资产。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QLabel, QWidget

from src.core.auto_save_manager import AutoSaveManager
from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig
from src.ui.file_loader_manager import FileLoaderManager
from src.ui.plot_config_manager import PlotConfigManager


class _Button:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def setEnabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self.enabled


class _AutoSaveStub:
    def __init__(self):
        self.config = PlotSessionConfig(plots=[PlotConfig(curves=["a"])])
        self.prepare_calls: list[tuple[list[str], object]] = []
        self.load_calls = 0

    def prepare_auto_save(self, current_vars, loader=None):
        self.prepare_calls.append((list(current_vars), loader))
        return SimpleNamespace(
            should=True,
            reason="matched",
            config=self.config,
            identity_snapshot=None,
            failed=False,
        )

    def load_auto_save(self):
        self.load_calls += 1
        return self.config


class _PostLoadHost:
    def __init__(self):
        self.loaded_path = None
        self.var_names = ["a"]
        self.loader = SimpleNamespace(
            fffd_unit_fixes=2, var_names=["a"], datalength=3
        )
        self.data = None
        self.broadcasts: list[tuple[str, str]] = []
        self.file_status: list[tuple] = []
        self.auto_save = _AutoSaveStub()
        self.applied: list[PlotSessionConfig] = []
        self.plot_config_manager = SimpleNamespace(
            auto_save_manager=self.auto_save,
            apply_config=self._apply_config,
        )
        for name in (
            "clear_all_plots_btn",
            "auto_range_btn",
            "auto_y_btn",
            "cursor_btn",
            "mark_region_btn",
            "grid_layout_btn",
            "curve_shelf_btn",
        ):
            setattr(self, name, _Button())

    def _apply_config(self, mw, config, **_kwargs):
        assert mw is self
        self.applied.append(config)
        return True

    def _broadcast(self, text, level="info"):
        self.broadcasts.append((text, level))

    def end_load_feedback(self, *args, **kwargs):
        pass

    def update_file_status(self, *args, **kwargs):
        self.file_status.append((args, kwargs))


@pytest.fixture()
def post_load_env(monkeypatch):
    host = _PostLoadHost()
    flm = FileLoaderManager(host)
    flm._last_replot_result = "empty"
    monkeypatch.setattr(flm, "_window_going_away", lambda: False)
    monkeypatch.setattr(flm, "_remember_last_open_dir", lambda path: None)
    monkeypatch.setattr(flm, "_announce_auto_restore", lambda cfg, vars, **kwargs: None)
    return flm, host


# ---------------------------------------------------------------------------
# R4：恢复判据消费显式 replots 结果，不从可见 curves 反推
# ---------------------------------------------------------------------------


def test_post_load_uses_explicit_restored_status_without_inspecting_assets(
    post_load_env, monkeypatch
):
    flm, host = post_load_env
    flm._last_replot_result = "restored"
    inspections: list[str] = []

    def explode():
        inspections.append("containers")
        raise RuntimeError("assets must not be inspected")

    monkeypatch.setattr(flm, "_iter_all_containers", explode)

    flm._post_load_actions("demo.csv")

    assert inspections == [], "restored 状态必须直接短路 auto-save 判据"
    assert host.auto_save.prepare_calls == []
    assert host.auto_save.load_calls == 0
    assert host.applied == []


@pytest.mark.parametrize(
    ("curves", "invalid", "hidden"),
    [
        ({}, {"ghost": object()}, False),
        ({"a": object()}, {}, True),
        ({}, {"ghost": object()}, True),
    ],
    ids=["visible-placeholder", "hidden-page-curve", "hidden-cell-placeholder"],
)
def test_restored_placeholder_or_hidden_assets_skip_auto_save(
    post_load_env, monkeypatch, curves, invalid, hidden
):
    flm, host = post_load_env
    flm._last_replot_result = "restored"
    container = SimpleNamespace(
        isHidden=lambda: hidden,
        plot_widget=SimpleNamespace(curves=curves, invalid_curves=invalid),
    )
    monkeypatch.setattr(flm, "_iter_all_containers", lambda: [container])

    flm._post_load_actions("demo.csv")

    assert host.auto_save.prepare_calls == []
    assert host.auto_save.load_calls == 0
    assert host.applied == []


@pytest.mark.parametrize("status", ["restored", "empty", "failed"])
def test_apply_loader_records_explicit_replot_result(qapp, monkeypatch, status):
    """replots 的返回值必须接力到 post-load，不能丢失后再看 curves。"""

    class ApplyHost(QWidget):
        def __init__(self):
            super().__init__()
            self.loader = SimpleNamespace(
                var_names=["a"],
                units={},
                time_channels_info={},
                time_column_name=None,
                time_axis_label="Index",
                df_validity={"a": 1},
                df=object(),
                datalength=3,
            )
            self.list_widget = SimpleNamespace(populate=lambda *args: None)
            self.placeholder_label = QLabel()
            self.filter_input = SimpleNamespace(text=lambda: "")
            self.unit_filter_input = SimpleNamespace(text=lambda: "")
            self.mark_region_btn = SimpleNamespace(isChecked=lambda: False)
            widget = SimpleNamespace(update_x_axis_label=lambda: None)
            self.plot_widgets = [SimpleNamespace(plot_widget=widget)]
            self.cursor_sync_manager = SimpleNamespace(
                _compute_baseline_density=lambda: None,
                _sync_min_xrange=lambda: None,
                replots_after_loading=lambda **kwargs: status,
            )

    host = ApplyHost()
    flm = FileLoaderManager(host)
    monkeypatch.setattr(flm, "_update_can_db_ui", lambda: None)
    monkeypatch.setattr(flm, "_refresh_table_dialog", lambda loader: None)

    flm._apply_loader()

    assert flm._last_replot_result == status
    host.deleteLater()


# ---------------------------------------------------------------------------
# R6：单次读取、backup 回退与失败反馈
# ---------------------------------------------------------------------------


def test_auto_restore_match_and_apply_share_one_loaded_config(
    post_load_env, monkeypatch, tmp_path
):
    flm, host = post_load_env
    manager = AutoSaveManager(tmp_path)
    first = PlotSessionConfig(plots=[PlotConfig(curves=["a"])])
    decoy = PlotSessionConfig(plots=[PlotConfig(curves=["b"])])
    loaded_configs = iter([first, decoy])
    loads: list[PlotSessionConfig] = []

    def load_once():
        config = next(loaded_configs)
        loads.append(config)
        return config

    monkeypatch.setattr(manager, "is_auto_save_enabled", lambda: True)
    monkeypatch.setattr(manager, "load_auto_save", load_once)
    host.plot_config_manager.auto_save_manager = manager

    flm._post_load_actions("demo.csv")

    assert loads == [first], "匹配和套用不得分别读盘得到两份配置"
    assert host.applied == [first]
    assert host.applied[0] is loads[0]


def test_corrupt_primary_auto_save_falls_back_to_backup(tmp_path):
    manager = AutoSaveManager(tmp_path)
    older = PlotSessionConfig(plots=[PlotConfig(curves=["a"])])
    manager.auto_save(older)
    manager.auto_save(PlotSessionConfig(plots=[PlotConfig(curves=["a", "b"])]))
    manager._auto_save_file.write_text("pages: [broken", encoding="utf-8")

    loaded = manager.load_auto_save()

    assert loaded is not None
    assert loaded.normalized_pages()[0].plots[0].curves == ["a"]


def test_corrupt_primary_and_backup_warn_once_and_merge_fffd_disclosure(
    post_load_env, monkeypatch, tmp_path
):
    flm, host = post_load_env
    manager = AutoSaveManager(tmp_path)
    manager.auto_save(PlotSessionConfig(plots=[PlotConfig(curves=["a"])]))
    manager._auto_save_file.write_text("pages: [broken", encoding="utf-8")
    manager._auto_save_backup_file.write_text("pages: [also-broken", encoding="utf-8")
    monkeypatch.setattr(manager, "is_auto_save_enabled", lambda: True)
    host.plot_config_manager.auto_save_manager = manager

    flm._post_load_actions("demo.csv")

    failures = [
        (text, level)
        for text, level in host.broadcasts
        if level in ("warn", "error") and not text.startswith("已推断修复")
    ]
    assert len(failures) == 1, host.broadcasts
    assert "自动恢复" in failures[0][0]
    assert host.applied == []
    assert host.broadcasts[-1][0] == failures[0][0]
    assert "已推断修复 2 处单位乱码" in host.broadcasts[-1][0]
    assert host.broadcasts[-1][1] == "warn"


def test_auto_save_write_failure_is_broadcast_once(post_load_env):
    flm, host = post_load_env
    config = PlotSessionConfig(plots=[PlotConfig(curves=["a"])])
    auto_save = SimpleNamespace(
        is_auto_save_enabled=lambda: True,
        auto_save=lambda _config: False,
    )
    manager = SimpleNamespace(
        _auto_save_manager=auto_save,
        export_current_config=lambda _mw: config,
    )

    PlotConfigManager.save_auto_save(manager, host)

    assert host.broadcasts == [
        ("自动恢复配置保存失败，当前布局未能写入", "error")
    ]


# ---------------------------------------------------------------------------
# R10：零行成功加载的按钮与空态契约
# ---------------------------------------------------------------------------


def test_zero_row_load_disables_data_actions_but_keeps_asset_cleanup(
    post_load_env, monkeypatch
):
    flm, host = post_load_env
    host.loader.datalength = 0
    monkeypatch.setattr(
        host.auto_save,
        "prepare_auto_save",
        lambda *args, **kwargs: SimpleNamespace(
            should=False, reason="empty", config=None,
            identity_snapshot=None, failed=False,
        ),
    )

    flm._post_load_actions("demo.csv")

    assert flm._has_valid_data is False
    for name in (
        "auto_range_btn",
        "auto_y_btn",
        "cursor_btn",
        "mark_region_btn",
        "grid_layout_btn",
    ):
        assert getattr(host, name).isEnabled() is False, name
    assert host.clear_all_plots_btn.isEnabled() is True
    assert host.curve_shelf_btn.isEnabled() is True
    empty_feedback = [
        (text, level)
        for text, level in host.broadcasts
        if not text.startswith("已推断修复")
        and ("空" in text or "0 行" in text or "没有有效数据" in text)
    ]
    assert len(empty_feedback) == 1, host.broadcasts
    assert empty_feedback[0][1] in ("info", "warn", "error")
    assert len(host.broadcasts) == 1, "空态与乱码披露必须合并为最终一条播报"
    assert "0 行" in host.broadcasts[0][0]
    assert "已有曲线与占位保持不变" in host.broadcasts[0][0]
    assert "已推断修复 2 处单位乱码" in host.broadcasts[0][0]
    assert host.broadcasts[0][1] == "warn"


def test_auto_restore_page_reconcile_failure_is_broadcast_once(
    post_load_env, monkeypatch
):
    flm, host = post_load_env
    manager = PlotConfigManager()
    manager._auto_save_manager = host.auto_save
    host.plot_config_manager = manager
    monkeypatch.setattr(
        manager, "_reconcile_pages",
        lambda _mw, _count: (1, "模板需要 2 页，最多支持 1 页"),
    )

    flm._post_load_actions("demo.csv")

    failures = [item for item in host.broadcasts if item[1] in ("warn", "error")]
    assert len(failures) == 1, host.broadcasts
    assert "自动恢复失败" in failures[0][0]
    assert host.applied == []


# ---------------------------------------------------------------------------
# 非事务风险：只记录 apply_config 当前真实部分应用边界，不要求回滚
# ---------------------------------------------------------------------------


def test_apply_config_exception_after_clear_records_partial_application(
    app_settings, plot_factory
):
    manager = PlotConfigManager()
    widget = plot_factory()
    assert widget.plot_variable("c")
    widget.demote_to_invalid("old_ghost", "gray", True)
    container = SimpleNamespace(plot_widget=widget)
    page = SimpleNamespace(
        title="旧页",
        row_height_factors={},
        rows_current=1,
        cols_current=1,
        grid_cols=1,
        plot_widgets=[container],
        _range_dirty=False,
        apply_visibility=lambda rows, cols: None,
        visible_containers=lambda: [container],
        cell_visible=lambda index: True,
    )
    host = SimpleNamespace(
        loader=SimpleNamespace(var_names=["a", "b", "c"]),
        pages=[page],
        current_page=page,
        factor=1.0,
        offset=0.0,
    )
    config = PlotSessionConfig(
        pages=[
            PlotPageConfig(
                layout_rows=1,
                layout_cols=1,
                plots=[PlotConfig(curves=["a"]), PlotConfig(curves=["b"])],
            )
        ]
    )
    real_add = widget.add_variable_to_plot

    def fail_second(name, *args, **kwargs):
        if name == "b":
            raise RuntimeError("second plot failed after clear")
        return real_add(name, *args, **kwargs)

    widget.add_variable_to_plot = fail_second

    assert manager.apply_config(host, config) is False

    # 当前真实边界：旧曲线/旧占位已清，第一条新曲线已落地，第二条未建。
    assert list(widget.curves) == ["a"]
    assert not widget.invalid_curves
    assert "c" not in widget.curves
    assert host.factor == 1.0 and host.offset == 0.0
    assert page._range_dirty is True
    assert widget._batch_adding is False
