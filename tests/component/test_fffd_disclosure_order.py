"""第 6 轮 P1-B：FFFD 披露必须在自动恢复播报之后（同回调覆盖问题）。

259e2ba 的 D1 披露原放在 _post_load_actions 前段，auto-save 命中时
同一回调内「自动恢复…」播报紧随其后，两次 setText 之间无事件循环迭代，
披露从未上屏 → 推断性数据改写静默化。修复：披露移到回调末尾。
"""
from __future__ import annotations

import types

import pytest


class _MW:
    """可弱引用的主窗口替身：只记录播报顺序。"""

    def __init__(self):
        self.broadcasts = []
        self.loaded_path = None
        self.var_names = ["a"]
        self.loader = types.SimpleNamespace(
            fffd_unit_fixes=3, var_names=["a"], datalength=3,
        )
        self.data = None
        self.plot_config_manager = types.SimpleNamespace(
            auto_save_manager=types.SimpleNamespace(
                prepare_auto_save=lambda vars, loader=None: types.SimpleNamespace(
                    should=True,
                    reason="matched",
                    config={"pages": []},
                    identity_snapshot=None,
                    failed=False,
                ),
            ),
            apply_config=lambda mw, cfg, **kwargs: True,
        )

    def _broadcast(self, text, level="info"):
        self.broadcasts.append((text, level))

    def end_load_feedback(self, *a, **k):
        pass

    def update_file_status(self, *a, **k):
        pass


@pytest.fixture()
def flm_env(monkeypatch):
    from src.ui.file_loader_manager import FileLoaderManager

    mw = _MW()
    flm = FileLoaderManager(mw)
    flm._last_replot_result = "empty"
    monkeypatch.setattr(flm, "_window_going_away", lambda: False)
    monkeypatch.setattr(flm, "_remember_last_open_dir", lambda p: None)
    monkeypatch.setattr(flm, "set_button_status", lambda *a, **k: None)
    monkeypatch.setattr(flm, "_iter_all_containers", lambda: [])
    monkeypatch.setattr(
        flm, "_announce_auto_restore",
        lambda cfg, vars, **kwargs: mw._broadcast("自动恢复配置 · 匹配 100%"),
    )
    return flm, mw


def test_fffd_disclosure_survives_auto_restore_broadcast(flm_env):
    flm, mw = flm_env
    flm._post_load_actions("/tmp/demo.csv")
    texts = [t for t, _ in mw.broadcasts]
    assert any("自动恢复" in t for t in texts), "自动恢复播报必须发生"
    assert texts[-1].startswith("已推断修复 3 处单位乱码"), (
        f"披露必须是最后一条播报（否则被顶掉从未上屏），实际={texts}"
    )
    assert mw.broadcasts[-1][1] == "warn", "披露 level=warn 保 8s 回收"


def test_fffd_disclosure_still_fires_without_auto_restore(flm_env, monkeypatch):
    flm, mw = flm_env
    monkeypatch.setattr(
        mw.plot_config_manager.auto_save_manager,
        "prepare_auto_save",
        lambda vars, loader=None: types.SimpleNamespace(
            should=False, reason="no config", config=None,
            identity_snapshot=None, failed=False,
        ),
    )
    flm._post_load_actions("/tmp/demo.csv")
    texts = [t for t, _ in mw.broadcasts]
    assert texts and texts[-1].startswith("已推断修复"), f"实际={texts}"


def test_no_fffd_no_disclosure(flm_env):
    flm, mw = flm_env
    mw.loader.fffd_unit_fixes = 0
    flm._post_load_actions("/tmp/demo.csv")
    assert not any("已推断修复" in t for t, _ in mw.broadcasts)


def test_disclosure_fires_when_assets_already_restored(flm_env):
    """replots 已恢复资产 → 无自动恢复播报，披露仍必须发出。"""
    flm, mw = flm_env
    flm._last_replot_result = "restored"
    flm._post_load_actions("/tmp/demo.csv")
    texts = [t for t, _ in mw.broadcasts]
    assert not any("自动恢复" in t for t in texts)
    assert texts and texts[-1].startswith("已推断修复"), f"实际={texts}"


def test_disclosure_survives_auto_restore_exception(flm_env, monkeypatch):
    """评审 N4：自动恢复阶段抛异常（容器销毁竞态等）不得跳过披露。"""
    flm, mw = flm_env

    def boom():
        raise RuntimeError("auto-save manager deleted")

    monkeypatch.setattr(
        mw.plot_config_manager.auto_save_manager, "prepare_auto_save", boom
    )
    flm._post_load_actions("/tmp/demo.csv")
    texts = [t for t, _ in mw.broadcasts]
    assert texts and texts[-1].startswith("已推断修复"), f"实际={texts}"


@pytest.mark.parametrize("stage", ["load", "apply_false", "apply_raise"])
def test_restore_failure_warns_once_and_fffd_stays_last(flm_env, monkeypatch, stage):
    """R6：读档/套用失败仅日志不够；一次 warn/error 后仍须披露 FFFD。"""
    flm, mw = flm_env
    manager = mw.plot_config_manager

    if stage == "load":
        def fail_load(*args, **kwargs):
            raise OSError("synthetic auto-save read failure")

        monkeypatch.setattr(
            manager.auto_save_manager, "prepare_auto_save", fail_load
        )
    elif stage == "apply_false":
        monkeypatch.setattr(manager, "apply_config", lambda mw, cfg, **kwargs: False)
    else:
        def fail_apply(mw, cfg, **kwargs):
            raise RuntimeError("synthetic auto-save apply failure")

        monkeypatch.setattr(manager, "apply_config", fail_apply)

    flm._post_load_actions("demo.csv")

    failures = [
        (text, level)
        for text, level in mw.broadcasts
        if level in ("warn", "error") and not text.startswith("已推断修复")
    ]
    assert len(failures) == 1, f"失败必须只播报一次，实际={mw.broadcasts}"
    assert "自动恢复" in failures[0][0], failures[0]
    assert mw.broadcasts[-1][0].startswith("已推断修复 3 处单位乱码")
    assert mw.broadcasts[-1][1] == "warn"