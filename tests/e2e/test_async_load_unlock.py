"""异步加载收尾解锁的防回归（V6.0 P0-3）。

后台加载完成回调 `_on_load_done` 里，`_swap_loader → _apply_loader →
replots_after_loading` 链上任一异常都不得跳过解锁 —— 否则
`_is_loading_new_data` / `_reload_in_progress` 永久为真、load/reload
按钮全灭，且 `_end_data_reload` 未执行意味着 safety timer 从未启动，
应用只能重启。同步路径早有 `finally: _end_data_reload()`，异步路径必须对齐。
"""

import pytest

from PySide6.QtWidgets import QProgressDialog


@pytest.fixture()
def flm(main_window, monkeypatch):
    """处于「加载中」状态的 FileLoaderManager（含 progress 替身）"""
    manager = main_window.file_loader_manager
    monkeypatch.setattr(
        main_window, "_progress", QProgressDialog(), raising=False
    )
    manager._begin_data_reload()
    manager._reload_in_progress = True
    main_window.load_btn.setEnabled(False)
    main_window.reload_btn.setEnabled(False)
    assert main_window._is_loading_new_data is True
    return manager


def test_on_load_done_unlocks_when_swap_raises(main_window, flm, monkeypatch):
    """_swap_loader 抛异常：仍必须解锁并恢复按钮"""

    def boom(new_loader, is_reload=False):
        raise RuntimeError("应用数据失败")

    monkeypatch.setattr(flm, "_swap_loader", boom)
    post_actions = []
    monkeypatch.setattr(
        flm, "_post_load_actions", lambda *a, **k: post_actions.append(1)
    )

    flm._on_load_done(object(), "/tmp/does_not_matter.csv")

    assert main_window.load_btn.isEnabled() is True
    assert main_window.reload_btn.isEnabled() is True
    assert flm._reload_in_progress is False
    # 失败路径不得继续走加载后动作
    assert post_actions == []


def test_on_load_done_unlocks_on_success(main_window, flm, monkeypatch):
    """正常路径：解锁 + 恢复按钮 + 调度加载后动作（回归基线）"""
    monkeypatch.setattr(flm, "_swap_loader", lambda *a, **k: None)
    post_actions = []
    monkeypatch.setattr(
        flm, "_post_load_actions", lambda *a, **k: post_actions.append(1)
    )

    flm._on_load_done(object(), "/tmp/does_not_matter.csv")

    assert main_window.load_btn.isEnabled() is True
    assert flm._reload_in_progress is False
    assert post_actions == []  # 仅经 singleShot 调度，未在同步路径内执行
