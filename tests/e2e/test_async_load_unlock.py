"""加载链路互斥锁收尾的防回归（V6.0 P0-3 / P0-4）。

异步加载完成回调 `_on_load_done` 里，`_swap_loader → _apply_loader →
replots_after_loading` 链上任一异常都不得跳过解锁 —— 否则
`_is_loading_new_data` / `_reload_in_progress` 永久为真、load/reload
按钮全灭，且 `_end_data_reload` 未执行意味着 safety timer 从未启动，
应用只能重启。同步路径早有 `finally: _end_data_reload()`，异步路径必须对齐。

P0-4 同属该死锁族：`os.path.getsize` 若在 `_begin_data_reload()` 之后抛出，
锁已置位却永不释放。
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


def test_getsize_failure_before_lock_does_not_deadlock(
    main_window, monkeypatch, tmp_path, dialog_stubs
):
    """V6.0 P0-4：上锁前取文件尺寸失败（文件被移动/删除）必须报错返回，
    不得留下 _is_loading_new_data=True 使后续所有加载被静默拒绝。"""
    import os

    from tests.fixtures.data_factory import make_simple_rows, write_csv

    csv = write_csv(
        tmp_path / "p0_4.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(20),
    )

    def fake_getsize(path):
        raise OSError("文件已被删除")

    monkeypatch.setattr(os.path, "getsize", fake_getsize)

    main_window.file_loader_manager._load_file(str(csv))

    assert main_window._is_loading_new_data is False
    assert [t for t, _ in dialog_stubs["critical"]] == ["文件不可读"]
    assert main_window.loader is None

