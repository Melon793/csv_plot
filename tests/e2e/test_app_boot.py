"""e2e 启动冒烟测试：主窗口构建 → 初始状态 → 正常退出。

对应 help.md「基本操作流程」的启动前提：窗口就绪、按钮初始
状态正确（未加载数据时分析类按钮禁用）。
"""


def test_main_window_boots(main_window):
    """MainWindow 构建成功：标题/关键控件就绪"""
    mw = main_window
    assert mw.windowTitle() == mw.defaultTitle
    assert mw.load_btn is not None
    assert mw.reload_btn is not None
    assert mw.list_widget is not None
    assert mw.file_loader_manager is not None
    assert mw.layout_manager is not None
    assert mw.cursor_sync_manager is not None


def test_initial_state_before_load(main_window):
    """加载前：无数据、无子图、变量列表为空、分析按钮禁用"""
    mw = main_window
    assert mw.loader is None
    assert mw.plot_widgets == []
    assert mw.list_widget.rowCount() == 0
    # set_button_status(False)：时间修正/清除/缩放/光标/标记/布局均禁用
    for btn in (
        mw.time_correction_btn,
        mw.clear_all_plots_btn,
        mw.auto_range_btn,
        mw.auto_y_btn,
        mw.cursor_btn,
        mw.mark_region_btn,
        mw.grid_layout_btn,
    ):
        assert not btn.isEnabled(), f"{btn.text()} 应在加载前禁用"


def test_shortcuts_noop_before_load(main_window, qtbot):
    """加载前触发快捷键：全部静默无操作（守卫 plot_widgets 为空）"""
    from PySide6.QtCore import Qt

    mw = main_window
    mw.activateWindow()
    for key in (Qt.Key_R, Qt.Key_Y, Qt.Key_T, Qt.Key_L):
        qtbot.keyClick(mw, key, Qt.KeyboardModifier.ControlModifier)
    # 无异常即通过；光标/标记按钮状态不受影响
    assert not mw.cursor_btn.isChecked()
    assert not mw.mark_region_btn.isChecked()


def test_reload_without_data_is_safe(main_window, dialog_stubs):
    """无数据时点「重载」：弹错误提示、不崩溃"""
    mw = main_window
    mw.file_loader_manager.reload_data()

    assert len(dialog_stubs["critical"]) == 1
    assert "没有可重新加载的数据" in dialog_stubs["critical"][0][1]
