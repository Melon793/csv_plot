"""P1-1：主窗口退出与矩阵重建都必须把销毁期标志传到 plot 上

`_is_being_destroyed` 此前只在 `__init__` 置 False，全项目无处置真，8 处
销毁期守卫空转。本用例走真实 MainWindow，验证两条拆卸路径都补上了置位。
"""

from PySide6.QtGui import QCloseEvent


def test_close_event_marks_window_and_live_plots(main_window, qapp):
    main_window.layout_manager.create_subplots_matrix(1, 2)
    qapp.processEvents()
    plots = [c.plot_widget for c in main_window.plot_widgets]
    assert len(plots) == 2
    assert all(not p._is_being_destroyed for p in plots)

    # 直接调 closeEvent 而不是 close()：窗口留给夹具收尾，避免与 qtbot 的
    # 生命周期跟踪打架，走的是同一段退出代码。
    main_window.closeEvent(QCloseEvent())

    assert main_window._is_being_destroyed is True
    assert all(p._is_being_destroyed for p in plots)


def test_matrix_rebuild_marks_old_plots_before_deletion(main_window):
    """旧 plot 要在 deleteLater 之后、事件循环转回来之前就拿到标志

    断言前不跑事件循环：否则置位可能来自 C++ 析构的兜底连接，测不到显式点。
    """
    main_window.layout_manager.create_subplots_matrix(1, 1)
    old_plot = main_window.plot_widgets[0].plot_widget
    assert old_plot._is_being_destroyed is False

    main_window.layout_manager.create_subplots_matrix(2, 1)

    assert old_plot._is_being_destroyed is True
