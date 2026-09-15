"""变量操作的共享实现。

从 MyTableWidget 抽取，使"变量列表右键菜单"与"变量信息窗口的联动按钮"
共用同一套逻辑，避免两处行为分叉。

本次抽取为纯等价改写：函数体逐字搬迁，仅做两处参数化——
原先隐式的 self.window() 改为显式 main_window 参数，
原先作为 QMessageBox 父窗口的 self 改为 msg_parent 参数。
"""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox


def normalize_var_list(var_names) -> list[str]:
    """将输入标准化为不重复的变量名列表"""
    if isinstance(var_names, str):
        candidates = [var_names]
    else:
        candidates = list(var_names) if var_names is not None else []

    normalized = []
    seen = set()
    for name in candidates:
        clean_name = (name or "").strip()
        if not clean_name or clean_name in seen:
            continue
        normalized.append(clean_name)
        seen.add(clean_name)
    return normalized


def add_variables_to_data_table(var_names, main_window) -> None:
    """添加至变量数值表（非模态单例窗口，内部会跳过已存在的列）"""
    var_list = normalize_var_list(var_names)
    if not var_list:
        return

    from src.ui.table_dialog import DataTableDialog

    DataTableDialog.add_variables(var_list, parent=main_window)


def add_variables_to_blank_plot(var_names, main_window, msg_parent=None) -> bool:
    """在当前 (m×n) 布局内找到首个空白绘图区并添加变量。

    返回是否成功提交（找到空白区并已排程添加）。失败时通过 msg_parent
    弹出提示；msg_parent 为 None 时静默返回 False，便于调用方自行处理。
    """
    var_list = normalize_var_list(var_names)
    if not var_list:
        return False

    # 获取 MainWindow 实例
    if not (main_window and hasattr(main_window, "loader")):
        _warn(msg_parent, "错误", "未找到主窗口实例")
        return False

    # 2. 在用户设置的当前布局(mxn)中查找空白绘图区，无论绘图区整体是否可见
    blank_plot = None
    rows, cols = main_window._plot_row_current, main_window._plot_col_current
    max_cols = main_window._plot_col_max_default  # 这是完整网格的列数，用于计算索引

    for idx, container in enumerate(main_window.plot_widgets):
        # 根据一维索引计算其在完整网格(pxq)中的二维坐标(r, c)
        r = idx // max_cols
        c = idx % max_cols

        # 判断这个坐标是否在用户当前的(mxn)布局内
        if r < rows and c < cols:
            # 如果在布局内，再判断是否为空白（统一版：仅检查 curves 字典）
            pw = container.plot_widget
            is_blank = not getattr(pw, "curves", None)
            if is_blank:
                blank_plot = pw
                break  # 找到第一个可用的就退出

    if blank_plot is None:
        _warn(msg_parent, "提示", "当前布局中已无空白绘图区")
        return False

    _delay = 0
    if not main_window._plot_area_visible:
        main_window.toggle_plot_btn.setChecked(False)
        _delay = 300

    def _job():
        # 4. 将变量添加至空白图中
        blank_plot.add_variables_to_plot(var_list)
        main_window.layout_manager.request_mark_stats_refresh()

    QTimer.singleShot(_delay, _job)
    return True


def _warn(parent, title: str, text: str) -> None:
    """parent 为 None 时静默跳过，避免调用方被迫提供消息父窗口。"""
    if parent is None:
        return
    QMessageBox.warning(parent, title, text)
