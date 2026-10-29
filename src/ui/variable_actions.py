"""变量操作的共享实现。

从 MyTableWidget 抽取，使"变量列表右键菜单"与"变量信息窗口的联动按钮"
共用同一套逻辑，避免两处行为分叉。

本次抽取为纯等价改写：函数体逐字搬迁，仅做两处参数化——
原先隐式的 self.window() 改为显式 main_window 参数，
原先作为 QMessageBox 父窗口的 self 改为 msg_parent 参数。
"""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from src.core.config import widget_alive


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

    # 2. 只在当前页、用户设置的可见布局(m×n)内查找空白绘图区（§6.6 一期）
    #    跨页全局搜索会把变量塞进用户看不见的页，语义等同“丢失”。
    page = getattr(main_window, "current_page", None)
    if page is not None:
        containers = list(page.plot_widgets)
        in_layout = (
            (lambda _idx: True) if not hasattr(page, "cell_visible")
            else (lambda idx: page.cell_visible(idx))
        )
    else:
        # 旧式替身（无 pages）保持原扫描口径
        containers = list(main_window.plot_widgets)
        rows, cols = main_window._plot_row_current, main_window._plot_col_current
        max_cols = getattr(main_window, "grid_cols", 0) or main_window._plot_col_max_default

        def in_layout(idx: int) -> bool:
            return idx // max_cols < rows and idx % max_cols < cols

    blank_plot = None
    for idx, container in enumerate(containers):
        if not in_layout(idx):
            continue
        pw = container.plot_widget
        if not getattr(pw, "curves", None):
            blank_plot = pw
            break

    if blank_plot is None:
        # 一期：不擅自扩布局 / 不跨页塞 / 不隐式建页
        _warn(
            msg_parent,
            "提示",
            "当前页已无空白绘图区，请先修改布局或新建页后再添加。",
        )
        return False

    _delay = 0
    if not main_window._plot_area_visible:
        main_window.toggle_plot_btn.setChecked(False)
        _delay = 300

    # 延迟回调带令牌：期间用户切页/关页都必须丢弃（§6.6 #2）
    target_page = page
    epoch = getattr(page, "epoch", None) if page is not None else None

    def _job():
        if target_page is not None:
            if epoch is not None and target_page.epoch != epoch:
                return
            if not widget_alive(target_page.widget):
                return
        if not widget_alive(blank_plot):
            return
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
