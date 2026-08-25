"""变量搜索栏增量标记更新 component 测试（由 tmp/verify_incremental_marks.py 转正）。

覆盖设计文档 v1.3 第 6 节验收标准：
- 查询词变化全量重建 / 键盘与鼠标路径的增量标记差异
- curves_changed 增量幂等 / 过滤外变量无副作用
- reset_session 与 update_data_source（重载场景）
"""

import pytest

from PySide6.QtCore import QObject, Qt, Signal

from src.ui.widgets.variable_search_bar import VariableSearchBar


class FakePlotWidget(QObject):
    """最小 plot_widget 替身：curves 字典 + curves_changed 信号"""

    curves_changed = Signal()

    def __init__(self):
        super().__init__()
        self.curves: dict = {}
        self.units = {}


@pytest.fixture()
def bar(qapp):
    pw = FakePlotWidget()
    var_names = [f"engine_rpm_{i}" for i in range(50)] + ["temp_coolant", "temp_oil"]
    units = {n: "rpm" for n in var_names}
    validity = {n: 1 for n in var_names}
    search_bar = VariableSearchBar(var_names, units, validity, pw)
    yield search_bar, pw
    search_bar.deleteLater()


def _item_names(search_bar):
    return [
        search_bar.candidate_list.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(search_bar.candidate_list.count())
    ]


def test_full_refresh_on_query(bar):
    """查询词变化 → 全量重建 + 选中第一个未添加项"""
    search_bar, _ = bar
    search_bar.search_edit.setText("rpm")
    search_bar._refresh_list()  # 直接调，跳过防抖
    assert search_bar.candidate_list.count() == 50
    assert search_bar.candidate_list.currentRow() == 0


def test_incremental_mark_added_keyboard(bar):
    """键盘 Enter 添加 → 增量标记 + 选中后移，列表不重建"""
    search_bar, pw = bar
    search_bar.search_edit.setText("rpm")
    search_bar._refresh_list()
    items_before = [search_bar.candidate_list.item(i)
                    for i in range(search_bar.candidate_list.count())]
    var = _item_names(search_bar)[0]

    pw.curves[var] = object()
    search_bar._last_input_was_mouse = False
    search_bar.mark_added(var)

    items_after = [search_bar.candidate_list.item(i)
                   for i in range(search_bar.candidate_list.count())]
    assert items_before == items_after  # item 对象未重建（同一批实例）
    assert search_bar.candidate_list.count() == 50
    text0 = search_bar.candidate_list.item(0).text()
    assert text0.endswith("(已添加)")
    assert "(rpm)" in text0  # 单位保留在文本中
    assert bool(search_bar.candidate_list.item(0).data(Qt.ItemDataRole.UserRole + 2))
    # 选中移动到下一个未添加项（索引 1）
    assert search_bar.candidate_list.currentRow() == 1


def test_incremental_mark_removed_keyboard(bar):
    """键盘 Enter 移除 → 增量标记，选中停留原项"""
    search_bar, pw = bar
    search_bar.search_edit.setText("rpm")
    search_bar._refresh_list()
    var = _item_names(search_bar)[3]
    pw.curves[var] = object()
    search_bar._last_input_was_mouse = False
    search_bar.mark_added(var)  # 选中会跳到索引 4
    search_bar.candidate_list.setCurrentRow(3)  # 模拟用户回到该项

    del pw.curves[var]
    search_bar._last_input_was_mouse = False
    search_bar.mark_removed(var)

    assert not search_bar.candidate_list.item(3).text().endswith("(已添加)")
    assert search_bar.candidate_list.currentRow() == 3  # 选中停留原项


def test_mouse_path_no_selection_change(bar):
    """鼠标添加 → 滚动条与选中完全不动"""
    search_bar, pw = bar
    search_bar.search_edit.setText("rpm")
    search_bar._refresh_list()
    search_bar.candidate_list.setCurrentRow(10)
    search_bar.candidate_list.scrollToBottom()
    scroll_before = search_bar.candidate_list.verticalScrollBar().value()
    var = _item_names(search_bar)[12]

    pw.curves[var] = object()
    search_bar._last_input_was_mouse = True
    search_bar.mark_added(var)

    assert search_bar.candidate_list.verticalScrollBar().value() == scroll_before
    assert search_bar.candidate_list.currentRow() == 10
    assert bool(search_bar.candidate_list.item(12).data(Qt.ItemDataRole.UserRole + 2))


def test_idempotent_and_curves_changed(bar):
    """curves_changed 增量路径 + 幂等"""
    search_bar, pw = bar
    search_bar.search_edit.setText("rpm")
    search_bar._refresh_list()
    pw.curves["engine_rpm_5"] = object()
    search_bar._on_curves_changed()
    # curves_changed 走增量：标记已更新
    assert bool(search_bar.candidate_list.item(5).data(Qt.ItemDataRole.UserRole + 2))
    snapshot = [search_bar.candidate_list.item(i).text()
                for i in range(search_bar.candidate_list.count())]
    search_bar._on_curves_changed()  # 第二次：应全部 skip
    snapshot2 = [search_bar.candidate_list.item(i).text()
                 for i in range(search_bar.candidate_list.count())]
    assert snapshot == snapshot2  # 幂等：第二次调用无变化


def test_external_var_outside_filter(bar):
    """过滤结果之外的变量被增删 → 列表不受影响"""
    search_bar, pw = bar
    search_bar.search_edit.setText("rpm")
    search_bar._refresh_list()
    pw.curves["temp_coolant"] = object()  # 不在当前过滤结果中
    search_bar._on_curves_changed()
    assert search_bar.candidate_list.count() == 50
    assert all(
        search_bar.candidate_list.item(i).data(Qt.ItemDataRole.UserRole + 2) is False
        for i in range(search_bar.candidate_list.count())
    )


def test_reset_session(bar):
    """reset_session（表格删除/清空）→ 仅样式更新，不跳顶"""
    search_bar, pw = bar
    search_bar.search_edit.setText("rpm")
    search_bar._refresh_list()
    pw.curves["engine_rpm_1"] = object()
    pw.curves["engine_rpm_2"] = object()
    search_bar._on_curves_changed()
    search_bar.candidate_list.setCurrentRow(20)
    search_bar.candidate_list.scrollToBottom()
    scroll_before = search_bar.candidate_list.verticalScrollBar().value()

    pw.curves.clear()
    search_bar.reset_session()

    assert all(
        not search_bar.candidate_list.item(i).data(Qt.ItemDataRole.UserRole + 2)
        for i in range(search_bar.candidate_list.count())
    )
    assert search_bar.candidate_list.currentRow() == 20  # 选中停留原位
    assert search_bar.candidate_list.verticalScrollBar().value() == scroll_before


def test_update_data_source(bar):
    """update_data_source（重载场景）→ 快照替换 + 保留搜索词全量重建"""
    search_bar, pw = bar
    search_bar.search_edit.setText("rpm")
    search_bar._refresh_list()
    assert search_bar.candidate_list.count() == 50

    new_names = [f"new_rpm_{i}" for i in range(10)] + ["new_temp"]
    pw.units = {n: "r/min" for n in new_names}
    search_bar.update_data_source(new_names, pw.units, {n: 1 for n in new_names})

    assert search_bar.search_edit.text() == "rpm"  # 搜索词保留
    assert search_bar.candidate_list.count() == 10
    assert "engine_rpm_0" not in _item_names(search_bar)  # 陈旧变量已消失
    assert "(r/min)" in search_bar.candidate_list.item(0).text()  # 新单位生效
    assert search_bar.candidate_list.currentRow() == 0


def test_no_match_and_empty_query(bar):
    """无匹配/空查询 → curves_changed 回退全量刷新无副作用"""
    search_bar, pw = bar
    search_bar.search_edit.setText("zzz_no_match")
    search_bar._refresh_list()
    assert search_bar.candidate_list.count() == 0
    pw.curves["engine_rpm_0"] = object()
    search_bar._on_curves_changed()  # 走 else 全量分支
    assert search_bar.candidate_list.count() == 0
    assert search_bar.candidate_list.currentRow() == -1

    search_bar.search_edit.clear()
    search_bar._refresh_list()
    assert search_bar.candidate_list.count() == 0
    search_bar._on_curves_changed()
    assert search_bar.candidate_list.count() == 0


def test_consecutive_keyboard_adds(bar):
    """连续 Enter 快速添加 5+ 个 → 选中逐项后移，末尾不回绕"""
    search_bar, pw = bar
    search_bar.search_edit.setText("rpm")
    search_bar._refresh_list()
    search_bar.candidate_list.setCurrentRow(46)  # 接近末尾
    for expected_row in (46, 47, 48):
        var = search_bar.candidate_list.currentItem().data(Qt.ItemDataRole.UserRole)
        pw.curves[var] = object()
        search_bar._last_input_was_mouse = False
        search_bar.mark_added(var)
        assert search_bar.candidate_list.currentRow() == expected_row + 1

    # 添加最后一个后无未添加项 → 清除选中（不回绕）
    var = search_bar.candidate_list.currentItem().data(Qt.ItemDataRole.UserRole)
    pw.curves[var] = object()
    search_bar._last_input_was_mouse = False
    search_bar.mark_added(var)
    assert search_bar.candidate_list.currentRow() == -1
