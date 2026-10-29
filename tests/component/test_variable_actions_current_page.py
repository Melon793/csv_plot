"""空白区添加变量的页作用域（§6.6 一期）。

- 只在**当前页**、用户设置的可见行列内找空白格；隐藏页/布局外格子不参与，
  否则变量会被塞进用户看不见的地方，语义等同“丢失”。
- 延迟回调带页 epoch 令牌：期间切页/关页必须丢弃这次添加。
"""

from __future__ import annotations

import pytest

from PySide6.QtWidgets import QWidget

from src.ui import variable_actions


class _FakePlot:
    def __init__(self, curves: bool = False):
        self.curves = {"old": object()} if curves else {}
        self.added: list[list[str]] = []

    def add_variables_to_plot(self, names):
        self.added.append(list(names))


class _FakeContainer:
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget


class _FakePage:
    """只提供 add_variables_to_blank_plot 访问的页级接口。"""

    def __init__(self, plots, *, rows: int, cols: int, grid_cols: int):
        self.widget = QWidget()
        self.plot_widgets = [_FakeContainer(p) for p in plots]
        self.rows_current = rows
        self.cols_current = cols
        self.grid_cols = grid_cols
        self.epoch = 0

    def cell_visible(self, index: int) -> bool:
        row, col = divmod(index, self.grid_cols)
        return row < self.rows_current and col < self.cols_current


class _FakeLayoutManager:
    def __init__(self):
        self.refresh_calls = 0

    def request_mark_stats_refresh(self, immediate: bool = False):
        self.refresh_calls += 1


class _FakeMainWindow:
    def __init__(self, pages, current):
        self.pages = list(pages)
        self.current_page = current
        self.loader = object()
        self._plot_area_visible = True
        self.layout_manager = _FakeLayoutManager()


@pytest.fixture()
def pages_cleanup(qapp):
    created: list[_FakePage] = []
    yield created
    for page in created:
        page.widget.deleteLater()


def _page(pages_cleanup, plots, **kwargs) -> _FakePage:
    page = _FakePage(plots, **kwargs)
    pages_cleanup.append(page)
    return page


def test_blank_search_uses_current_page_visible_cells_only(
    qapp, qtbot, pages_cleanup
):
    occupied = _FakePlot(curves=True)
    blank = _FakePlot()
    page = _page(pages_cleanup, [occupied, blank], rows=1, cols=2, grid_cols=2)
    hidden_page = _page(
        pages_cleanup, [_FakePlot()], rows=1, cols=1, grid_cols=1
    )
    mw = _FakeMainWindow([hidden_page, page], current=page)

    assert variable_actions.add_variables_to_blank_plot(["speed"], mw) is True
    qtbot.wait(20)

    assert blank.added == [["speed"]]
    assert occupied.added == []
    assert hidden_page.plot_widgets[0].plot_widget.added == [], "不得塞进隐藏页"
    assert mw.layout_manager.refresh_calls == 1


def test_no_blank_within_visible_layout_does_not_touch_outside_cell(
    qapp, qtbot, pages_cleanup, monkeypatch
):
    occupied = _FakePlot(curves=True)
    outside_layout_blank = _FakePlot()
    # 建了 1×2 网格，但用户只让第一列可见：第二列虽有空白格也不算
    page = _page(
        pages_cleanup, [occupied, outside_layout_blank], rows=1, cols=1, grid_cols=2
    )
    mw = _FakeMainWindow([page], current=page)

    warnings: list[tuple[str, str]] = []

    def fake_warning(parent, title, text, *args, **kwargs):
        warnings.append((title, text))

    monkeypatch.setattr(variable_actions.QMessageBox, "warning", staticmethod(fake_warning))

    assert variable_actions.add_variables_to_blank_plot(
        "speed", mw, msg_parent=mw
    ) is False
    qtbot.wait(20)

    assert outside_layout_blank.added == []
    assert len(warnings) == 1
    assert "当前页已无空白绘图区" in warnings[0][1]
    assert mw.layout_manager.refresh_calls == 0


def test_pending_add_is_dropped_when_page_epoch_changes(qapp, qtbot, pages_cleanup):
    blank = _FakePlot()
    page = _page(pages_cleanup, [blank], rows=1, cols=1, grid_cols=1)
    mw = _FakeMainWindow([page], current=page)

    assert variable_actions.add_variables_to_blank_plot(["rpm"], mw) is True
    # 模拟切页/关页：epoch 自增后，已排队的回调必须作废
    page.epoch += 1
    qtbot.wait(20)

    assert blank.added == []
    assert mw.layout_manager.refresh_calls == 0
