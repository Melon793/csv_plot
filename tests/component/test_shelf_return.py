"""暂存区「退回原图」：分组 / 场景口径 / 只删成功 / 播报（offscreen）。"""

import weakref

import pytest

from PySide6.QtWidgets import QWidget

from src.ui.transfer_broadcast import build_return_message
from src.ui.widgets.curve_shelf import CurveShelfWidget, ShelfEntry
from src.ui.widgets.plot_widget import TransferResult
from tests.fixtures.waits import force_delete


class FakeContainer:
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget


class FakePage:
    def __init__(self, index: int, containers, *, cell_visible: bool = True):
        self.index = index
        self.plot_widgets = list(containers)
        self._cell_visible = bool(cell_visible)
        self.dirty = False

    def mark_dirty_for_activation(self):
        self.dirty = True

    def cell_visible(self, index: int) -> bool:
        return 0 <= index < len(self.plot_widgets) and self._cell_visible


class ReturnHost(QWidget):
    """最小宿主：页索引 + 当前页 + 播报记录 + 加载闸门。"""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []
        self.loading = False
        self.pages: list[FakePage] = []
        self.current_page = None

    def _show_status_message(self, message: str) -> None:
        self.messages.append(message)

    def is_data_loading(self) -> bool:
        return self.loading

    def reject_when_loading(self, action: str) -> bool:
        if not self.loading:
            return False
        self.messages.append(f"正在加载数据，请稍候再{action}")
        return True

    def page_index(self, page) -> int:
        try:
            return self.pages.index(page)
        except ValueError:
            return -1


class FakePlot:
    """记录 land 调用并按需返回 added/skipped/failed。"""

    def __init__(self, page=None):
        self.page = page
        self.curves: dict[str, object] = {}
        self.calls: list[tuple[list[ShelfEntry], dict]] = []
        self._is_being_destroyed = False
        self.raise_error = False
        self.skip_names: set[str] = set()

    def land_shelf_entries(self, entries, **kwargs):
        self.calls.append((list(entries), kwargs))
        if self.raise_error:
            raise RuntimeError("boom")
        result = TransferResult(requested=[entry.var_name for entry in entries])
        for entry in entries:
            if entry.var_name in self.curves or entry.var_name in self.skip_names:
                result.skipped.append(entry.var_name)
                continue
            self.curves[entry.var_name] = object()
            result.added.append(entry.var_name)
        callback = kwargs.get("remove_entries_cb")
        if callback is not None and result.added:
            callback(list(result.added))
        return result


def _entry(name: str, plot: FakePlot, key: str | None = None) -> ShelfEntry:
    return ShelfEntry(
        key=key or f"k-{name}",
        var_name=name,
        origin_label="页1·图1",
        origin_plot_ref=weakref.ref(plot),
    )


@pytest.fixture()
def host(qapp):
    widget = ReturnHost()
    widget.show()
    yield widget
    widget.deleteLater()


@pytest.fixture()
def shelf(host):
    widget = CurveShelfWidget(main_window=host)
    widget.show()
    yield widget
    widget.deleteLater()


def _make_plot(host, *, page_index: int = 0, current: bool = True, **page_kw):
    page = FakePage(page_index, [], **page_kw)
    plot = FakePlot(page)
    page.plot_widgets = [FakeContainer(plot)]
    host.pages.append(page)
    if current:
        host.current_page = page
    return page, plot


def test_return_single_entry_lands_and_removes(shelf, host):
    page, plot = _make_plot(host)
    entry = _entry("a", plot)
    shelf.add_entry(entry)

    returned = shelf.return_entries([entry.key])

    assert returned == ["a"]
    assert "a" in plot.curves
    assert shelf.count() == 0
    assert len(plot.calls) == 1
    _, kwargs = plot.calls[0]
    assert kwargs["scene"] is True
    assert kwargs["batch"] is False
    assert host.messages == ["已退回 a：暂存区 → 页1·图1"]


def test_return_hidden_page_defers_scene_and_batches(shelf, host):
    page, plot = _make_plot(host, current=False)
    other = FakePage(1, [FakeContainer(FakePlot())])
    host.pages.append(other)
    host.current_page = other
    entry = _entry("a", plot)
    shelf.add_entry(entry)

    shelf.return_entries([entry.key])

    _, kwargs = plot.calls[0]
    assert kwargs["scene"] is False
    assert kwargs["batch"] is True  # 单条也走 finalize(scene=False)
    assert host.messages == ["已退回 a：暂存区 → 页1·图1"]


def test_return_groups_same_plot_once(shelf, host):
    page, plot = _make_plot(host)
    first = _entry("a", plot, "k1")
    second = _entry("b", plot, "k2")
    shelf.add_entry(first)
    shelf.add_entry(second)

    returned = shelf.return_entries([first.key, second.key])

    assert returned == ["a", "b"]
    assert len(plot.calls) == 1
    entries, kwargs = plot.calls[0]
    assert [entry.var_name for entry in entries] == ["a", "b"]
    assert kwargs["batch"] is True
    assert shelf.count() == 0
    assert host.messages == ["已退回 a、b：暂存区 → 页1·图1"]


def test_return_present_at_origin_skips_without_landing(shelf, host):
    page, plot = _make_plot(host)
    plot.curves["a"] = object()
    entry = _entry("a", plot)
    shelf.add_entry(entry)

    returned = shelf.return_entries([entry.key])

    assert returned == []
    assert plot.calls == []
    assert shelf.count() == 1
    assert host.messages == ["未退回；原图已有 a，条目已保留"]


def test_return_no_origin_and_closed_keep_entries(shelf, host):
    page, plot = _make_plot(host)
    no_origin = ShelfEntry(key="k1", var_name="a", origin_label="变量列表")

    dead = QWidget()
    dead_entry = _entry("b", FakePlot())
    dead_entry.origin_plot_ref = weakref.ref(dead)
    force_delete(dead)

    shelf.add_entry(no_origin)
    shelf.add_entry(dead_entry)

    returned = shelf.return_entries([no_origin.key, dead_entry.key])

    assert returned == []
    assert shelf.count() == 2
    assert host.messages == [
        "未退回；b 的原绘图已关闭，未退回（条目已保留）；"
        "a 无原绘图来源，仍留在暂存区"
    ]


def test_return_removes_only_added(shelf, host):
    page, plot = _make_plot(host)
    plot.skip_names.add("b")
    first = _entry("a", plot, "k1")
    second = _entry("b", plot, "k2")
    shelf.add_entry(first)
    shelf.add_entry(second)

    returned = shelf.return_entries([first.key, second.key])

    assert returned == ["a"]
    assert shelf.count() == 1
    assert shelf.selected_keys() == []
    assert host.messages == [
        "已退回 a：暂存区 → 页1·图1；原图已有 b，条目已保留"
    ]


def test_return_group_exception_isolated(shelf, host):
    page_a, plot_a = _make_plot(host, page_index=0)
    page_b, plot_b = _make_plot(host, page_index=1, current=False)
    host.current_page = page_a
    plot_a.raise_error = True
    first = _entry("a", plot_a, "k1")
    second = _entry("b", plot_b, "k2")
    shelf.add_entry(first)
    shelf.add_entry(second)

    returned = shelf.return_entries([first.key, second.key])

    assert returned == ["b"]
    assert shelf.count() == 1
    assert shelf.entry_for_name("a") is not None
    assert "未退回" in host.messages[0]
    assert "a 数据校验失败，未退回" in host.messages[0]
    assert "已退回 b：暂存区 → 页2·图1" in host.messages[0]


def test_return_loading_rejected(shelf, host):
    page, plot = _make_plot(host)
    entry = _entry("a", plot)
    shelf.add_entry(entry)
    host.loading = True

    returned = shelf.return_entries([entry.key])

    assert returned == []
    assert plot.calls == []
    assert shelf.count() == 1
    assert host.messages == ["数据加载中，暂不可搬运变量"]


def test_return_hidden_cell_appends_hint(shelf, host):
    page, plot = _make_plot(host, cell_visible=False)
    entry = _entry("a", plot)
    shelf.add_entry(entry)

    shelf.return_entries([entry.key])

    assert host.messages == [
        "已退回 a：暂存区 → 页1·图1（该子图当前被布局隐藏）"
    ]


def test_return_selected_uses_selection(shelf, host):
    page, plot = _make_plot(host)
    entry = _entry("a", plot)
    shelf.add_entry(entry)
    shelf.item(0).setSelected(True)

    assert shelf.return_selected() == ["a"]
    assert shelf.count() == 0


def test_build_return_message_classification():
    assert build_return_message([]) == ""
    assert build_return_message(
        [], skipped=["a"], failed=["b"], origin_closed=["c"], no_origin=["d"]
    ) == (
        "未退回；原图已有 a，条目已保留；b 数据校验失败，未退回；"
        "c 的原绘图已关闭，未退回（条目已保留）；d 无原绘图来源，仍留在暂存区"
    )