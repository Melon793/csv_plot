from src.ui.action_receipts import (
    page_closed,
    pages_closed,
    shelf_cleared,
    shelf_entries_removed,
)


def test_page_closed_uses_title_without_curve_count():
    assert page_closed("页 2") == "已关闭「页 2」"


def test_pages_closed_uses_count():
    assert pages_closed(3) == "已关闭 3 个绘图页"


def test_shelf_entries_removed_uses_single_name():
    assert shelf_entries_removed(1, "Demo") == "已删除暂存条目「Demo」"


def test_shelf_entries_removed_uses_batch_count():
    assert shelf_entries_removed(3) == "已删除 3 个暂存条目"


def test_shelf_entries_removed_falls_back_to_count_without_name():
    assert shelf_entries_removed(1) == "已删除 1 个暂存条目"


def test_shelf_cleared_includes_count():
    assert shelf_cleared(2) == "已清空暂存区 · 2 个条目"
