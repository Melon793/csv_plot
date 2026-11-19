"""表格选中 QSS 的样式门控：只对 windows11 生效。"""

from src.ui import theme


def test_applies_for_windows11(monkeypatch):
    monkeypatch.setattr(theme, "_base_style_name", "windows11")
    assert theme.table_selection_qss() == theme.TABLE_SELECTION_QSS


def test_skips_other_styles(monkeypatch):
    # Win10 的 windowsvista、macOS、Fusion 都没有该竖条，不该改选中观感
    for name in ("", "windowsvista", "fusion", "macos"):
        monkeypatch.setattr(theme, "_base_style_name", name)
        assert theme.table_selection_qss() == ""


def test_disables_win11_focus_ring_for_both_views():
    # 焦点框是 PE_FrameFocusRect，QSS 的 ::item 规则管不到，必须显式 outline: 0
    assert "QTableWidget, QTableView { outline: 0; }" in theme.TABLE_SELECTION_QSS


def test_set_base_style_name_normalizes_empty(monkeypatch):
    monkeypatch.setattr(theme, "_base_style_name", "")
    theme.set_base_style_name(None)
    assert theme.table_selection_qss() == ""
    theme.set_base_style_name("windows11")
    assert theme.table_selection_qss() == theme.TABLE_SELECTION_QSS