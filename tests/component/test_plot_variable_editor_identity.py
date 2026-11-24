"""绘图变量编辑器：顶部身份行（页名 + 行列位次）防回归。

口径（设计 C/D1/D2/D5/D6 + 可见网格修订）：
- 文案 = ``{page.title} · 位次``；位次取 QGridLayout 真实格位（1 起算）
- 按**可见**网格形状退化：1×1 只给页名 / 单行只给列 / 单列只给行 /
  多行多列写全；可见行列缺失或非法时回退已建网格
- 目标格被布局隐藏（缩行列只 setVisible(False)）时写全已建行列并标注隐藏
- 解析不到身份时整行隐藏（不留占位、不写"未知绘图区"）
- 顶部不再出现与窗口标题重复的 16px 粗体「绘图变量编辑器」
"""

import pytest
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from src.core.settings import ConfigKey
from src.ui import theme
from src.ui.plot_variable_editor import PlotVariableEditorDialog


class _Container(QWidget):
    """网格格位替身：只需 plot_widget 反向指针（同 PlotContainerWidget）。"""

    def __init__(self, plot_widget=None):
        super().__init__()
        self.plot_widget = plot_widget


class _FakePage:
    """最小页替身：title + 网格 + 容器列表（与 PlotPage 同名接口）。

    可见行列只在显式传入时挂到 ``rows_current/cols_current``；不传则让
    身份行走"回退已建网格"分支（独立 widget / 旧替身场景）。
    """

    def __init__(self, title="page 1", rows_visible=None, cols_visible=None):
        self.title = title
        self.widget = QWidget()
        self.plot_layout = QGridLayout(self.widget)
        self.plot_widgets = []
        if rows_visible is not None:
            self.rows_current = rows_visible
        if cols_visible is not None:
            self.cols_current = cols_visible


def _attach_page(
    pw, rows, cols, target_rc, title="page 1", rows_visible=None, cols_visible=None
):
    """把 pw 放进 rows×cols 网格的 target_rc 格，并挂到伪宿主上。"""
    page = _FakePage(title, rows_visible=rows_visible, cols_visible=cols_visible)
    for row in range(rows):
        for col in range(cols):
            container = _Container(pw if (row, col) == target_rc else None)
            page.plot_layout.addWidget(container, row, col)
            page.plot_widgets.append(container)
    pw.page = page
    host = pw.window()
    host.page_index = lambda candidate: 0 if candidate is page else -1
    return page


@pytest.fixture()
def make_editor(plot_factory, qapp, app_settings):
    created = []
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY, "")

    def _make(
        rows=1,
        cols=1,
        target_rc=(0, 0),
        title="page 1",
        attach=True,
        rows_visible=None,
        cols_visible=None,
    ):
        pw = plot_factory()
        if attach:
            _attach_page(
                pw,
                rows,
                cols,
                target_rc,
                title,
                rows_visible=rows_visible,
                cols_visible=cols_visible,
            )
        dlg = PlotVariableEditorDialog(pw)
        created.append(dlg)
        qapp.processEvents()
        return dlg

    yield _make
    for dlg in created:
        if dlg.isVisible():
            dlg.close()
        dlg.deleteLater()


def _text(dlg) -> str:
    return dlg.identity_label.text()


def test_single_plot_page_shows_page_name_only(make_editor):
    """1×1：只给页名，不写"第 1 行第 1 列"这类零信息长句。"""
    dlg = make_editor()
    assert _text(dlg) == "page 1"
    assert not dlg.identity_label.isHidden()
    assert dlg.identity_label.toolTip() == "page 1"


def test_multi_row_col_uses_row_and_col(make_editor):
    """可见 2×2 的第 (2,1) 格：写全行列，行与列之间只有一个空格、不加标点。"""
    dlg = make_editor(rows=2, cols=2, target_rc=(1, 0), rows_visible=2, cols_visible=2)
    assert _text(dlg) == "page 1 · 第 2 行 第 1 列"


def test_hidden_columns_do_not_add_column(make_editor):
    """已建 3×2 但可见 3×1：退化成单列，不再多印「第 1 列」。"""
    dlg = make_editor(rows=3, cols=2, target_rc=(1, 0), rows_visible=3, cols_visible=1)
    assert _text(dlg) == "page 1 · 第 2 行"


def test_hidden_rows_do_not_add_row(make_editor):
    """已建 2×3 但可见 1×3：退化成单行，不再多印「第 1 行」。"""
    dlg = make_editor(rows=2, cols=3, target_rc=(0, 2), rows_visible=1, cols_visible=3)
    assert _text(dlg) == "page 1 · 第 3 列"


def test_single_visible_cell_shows_page_name_only(make_editor):
    """已建 1×2 但可见 1×1：只给页名。"""
    dlg = make_editor(rows=1, cols=2, target_rc=(0, 0), rows_visible=1, cols_visible=1)
    assert _text(dlg) == "page 1"


def test_hidden_target_cell_shows_full_coords_with_note(make_editor):
    """目标格本身被布局隐藏：按已建网格写全并标注，避免印出越界位次。"""
    dlg = make_editor(rows=3, cols=2, target_rc=(1, 1), rows_visible=3, cols_visible=1)
    assert _text(dlg) == "page 1 · 第 2 行 第 2 列（当前被布局隐藏）"


def test_invalid_visible_dims_fall_back_to_built_grid(make_editor):
    """可见行列缺失/非法（0）：回退已建网格，不抛异常、文案按已建算。"""
    dlg = make_editor(
        rows=3, cols=2, target_rc=(1, 0), rows_visible=0, cols_visible="bad"
    )
    assert _text(dlg) == "page 1 · 第 2 行 第 1 列"


def test_single_row_degrades_to_column(make_editor):
    """单行多列：行不言自明，只给列。"""
    dlg = make_editor(rows=1, cols=3, target_rc=(0, 2))
    assert _text(dlg) == "page 1 · 第 3 列"


def test_single_col_degrades_to_row(make_editor):
    """单列多行：只给行。"""
    dlg = make_editor(rows=3, cols=1, target_rc=(2, 0))
    assert _text(dlg) == "page 1 · 第 3 行"


def test_identity_uses_muted_text_style(make_editor):
    """身份行是 12px 次要文字（D4 的 16px 粗体标题已撤）。"""
    dlg = make_editor(rows=2, cols=2, target_rc=(0, 1))
    assert dlg.identity_label.styleSheet() == theme.muted_text()
    assert not [
        label for label in dlg.findChildren(QLabel) if label.text() == "绘图变量编辑器"
    ]


def test_unresolvable_identity_hides_line(make_editor):
    """无 page（独立 widget/页已销毁）：整行隐藏，不留残缺文案。"""
    dlg = make_editor(attach=False)
    assert _text(dlg) == ""
    assert dlg.identity_label.isHidden()


def test_rename_page_refreshes_identity(make_editor):
    """页重命名后经 refresh_data_source 顺带校正文案（保留位次）。"""
    dlg = make_editor(rows=2, cols=2, target_rc=(1, 0), title="page 1")
    dlg.plot_widget.page.title = "温度工况"
    dlg.refresh_data_source()
    assert _text(dlg) == "温度工况 · 第 2 行 第 1 列"
