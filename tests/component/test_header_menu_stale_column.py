"""P2-7：表头右键菜单收尾时不得再用 exec 之前捕获的列号。

`menu.exec` 跑的是嵌套事件循环，期间后台重载（`update_data` 经 singleShot 到达）
可以增删列。tab 分支早就改成按稳定 `var_name` 作用于 `state.df`，单表分支却仍把
exec 前算好的 `logical_col` 传进 `_remove_column` / `freeze_column` —— 列一变，
轻则 `self._df.columns[col]` 越界 IndexError，重则**删掉用户没点的那一列**。
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from PySide6.QtCore import QPoint, QCoreApplication
from PySide6.QtWidgets import QMenu

from src.ui.table_dialog import DataTableDialog, PandasTableModel


class _MutatingMenu(QMenu):
    """替身菜单：exec 落地前先跑 mutate()，再按文案前缀返回用户选中的动作。

    打桩在 `exec` 内部正是为了复现「菜单还开着，底下的表已经变了」。
    """

    menus: list[QMenu] = []
    pick_prefix: str | None = None
    mutate = None

    def __init__(self, parent=None):
        super().__init__(parent)
        _MutatingMenu.menus.append(self)

    def exec(self, *args, **kwargs):
        if _MutatingMenu.mutate is not None:
            _MutatingMenu.mutate()
        for act in self.actions():
            if act.text().startswith(_MutatingMenu.pick_prefix):
                return act
        raise AssertionError(f"菜单里没有 {self.pick_prefix!r}: "
                             f"{[a.text() for a in self.actions()]}")


@pytest.fixture()
def stub(qapp, app_settings, monkeypatch):
    _MutatingMenu.menus.clear()
    _MutatingMenu.pick_prefix = None
    _MutatingMenu.mutate = None
    monkeypatch.setattr("src.ui.table_dialog.QMenu", _MutatingMenu)

    dlg = DataTableDialog()
    for name in ("a", "b", "c"):
        dlg._add_variable_to_table(name, pd.Series([1.0, 2.0, 3.0], name=name))
    dlg.set_skip_close_confirmation(True)
    try:
        yield dlg
    finally:
        dlg.hide()
        dlg.deleteLater()
        end = time.monotonic() + 0.02
        while time.monotonic() < end:
            QCoreApplication.processEvents()


def _reload_without(dlg, *gone: str):
    """替身版的「后台重载」：删列并连 model/视图一起重建（与 update_data 同形）。

    只动 `_df` 会造成 df 与 model 列数不一致，`freeze_column` 内部要读视觉顺序，
    那时炸的是这个不自洽，而不是本项要测的陈旧列号。
    """
    def mutate():
        dlg._df.drop(columns=list(gone), inplace=True)
        dlg.model = PandasTableModel(dlg._df, dlg.units)
        dlg.main_view.setModel(dlg.model)
        dlg.frozen_view.setModel(dlg.model)

    _MutatingMenu.mutate = mutate


def _hit(view, logical_col: int) -> QPoint:
    header = view.horizontalHeader()
    return QPoint(header.sectionViewportPosition(logical_col) + 2, header.height() // 2)


class TestSingleTableBranchResolvesByName:
    def test_delete_hits_the_column_the_user_pointed_at(self, stub):
        """用户点的是 b；菜单打开期间 a 被删 → b 挪到 0 号位，c 顶到 1 号位。"""
        _reload_without(stub, "a")
        _MutatingMenu.pick_prefix = '删除列 "b"'

        stub._on_header_right_click(_hit(stub.main_view, 1), stub.main_view)

        assert "b" not in stub._df.columns, "用户点中的列必须被删掉"
        assert "c" in stub._df.columns, "旧写法会误删顶上来的 c"
        assert list(stub._df.columns) == ["c"]

    def test_missing_column_is_skipped_without_indexerror(self, stub):
        """目标列自己没了：静默跳过，不能 IndexError，也不能顺手删别列。"""
        _reload_without(stub, "b")
        _MutatingMenu.pick_prefix = '删除列 "b"'

        stub._on_header_right_click(_hit(stub.main_view, 1), stub.main_view)

        assert list(stub._df.columns) == ["a", "c"]

    def test_freeze_follows_the_name_not_the_position(self, stub):
        _reload_without(stub, "a")
        _MutatingMenu.pick_prefix = "冻结列"

        stub._on_header_right_click(_hit(stub.main_view, 1), stub.main_view)

        assert list(stub.frozen_columns) == ["b"], f"该冻的是 b，实际 {list(stub.frozen_columns)}"

    def test_other_actions_do_not_touch_the_table(self, stub):
        """早退分支只认删/冻两个动作；其余动作（复制、清空由各自 triggered 处理）
        从这条路径返回时不得顺带改表。替身 `exec` 不会真的 triggered 动作，
        故这里只钉「表没被动过」。"""
        _MutatingMenu.mutate = None
        _MutatingMenu.pick_prefix = "复制变量名"

        stub._on_header_right_click(_hit(stub.main_view, 1), stub.main_view)

        assert list(stub._df.columns) == ["a", "b", "c"]
        assert list(stub.frozen_columns) == []
