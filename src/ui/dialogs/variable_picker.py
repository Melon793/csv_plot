"""轻量变量选择对话框（身份匹配 C4 / D7，占位"替换变量…"专用）。

搜索框 + 变量列表，过滤复用 :class:`KeywordMatcher`（与变量列表/编辑器
搜索栏同语义：子串 + `*`/`?` 通配符）。双击条目或点「替换」确认，
取消/关闭返回 None。不做多选、不做预览——占位替换只需要一个名字。
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QVBoxLayout,
)

from src.core.keyword_match import KeywordMatcher


class VariablePickerDialog(QDialog):
    """从 var_names 里挑一个变量名（模态，exec 后经 pick 取结果）。"""

    def __init__(self, var_names: list[str], parent=None, *,
                 title: str = "替换变量", prompt: str = "选择要替换占位的变量："):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(360, 420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(prompt))
        self._search = QLineEdit()
        self._search.setPlaceholderText("搜索（支持 * / ? 通配符）")
        layout.addWidget(self._search)
        self._list = QListWidget()
        for name in var_names:
            self._list.addItem(str(name))
        layout.addWidget(self._list)

        self._buttons = QDialogButtonBox()
        self._ok = self._buttons.addButton("替换", QDialogButtonBox.ButtonRole.AcceptRole)
        self._buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self._buttons)

        self._search.textChanged.connect(self._refilter)
        self._list.itemDoubleClicked.connect(lambda _item: self._on_accept())
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)

        if self._list.count():
            self._list.setCurrentRow(0)

    def _refilter(self) -> None:
        matcher = KeywordMatcher(self._search.text())
        first_visible = None
        for i in range(self._list.count()):
            item = self._list.item(i)
            hidden = not matcher.matches_lower(item.text().lower())
            item.setHidden(hidden)
            if not hidden and first_visible is None:
                first_visible = item
        current = self._list.currentItem()
        if current is None or current.isHidden():
            # 选中项被过滤掉时落到首个可见项，输入后回车即可确认
            self._list.setCurrentItem(first_visible)

    def _on_accept(self) -> None:
        # 无选中（含搜索把选中项过滤掉）时不关窗，避免空替换
        if self.selected_name() is not None:
            self.accept()

    def selected_name(self) -> str | None:
        item = self._list.currentItem()
        if item is None or item.isHidden():
            return None
        return item.text()

    @classmethod
    def pick(cls, var_names: list[str], parent=None, **kwargs) -> str | None:
        """模态挑选：确认返回变量名，取消/关闭返回 None。"""
        dlg = cls(var_names, parent, **kwargs)
        try:
            if dlg.exec() == QDialog.DialogCode.Accepted:
                return dlg.selected_name()
            return None
        finally:
            dlg.deleteLater()