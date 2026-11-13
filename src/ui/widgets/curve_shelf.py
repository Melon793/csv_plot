"""曲线暂存区（Curve Shelf）：跨绘图页搬运曲线的非模态浮窗托盘。

设计口径（tmp/2026-11-13/curve-transfer-preview-plan.md）：
- 托盘条目自持全量样式快照（入区即拷完，出区零源依赖）；
- legend → 托盘：无修饰=剪切（源端延迟删除）/ Ctrl=复制；
- 变量列表 → 托盘：名字入区，默认样式，复制语义；
- 托盘 → plot：无修饰=移出（条目消失）/ Ctrl=复制出（条目保留）；
- stale（reload 后变量不在新数据中）置灰不可拖，删除入口保留。

本模块只放数据模型 + 托盘 QListWidget + 浮窗壳；plot 侧批量协议见
PlotWidget._transfer_curves 与 _handle_shelf_drop。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDrag, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_SOURCE_AUTO,
    SYMBOL_POLICY_AUTO,
)
from src.core.logger import get_logger
from src.ui.drag_drop import (
    LEGEND_MIME_FORMAT,
    SHELF_MIME_FORMAT,
    build_shelf_mimedata,
    clear_active_shelf_drag,
    create_drag_pixmap,
    get_active_legend_drag_source,
    parse_var_names_from_mimedata,
    set_active_shelf_drag,
)
from src.ui.transfer_broadcast import (
    broadcast_text,
    broadcast_transfer,
    loading_reject_message,
    source_closed_message,
)

logger = get_logger("widget.curve_shelf")

SHELF_ITEM_HEIGHT = 26
SHELF_KEY_ROLE = Qt.ItemDataRole.UserRole + 101
SHELF_STALE_MARK = " ✕"


@dataclass
class ShelfEntry:
    """托盘条目：入区时从源 CurveInfo 拷贝的全量样式快照。"""

    key: str
    var_name: str
    color: str = ""
    visible: bool = True
    draw_mode: str = DRAW_MODE_AUTO
    symbol_policy: str = SYMBOL_POLICY_AUTO
    # 快照保真：出区落地后必须回写，不得被 apply 入口误钉成 'user'
    draw_mode_source: str = DRAW_MODE_SOURCE_AUTO
    origin_label: str = ""
    stale: bool = False

    @classmethod
    def from_curve_info(cls, var_name: str, ci: Any, origin_label: str = "") -> "ShelfEntry":
        """从源曲线快照全量样式（须在 drop 调用栈内、源存活时读完）。"""
        return cls(
            key=uuid.uuid4().hex,
            var_name=var_name,
            color=getattr(ci, "color", "") or "",
            visible=bool(getattr(ci, "visible", True)),
            draw_mode=getattr(ci, "draw_mode", DRAW_MODE_AUTO),
            symbol_policy=getattr(ci, "symbol_policy", SYMBOL_POLICY_AUTO),
            draw_mode_source=getattr(
                ci, "draw_mode_source", DRAW_MODE_SOURCE_AUTO
            ),
            origin_label=origin_label,
        )

    @classmethod
    def from_name(cls, var_name: str, origin_label: str = "变量列表") -> "ShelfEntry":
        """变量列表来源：默认样式，复制语义。"""
        return cls(key=uuid.uuid4().hex, var_name=var_name, origin_label=origin_label)

    def style_summary(self) -> str:
        """tooltip 用的样式摘要。"""
        vis = "显示" if self.visible else "隐藏"
        return (
            f"{self.var_name}\n来源：{self.origin_label or '未知'}\n"
            f"颜色：{self.color or '默认'} · {vis}\n"
            f"绘制方式：{self.draw_mode} · 数据点：{self.symbol_policy}"
        )


def _color_icon(color: str, size: int = 12) -> QIcon:
    """色块小图标（QListWidget 条目左侧）。"""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color) if color else QColor("#808080"))
    painter.setPen(QColor(0, 0, 0, 60))
    painter.drawRoundedRect(0, 0, size - 1, size - 1, 3, 3)
    painter.end()
    return QIcon(pixmap)


class CurveShelfWidget(QListWidget):
    """托盘列表：多选拖出 + legend/变量列表拖入 + stale 置灰。

    宿主窗口（MainWindow）经 ``main_window`` 注入：用于加载闸门与统一
    播报；独立测试可留空，闸门/播报自动降级为 no-op。
    """

    entries_changed = Signal()

    def __init__(self, main_window: Any = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._main_window = main_window
        self._entries: dict[str, ShelfEntry] = {}
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setAlternatingRowColors(False)

    # ------------------------------------------------------------------
    # 查询 / 条目管理
    # ------------------------------------------------------------------
    @property
    def entries(self) -> list[ShelfEntry]:
        """按界面顺序返回条目。"""
        result = []
        for row in range(self.count()):
            item = self.item(row)
            entry = self._entries.get(item.data(SHELF_KEY_ROLE)) if item else None
            if entry is not None:
                result.append(entry)
        return result

    def entry_for_key(self, key: str) -> ShelfEntry | None:
        return self._entries.get(key)

    def entries_for_keys(self, keys: list[str]) -> list[ShelfEntry]:
        return [self._entries[k] for k in keys if k in self._entries]

    def entry_for_name(self, var_name: str) -> ShelfEntry | None:
        for entry in self._entries.values():
            if entry.var_name == var_name:
                return entry
        return None

    def selected_entries(self) -> list[ShelfEntry]:
        """按界面行序返回选中条目（拖出 MIME/预览图顺序稳定）。"""
        items = sorted(self.selectedItems(), key=self.row)
        return [
            entry
            for item in items
            if (entry := self._entries.get(item.data(SHELF_KEY_ROLE))) is not None
        ]

    def selected_keys(self) -> list[str]:
        return [entry.key for entry in self.selected_entries()]

    def add_entry(self, entry: ShelfEntry) -> str:
        """入区（按变量名去重）：返回 "added" / "updated"。"""
        existing = self.entry_for_name(entry.var_name)
        if existing is not None:
            # D10：按名去重更新（保留原 key，避免在飞 MIME 失效）
            for attr in (
                "color",
                "visible",
                "draw_mode",
                "symbol_policy",
                "draw_mode_source",
                "origin_label",
            ):
                setattr(existing, attr, getattr(entry, attr))
            existing.stale = False
            self._refresh_item(existing)
            self.entries_changed.emit()
            return "updated"

        self._entries[entry.key] = entry
        item = QListWidgetItem()
        item.setData(SHELF_KEY_ROLE, entry.key)
        item.setSizeHint(QSize(0, SHELF_ITEM_HEIGHT))
        self.addItem(item)
        self._refresh_item(entry)
        self.entries_changed.emit()
        return "added"

    def remove_entries(self, keys: list[str]) -> int:
        """删除指定 key 的条目；返回实际删除数。"""
        removed = 0
        key_set = {str(k) for k in keys}
        for row in reversed(range(self.count())):
            item = self.item(row)
            if item is None or item.data(SHELF_KEY_ROLE) not in key_set:
                continue
            self._entries.pop(item.data(SHELF_KEY_ROLE), None)
            self.takeItem(row)
            removed += 1
        if removed:
            self.entries_changed.emit()
        return removed

    def clear_entries(self) -> int:
        count = self.count()
        if count:
            self._entries.clear()
            self.clear()
            self.entries_changed.emit()
        return count

    def revalidate(self, var_names: set[str]) -> list[str]:
        """reload 后置灰：变量不在新数据中 → stale；重新出现则恢复。

        Returns:
            本次新变为 stale 的变量名列表（供调用方汇总播报）。
        """
        newly_stale: list[str] = []
        changed = False
        for entry in self._entries.values():
            want_stale = entry.var_name not in var_names
            if entry.stale != want_stale:
                entry.stale = want_stale
                changed = True
                if want_stale:
                    newly_stale.append(entry.var_name)
            self._refresh_item(entry)
        if changed:
            self.entries_changed.emit()
        return newly_stale

    def _refresh_item(self, entry: ShelfEntry) -> None:
        """按条目状态重绘文本 / 颜色 / 拖拽标志。"""
        item = self.item_for_key(entry.key)
        if item is None:
            return
        text = entry.var_name
        if entry.origin_label:
            text += f"    {entry.origin_label}"
        if entry.stale:
            text += SHELF_STALE_MARK
        item.setText(text)
        item.setIcon(_color_icon(entry.color))
        if entry.stale:
            item.setForeground(QColor("#9a9a9a"))
            item.setToolTip(entry.style_summary() + "\n当前数据中不存在（不可拖出）")
        else:
            item.setForeground(QColor("#1f1f1f"))
            item.setToolTip(entry.style_summary())
        flags = (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        if entry.stale:
            flags &= ~Qt.ItemFlag.ItemIsDragEnabled
        item.setFlags(flags)

    def item_for_key(self, key: str) -> QListWidgetItem | None:
        for row in range(self.count()):
            item = self.item(row)
            if item is not None and item.data(SHELF_KEY_ROLE) == key:
                return item
        return None

    # ------------------------------------------------------------------
    # 拖出
    # ------------------------------------------------------------------
    def startDrag(self, supportedActions):
        """拖出：只取非 stale 的选中条目，MIME 携带 key 列表 + 文本兼容层。"""
        entries = [entry for entry in self.selected_entries() if not entry.stale]
        if not entries:
            return
        names = [entry.var_name for entry in entries]
        keys = [entry.key for entry in entries]
        drag = QDrag(self)
        drag.setMimeData(build_shelf_mimedata(keys, names, self))
        pixmap = create_drag_pixmap(names, self.font())
        if pixmap:
            drag.setPixmap(pixmap)
            drag.setHotSpot(
                QPoint(pixmap.width() // 2, pixmap.height() // 2)
            )
        set_active_shelf_drag(self, keys)
        try:
            # 无修饰=移出、Ctrl=复制出，由 drop 端按修饰键决定语义
            drag.exec(
                Qt.DropAction.CopyAction | Qt.DropAction.MoveAction
            )
        finally:
            clear_active_shelf_drag()

    # ------------------------------------------------------------------
    # 拖入（legend / 变量列表）
    # ------------------------------------------------------------------
    def _host_window(self) -> Any:
        if self._main_window is not None:
            try:
                self._main_window.window()
            except RuntimeError:
                self._main_window = None
            else:
                return self._main_window
        return self.window()

    def _announce(self, message: str, level: str = "info") -> None:
        show = getattr(self._host_window(), "_show_status_message", None)
        if callable(show):
            show(message)
        else:
            logger.info(message)

    def _reject_when_loading(self, action: str) -> bool:
        host = self._host_window()
        is_loading = getattr(host, "is_data_loading", None)
        if callable(is_loading) and is_loading():
            # 搬运口径的统一文案（§5.3），不复用通用闸门的「请稍候再…」
            broadcast_text(host, loading_reject_message())
            return True
        reject = getattr(host, "reject_when_loading", None)
        return bool(reject and reject(action))

    def _can_accept_mime(self, mime) -> bool:
        if mime is None or not mime.hasText():
            return False
        # 托盘自身拖回托盘不做重排（无意义），交给 plot drop 端消费
        if mime.hasFormat(SHELF_MIME_FORMAT):
            return False
        return True

    def dragEnterEvent(self, event):
        if self._can_accept_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._can_accept_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()
        if not self._can_accept_mime(mime):
            event.ignore()
            return
        if self._reject_when_loading("暂存变量"):
            event.ignore()
            return
        names = parse_var_names_from_mimedata(mime)
        if not names:
            event.ignore()
            return

        if mime.hasFormat(LEGEND_MIME_FORMAT):
            self._drop_from_legend(event, names)
        else:
            self._drop_from_variable_list(event, names)

    def _drop_from_legend(self, event, names: list[str]) -> None:
        """legend 来源：无修饰=剪切入区（源端延迟删除）/ Ctrl=复制入区。"""
        source = get_active_legend_drag_source(event.mimeData())
        if source is None:
            # 源已销毁：快照不可读，拒绝入区（区别于 plot drop 端的按复制降级）
            broadcast_text(self._host_window(), source_closed_message(names))
            event.ignore()
            return

        copy_mode = bool(
            event.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier
        )
        origin = self._origin_label_for(source)
        # 快照必须在 drop 调用栈内读完（源端删除延迟一拍）
        added: list[str] = []
        updated = 0
        for name in names:
            ci = source.curves.get(name)
            if ci is None:
                continue
            entry = ShelfEntry.from_curve_info(name, ci, origin_label=origin)
            status = self.add_entry(entry)
            if status == "updated":
                updated += 1
            added.append(name)

        if not added:
            self._announce(
                f"曲线不可用，无法暂存 {'、'.join(names)}", level="warn"
            )
            event.ignore()
            return

        if not copy_mode:
            self._schedule_source_remove(source, added)

        event.acceptProposedAction()
        self._announce_shelf_intake(
            cut=not copy_mode, names=added, origin=origin, updated=updated
        )

    def _drop_from_variable_list(self, event, names: list[str]) -> None:
        """变量列表来源：名字入区，默认样式，复制语义。"""
        updated = 0
        for name in names:
            if self.add_entry(ShelfEntry.from_name(name)) == "updated":
                updated += 1
        event.acceptProposedAction()
        self._announce_shelf_intake(
            cut=False, names=names, origin="变量列表", updated=updated
        )

    def _schedule_source_remove(self, source: Any, names: list[str]) -> None:
        """剪切语义：延迟一拍删除源曲线（避开 drop 栈内场景中间态）。"""

        def _make_remove(src: Any, var: str):
            def _remove():
                try:
                    src.remove_variable_from_plot(var)
                except RuntimeError:
                    logger.debug("暂存区入区：源绘图已销毁，跳过删除")
                except Exception:
                    logger.debug(
                        "暂存区入区：源曲线删除失败 %s", var, exc_info=True
                    )

            return _remove

        for name in names:
            QTimer.singleShot(0, _make_remove(source, name))

    def _announce_shelf_intake(
        self, *, cut: bool, names: list[str], origin: str, updated: int
    ) -> None:
        broadcast_transfer(
            self._host_window(),
            "cut" if cut else "shelf_copy",
            names,
            src_label=origin,
            updated=updated,
        )

    def _origin_label_for(self, source: Any) -> str:
        """反算来源徽章「页N·图M」（快照语义：之后重命名/重排不追更）。"""
        page = getattr(source, "page", None)
        host = self._host_window()
        if page is None or host is None:
            return ""
        try:
            page_index = host.page_index(page)
            plot_index = next(
                (
                    index
                    for index, container in enumerate(page.plot_widgets)
                    if getattr(container, "plot_widget", None) is source
                ),
                -1,
            )
        except Exception:
            return ""
        if page_index < 0 or plot_index < 0:
            return ""
        return f"页{page_index + 1}·图{plot_index + 1}"

    # ------------------------------------------------------------------
    # 条目删除（中键 / 右键 / 清空）
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                key = item.data(SHELF_KEY_ROLE)
                self.remove_entries([key])
                event.accept()
                return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        menu = QMenu(self)
        delete_act = None
        delete_keys: list[str] = []
        if item is not None:
            key = item.data(SHELF_KEY_ROLE)
            if item.isSelected():
                delete_keys = [k for k in self.selected_keys() if k]
            else:
                delete_keys = [key]
            label = (
                f"删除所选条目（{len(delete_keys)}）"
                if len(delete_keys) > 1
                else "删除条目"
            )
            delete_act = menu.addAction(label)
        clear_act = menu.addAction("清空暂存区")
        clear_act.setEnabled(self.count() > 0)

        chosen = menu.exec(event.globalPos())
        menu.deleteLater()
        if chosen is delete_act and delete_keys:
            if len(delete_keys) > 1:
                reply = QMessageBox.question(
                    self,
                    "删除暂存条目",
                    f"确定要删除暂存的 {len(delete_keys)} 个条目吗？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
            self.remove_entries(delete_keys)
        elif chosen is clear_act:
            self.request_clear()

    def request_clear(self) -> int:
        """清空入口（按钮/菜单共用）：N>1 一次汇总确认。"""
        count = self.count()
        if count == 0:
            return 0
        if count > 1:
            reply = QMessageBox.question(
                self,
                "清空暂存区",
                f"确定要清空暂存的 {count} 个条目吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return 0
        return self.clear_entries()


class CurveShelfDialog(QDialog):
    """非模态浮窗壳（Qt.Tool）：几何记忆 + 计数标题 + 清空/收起。"""

    visibility_changed = Signal(bool)

    def __init__(self, main_window: Any = None, parent: QWidget | None = None):
        super().__init__(parent if parent is not None else main_window)
        self.main_window = main_window
        self.setWindowTitle("曲线暂存区")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setModal(False)
        self.setMinimumSize(260, 220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        self.title_label = QLabel("暂存区 (0)", self)
        self.title_label.setStyleSheet("font-weight: bold;")
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self.shelf = CurveShelfWidget(main_window=main_window, parent=self)
        layout.addWidget(self.shelf, 1)

        btn_row = QHBoxLayout()
        self.clear_btn = QPushButton("清空", self)
        self.clear_btn.clicked.connect(self.shelf.request_clear)
        self.close_btn = QPushButton("收起", self)
        self.close_btn.clicked.connect(self.hide)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self.shelf.entries_changed.connect(self._on_entries_changed)
        self._restore_geometry()

    # ------------------------------------------------------------------
    # 标题 / 可见性
    # ------------------------------------------------------------------
    def _on_entries_changed(self) -> None:
        count = self.shelf.count()
        self.title_label.setText(f"暂存区 ({count})")
        self.setWindowTitle(f"曲线暂存区 ({count})")
        self.clear_btn.setEnabled(count > 0)

    def showEvent(self, event):
        super().showEvent(event)
        self._on_entries_changed()
        self.visibility_changed.emit(True)

    def hideEvent(self, event):
        self._save_geometry()
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def changeEvent(self, event):
        """最小化/恢复时同步开合态。

        部分平台最小化不伴随 hideEvent，单靠 hideEvent 会让胶囊停在勾选态，
        下一次点击被当成"关闭"，窗口永远唤不回。这里把"最小化"也视为关。
        """
        if event.type() == QEvent.Type.WindowStateChange:
            self.visibility_changed.emit(
                self.isVisible() and not self.isMinimized()
            )
        super().changeEvent(event)

    def closeEvent(self, event):
        """关闭 = 隐藏单例（条目保留），几何落盘。"""
        self._save_geometry()
        event.accept()

    # ------------------------------------------------------------------
    # reload 生命周期
    # ------------------------------------------------------------------
    def revalidate_against_loader(self, loader: Any) -> list[str]:
        """reload 后按新 loader 变量集置灰；返回新 stale 列表。"""
        var_names = set(getattr(loader, "var_names", []) or [])
        return self.shelf.revalidate(var_names)

    # ------------------------------------------------------------------
    # 几何记忆（抄 plot_variable_editor 的多显示器拔插守卫）
    # ------------------------------------------------------------------
    def _geometry_on_screen(self) -> bool:
        from PySide6.QtGui import QGuiApplication

        geometry = self.normalGeometry()
        if geometry.isNull():
            return True
        return any(
            geometry.intersects(screen.availableGeometry())
            for screen in QGuiApplication.screens()
        )

    def _restore_geometry(self) -> None:
        from PySide6.QtGui import QGuiApplication

        from src.core.settings import AppSettings

        settings = AppSettings()
        geometry = settings.get_curve_shelf_geometry()
        if (
            geometry is not None
            and self.restoreGeometry(geometry)
            and self._geometry_on_screen()
        ):
            return
        self.resize(300, 260)
        # 默认落点：页签栏右端胶囊下方 6px、右对齐。getattr 兜底是硬要求：
        # 组件测试的 ShelfHost 没有这个方法，不能 AttributeError。
        anchor_getter = getattr(self.main_window, "curve_shelf_anchor_global", None)
        anchor = anchor_getter() if callable(anchor_getter) else None
        if anchor is not None:
            size = self.size()
            x = anchor.x() - size.width() + 1
            y = anchor.y() + 6
            # G7：screenAt 会返回 None（虚拟屏 / 跨屏边缘），必须退回主屏
            screen = QGuiApplication.screenAt(anchor) or QGuiApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                x = min(max(x, available.left()), available.right() - size.width() + 1)
                y = min(max(y, available.top()), available.bottom() - size.height() + 1)
            self.move(x, y)
            return
        parent = self.parentWidget()
        if parent is not None:
            geo = self.geometry()
            self.move(
                parent.x() + parent.width() - geo.width() - 40,
                parent.y() + 80,
            )

    def _save_geometry(self) -> None:
        # 最小化/最大化时 saveGeometry 会把窗口状态一并存盘，下次启动会
        # 直接以最小化出现（等于丢窗）；与 mark_stats.save_geom 同口径跳过。
        if self.isMinimized() or self.isMaximized():
            return
        from src.core.settings import AppSettings

        AppSettings().set_curve_shelf_geometry(self.saveGeometry())
