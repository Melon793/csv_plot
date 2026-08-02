"""内嵌式变量搜索栏组件

常驻于 PlotVariableEditorDialog 顶部，提供 Spotlight 风格的变量搜索添加能力。
- 搜索框 + 候选列表（限高、可折叠） + 状态栏
- QTimer 100ms 防抖
- 子串匹配 + 关键词分割 + 分级排序
- 有效性色块（与左侧变量列表 NoHoverDelegate 一致）
- 已添加变量置灰可选可移除（点击/Enter 已添加项触发移除）
- 已添加变量保持原位（仅样式区分，不挪末尾），add/remove 行为对称
- 弱引用持有 plot_widget，实时查询已添加变量集合
"""

from __future__ import annotations

import sys
import weakref
from typing import Any

from PySide6.QtCore import QEvent, Qt, QTimer, QRect, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

# 平台标识：macOS 上用 ⌘F，其他平台用 Ctrl+F（避免 ⌘ 符号在 Win/Linux 上不自然）
_IS_MAC = sys.platform == "darwin"


class _CandidateItemDelegate(QStyledItemDelegate):
    """候选列表项委托：绘制有效性色块 + 变量名

    与左侧变量列表 NoHoverDelegate 的色块样式保持一致：
    - 绿色方块 = 有效变量 (valid=1)
    - 橙色方块 = 常数变量 (valid=0)
    - 红色方块 = 无效变量 (valid=-1)
    - 已添加变量：灰色斜体 + "(已添加)" 后缀
    """

    def paint(self, painter, option, index):
        from PySide6.QtWidgets import QStyle

        # 移除悬停和焦点状态，避免视觉干扰
        option.state &= ~QStyle.StateFlag.State_MouseOver
        option.state &= ~QStyle.StateFlag.State_HasFocus

        # 绘制背景
        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.palette.base())
        painter.restore()

        # 取数据
        var_name = index.data(Qt.ItemDataRole.UserRole)
        valid = index.data(Qt.ItemDataRole.UserRole + 1)
        is_existing = index.data(Qt.ItemDataRole.UserRole + 2)
        display_text = index.data(Qt.ItemDataRole.DisplayRole)
        if not display_text:
            return

        # 判定色块颜色（父容器全为未知有效性时不显示色块）
        parent_bar = self.parent()
        all_unknown = getattr(parent_bar, "_all_validity_unknown", False)
        color = None
        if not all_unknown and valid is not None:
            color = self._get_validity_color(valid)

        # 计算文本矩形
        text_rect = option.rect
        if color is not None:
            square_size = min(option.rect.height() - 4, 12)
            square_x = option.rect.left() + 8
            square_y = option.rect.top() + (option.rect.height() - square_size) // 2

            painter.save()
            painter.setPen(QPen(color, 1))
            painter.setBrush(color)
            painter.drawRect(square_x, square_y, square_size, square_size)
            painter.restore()

            text_rect = QRect(option.rect)
            text_rect.setLeft(option.rect.left() + square_size + 16)
        else:
            text_rect = option.rect.adjusted(10, 0, -8, 0)

        # 绘制文本
        painter.save()
        if is_existing:
            # 已添加：灰色斜体
            gray = QColor(150, 150, 150)
            painter.setPen(gray)
            font = painter.font()
            font.setItalic(True)
            painter.setFont(font)
        elif option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(QColor(255, 255, 255))
        else:
            painter.setPen(option.palette.text().color())

        elided_text = painter.fontMetrics().elidedText(
            display_text, Qt.TextElideMode.ElideRight, text_rect.width()
        )
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            elided_text,
        )
        painter.restore()

    @staticmethod
    def _get_validity_color(valid: int) -> QColor | None:
        """根据有效性返回色块颜色（与 NoHoverDelegate._get_validity_color 一致）"""
        if valid == 1:
            return QColor(0, 200, 0)      # 绿色：有效
        if valid == 0:
            return QColor(255, 140, 0)    # 橙色：常数
        if valid == -1:
            return QColor(255, 0, 0)       # 红色：无效
        return None


class VariableSearchBar(QWidget):
    """内嵌式变量搜索栏组件

    通过弱引用持有 plot_widget，每次刷新候选列表时实时查询当前已添加变量集合
    （`plot_widget.curves.keys()` 与 `plot_widget.y_name`），避免快照失效。

    Signals:
        variable_selected(str): 用户选中某候选变量（未添加 → 添加）
        variable_removed(str): 用户在搜索栏中移除已添加变量（已添加 → 移除）
        escape_pressed(): 用户在搜索框为空时按 Esc，父容器可据此把焦点交还表格
    """

    variable_selected = Signal(str)
    variable_removed = Signal(str)
    escape_pressed = Signal()

    def __init__(
        self,
        var_names: list[str],
        units: dict[str, str],
        validity: dict[str, int],
        plot_widget: Any,
        parent: QWidget | None = None,
    ):
        """
        Args:
            var_names: 所有可用变量名列表（按原始顺序）
            units: 变量名 → 单位映射
            validity: 变量名 → 有效性映射（1=有效，0=常数，-1=无效，-2=未知）
            plot_widget: 当前 plot 对象（弱引用持有，用于实时查询已添加变量集合）
            parent: 父窗口
        """
        super().__init__(parent)
        self._all_var_names: list[str] = list(var_names)
        self._lower_names: list[str] = [n.lower() for n in self._all_var_names]
        self._units: dict[str, str] = dict(units) if units else {}
        self._validity: dict[str, int] = dict(validity) if validity else {}
        # 全为 -2（未知）时不显示色块，与 variable_list._all_validity_unknown 一致
        self._all_validity_unknown: bool = all(v == -2 for v in self._validity.values())
        # 弱引用持有 plot_widget，避免延长其生命周期
        self._plot_ref = weakref.ref(plot_widget) if plot_widget is not None else None
        # 连接曲线变化信号：拖拽添加等外部变化时自动刷新候选列表
        if plot_widget is not None:
            plot_widget.curves_changed.connect(self._on_curves_changed)

        self._filtered: list[str] = []
        # 输入来源标志：键盘 Enter 添加后选中下一个可选项；鼠标单击添加后保持当前位置
        self._last_input_was_mouse = False
        # 鼠标场景预存"emit 前的滚动条 value（像素值）"：emit variable_selected 之前保存
        # 因为 emit 会同步触发 curves_changed → 第一次 refresh（select_first=True），
        # 该 refresh 会重置滚动条；mark_added 触发的第二次 refresh（select_first=False）
        # 需使用 emit 前的像素值恢复位置。
        # 用像素值而非 var_name：scrollToItem(PositionAtTop) 只能对齐到项边界，
        # 若原位置让项顶部"高于"视口顶部（部分截断），恢复后会丢失像素偏移，产生细微跳动；
        # setValue(像素) 精确到 1px，无偏移。
        # 异步执行（QTimer.singleShot(0)）：QListWidget 几何布局惰性更新，
        # addItem 后立即读 verticalScrollBar().maximum() 仍是 0，setValue 会被 clamp；
        # singleShot(0) 让 Qt 先完成 updateGeometries 计算真实 scrollbar range
        self._pending_mouse_top: tuple[str, int] | None = None
        # 最近一次操作的变量名（add 或 remove 合一）：键盘操作后从该项往后找下一个未添加项，
        # 避免每次都从头查找导致跳顶。textChanged / reset_session 时清除。
        # 替代了原 _last_added_var + _last_removed_var 两个互斥字段：
        # - 无需"设置时清对方"
        # - select_first 锚点查询从 `a or b` 简化为单一字段
        # - remove 操作后变量已不在 existing，rank 自然为 0（回原位），无需额外集合区分
        self._last_op_var: str | None = None

        self._build_ui()
        self._connect_signals()

        # 初始状态：候选列表折叠
        self._collapse_candidates()

    # ---------------- UI 构建 ----------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 搜索框容器（带清空按钮）
        search_row = QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 6)
        search_row.setSpacing(6)

        self.search_edit = QLineEdit(self)
        # placeholder 平台化：macBook 无独立 Insert 键，仅显示 ⌘F；其他平台显示 Ctrl+F 或 Insert
        if _IS_MAC:
            placeholder = "🔍 搜索变量...（⌘F 聚焦）"
        else:
            placeholder = "🔍 搜索变量...（Ctrl+F 或 Insert 聚焦）"
        self.search_edit.setPlaceholderText(placeholder)
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setStyleSheet("""
            QLineEdit {
                font-size: 14px;
                padding: 8px 12px;
                border: 1px solid #ccc;
                border-radius: 6px;
                background: #f8f8f8;
            }
            QLineEdit:focus {
                border: 1px solid #4a90d9;
                background: #ffffff;
            }
        """)
        search_row.addWidget(self.search_edit, 1)
        layout.addLayout(search_row)

        # 候选列表
        self.candidate_list = QListWidget(self)
        self.candidate_list.setMaximumHeight(120)
        # 列表不获取键盘焦点：鼠标点击项时焦点留在搜索框，用户可继续输入/删除
        # ↑↓ 导航通过 eventFilter 拦截搜索框按键实现，不依赖列表焦点
        self.candidate_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.candidate_list.setItemDelegate(_CandidateItemDelegate(self.candidate_list))
        self.candidate_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-top: none;
                background: #ffffff;
                font-size: 12px;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 12px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background: #e3f2fd;
            }
            QListWidget::item:disabled {
                color: #999;
                background: #fafafa;
            }
        """)
        self.candidate_list.setMouseTracking(True)
        self.candidate_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.candidate_list)

        # 状态栏
        self.status_label = QLabel(self)
        self.status_label.setObjectName("search_status")
        self.status_label.setStyleSheet("""
            QLabel#search_status {
                color: #888;
                font-size: 11px;
                padding: 4px 12px;
                background: #fafafa;
                border: 1px solid #ddd;
                border-top: none;
            }
        """)
        layout.addWidget(self.status_label)

        # 外层框，视觉融入编辑器
        self.setFrameShape(QFrame.Shape.NoFrame)

    def setFrameShape(self, shape):
        """兼容性方法：避免 setStyleSheet 覆盖"""
        # No-op：组件整体无边框，融入编辑器布局
        pass

    def _connect_signals(self):
        # 防抖定时器
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(100)
        self._debounce_timer.timeout.connect(self._refresh_list)
        self.search_edit.textChanged.connect(self._on_text_changed)

        # 候选项单击 → 添加/移除（Spotlight 心智模型；已添加项单击触发移除）
        # 注意：不连接 itemActivated，避免双击时单击与激活重复操作
        self.candidate_list.itemClicked.connect(self._on_item_clicked)

        # 鼠标点击防抖：仅对"同一变量的快速二次点击"防抖（避免双击同一变量 toggle）
        # 不同变量的快速点击是合法的连续操作，不阻塞，零延迟响应。
        # 用时间戳比较替代全局布尔 + QTimer：
        #   - 旧方案：300ms 内 click 全部阻塞（误伤不同变量的连续操作）
        #   - 新方案：仅当 var_name 相同且间隔 < threshold 才阻塞，精准命中双击 toggle
        from PySide6.QtGui import QGuiApplication
        import time
        self._click_interval_ms = max(QGuiApplication.styleHints().mouseDoubleClickInterval(), 300)
        self._last_click_var: str | None = None
        self._last_click_time: float = 0.0
        self._click_monotonic = time.monotonic  # 注入便于测试（不直接 import time 在方法内）

        # 安装事件过滤器：拦截搜索框的方向键 / Enter / Esc
        self.search_edit.installEventFilter(self)
        self.candidate_list.installEventFilter(self)

    # ---------------- 公开接口 ----------------

    def focus_search_edit(self):
        """Insert 键调用：聚焦搜索框并全选文本"""
        self.search_edit.setFocus()
        self.search_edit.selectAll()

    def refresh(self):
        """父容器在添加变量后调用，刷新候选列表的"已添加"状态"""
        self._refresh_list()

    def _on_curves_changed(self):
        """plot_widget 曲线变化时刷新候选列表（拖拽添加等外部来源）"""
        self._refresh_list()

    def mark_added(self, var_name: str):
        """标记变量为"本次刚添加"：保持原位置，只更新样式（置灰）

        设计简化：所有变量始终按原始顺序排序，已添加仅通过样式区分（灰色斜体），
        不再挪到末尾。这样 add/remove 行为完全对称，无需 _recently_added 集合
        区分"刚添加"vs"之前已添加"。

        输入来源区分：
        - 键盘 Enter：refresh 后从 _last_op_var 往后找下一个未添加项
        - 鼠标单击：refresh 后保持当前位置
        """
        self._last_op_var = var_name
        self._refresh_list(select_first=not self._last_input_was_mouse)

    def mark_removed(self, var_name: str):
        """标记变量为"本次刚移除"：变量回原位（与未添加同等对待）

        与 mark_added 对称：移除后变量不在 existing 集合中，rank 自然为 0，
        无需额外集合。仅保留 _last_op_var 锚点供键盘移除后 select_first 定位光标。

        输入来源区分：
        - 键盘 Enter：refresh 后从 _last_op_var 往后找下一个未添加项
        - 鼠标单击：refresh 后保持当前位置

        Args:
            var_name: 刚被移除的变量名
        """
        self._last_op_var = var_name
        self._refresh_list(select_first=not self._last_input_was_mouse)

    def reset_session(self):
        """重置搜索会话：清空 _last_op_var 锚点并刷新

        此方法负责清锚点 + 刷新样式。保留刷新是为了：删除/清空变量后
        重新查询 curves 更新"已添加"状态。

        触发时机：表格删除按钮、清空所有变量。
        注意：搜索栏内移除变量不调本方法（见 _on_variable_removed 注释）。
        """
        self._last_op_var = None
        self._refresh_list()

    # ---------------- 防抖与刷新 ----------------

    def _on_text_changed(self, _text: str):
        """textChanged 触发：重置防抖定时器；清锚点"""
        self._debounce_timer.start()  # 已在运行则重新计时
        if self.search_edit.text().strip():
            self._expand_candidates()
        else:
            self._collapse_candidates()
        # 搜索词变化时原 pending 已无意义（列表内容将重新过滤），
        # 防御性清空：避免 add 失败时 _pending_mouse_top 遗留
        self._pending_mouse_top = None
        # 清除操作锚点，回到"选中第一个可选项"的默认行为
        # （不再需要关键词集合变化检测：已添加始终在原位，无"暂存→提交"语义）
        self._last_op_var = None

    def _refresh_list(self, select_first: bool = True):
        """实际执行过滤 + 排序 + 渲染

        Args:
            select_first: True 时选中第一个可选项（键盘场景），
                         False 时保持当前位置（鼠标场景）
        """
        query = self.search_edit.text()
        # 空查询时不填充候选列表：避免折叠后隐藏项仍被 currentItem() 选中
        # 导致输入框为空时按 Enter 误添加第一个变量
        if not query.strip():
            self._filtered = []
            self.candidate_list.clear()
            self._update_status_bar(0)
            return
        self._filtered = self._filter_and_sort(query)
        self._populate_candidate_list(self._filtered, select_first=select_first)
        self._update_status_bar(len(self._filtered))

    # ---------------- 过滤与排序 ----------------

    def _filter_and_sort(self, query: str) -> list[str]:
        """根据关键词过滤并分级排序（OR 关系）

        过滤：空格分隔多个关键字，任一关键字命中即保留（OR 逻辑）。

        排序优先级（统一多态，不区分单/多关键词）：
        1. last_hit_rank：命中的最后一个关键词索引越大越优先
           （越后输入的关键词匹配的变量越靠前，多关键词时自然生效）
        2. hit_count_rank：总命中数多 → 少
        3. validity_rank：有效 > 常数 > 无效
        4. 原始顺序（stable sort 保证）

        设计简化：已不再用 existing_rank 把"之前已添加"挪到末尾。
        所有变量（无论是否已添加）都按原始顺序排序，已添加仅通过样式区分
        （委托绘制灰色斜体 + "(已添加)" 后缀）。这样 add/remove 行为完全对称，
        列表永不因 add/commit 跳动。existing 集合只在 _populate_candidate_list
        中查询一次用于设置样式，不影响排序。

        与左侧变量列表 hide_non_matching 的 AND 逻辑不同：搜索栏用 OR，
        让用户输入多个关键字时能扩大搜索范围而非收窄。
        """
        q = query.strip()
        if not q:
            return list(self._all_var_names)

        keywords = q.split()
        keywords_lower = [kw.lower() for kw in keywords]

        results: list[tuple[int, int, int, int, str]] = []
        # 元组：(last_hit_rank, hit_count_rank, validity_rank, original_idx, name)
        for idx, (var_name, name_lower) in enumerate(zip(self._all_var_names, self._lower_names)):
            # OR 逻辑：任一关键字命中即保留
            # 同时计算"命中的最后一个关键词的索引"
            last_hit_idx = -1
            for i, kw in enumerate(keywords_lower):
                if kw in name_lower:
                    last_hit_idx = i
            if last_hit_idx == -1:
                continue  # 无任何命中

            hit_count = sum(1 for kw in keywords_lower if kw in name_lower)

            # validity_rank: 0=有效, 1=常数, 2=无效；未知(-2)按有效对待
            valid = self._validity.get(var_name, -2)
            if valid == 1 or valid == -2:
                validity_rank = 0
            elif valid == 0:
                validity_rank = 1
            else:  # -1
                validity_rank = 2

            # hit_count_rank：命中数越多越靠前（取负让 sort 升序时多的在前）
            hit_count_rank = -hit_count
            # last_hit_rank：命中的最后一个关键词索引越大越靠前（取负让大的排前）
            last_hit_rank = -last_hit_idx

            results.append((last_hit_rank, hit_count_rank, validity_rank, idx, var_name))

        # stable sort：按优先级升序，原始顺序作稳定兜底
        results.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        return [name for _, _, _, _, name in results]

    # ---------------- 渲染 ----------------

    def _populate_candidate_list(self, filtered: list[str], select_first: bool = True):
        """渲染候选列表

        Args:
            select_first: True 时选中第一个可选项（键盘场景，方便连续 Enter 添加）；
                         False 时保持当前位置（鼠标场景，不跳到列表顶部）
        """
        self.candidate_list.blockSignals(True)
        # 鼠标场景使用 emit 前预存的"视口顶部可见项"恢复位置（_pending_mouse_top）。
        # 不用比例（value/max）：添加变量会触发窗口布局调整，视口高度可能变化（max 1951→901），
        # 比例恢复后用户看到的"列表 X% 处"对应的项完全变了，不符合"看到原来项"的预期。
        # 用 var_name + 像素偏移：scrollToItem 让该项对齐视口顶部，setValue 调整偏移
        # 恢复精确视觉位置。无论视口大小如何变化，项在列表中的索引不变，恢复正确。
        # 异步执行（QTimer.singleShot(0)）：QListWidget 几何布局惰性更新，
        # addItem 后立即 scrollToItem 可能失效；singleShot(0) 让 Qt 先完成几何更新
        mouse_top = None
        if not select_first and self._pending_mouse_top is not None:
            mouse_top = self._pending_mouse_top
            self._pending_mouse_top = None  # 消费一次，避免遗留
        self.candidate_list.clear()

        existing = self._get_existing_set()

        for var_name in filtered:
            unit = self._units.get(var_name, "")
            is_existing = var_name in existing
            valid = self._validity.get(var_name, -2)

            if unit:
                display_text = f"{var_name} ({unit})"
            else:
                display_text = var_name
            if is_existing:
                display_text = f"{display_text} (已添加)"

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, var_name)
            item.setData(Qt.ItemDataRole.UserRole + 1, valid)
            item.setData(Qt.ItemDataRole.UserRole + 2, is_existing)
            # 已添加项不再禁用：保持可选可激活，点击/Enter 触发移除（与添加逻辑对称）
            # 视觉上仍通过委托灰色斜体 + "(已添加)" 后缀区分

            self.candidate_list.addItem(item)

        if select_first:
            # 键盘场景：从"最近操作项"开始找第一个"未添加"项，找不到则不选中
            # _last_op_var 合一锚点：add/remove 共用，无需 `a or b` 优先级表达式
            #
            # add/remove 不对称设计（符合心智）：
            # - Enter 添加：anchor 现在是已添加 → 循环跳过它 → 从下一个未添加项开始选中
            # - Enter 移除：anchor 现在是未添加（刚被移除）→ 循环停在它身上 → 保持原位
            #   （误删后可立即重新 Enter 添加，或继续 ↓ 操作下一项）
            anchor_var = self._last_op_var
            start_idx = -1
            if anchor_var is not None:
                for i, name in enumerate(filtered):
                    if name == anchor_var:
                        start_idx = i
                        break
            # 从 start_idx 往后找第一个"未添加"项（add 跳过已添加的 anchor，remove 停在未添加的 anchor）
            for i in range(max(0, start_idx), self.candidate_list.count()):
                item = self.candidate_list.item(i)
                is_item_existing = item.data(Qt.ItemDataRole.UserRole + 2)
                if not is_item_existing:
                    self.candidate_list.setCurrentRow(i)
                    break
            # 找不到下一个未添加项：保持不选中（currentRow=-1），用户按 ↓ 可重新从头选择
        elif mouse_top is not None:
            # 鼠标场景：异步恢复滚动条到 emit 前的"视口顶部可见项"位置
            # 用 var_name + 像素偏移：
            #   1. scrollToItem(item, PositionAtTop) 让目标项对齐视口顶部
            #   2. setValue(scrollbar.value - offset) 恢复像素偏移（offset<=0，项原本可能部分截断）
            # 自适应视口大小变化：项在列表中的索引不变，无论视口多大都能滚到该项
            # QTimer.singleShot(0) 让 Qt 先完成 updateGeometries 再操作
            top_var, offset = mouse_top
            lst = self.candidate_list

            def _restore_scroll(var=top_var, off=offset, lst=lst):
                # 找到目标项
                target_item = None
                for i in range(lst.count()):
                    it = lst.item(i)
                    if it.data(Qt.ItemDataRole.UserRole) == var:
                        target_item = it
                        break
                if target_item is None:
                    return
                # 滚动到目标项对齐视口顶部
                bar = lst.verticalScrollBar()
                lst.scrollToItem(
                    target_item,
                    QListWidget.ScrollHint.PositionAtTop,
                )
                # 调整像素偏移：offset 是该项顶部相对视口顶部的偏移（<=0）
                # 原始状态：scrollbar.value = S, item_top_y = S + offset
                # scrollToItem 后：scrollbar.value = item_top_y（项对齐视口顶部）
                # 恢复原始位置：setValue(item_top_y - offset) = S
                # 注意符号：offset 是负值，- offset 是正值（往下滚回到原位置）
                bar.setValue(bar.value() - off)

            QTimer.singleShot(0, _restore_scroll)

        self.candidate_list.blockSignals(False)

    def _update_status_bar(self, count: int):
        self.status_label.setText(
            f"找到 {count} 个变量 | Enter 添加/移除 | ↑↓ 移动 | Esc 清空"
        )

    # ---------------- 展开/折叠 ----------------

    def _collapse_candidates(self):
        self.candidate_list.hide()
        self.status_label.hide()

    def _expand_candidates(self):
        self.candidate_list.show()
        self.status_label.show()

    # ---------------- 弱引用查询 ----------------

    def _get_existing_set(self) -> set[str]:
        """通过弱引用实时查询当前 plot 已添加的变量集合"""
        pw = self._plot_ref() if self._plot_ref is not None else None
        if pw is None:
            return set()
        existing: set[str] = set()
        curves = getattr(pw, "curves", None)
        if curves:
            existing.update(curves.keys())
        y_name = getattr(pw, "y_name", "")
        if y_name:
            existing.add(y_name)
        return existing

    # ---------------- 候选项操作 ----------------

    def _move_candidate_selection(self, offset: int):
        """在候选列表内移动高亮，跳过已禁用项"""
        count = self.candidate_list.count()
        if count == 0:
            return
        current = self.candidate_list.currentRow()
        if current < 0:
            current = 0 if offset > 0 else count - 1
        # 向前/向后找下一个可选项
        for step in range(1, count + 1):
            target = (current + offset * step) % count
            if target < 0:
                target += count
            item = self.candidate_list.item(target)
            if item is not None and (item.flags() & Qt.ItemFlag.ItemIsEnabled):
                self.candidate_list.setCurrentRow(target)
                return

    def _accept_current_candidate(self):
        """Enter 触发：发射当前高亮变量（已添加 → 移除；未添加 → 添加）"""
        # 双保险：候选列表隐藏或为空时不响应
        # （_refresh_list 已保证空查询时不填充列表，此处防御性兜底）
        if not self.candidate_list.isVisible() or self.candidate_list.count() == 0:
            return
        item = self.candidate_list.currentItem()
        if item is None:
            return
        if not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        var_name = item.data(Qt.ItemDataRole.UserRole)
        if var_name:
            is_existing = item.data(Qt.ItemDataRole.UserRole + 2)
            # 键盘触发：refresh 后选中下一个未添加项
            self._last_input_was_mouse = False
            if is_existing:
                # 已添加 → 移除
                # 键盘场景不需要滚动恢复：清空 pending 防御性兜底，避免遗留被后续 refresh 误用
                self._pending_mouse_top = None
                self.variable_removed.emit(var_name)
            else:
                # 未添加 → 添加
                self.variable_selected.emit(var_name)

    def _on_item_clicked(self, item: QListWidgetItem):
        """单击候选项 → 已添加则移除；未添加则添加

        Spotlight 心智模型：单击即操作。已添加项通过委托灰色斜体 + "(已添加)" 后缀区分，
        点击已添加项触发移除（与添加逻辑对称）。

        防抖：仅对"同一变量的快速二次点击"防抖（避免双击同一变量 toggle）。
        不同变量的快速点击是合法的连续操作，零延迟响应。
        """
        if not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        var_name = item.data(Qt.ItemDataRole.UserRole)
        if not var_name:
            return
        # 防抖判断：同一变量在间隔内再次点击才阻塞
        now = self._click_monotonic()
        if (self._last_click_var == var_name
                and (now - self._last_click_time) * 1000 < self._click_interval_ms):
            return
        self._last_click_var = var_name
        self._last_click_time = now

        is_existing = item.data(Qt.ItemDataRole.UserRole + 2)
        # 鼠标触发：refresh 后保持当前位置，不跳到列表顶部
        self._last_input_was_mouse = True
        # emit 前预存"视口顶部可见项"的 var_name + 像素偏移：
        # emit 内部会同步触发 curves_changed → 一次 select_first=True 的 refresh，
        # mark_added/mark_removed 触发的第二次 refresh（select_first=False）使用此信息
        # 通过 QTimer.singleShot(0) + scrollToItem + setValue 恢复位置。
        # 用 var_name + offset 而非像素值或比例：
        #   - 像素值：视口高度变化时 setValue 会被 clamp
        #   - 比例：视口变化后"列表 X% 处"对应的项完全变了
        #   - var_name + offset：基于"看到原来的项"恢复，自适应视口大小变化
        # offset = visualItemRect(item).top()：项顶部相对视口顶部的偏移（<=0）
        #   0 表示项顶部对齐视口顶部；负值表示项顶部在视口上方（部分截断）
        top_item = self.candidate_list.itemAt(0, 0)
        if top_item is not None:
            top_var = top_item.data(Qt.ItemDataRole.UserRole)
            offset = self.candidate_list.visualItemRect(top_item).top()
            self._pending_mouse_top = (top_var, offset)
        else:
            self._pending_mouse_top = None
        if is_existing:
            # 已添加 → 移除
            self.variable_removed.emit(var_name)
        else:
            # 未添加 → 添加
            self.variable_selected.emit(var_name)

    # ---------------- 事件过滤（焦点分区） ----------------

    def eventFilter(self, obj, event):
        """拦截搜索框的方向键 / Enter / Esc，避免穿透到下方表格"""
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if obj is self.search_edit:
                if key == Qt.Key.Key_Up:
                    self._move_candidate_selection(-1)
                    return True
                if key == Qt.Key.Key_Down:
                    self._move_candidate_selection(1)
                    return True
                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    self._accept_current_candidate()
                    return True
                if key == Qt.Key.Key_Escape:
                    if self.search_edit.text():
                        self.search_edit.clear()
                    else:
                        self.escape_pressed.emit()
                    return True
                # Ctrl+A 让 QLineEdit 默认处理（全选）
                # Tab 默认行为：切到下一个控件
            elif obj is self.candidate_list:
                if key == Qt.Key.Key_Escape:
                    # 候选列表聚焦时 Esc → 焦点回到搜索框
                    self.search_edit.setFocus()
                    return True
                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    # 候选列表聚焦时 Enter → 添加当前高亮项
                    self._accept_current_candidate()
                    return True
        return super().eventFilter(obj, event)
