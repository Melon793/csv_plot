"""
MultiCurveManager - 曲线绘图管理（统一版）

负责 DraggableGraphicsLayoutWidget 的曲线绘图、样式和可见性管理功能：
- 曲线添加和移除（统一路径，无模式切换）
- 曲线样式管理
- 可见性切换
- 图例管理

单线/多线模式统一后，所有曲线均存储在 pw.curves 字典中。
"""

from __future__ import annotations
import html
from typing import Any, TYPE_CHECKING
from urllib.parse import quote

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QMessageBox
from src.core.config import DEFAULT_LINE_WIDTH, THICK_LINE_WIDTH, THIN_LINE_WIDTH
from src.core.data_types import CurveInfo
from src.core.logger import get_logger
from src.ui.drag_drop import ANCHOR_SCHEME, parse_anchor_var_name

logger = get_logger("widget.multi_curve")

if TYPE_CHECKING:
    from src.ui.widgets.plot_data_manager import PlotDataManager


class MultiCurveManager:
    """负责曲线绘图和样式管理（统一版）"""

    def __init__(self, plot_data_manager: PlotDataManager):
        if plot_data_manager is None:
            raise ValueError(
                "MultiCurveManager requires a valid PlotDataManager instance"
            )
        self._data_manager = plot_data_manager

    @property
    def pw(self) -> Any:
        dm = self._data_manager
        if dm is None:
            raise RuntimeError(
                "MultiCurveManager: dependency chain broken (_data_manager is None)"
            )
        return dm.pw

    def _assign_curve_color(self, preferred_color: str | None = None) -> str:
        """为曲线分配颜色。

        - 首条曲线固定蓝色（无 preferred_color 时）
        - 指定 preferred_color 时使用该颜色
        - 否则按 curve_colors 顺序轮转
        - 索引在所有分支统一推进（含 preferred_color 场景），
          避免 reload 恢复 N 条曲线后索引停留导致颜色碰撞
        """
        pw = self.pw
        is_first_curve = len(pw.curves) == 0

        if preferred_color:
            color = preferred_color
        elif is_first_curve:
            color = "blue"
        else:
            color = pw.curve_colors[pw.current_color_index % len(pw.curve_colors)]

        pw.current_color_index = max(pw.current_color_index + 1, 1)
        return color

    def _update_header_for_curves(self):
        """统一的图例更新逻辑

        始终使用 HTML 图例格式（颜色方块 + 文字），支持点击切换可见性。
        """
        pw = self.pw
        curve_count = len(pw.curves)
        logger.debug(
            "[HEADER] _update_header_for_curves: curve_count=%d, 显示模式=%s",
            curve_count, "图例" if curve_count > 0 else "空",
        )

        if curve_count > 0:
            self.update_legend()
        else:
            pw.update_legend_label("channel name")

    def update_multi_curve_mode(self):
        """兼容别名：委托到 _update_header_for_curves"""
        self._update_header_for_curves()

    def _build_legend_html(self) -> str:
        """构建图例 HTML（唯一样式来源，供显示与测试复用）"""
        pw = self.pw
        legend_items = []
        for var_name, ci in pw.curves.items():
            color = ci.color
            unit = pw.units.get(var_name, "")
            legend_text = html.escape(f"{var_name} ({unit})" if unit else var_name)
            encoded = quote(var_name, safe="")

            if ci.visible:
                legend_items.append(
                    f"<a href='{ANCHOR_SCHEME}:///{encoded}' "
                    f"style='color: {color}; text-decoration: none;'>"
                    f"<span style='font-weight: bold;'>■</span>&nbsp;{legend_text}</a>"
                )
            else:
                legend_items.append(
                    f"<a href='{ANCHOR_SCHEME}:///{encoded}' "
                    f"style='text-decoration: none;'>"
                    f"<span style='color: {color}; opacity: 0.5;'>□</span>&nbsp;"
                    f"<span style='color: gray;'>{legend_text}</span></a>"
                )
        return " | ".join(legend_items)

    def update_legend(self):
        """更新图例显示（多行自适应 + 锚点点击切换显隐）"""
        pw = self.pw
        legend_html = self._build_legend_html()
        if legend_html:
            pw.update_legend_label(legend_html)
        else:
            pw.update_legend_label("channel name")

    def toggle_curve_visibility_by_name(self, var_name: str):
        """通过变量名切换曲线可见性

        点击图例中的曲线名时调用，切换该曲线的显示/隐藏状态。
        如果曲线对象失效（不在scene中），会尝试重新创建。

        Args:
            var_name: 要切换可见性的变量名
        """
        pw = self.pw

        if var_name not in pw.curves:
            return

        ci = pw.curves[var_name]
        ci.visible = not ci.visible
        new_visible = ci.visible

        if ci.curve is not None:
            curve_obj = ci.curve
            try:
                if curve_obj.scene() is not None:
                    curve_obj.setVisible(new_visible)
                else:
                    self._recreate_curve(var_name)
            except Exception:
                self._recreate_curve(var_name)

        self.update_legend()

        self._update_axes_for_multi_curve()

        if pw.vline.isVisible():
            pw.update_cursor_label()

    def _finalize_batch_add(
        self,
        *,
        update_header: bool = True,
        vline_fallback: tuple[float, float] | None = None,
        sync_min_xrange: bool = True,
    ) -> None:
        """批量添加收尾：补偿 _batch_adding 模式跳过的副作用（唯一权威实现）。

        普通批量添加（add_variables_to_plot）与 reload 批量重建
        （cursor_sync_manager.replots_after_loading）共用，避免双实现漂移。

        Args:
            update_header: 是否刷新 legend/header（调用方已刷新时传 False）
            vline_fallback: 无可见曲线（全隐藏）时的 vline bounds 回退范围；
                不传则保持原 bounds 不变（普通批量路径行为）
            sync_min_xrange: 是否触发全局 minXRange 同步（reload 路径在
                所有 plot 重建后统一触发一次，传 False）
        """
        pw = self.pw
        if update_header:
            self._update_header_for_curves()

        self._update_axes_for_multi_curve()

        x_arrays = pw._collect_visible_curve_arrays('x_data')
        if x_arrays:
            combined = np.concatenate(x_arrays)
            min_x, max_x = np.nanmin(combined), np.nanmax(combined)
        else:
            min_x, max_x = vline_fallback if vline_fallback is not None else (None, None)

        if min_x is not None:
            pw._set_vline_bounds([min_x, max_x])
            pw._update_cursor_after_plot(min_x, max_x)

        if pw.vline.isVisible():
            pw.update_cursor_label()

        pw._recalc_max_point_density()

        if sync_min_xrange:
            main_window = pw.window()
            if main_window is not None and hasattr(main_window, 'cursor_sync_manager'):
                main_window.cursor_sync_manager._sync_min_xrange()

    def _recreate_curve(self, var_name: str):
        """重新创建失效的曲线"""
        pw = self.pw
        try:
            if var_name in pw.curves:
                ci = pw.curves[var_name]
                pw.add_variable_to_plot(
                    var_name,
                    skip_existence_check=True,
                    preferred_color=ci.color,
                )
        except Exception:
            logger.warning("重建曲线 '%s' 失败", var_name, exc_info=True)

    def _collect_visible_curve_arrays(self, key: str) -> list:
        """收集可见曲线的数据数组"""
        pw = self.pw
        arrays = []
        if not getattr(pw, "curves", None):
            return arrays
        for ci in pw.curves.values():
            if not ci.visible:
                continue
            data = getattr(ci, key, None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.size == 0:
                continue
            arrays.append(arr)
        return arrays

    def _collect_visible_curve_pairs(self) -> list:
        """收集可见曲线的 x-y 数据对"""
        pw = self.pw
        pairs = []
        if not getattr(pw, "curves", None):
            return pairs
        for ci in pw.curves.values():
            if not ci.visible:
                continue
            x_data = ci.x_data
            y_data = ci.y_data
            if x_data is None or y_data is None:
                continue
            x_arr = np.asarray(x_data)
            y_arr = np.asarray(y_data)
            if x_arr.size == 0 or y_arr.size == 0:
                continue
            pairs.append((x_arr, y_arr))
        return pairs

    def _get_x_window_intersected(self, x_values) -> tuple:
        """计算用户当前 X viewRange 与新数据范围的交集。

        v5.x 修复问题 B：reload 后 Y viewRange 应基于"用户可见窗口与新数据的交集"，
        而非旧的 X viewRange（可能包含已不存在的数据范围）。
        交集为空时返回 (min > max) 的反转范围，使 _get_y_range_in_x_window
        内部 mask 为空，触发其回退逻辑（基于全数据范围）。
        """
        pw = self.pw
        try:
            data_x_min = float(np.min(x_values))
            data_x_max = float(np.max(x_values))
            view_x_min, view_x_max = pw.view_box.viewRange()[0]
            x_min = max(float(view_x_min), data_x_min)
            x_max = min(float(view_x_max), data_x_max)
            if x_min > x_max:
                # 交集为空：返回反转范围使 mask 为空，触发全数据回退
                return data_x_max + 1.0, data_x_min - 1.0
            return x_min, x_max
        except Exception:
            # 异常时返回反转范围，触发全数据回退（保守策略）
            try:
                return float(np.max(x_values)) + 1.0, float(np.min(x_values)) - 1.0
            except Exception:
                return 1.0, 0.0

    def get_curve_x_limits(self, curves_filter: str = "visible") -> tuple:
        """获取曲线 X 轴限制（统一版：始终从 curves 字典获取）

        Args:
            curves_filter: "visible" — 仅可见曲线；"all" — 所有曲线（含隐藏）

        Returns:
            (min_x, max_x) 或 (None, None) 当无数据时
        """
        pw = self.pw
        mins = []
        maxs = []

        for ci in pw.curves.values():
            if curves_filter == "visible" and not ci.visible:
                continue
            mins.append(ci.x_min)
            maxs.append(ci.x_max)

        if not mins:
            return (None, None)
        return (min(mins), max(maxs))

    def _update_axes_for_multi_curve(self, _update_x_range: bool = False):
        """为多曲线更新坐标轴范围。

        计算所有可见曲线的数据范围，并更新坐标轴显示范围。
        只考虑 visible=True 的曲线，忽略隐藏的曲线。

        注意:
            _update_x_range 参数已废弃（保留仅为 API 兼容）。
            X 轴 limits 由 compute_global_x_limits 统一管理。
        """
        pw = self.pw

        if not pw.curves:
            return

        pairs = self._collect_visible_curve_pairs()
        if not pairs:
            return
        x_values = np.concatenate([p[0] for p in pairs])
        y_values = np.concatenate([p[1] for p in pairs])
        if x_values.size == 0 or y_values.size == 0:
            return

        all_data_min_y = np.nanmin(y_values)
        all_data_max_y = np.nanmax(y_values)
        self.pw._set_safe_y_range(
            all_data_min_y, all_data_max_y, set_limits=True
        )

        special_limits = self._data_manager.handle_single_point_limits(
            x_values, y_values
        )
        if special_limits:
            min_x, max_x, min_y, max_y = special_limits
            self.pw._set_safe_y_range(min_y, max_y, set_limits=False)
        else:
            # v5.x 修复问题 B：用 viewRange 与新数据范围的交集，而非旧 viewRange
            x_min, x_max = self._get_x_window_intersected(x_values)

            all_y_in_range = []
            for x_arr, y_arr in pairs:
                min_y, max_y = pw._get_y_range_in_x_window(
                    x_arr, y_arr, x_min, x_max
                )
                all_y_in_range.extend([min_y, max_y])

            if all_y_in_range:
                final_min_y = np.nanmin(all_y_in_range)
                final_max_y = np.nanmax(all_y_in_range)
                self.pw._set_safe_y_range(
                    final_min_y, final_max_y, set_limits=False
                )

    def _on_legend_anchor_clicked(self, url):
        """锚点点击：解析 href 中的变量名并切换曲线显隐

        变量名存放在 URL path 部分（curve:///name）：QUrl 会将 host 强制
        转小写并对非 ASCII 做 IDNA 编码，故用 toString + unquote 解析
        （封装在 parse_anchor_var_name，与 legend 拖拽发起端共用）。
        """
        var_name = parse_anchor_var_name(url.toString())
        if var_name and var_name in self.pw.curves:
            self.toggle_curve_visibility_by_name(var_name)

    def _apply_plot_style(self, show_symbols: bool = False):
        """应用绘图样式 - 细线+symbol 或 粗线无symbol（含内存优化pen缓存，统一版）"""
        pw = self.pw
        try:
            for var_name, ci in pw.curves.items():
                if ci.curve is None:
                    continue

                curve = ci.curve
                color = ci.color

                if show_symbols:
                    cache_key = f'thin_{color}'
                    if not hasattr(curve, '_cached_pen_key') or curve._cached_pen_key != cache_key:
                        pen = pg.mkPen(color=color, width=THIN_LINE_WIDTH)
                        curve.setPen(pen)
                        curve._cached_pen_key = cache_key

                    if not hasattr(curve, '_has_symbols') or not curve._has_symbols:
                        curve.setSymbol('s')
                        curve.setSymbolSize(3)
                        curve.setSymbolPen(color)
                        curve.setSymbolBrush(color)
                        curve._has_symbols = True
                else:
                    cache_key = f'thick_{color}'
                    if not hasattr(curve, '_cached_pen_key') or curve._cached_pen_key != cache_key:
                        pen = pg.mkPen(color=color, width=THICK_LINE_WIDTH)
                        curve.setPen(pen)
                        curve._cached_pen_key = cache_key

                    if not hasattr(curve, '_has_symbols') or curve._has_symbols:
                        curve.setSymbol(None)
                        curve._has_symbols = False
        except Exception as e:
            logger.error("应用绘图样式时出错: %s", e)

    def add_variables_to_plot(self, var_names: list[str]):
        """批量添加变量到当前绘图区（统一版）"""
        names = [name.strip() for name in (var_names or []) if isinstance(name, str) and name.strip()]
        if not names:
            return

        if len(names) > 1:
            pw = self.pw
            invalid_vars = []   # 数据无效/不存在的变量
            duplicate_vars = [] # 已在绘图中的变量
            success_vars = []
            variables_data = []

            for var_name in names:
                is_valid, _ = pw._validate_plot_data(var_name)
                if not is_valid:
                    invalid_vars.append(var_name)
                    continue

                success, _, x_array, y_array, y_format = pw._prepare_plot_data(var_name)
                if not success:
                    invalid_vars.append(var_name)
                    continue

                # 统一重复检测：始终检查 curves 字典
                if var_name in pw.curves:
                    duplicate_vars.append(var_name)
                    continue

                variables_data.append((var_name, x_array, y_array, y_format))

            pw._batch_adding = True
            try:
                if variables_data:
                    for var_name, x_array, y_array, y_format in variables_data:
                        original_index = np.ascontiguousarray(x_array, dtype=np.float32)
                        x_values = pw.offset + pw.factor * original_index
                        y_contiguous = np.ascontiguousarray(y_array)

                        # 颜色分配（统一方法）
                        color = self._assign_curve_color()
                        logger.debug(
                            "[COLOR] 批量添加颜色分配: var=%s, color=%s",
                            var_name, color,
                        )

                        pen = pg.mkPen(color=color, width=DEFAULT_LINE_WIDTH)
                        curve = pw.plot_item.plot(
                            x_values, y_contiguous, pen=pen, name=var_name,
                            skipFiniteCheck=True, connect="all",
                        )

                        pw.curves[var_name] = CurveInfo(
                            var_name=var_name,
                            curve=curve,
                            x_data=x_values,
                            y_data=y_contiguous,
                            original_index=original_index,
                            color=color,
                            y_format=y_format or '',
                            visible=True
                        )

                        success_vars.append(var_name)

                if success_vars:
                    self._finalize_batch_add()
            finally:
                # try/finally 保证异常路径也复位 flag（对齐 cursor_sync_manager
                # 批量重建路径）：否则 flag 残留 True 会使后续单曲线添加/
                # 显隐切换静默跳过 header/vline/密度等全部刷新
                pw._batch_adding = False

            if invalid_vars or duplicate_vars:
                msg_parts = []
                if invalid_vars:
                    msg_parts.append("以下变量没有有效数据:\n" + "\n".join(invalid_vars))
                if duplicate_vars:
                    msg_parts.append("以下变量已在绘图中:\n" + "\n".join(duplicate_vars))
                QMessageBox.information(pw, "提示", "\n\n".join(msg_parts))
        else:
            self.pw.plot_variable(names[0])

    def add_variable_to_plot(self, var_name: str, x_values: np.ndarray = None, y_values: np.ndarray = None,
                             y_format: str = None, skip_existence_check: bool = False,
                             show_duplicate_warning: bool = True, preferred_color: str | None = None) -> bool:
        """添加变量到曲线绘图（统一版：无模式切换，始终写入 curves 字典）"""
        pw = self.pw
        logger.debug(
            "[ADD_VAR] add_variable_to_plot 入口: var_name=%s, 当前 curves 数量=%d, "
            "preferred_color=%s, skip_existence_check=%s",
            var_name, len(pw.curves), preferred_color, skip_existence_check,
        )
        try:
            original_index = None
            if x_values is None or y_values is None:
                success, error_msg, x_array, y_array, y_format = pw._prepare_plot_data(var_name)
                if not success:
                    QMessageBox.warning(pw, "错误", error_msg)
                    return False
                original_index = np.ascontiguousarray(x_array, dtype=np.float32)
                x_values = pw.offset + pw.factor * original_index
                y_values = np.ascontiguousarray(y_array)
            else:
                # 外部传入的 x_values 已含 factor/offset，尝试反算 original_index
                if pw.factor != 0:
                    original_index = np.ascontiguousarray(
                        (np.asarray(x_values) - pw.offset) / pw.factor, dtype=np.float32
                    )

            # 统一重复检测：始终检查 curves 字典
            if not skip_existence_check and var_name in pw.curves:
                if show_duplicate_warning:
                    QMessageBox.information(pw, "提示", f"变量 {var_name} 已在绘图中")
                return False

            # 颜色分配（统一方法，含 preferred_color 场景）
            color = self._assign_curve_color(preferred_color)
            logger.debug(
                "[COLOR] 颜色分配: var=%s, preferred=%s, assigned=%s, next_index=%d",
                var_name, preferred_color, color, self.pw.current_color_index,
            )

            pen = pg.mkPen(color=color, width=DEFAULT_LINE_WIDTH)
            curve = pw.plot_item.plot(
                x_values, y_values,
                pen=pen,
                name=var_name,
                skipFiniteCheck=True,
                connect="all",
            )

            pw.curves[var_name] = CurveInfo(
                var_name=var_name,
                curve=curve,
                x_data=x_values,
                y_data=y_values,
                original_index=original_index,
                color=color,
                y_format=y_format or '',
                visible=True
            )
            logger.debug(
                "[ADD_VAR] CurveInfo 已创建: var_name=%s, x_shape=%s, y_shape=%s, color=%s",
                var_name,
                x_values.shape if x_values is not None else None,
                y_values.shape if y_values is not None else None,
                color,
            )

            batch_adding = getattr(pw, '_batch_adding', False)
            # batch 模式下 header/legend 由调用方在批量循环后统一刷新，
            # 避免逐曲线重拼 legend 的 O(N²) 开销
            if not batch_adding:
                self._update_header_for_curves()
                y_arrays = pw._collect_visible_curve_arrays('y_data')
                if y_arrays:
                    combined_y = np.concatenate(y_arrays)
                    if combined_y.size:
                        all_data_min_y = np.nanmin(combined_y)
                        all_data_max_y = np.nanmax(combined_y)
                        pw._set_safe_y_range(all_data_min_y, all_data_max_y, set_limits=True)

                special_limits = pw.handle_single_point_limits(x_values, y_values)
                if special_limits:
                    min_x, max_x, min_y, max_y = special_limits
                    has_other_curves = len(pw.curves) > 1

                    if not has_other_curves:
                        pw._set_safe_y_range(min_y, max_y, set_limits=False)
                    else:
                        current_y_range = pw.view_box.viewRange()[1]
                        current_min_y, current_max_y = current_y_range
                        final_min_y = min(current_min_y, min_y)
                        final_max_y = max(current_max_y, max_y)
                        pw._set_safe_y_range(final_min_y, final_max_y, set_limits=False)
                else:
                    # v5.x 修复问题 B：用 viewRange 与新数据范围的交集，而非旧 viewRange
                    x_min, x_max = self._get_x_window_intersected(x_values)

                    new_min_y, new_max_y = pw._get_y_range_in_x_window(x_values, y_values, x_min, x_max)
                    has_other_curves = len(pw.curves) > 1

                    if not has_other_curves:
                        pw._set_safe_y_range(new_min_y, new_max_y, set_limits=False)
                    else:
                        current_y_range = pw.view_box.viewRange()[1]
                        current_min_y, current_max_y = current_y_range
                        final_min_y = min(current_min_y, new_min_y)
                        final_max_y = max(current_max_y, new_max_y)
                        pw._set_safe_y_range(final_min_y, final_max_y, set_limits=False)

            # batch 模式下跳过 vline bounds / cursor 更新 / 密度重算，
            # 由调用方在批量循环完成后统一补偿（可见性已恢复）
            if not batch_adding:
                x_arrays = pw._collect_visible_curve_arrays('x_data')
                if x_arrays:
                    combined_x = np.concatenate(x_arrays)
                    min_x, max_x = np.nanmin(combined_x), np.nanmax(combined_x)
                else:
                    min_x, max_x = np.min(x_values), np.max(x_values)
                pw._set_vline_bounds([min_x, max_x])

                pw._update_cursor_after_plot(min_x, max_x)

                if pw.vline.isVisible():
                    pw.update_cursor_label()

                pw._recalc_max_point_density()
                main_window = pw.window()
                if main_window is not None and hasattr(main_window, 'cursor_sync_manager'):
                    main_window.cursor_sync_manager._sync_min_xrange()

            return True

        except Exception as e:
            QMessageBox.critical(pw, "绘图错误", f"添加变量时发生错误: {str(e)}")
            return False
