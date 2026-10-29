"""Plot 配置管理器 - 协调各模块的入口"""

from __future__ import annotations
from contextlib import ExitStack, contextmanager
from typing import Optional
from PySide6.QtCore import QObject, QSignalBlocker
from src.core.config import RATIO_RESET_PLOTS, INVALID_CURVE_COLOR
from src.core.template_manager import TemplateManager
from src.core.auto_save_manager import AutoSaveManager
from src.core.plot_config import (
    PlotSessionConfig,
    PlotConfig,
    PlotPageConfig,
)
from src.core.template_models import PlotTemplate
from src.core.logger import get_logger


logger = get_logger(__name__)


class PlotConfigManager(QObject):
    """Plot 配置管理器 - 作为 MainWindow 和下层模块之间的桥梁"""

    def __init__(self):
        super().__init__()
        self._template_manager = TemplateManager()
        self._auto_save_manager = AutoSaveManager()

    @property
    def template_manager(self) -> TemplateManager:
        return self._template_manager

    @property
    def auto_save_manager(self) -> AutoSaveManager:
        return self._auto_save_manager

    def export_current_config(self, main_window) -> PlotSessionConfig:
        """从 MainWindow 导出当前状态为配置对象（模板 v2，设计文档 §6.5）。

        关键在于可见性口径：隐藏页整体不可见，``isVisible()`` 恒为 False，
        必须用 ``isHidden()`` + 页的显式行列，否则整页会被导成空。
        """
        config = PlotSessionConfig(
            time_factor=main_window.factor,
            time_offset=main_window.offset,
        )

        pages = list(getattr(main_window, "pages", []) or [])
        if pages:
            for page in pages:
                config.pages.append(self._export_page(page))
            current = getattr(main_window, "current_page", None)
            config.active_page = max(0, main_window.page_index(current))
            # 顶层字段是旧版软件的回退视图（第 1 页）
            first = config.pages[0]
            config.layout_rows = first.layout_rows
            config.layout_cols = first.layout_cols
            config.plots = list(first.plots)
        else:
            # 旧式替身（没有 pages）：保持单页导出行为
            config.layout_rows = main_window._plot_row_current
            config.layout_cols = main_window._plot_col_current
            plots = []
            for container in main_window.plot_widgets:
                if container.isHidden():
                    continue
                strategy = container.plot_widget.curve_strategy
                curve_names = strategy.get_curve_names()
                invalid_names = strategy.get_invalid_curve_names()
                if curve_names or invalid_names:
                    plots.append(
                        PlotConfig(
                            curves=list(curve_names),
                            invalid_curves=list(invalid_names),
                        )
                    )
            config.plots = plots
            config.pages = [
                PlotPageConfig(
                    title="图 1",
                    layout_rows=config.layout_rows,
                    layout_cols=config.layout_cols,
                    plots=list(config.plots),
                )
            ]

        # 全局视图态（可选字段：缺省不写，应用时不覆盖当前值）
        cursor_btn = getattr(main_window, "cursor_btn", None)
        if cursor_btn is not None and cursor_btn.isChecked():
            config.cursor_mode = getattr(main_window, "cursor_mode", None)
        mark_btn = getattr(main_window, "mark_region_btn", None)
        saved_range = getattr(main_window, "saved_mark_range", None)
        if mark_btn is not None and mark_btn.isChecked() and saved_range:
            config.saved_mark_range = [float(saved_range[0]), float(saved_range[1])]

        return config

    @staticmethod
    def _export_page(page) -> PlotPageConfig:
        page_config = PlotPageConfig(
            title=getattr(page, "title", ""),
            layout_rows=int(page.rows_current),
            layout_cols=int(page.cols_current),
            row_height_factors=dict(page.row_height_factors or {}),
        )
        for index, container in enumerate(page.plot_widgets):
            if container.isHidden():
                continue
            if hasattr(page, "cell_visible") and not page.cell_visible(index):
                continue
            plot_widget = getattr(container, "plot_widget", None)
            if plot_widget is None:
                continue
            strategy = plot_widget.curve_strategy
            curve_names = strategy.get_curve_names()
            invalid_names = strategy.get_invalid_curve_names()
            # 判空口径含占位（§8.2/R16）：纯占位格子也是用户资产，不得静默丢弃
            if curve_names or invalid_names:
                page_config.plots.append(
                    PlotConfig(
                        curves=list(curve_names),
                        invalid_curves=list(invalid_names),
                    )
                )
        return page_config

    def _reconcile_pages(self, main_window, page_count: int) -> None:
        """让页数量与模板一致：复用已有页、多的删、少的建（§6.5）。"""
        if not hasattr(main_window, "pages"):
            return
        while len(main_window.pages) > page_count:
            main_window.close_page(main_window.pages[-1], confirm=False)
        while len(main_window.pages) < page_count:
            if main_window.loader is None:
                logger.warning("未加载数据，无法新建模板中的绘图页")
                return
            if main_window.new_page() is None:
                logger.warning("新建绘图页失败，模板页数将被截断")
                return

    def _apply_page_title(self, main_window, page, title: str) -> None:
        if not title:
            return
        page.title = title
        tabs = getattr(main_window, "plot_tabs", None)
        if tabs is not None:
            index = tabs.indexOf(page.widget)
            if index >= 0:
                tabs.setTabText(index, title)

    @staticmethod
    @contextmanager
    def _quiet_cursor_signals(plot_widget):
        """隐藏页的清除/重建必须静默执行（契约 2）。

        ``clear_plot_item`` → ``apply_cursor_x_domain`` → ``_set_vline_bounds``
        内部的 ``setBounds`` 会把越界 vline 钳到新边界并 emit
        ``sigPositionChanged``；pinned 状态下这条回调会把钳制值写回全局
        ``pinned_x_values``，把用户游标悄悄挪走。信号阻断 + 抑制回写双保险。
        """
        previous = getattr(plot_widget, "_suppress_pin_update", False)
        plot_widget._suppress_pin_update = True
        try:
            with ExitStack() as stack:
                for attr in ("vline", "vline2"):
                    line = getattr(plot_widget, attr, None)
                    if line is None:
                        continue
                    try:
                        stack.enter_context(QSignalBlocker(line))
                    except TypeError:
                        # 组件测试替身非 QObject：抑制标志仍生效
                        pass
                yield
        finally:
            plot_widget._suppress_pin_update = previous

    def apply_config(self, main_window, config: PlotSessionConfig) -> bool:
        """将配置应用到 MainWindow（模板 v2：逐页应用）。"""
        try:
            page_cfgs = config.normalized_pages()
            logger.info("开始应用模板配置")
            logger.info("  页数: %d", len(page_cfgs))
            logger.info("  时间因子: %s, 偏移: %s", config.time_factor, config.time_offset)

            self._reconcile_pages(main_window, len(page_cfgs))
            pages = list(getattr(main_window, "pages", []) or [])
            if not pages:
                # 旧式替身：保留单页路径
                return self._apply_single_page_config(main_window, config)

            current = getattr(main_window, "current_page", None)
            applied_count = 0
            total_plots = 0

            for page, page_config in zip(pages, page_cfgs):
                # 模板 = 整套替换曲线：恢复后视图对齐数据域，
                # 由页激活收尾 _post_activation 消费（覆盖 saved_x_range 跟随值）
                page._range_dirty = True
                self._apply_page_title(main_window, page, page_config.title)

                factors = {
                    int(row): int(value)
                    for row, value in (page_config.row_height_factors or {}).items()
                }
                if factors:
                    page.row_height_factors = factors
                page.apply_visibility(page_config.layout_rows, page_config.layout_cols)

                # 模板 = 完整还原：页内未列出的可见格子也要清掉旧曲线与旧占位
                visible_containers = page.visible_containers()
                scene = page is current
                for container in visible_containers:
                    if scene:
                        container.plot_widget.clear_plot_item(clear_invalid=True)
                    else:
                        with self._quiet_cursor_signals(container.plot_widget):
                            container.plot_widget.clear_plot_item(clear_invalid=True)

                for index, plot_config in enumerate(page_config.plots):
                    total_plots += 1
                    if index >= len(visible_containers):
                        logger.warning("Plot[%d] 超出页面可见格子，跳过", index)
                        continue
                    plot_widget = visible_containers[index].plot_widget
                    if self._apply_plot_config(
                        plot_widget, plot_config, main_window, index, scene=scene
                    ):
                        applied_count += 1

            # 全局时间基准：所有页都要重算数据层，隐藏页由 scene=False 分级
            main_window.factor = config.time_factor
            main_window.offset = config.time_offset
            if config.time_factor != 1.0 or config.time_offset != 0.0:
                for page in pages:
                    is_current = page is current
                    for container in page.plot_widgets:
                        container.plot_widget.update_time_correction(
                            config.time_factor,
                            config.time_offset,
                            scene=is_current,
                            sync_min_xrange=False,
                        )
                # 全部页的局部密度都刷新后统一汇总一次；目标页的 limits
                # 由随后 activate_page 的场景恢复按新全局密度重写。
                cursor_sync = getattr(main_window, "cursor_sync_manager", None)
                if cursor_sync is not None:
                    cursor_sync._sync_min_xrange()

            target = pages[min(max(config.active_page, 0), len(pages) - 1)]
            main_window.activate_page(target)

            # 可选视图态：缺省保留当前值，不覆盖。放在激活之后，
            # set_cursor_mode / toggle_mark_region 的参考视图才是目标页。
            if config.cursor_mode is not None:
                main_window.cursor_mode = config.cursor_mode
                cursor_sync = getattr(main_window, "cursor_sync_manager", None)
                if cursor_sync is not None:
                    cursor_sync.set_cursor_mode(config.cursor_mode)
            if config.saved_mark_range:
                # 区间在保存时存在 ⇒ 当时标记区是开着的，恢复时一并打开
                main_window.saved_mark_range = tuple(config.saved_mark_range)
                main_window.layout_manager.toggle_mark_region(True)

            logger.info(
                "已成功应用 %d/%d 个 Plot 配置（%d 页）",
                applied_count, total_plots, len(pages),
            )
            return True
        except Exception as e:
            logger.error("❌ 应用配置失败: %s", e, exc_info=True)
            return False

    def _apply_single_page_config(self, main_window, config: PlotSessionConfig) -> bool:
        """旧式替身（无 pages）：保持原先单页应用路径。"""
        main_window.layout_manager.set_plots_visible(
            config.layout_rows, config.layout_cols
        )
        main_window.factor = config.time_factor
        main_window.offset = config.time_offset

        applied_count = 0
        visible_containers = [c for c in main_window.plot_widgets if not c.isHidden()]
        for i, plot_config in enumerate(config.plots):
            if i < len(visible_containers):
                if self._apply_plot_config(
                    visible_containers[i].plot_widget, plot_config, main_window, i
                ):
                    applied_count += 1
        logger.info("已成功应用 %d/%d 个 Plot 配置", applied_count, len(config.plots))
        return True

    def _apply_plot_config(
        self, plot_widget, plot_config: PlotConfig, main_window, plot_index: int,
        *, scene: bool = True,
    ):
        """将配置应用到单个 Plot；隐藏页只建数据层（scene=False）。

        模板 = 完整还原（§7）：开头 clear 连带清占位（clear_invalid=True）；
        ``curves`` 中变量存在则绘制、不存在则降级为占位（固定灰色，§8.3/§12.2）；
        ``invalid_curves`` 忠实还原为占位。占位不参与 applied_count（§8.3）。
        """
        curve_names = list(plot_config.curves)
        invalid_names = list(getattr(plot_config, "invalid_curves", []) or [])
        logger.debug(
            "处理 Plot[%d], 曲线: %s, 占位: %s", plot_index, curve_names, invalid_names
        )

        # 先清除现有曲线与占位（隐藏页静默：clear 链的 setBounds 钳制不得回写全局 pin）
        if scene:
            plot_widget.clear_plot_item(clear_invalid=True)
        else:
            with self._quiet_cursor_signals(plot_widget):
                plot_widget.clear_plot_item(clear_invalid=True)

        if not curve_names and not invalid_names:
            logger.debug("Plot[%d] 没有曲线配置，跳过", plot_index)
            return False

        has_loader = bool(main_window.loader)
        available = set(main_window.loader.var_names) if has_loader else set()
        # 单曲线首绘走 plot_variable（保留既有首绘语义），其余走批量入口
        single_scene_first = scene and len(curve_names) == 1

        if not scene:
            # 隐藏页：只建数据层；游标/标签收尾由切页时的 restore_scene 补
            plot_widget._batch_adding = True
        try:
            added_count = 0
            for i, var_name in enumerate(curve_names):
                if has_loader and var_name in available:
                    logger.debug("Plot[%d] 添加曲线: %s", plot_index, var_name)
                    if single_scene_first and i == 0:
                        plot_widget.plot_variable(var_name)
                    else:
                        plot_widget.add_variable_to_plot(var_name)
                    added_count += 1
                else:
                    # 变量不存在 → 降级为占位（模板来源落固定灰色，不用
                    # _assign_curve_color：避免同一模板多次应用占位色随索引漂移）
                    if not has_loader:
                        logger.warning("Plot[%d] 无法加载曲线 '%s': 未加载数据", plot_index, var_name)
                    else:
                        logger.warning("Plot[%d] 曲线 '%s' 不存在于数据中，降级为占位", plot_index, var_name)
                    plot_widget.demote_to_invalid(var_name, INVALID_CURVE_COLOR, True)

            # 模板自带占位：忠实还原为占位（固定灰色）；与 curves 天然互斥，
            # 已绘制成曲线的（手工编辑的畸形模板）跳过，避免同名两侧共存
            for var_name in invalid_names:
                if var_name in plot_widget.curves:
                    continue
                plot_widget.demote_to_invalid(var_name, INVALID_CURVE_COLOR, True)

            # scene 路径的曲线绘制已各自刷新过 header；若新增了占位，需再刷新
            # 一次让占位条目进图例（demote 是纯数据操作，不刷 header，§4.2）。
            # 非活动页由 finally 的 _finalize_batch_add(update_header=True) 统一刷新。
            if scene and getattr(plot_widget, "invalid_curves", None):
                plot_widget._update_header_for_curves()

            return added_count > 0
        finally:
            if not scene:
                plot_widget._batch_adding = False
                try:
                    plot_widget._multi_curve_manager._finalize_batch_add(
                        update_header=True,
                        sync_min_xrange=False,
                        scene=False,
                    )
                except Exception:
                    logger.debug(
                        "Plot[%d] 隐藏页批量收尾失败，留待切页补建",
                        plot_index, exc_info=True,
                    )
                plot_widget.mark_dirty_for_activation()

    @staticmethod
    def check_template_match(config: PlotSessionConfig, current_vars: list[str]) -> tuple[float, set[str], set[str]]:
        """检查模板变量与当前数据的匹配度

        返回 (match_ratio, matched_vars, unmatched_vars)
        """
        template_vars: set[str] = set()
        for page in config.normalized_pages():
            for plot in page.plots:
                template_vars.update(plot.curves)

        if not template_vars:
            return 1.0, set(), set()

        current_set = set(current_vars)
        matched = template_vars & current_set
        unmatched = template_vars - current_set
        total = len(template_vars)
        ratio = len(matched) / total

        return ratio, matched, unmatched

    def apply_template(
        self, main_window, template_id: str,
        check_match: bool = False, current_vars: list[str] | None = None
    ) -> bool:
        logger.info("========================================")
        logger.info("开始加载模板，ID: %s", template_id)
        template = self._template_manager.get_template(template_id)
        if not template:
            logger.error("❌ 找不到模板: %s", template_id)
            return False

        config = PlotSessionConfig.from_dict(template.config)

        if check_match and current_vars is not None:
            ratio, matched, unmatched = self.check_template_match(
                config, current_vars
            )
            if ratio < RATIO_RESET_PLOTS:
                logger.warning(
                    "模板[%s]匹配度 %.0f%% < %.0f%%",
                    template.metadata.name, ratio * 100, RATIO_RESET_PLOTS * 100,
                )
                return False

        logger.info("模板名称: %s", template.metadata.name)
        logger.info("模板描述: %s", template.metadata.description or "无")

        result = self.apply_config(main_window, config)

        if result:
            logger.info("✅ 模板加载成功")
        else:
            logger.error("❌ 模板加载失败")
        logger.info("========================================")

        return result

    def save_current_as_template(
        self, main_window, name: str, description: str = ""
    ) -> Optional[PlotTemplate]:
        """保存当前配置为模板"""
        config = self.export_current_config(main_window)
        try:
            return self._template_manager.save_template(config, name, description)
        except Exception as e:
            logger.error("Failed to save template: %s", e)
            raise

    def save_auto_save(self, main_window):
        """保存为自动恢复配置"""
        if not self._auto_save_manager.is_auto_save_enabled():
            return
        config = self.export_current_config(main_window)
        self._auto_save_manager.auto_save(config)
