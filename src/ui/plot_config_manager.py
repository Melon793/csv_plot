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
            for cell_index, container in enumerate(main_window.plot_widgets):
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
                            cell_index=cell_index,
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

        # 身份匹配 C2（D2/§3.1）：组装完成后统一后处理采集身份。不动
        # _export_page 签名（test_template_cell_stride 单参直调），天然
        # 覆盖 legacy 替身分支；auto-save 共用本出口，自动携带。
        self._collect_curve_identities(config, main_window)

        return config

    @staticmethod
    def _collect_curve_identities(config: PlotSessionConfig, main_window) -> None:
        """按对象身份去重遍历全部 PlotConfig，逐曲线名采集跨文件身份。

        守卫：无 loader / 无 get_channel_info（CSV/Excel，D8）→ 整体跳过；
        单曲线 KeyError/AttributeError（占位名不在 loader、数据源已关闭）
        → 跳过该曲线。占位永远拿不到身份（D2 披露），跨文件回退靠 L2。
        实测 get_channel_info 零磁盘 I/O（~30μs/次），百条约 3ms。
        """
        from src.core.template_identity import identity_from_channel_info

        loader = getattr(main_window, "loader", None)
        gci = getattr(loader, "get_channel_info", None)
        if not callable(gci):
            return
        seen: set[int] = set()
        plots = list(config.plots) + [
            p for page in config.pages for p in page.plots
        ]
        for plot in plots:
            if id(plot) in seen:
                continue
            seen.add(id(plot))
            for name in plot.curves:
                try:
                    info = gci(name)
                except (KeyError, AttributeError):
                    continue
                entry = identity_from_channel_info(info)
                if entry:
                    plot.curve_identities[name] = entry

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
                # P1-7（第 5 轮）：同时记 (row, col)——cell_index 的扁平步长是
                # 导出端 built stride，跨会话/缩列后与目标页不一致会错位丢弃
                row, col = divmod(index, page.grid_cols)
                page_config.plots.append(
                    PlotConfig(
                        curves=list(curve_names),
                        invalid_curves=list(invalid_names),
                        # P1-L：记真实格子索引——空格子不进列表，套用端按
                        # enumerate 对位会错位（上轮 P1-10，多页语境下页内错位）
                        cell_index=index,
                        cell_row=row,
                        cell_col=col,
                        # 编辑器强化轮 D5/D6：显示名/换算随模板（含占位）
                        curve_settings=PlotConfigManager._collect_curve_settings(plot_widget),
                    )
                )
        return page_config

    @staticmethod
    def _collect_curve_settings(plot_widget) -> dict:
        """收集逐曲线显示名/换算（仅非默认键；占位一并存，D6）。"""
        settings: dict = {}
        sources = list((getattr(plot_widget, "curves", {}) or {}).items()) + list(
            (getattr(plot_widget, "invalid_curves", {}) or {}).items()
        )
        for name, ci in sources:
            entry: dict = {}
            dn = getattr(ci, "display_name", "")
            if dn:
                entry["display_name"] = dn
            f = getattr(ci, "factor", 1.0)
            o = getattr(ci, "offset", 0.0)
            if f != 1.0:
                entry["factor"] = f
            if o != 0.0:
                entry["offset"] = o
            if entry:
                settings[name] = entry
        return settings

    @staticmethod
    def _apply_curve_settings(plot_widget, settings: dict) -> None:
        """静默还原显示名/换算（D5/D6）：直写字段 + 从 y_raw 重投影。

        套用是程序化批量路径，不走 apply_curve_* 入口（避免逐条播报/emit）；
        legend 由调用方收尾统一刷新。占位只写字段（复活时构造点消费）。
        """
        from src.core.curve_transform import reproject
        from src.core.data_types import step_mode_for

        for name, entry in (settings or {}).items():
            ci = plot_widget.curves.get(name)
            ref = (getattr(plot_widget, "invalid_curves", {}) or {}).get(name)
            target = ci if ci is not None else ref
            if target is None:
                continue
            if "display_name" in entry:
                target.display_name = entry["display_name"]
            if "factor" in entry or "offset" in entry:
                target.factor = entry.get("factor", 1.0)
                target.offset = entry.get("offset", 0.0)
                if ci is not None:
                    reproject(ci)
                    if ci.curve is not None:
                        try:
                            ci.curve.setData(
                                ci.x_data, ci.y_data,
                                stepMode=step_mode_for(ci.draw_mode),
                            )
                        except Exception:
                            logger.debug(
                                "模板还原换算失败: var=%s", name, exc_info=True
                            )

    def _reconcile_pages(
        self, main_window, page_count: int
    ) -> tuple[int, str | None]:
        """让页数量与模板一致，返回实际页数及失败原因。"""
        if not hasattr(main_window, "pages"):
            return 0, None

        needs_new_page = len(main_window.pages) < page_count
        if needs_new_page and main_window.loader is None:
            return len(main_window.pages), "未加载数据，无法新建模板中的绘图页"
        if needs_new_page:
            max_pages = getattr(main_window, "MAX_PLOT_PAGES", None)
            if max_pages is not None and page_count > max_pages:
                return (
                    len(main_window.pages),
                    f"模板需要 {page_count} 页，最多支持 {max_pages} 页",
                )

        while len(main_window.pages) > page_count:
            previous_count = len(main_window.pages)
            main_window.close_page(main_window.pages[-1], confirm=False)
            if len(main_window.pages) >= previous_count:
                return len(main_window.pages), "关闭多余绘图页失败"
        while len(main_window.pages) < page_count:
            if main_window.new_page() is None:
                return len(main_window.pages), "新建绘图页失败"
        return len(main_window.pages), None

    def _apply_page_title(self, main_window, page, title: str) -> None:
        if not title:
            return
        # 页标题是统计窗口分组键（StatRow.key 首段）：交互重命名入口已拒绝
        # 重名（main_window._on_tab_rename_requested，3cdb64c），模板路径同样
        # 要设防——手写/编辑器产出的重名模板会让两页统计行写进同一 key，
        # 表格静默显示错误数字（P1-G）。程序化路径不弹框：撞名自动加 " (N)"
        # 后缀，与 import_template 的后缀策略对齐。
        final = title
        suffix = 2
        pages = list(getattr(main_window, "pages", []) or [])
        while any(other is not page and other.title == final for other in pages):
            final = f"{title} ({suffix})"
            suffix += 1
        page.title = final
        tabs = getattr(main_window, "plot_tabs", None)
        if tabs is not None:
            index = tabs.indexOf(page.widget)
            if index >= 0:
                tabs.setTabText(index, final)

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

    def apply_config(
        self, main_window, config: PlotSessionConfig, *, identity_snapshot=None
    ) -> bool:
        """将配置应用到 MainWindow（模板 v2：逐页应用）。"""
        try:
            from src.core.template_identity import resolve_identity_snapshot

            snapshot = identity_snapshot or resolve_identity_snapshot(
                config, getattr(main_window, "loader", None)
            )
            config = snapshot.config
            remap = snapshot.remap
            demoted = snapshot.demoted
            if remap:
                logger.info("模板身份重映射 %d 条: %s", len(remap), remap)
            if demoted:
                logger.info("身份矛盾降占位 %d 条: %s", len(demoted), demoted)
            page_cfgs = config.normalized_pages()
            logger.info("开始应用模板配置")
            logger.info("  页数: %d", len(page_cfgs))
            logger.info("  时间因子: %s, 偏移: %s", config.time_factor, config.time_offset)

            actual_pages, failure_reason = self._reconcile_pages(
                main_window, len(page_cfgs)
            )
            if failure_reason is not None:
                logger.warning("模板页数核对失败: %s", failure_reason)
                broadcast = getattr(main_window, "_broadcast", None)
                if callable(broadcast):
                    broadcast(f"模板恢复失败：{failure_reason}", level="warn")
                return False
            pages = list(getattr(main_window, "pages", []) or [])
            if not pages:
                # 旧式替身：保留单页路径
                return self._apply_single_page_config(main_window, config)
            if actual_pages != len(page_cfgs):
                logger.warning(
                    "模板页数核对失败: 实际 %d 页，配置 %d 页",
                    actual_pages, len(page_cfgs),
                )
                return False

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
                    # P1-L：优先按 cell_index 落位（导出端记录的真实格子索引）；
                    # 旧档无该字段（None）时回退 enumerate 对位（旧行为，零回归）
                    cell = plot_config.cell_index
                    # P1-7（第 5 轮）：row/col 与 stride 无关，优先按目标页
                    # built stride 换算，修复跨会话/缩列后的错位与越界丢弃
                    if (
                        plot_config.cell_row is not None
                        and plot_config.cell_col is not None
                    ):
                        # 评审 B2：目标机容量钳制（cols_max < 模板列数）时
                        # col 可能 ≥ grid_cols，直接换算会回绕到下一行错位
                        # 覆盖；越界显式跳过（row 越界由下方 cell 校验兜底）
                        if (
                            plot_config.cell_row < 0
                            or plot_config.cell_col < 0
                            or plot_config.cell_col >= page.grid_cols
                        ):
                            logger.warning(
                                "Plot[%d] cell_row/col=%s/%s 超出目标页网格，跳过",
                                index, plot_config.cell_row, plot_config.cell_col,
                            )
                            continue
                        cell = (
                            plot_config.cell_row * page.grid_cols
                            + plot_config.cell_col
                        )
                    if cell is not None:
                        if (
                            cell < 0
                            or cell >= len(page.plot_widgets)
                            or not page.cell_visible(cell)
                        ):
                            logger.warning(
                                "Plot[%d] cell_index=%s 超出页面可见格子，跳过",
                                index, cell,
                            )
                            continue
                        container = page.plot_widgets[cell]
                    else:
                        if index >= len(visible_containers):
                            logger.warning("Plot[%d] 超出页面可见格子，跳过", index)
                            continue
                        container = visible_containers[index]
                    plot_widget = container.plot_widget
                    if self._apply_plot_config(
                        plot_widget, plot_config, main_window, index, scene=scene
                    ):
                        applied_count += 1

            # 全局时间基准：所有页都要重算数据层，隐藏页由 scene=False 分级
            main_window.factor = config.time_factor
            main_window.offset = config.time_offset
            # 状态栏 x 轴段是缓存文案，只在加载收尾/抽屉应用时重算；模板与
            # auto-save 套用同样改了全局基准，必须同规格补刷，否则抽屉
            # 现读 mw.factor 显示 0.1、底部还停在"x轴：Index"
            self._refresh_axis_segment(main_window)

            needs_time_recompute = (
                config.time_factor != 1.0 or config.time_offset != 0.0
            )
            if not needs_time_recompute:
                # 模板/auto-save 恢复默认基准时也要校正：新建页可能在建图时
                # 继承了旧 factor，而原先只按“模板值是否非默认”决定是否重算，
                # 会把整场恢复停在旧基准（曲线 X 与 mw.factor 不一致）。
                for page in pages:
                    for container in page.plot_widgets:
                        widget = getattr(container, "plot_widget", None)
                        if widget is None:
                            continue
                        if (
                            abs(
                                float(getattr(widget, "factor", config.time_factor))
                                - config.time_factor
                            ) > 1e-12
                            or abs(
                                float(getattr(widget, "offset", config.time_offset))
                                - config.time_offset
                            ) > 1e-12
                        ):
                            needs_time_recompute = True
                            break
                    if needs_time_recompute:
                        break

            if needs_time_recompute:
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
                # 区间在保存时存在 ⇒ 当时标记区是开着的，恢复时一并打开。
                # P1-J：不能在这里同步 toggle——activate_page 只排了 0ms 定时器，
                # 此刻视图还是套用前的旧视野，toggle_mark_region 会把落在旧视野外
                # 的模板区间钳成旧视野中间 1/3，对齐后钳制值又经 restore_mark_regions
                # 回写全局，模板区间永久丢失。改为挂 pending 标志，由
                # _post_activation 在 _range_dirty 对齐之后消费。
                main_window.saved_mark_range = tuple(config.saved_mark_range)
                main_window._pending_mark_region_restore = True

            logger.info(
                "已成功应用 %d/%d 个 Plot 配置（%d 页）",
                applied_count, total_plots, len(pages),
            )
            return True
        except Exception as e:
            logger.error("❌ 应用配置失败: %s", e, exc_info=True)
            return False

    @staticmethod
    def _refresh_axis_segment(main_window) -> None:
        """补刷状态栏 x 轴段（缓存文案）。

        替身可无该方法；补刷是装饰性的，控件已销毁等异常也不能把已经
        落地的模板套用变成「应用失败」——它发生在 factor/offset 写入之后。
        """
        update = getattr(main_window, "_update_axis_segment", None)
        if not callable(update):
            return
        try:
            update()
        except Exception:
            logger.debug("补刷状态栏 x 轴段失败", exc_info=True)

    def _apply_single_page_config(self, main_window, config: PlotSessionConfig) -> bool:
        """旧式替身（无 pages）：保持原先单页应用路径。"""
        main_window.layout_manager.set_plots_visible(
            config.layout_rows, config.layout_cols
        )
        main_window.factor = config.time_factor
        main_window.offset = config.time_offset
        self._refresh_axis_segment(main_window)

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
                    # quiet=True：套用是程序化批量路径，失败不在中途弹模态框
                    if single_scene_first and i == 0:
                        ok = plot_widget.plot_variable(var_name, quiet=True)
                    else:
                        ok = plot_widget.add_variable_to_plot(var_name, quiet=True)
                    if ok:
                        added_count += 1
                    elif var_name in plot_widget.curves:
                        # 畸形模板的重名条目（curves=["a","a"]）：失败原因是
                        # 「已在绘图中」而非「不可绘制」，不得 demote——demote
                        # 会经 _remove_single_curve_light 把已画好的曲线移除。
                        # 与下方 invalid_names 循环的既有防护同口径。
                        logger.debug(
                            "Plot[%d] 曲线 '%s' 在模板中重复，跳过",
                            plot_index, var_name,
                        )
                    else:
                        # P1-I：变量存在于数据但不可绘制（文本列 validity=-1、
                        # 全哨兵列被过滤成全 NaN 等）——与「不存在」同归宿降级
                        # 为占位，且不计数（修复前失败仍 added_count+=1，
                        # 「已应用 N/M」虚高、格子静默留空）。
                        logger.warning(
                            "Plot[%d] 曲线 '%s' 存在于数据但绘制失败，降级为占位",
                            plot_index, var_name,
                        )
                        plot_widget.demote_to_invalid(
                            var_name, INVALID_CURVE_COLOR, True
                        )
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

            # 编辑器强化轮 D5/D6：静默还原显示名/换算（含占位），legend 收尾统一刷
            curve_settings = getattr(plot_config, "curve_settings", {}) or {}
            if curve_settings:
                self._apply_curve_settings(plot_widget, curve_settings)
                # F4（D15）：上面按默认换算画完曲线后才套换算，Y limits 仍是
                # raw 范围；此处按投影数据重算并让视野纳入新数据（模板套用
                # 是从零还原视图，允许扩张）。隐藏页由 _finalize_batch_add 收尾。
                if scene:
                    try:
                        plot_widget._multi_curve_manager._update_axes_for_multi_curve()
                    except Exception:
                        logger.debug(
                            "Plot[%d] 模板换算后 Y 范围重算失败",
                            plot_index, exc_info=True,
                        )

            # scene 路径的曲线绘制已各自刷新过 header；若新增了占位或还原了
            # 显示名，需再刷新一次让占位/自定义名进图例（demote/直写不刷 header）。
            # 非活动页由 finally 的 _finalize_batch_add(update_header=True) 统一刷新。
            if scene and (
                getattr(plot_widget, "invalid_curves", None) or curve_settings
            ):
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

        try:
            config = PlotSessionConfig.from_dict(template.config)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            # P1-8（第 5 轮）：手写/损坏档的非法 int 字段不得让异常逃出 Qt 槽
            logger.error("模板[%s]配置解析失败: %s", template_id, exc)
            return False

        from src.core.template_identity import (
            merge_demoted,
            resolve_identity_snapshot,
        )

        snapshot = resolve_identity_snapshot(config, getattr(main_window, "loader", None))
        config = snapshot.config
        demoted = snapshot.demoted
        if check_match and current_vars is not None:
            # 身份匹配 D12/MN-1：与 _check_and_apply_template 同口径，
            # resolve 前置后再算匹配度（重映射计入 matched，D5；
            # 降占位计入失配，M-1 方案乙）。
            ratio, matched, unmatched = self.check_template_match(
                config, current_vars
            )
            ratio, matched, unmatched = merge_demoted(matched, unmatched, demoted)
            if ratio < RATIO_RESET_PLOTS:
                logger.warning(
                    "模板[%s]匹配度 %.0f%% < %.0f%%",
                    template.metadata.name, ratio * 100, RATIO_RESET_PLOTS * 100,
                )
                return False

        logger.info("模板名称: %s", template.metadata.name)
        logger.info("模板描述: %s", template.metadata.description or "无")

        result = self.apply_config(
            main_window, config, identity_snapshot=snapshot
        )

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
        if not self._auto_save_manager.auto_save(config):
            main_window._broadcast(
                "自动恢复配置保存失败，当前布局未能写入", level="error"
            )
