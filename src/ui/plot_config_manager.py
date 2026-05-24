"""Plot 配置管理器 - 协调各模块的入口"""

from __future__ import annotations
from typing import Optional
from PySide6.QtCore import QObject
from src.core.config import RATIO_RESET_PLOTS
from src.core.template_manager import TemplateManager
from src.core.auto_save_manager import AutoSaveManager
from src.core.plot_config import PlotSessionConfig, PlotConfig
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
        """从 MainWindow 导出当前状态为配置对象"""
        config = PlotSessionConfig()

        # 获取布局信息
        config.layout_rows = main_window._plot_row_current
        config.layout_cols = main_window._plot_col_current

        # 获取时间修正参数
        config.time_factor = main_window.factor
        config.time_offset = main_window.offset

        # 获取各 Plot 的变量名（按 row-major 顺序）
        config.plots = []
        for container in main_window.plot_widgets:
            plot_widget = container.plot_widget
            if not container.isVisible():
                continue
            curve_names = plot_widget.curve_strategy.get_curve_names()
            if curve_names:
                config.plots.append(PlotConfig(curves=curve_names))

        return config

    def apply_config(self, main_window, config: PlotSessionConfig) -> bool:
        """将配置应用到 MainWindow"""
        try:
            logger.info(f"开始应用模板配置")
            logger.info(f"  布局: {config.layout_rows} 行 × {config.layout_cols} 列")
            logger.info(f"  时间因子: {config.time_factor}, 偏移: {config.time_offset}")
            logger.info(f"  Plot 数量: {len(config.plots)}")
            
            # 设置布局
            from src.ui.layout_manager import LayoutManager

            # 先设置布局
            logger.debug("设置布局...")
            main_window.layout_manager.set_plots_visible(
                config.layout_rows, config.layout_cols
            )

            # 设置时间修正参数
            logger.debug("设置时间修正参数...")
            main_window.factor = config.time_factor
            main_window.offset = config.time_offset

            # 检查是否有数据
            has_data = main_window.loader is not None
            logger.info(f"数据加载状态: {'已加载' if has_data else '未加载'}")
            
            if not has_data:
                logger.warning("⚠️  未加载数据，只能设置布局和时间参数，无法加载曲线")
            else:
                logger.debug(f"可用数据列数: {len(main_window.loader.var_names)}")

            # 应用各 Plot 的配置 - 只遍历可见的 container（row-major 顺序）
            applied_count = 0
            visible_containers = [c for c in main_window.plot_widgets if c.isVisible()]
            logger.debug(f"可见容器数: {len(visible_containers)}")

            for i, plot_config in enumerate(config.plots):
                if i < len(visible_containers):
                    container = visible_containers[i]
                    plot_widget = container.plot_widget
                    success = self._apply_plot_config(plot_widget, plot_config, main_window, i)
                    if success:
                        applied_count += 1
                else:
                    logger.warning(f"Plot[{i}] 没有对应的可见容器，跳过")

            logger.info(f"已成功应用 {applied_count}/{len(config.plots)} 个 Plot 配置")

            logger.info("模板配置应用完成")
            return True
        except Exception as e:
            logger.error(f"❌ 应用配置失败: {e}", exc_info=True)
            return False

    def _apply_plot_config(self, plot_widget, plot_config: PlotConfig, main_window, plot_index: int):
        """将配置应用到单个 Plot"""
        logger.debug(f"处理 Plot[{plot_index}], 曲线: {plot_config.curves}")
        
        # 先清除现有曲线
        plot_widget.clear_plot_item()

        # 根据曲线数量设置模式
        if len(plot_config.curves) == 0:
            logger.debug(f"Plot[{plot_index}] 没有曲线配置，跳过")
            return False
        elif len(plot_config.curves) == 1:
            var_name = plot_config.curves[0]
            if main_window.loader and var_name in main_window.loader.var_names:
                logger.debug(f"Plot[{plot_index}] 设置单曲线: {var_name}")
                plot_widget.plot_variable(var_name)
                return True
            else:
                if not main_window.loader:
                    logger.warning(f"Plot[{plot_index}] 无法加载曲线 '{var_name}': 未加载数据")
                else:
                    logger.warning(f"Plot[{plot_index}] 无法加载曲线 '{var_name}': 变量名不存在于数据中")
                return False
        else:
            # 多曲线模式
            # 添加每条曲线
            added_count = 0
            for var_name in plot_config.curves:
                if main_window.loader and var_name in main_window.loader.var_names:
                    logger.debug(f"Plot[{plot_index}] 添加曲线: {var_name}")
                    plot_widget.add_variable_to_plot(var_name)
                    added_count += 1
                else:
                    if not main_window.loader:
                        logger.warning(f"Plot[{plot_index}] 无法加载曲线 '{var_name}': 未加载数据")
                    else:
                        logger.warning(f"Plot[{plot_index}] 无法加载曲线 '{var_name}': 变量名不存在于数据中")
            
            return added_count > 0

    @staticmethod
    def check_template_match(config: PlotSessionConfig, current_vars: list[str]) -> tuple[float, set[str], set[str]]:
        """检查模板变量与当前数据的匹配度

        返回 (match_ratio, matched_vars, unmatched_vars)
        """
        template_vars: set[str] = set()
        for plot in config.plots:
            template_vars.update(plot.curves)

        if not template_vars:
            return 1.0, set(), set()

        current_set = set(current_vars)
        matched = template_vars & current_set
        unmatched = template_vars - current_set
        total = len(template_vars)
        ratio = len(matched) / total

        return ratio, matched, unmatched

    def apply_template(self, main_window, template_id: str) -> bool:
        """应用模板"""
        logger.info(f"========================================")
        logger.info(f"开始加载模板，ID: {template_id}")
        template = self._template_manager.get_template(template_id)
        if not template:
            logger.error(f"❌ 找不到模板: {template_id}")
            return False
        
        logger.info(f"模板名称: {template.metadata.name}")
        logger.info(f"模板描述: {template.metadata.description or '无'}")
        
        config = PlotSessionConfig.from_dict(template.config)
        result = self.apply_config(main_window, config)
        
        if result:
            logger.info(f"✅ 模板加载成功")
        else:
            logger.error(f"❌ 模板加载失败")
        logger.info(f"========================================")
        
        return result

    def save_current_as_template(
        self, main_window, name: str, description: str = ""
    ) -> Optional[PlotTemplate]:
        """保存当前配置为模板"""
        config = self.export_current_config(main_window)
        try:
            return self._template_manager.save_template(config, name, description)
        except Exception as e:
            logger.error(f"Failed to save template: {e}")
            raise

    def save_auto_save(self, main_window):
        """保存为自动恢复配置"""
        if not self._auto_save_manager.is_auto_save_enabled():
            return
        config = self.export_current_config(main_window)
        self._auto_save_manager.auto_save(config)

    def try_auto_restore(self, main_window, current_vars: list[str]) -> bool:
        """尝试自动恢复配置"""
        should_apply, reason = self._auto_save_manager.should_apply_auto_save(
            current_vars
        )
        if not should_apply:
            logger.info(f"Auto-restore skipped: {reason}")
            return False
        config = self._auto_save_manager.load_auto_save()
        if not config:
            return False
        success = self.apply_config(main_window, config)
        if success:
            self._auto_save_manager.config_applied.emit()
        return success
