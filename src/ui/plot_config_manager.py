"""Plot 配置管理器 - 协调各模块的入口"""

from __future__ import annotations
from typing import Optional
from PySide6.QtCore import QObject
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
            curves = []
            if plot_widget.is_multi_curve_mode:
                for var_name in plot_widget.curves.keys():
                    curves.append(var_name)
            elif plot_widget.y_name:
                curves.append(plot_widget.y_name)
            if curves:
                config.plots.append(PlotConfig(curves=curves))

        return config

    def apply_config(self, main_window, config: PlotSessionConfig) -> bool:
        """将配置应用到 MainWindow"""
        try:
            # 设置布局
            from src.ui.layout_manager import LayoutManager

            # 先设置布局
            main_window.layout_manager.set_plots_visible(
                config.layout_rows, config.layout_cols
            )

            # 设置时间修正参数
            main_window.factor = config.time_factor
            main_window.offset = config.time_offset

            # 应用各 Plot 的配置
            for i, plot_config in enumerate(config.plots):
                if i < len(main_window.plot_widgets):
                    container = main_window.plot_widgets[i]
                    if container.isVisible():
                        plot_widget = container.plot_widget
                        self._apply_plot_config(plot_widget, plot_config, main_window)

            # 更新时间修正显示
            main_window.update_ui_after_time_correction()

            logger.debug(f"Config applied successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to apply config: {e}")
            return False

    def _apply_plot_config(self, plot_widget, plot_config: PlotConfig, main_window):
        """将配置应用到单个 Plot"""
        # 先清除现有曲线
        plot_widget.clear_plot()

        # 根据曲线数量设置模式
        if len(plot_config.curves) == 0:
            return
        elif len(plot_config.curves) == 1:
            # 单曲线模式
            var_name = plot_config.curves[0]
            if main_window.loader and var_name in main_window.loader.df.columns:
                plot_widget.set_data(
                    units_dict=main_window.units,
                    dataframe=main_window.loader.df,
                    time_channels_info=main_window.time_channels_infos,
                    var_name=var_name,
                    factor=main_window.factor,
                    offset=main_window.offset,
                    sync_cursor=main_window.sync_cursor,
                )
        else:
            # 多曲线模式
            # 先切换到多曲线模式
            plot_widget.set_multi_curve_mode(True)
            # 添加每条曲线
            for var_name in plot_config.curves:
                if main_window.loader and var_name in main_window.loader.df.columns:
                    plot_widget.add_curve(var_name)

    def apply_template(self, main_window, template_id: str) -> bool:
        """应用模板"""
        template = self._template_manager.get_template(template_id)
        if not template:
            return False
        config = PlotSessionConfig.from_dict(template.config)
        return self.apply_config(main_window, config)

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
        config = self.export_current_config(main_window)
        self._auto_save_manager.auto_save(config)

    def try_auto_restore(self, main_window, current_vars: list[str]) -> bool:
        """尝试自动恢复配置"""
        should_apply, reason = self._auto_save_manager.should_apply_auto_save(
            current_vars
        )
        if should_apply:
            config = self._auto_save_manager.load_auto_save()
            if config:
                success = self.apply_config(main_window, config)
                if success:
                    self._auto_save_manager.config_applied.emit()
                return success
        return False, reason
