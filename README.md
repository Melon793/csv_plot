# CSV Plot (PyQt6)

一个基于 PyQt6 + pyqtgraph 的高性能交互式数据可视化工具，支持 CSV 和 MDF（ASAM MDF4/DAT）格式文件，专为快速加载、浏览和分析时序数据而设计。支持百万级数据点的流畅绘制和多曲线模式。

## 主要特性

### 数据加载与显示
- **多格式支持**: CSV / MDF（.mf4 / .mdf / .dat）、TSV 及兼容格式
- **MDF 专属**: 枚举类型通道自动映射文本标签、多 Channel Group 聚合、跨 Group 同名变量智能后缀
- **拖拽加载**: 直接将文件拖拽到窗口中即可加载
- **智能解析**: CSV 自动检测编码、分隔符、标题行/单位行；MDF 自动识别枚举通道
- **大数据支持**: 后台线程加载，进度显示，支持百万级数据点
- **数据质量指示**:
  - 🟢 绿色：变化的有效数值
  - 🟡 黄色：无变化的有效数值（常数）
  - 🔴 红色：非有效数值
- **快速浏览**: 双击变量名查看详细数值表格

### 强大的图表功能
- **拖拽式绘图**: 从变量列表拖拽变量到绘图区域即可绘图
- **多曲线模式**: 支持在同一图表中同时显示多条曲线，自动颜色分配
- **变量编辑器**: 可视化编辑曲线顺序、颜色、可见性
- **灵活布局**: 支持多子图网格排列（行列数可配置），可自由调整布局
- **同步缩放**: 所有图表 X 轴自动同步（XLink），支持鼠标交互缩放
- **高性能渲染**:
  - 智能降采样（peak 模式保留峰值特征）
  - 交互期间自适应降低样式更新频率
  - 支持百万级数据点流畅绘制
  - 曲线样式自动切换：缩放至细节时自动显示符号点
- **精确控制**:
  - 鼠标滚轮缩放 X 轴（以鼠标位置为中心）
  - 鼠标左键拖拽移动视图
  - 双击中键清除单个图表
  - Ctrl/Shift + 滚轮缩放特定轴
- **右键菜单**: 跳转到数据、自动调节 Y 轴、游标模式、调整行高等
- **坐标轴设置**: 双击坐标轴标签可设置范围/刻度

### 游标与交互功能
- **十字游标**: 实时显示当前鼠标位置的 X/Y 坐标值
- **多游标模式**: 1 个自由游标 / 1 个锚定游标 / 2 个锚定游标
- **多曲线游标**: 在多曲线模式下同时显示所有曲线在当前 X 位置的值
- **MDF 枚举游标**: 枚举通道自动显示文本标签而非原始整数值
- **同步游标**: 多个图表之间的游标位置自动同步
- **交互优化**: 缩放/拖动期间自动禁用游标更新

### 标记与统计分析
- **区间标记**: 在图表上标记感兴趣的数据区间
- **自动统计**: 实时计算标记区域的统计值（最小/最大/平均值/斜率等）
- **标记统计窗口**: 汇总显示所有图表的标记统计信息

### 智能数据表格
- **快速访问**: 双击变量名或点击「Jump to Data」打开表格
- **冻结列功能**: 支持列冻结，方便对比查看
- **智能高亮**: 选中单元格高亮 + 所在行/列浅蓝高亮
- **XY 散点图**: 选中两列数据可快速绘制散点图
- **同步滚动**: 冻结列与非冻结列保持同步

## 项目结构

```
csv_plot/
├── csv_plot_pyqt6.py            # 主入口（~6600 行，3 个核心类）
├── pyproject.toml               # 项目依赖与构建配置
├── README.md
├── scripts/                     # 打包脚本
│   ├── build_exe_pyinstaller
│   └── build_exe_nuitka
├── assets/                      # 图标资源
│   ├── icon.png
│   ├── icon.ico
│   └── icon.icns
├── data/                        # 示例数据
└── src/                         # 模块化源码
    ├── app/
    │   └── plot_context.py      # PlotContext 服务层（依赖注入）
    ├── core/
    │   ├── config.py            # 全局常量、调试钩子、float32 安全检查
    │   ├── types.py             # AutoDetectError / FormatInfo / CurveInfo 数据类型
    │   └── scheduler.py         # UnifiedUpdateScheduler 防抖调度器
    ├── data/
    │   ├── loader.py            # FastDataLoader CSV 加载 + DataLoadThread 后台线程
    │   └── mdf_loader.py        # MDFDataLoader MDF4/DAT 加载 + 枚举通道支持
    └── ui/
        ├── drag_drop.py         # 拖放解析（parse_var_names / build_mimedata / create_pixmap）
        ├── table_dialog.py      # DataTableDialog + PandasTableModel + CustomDelegate + XYScatterPlotDialog
        ├── variable_list.py     # MyTableWidget 变量列表面板
        ├── mark_stats.py        # MarkStatsWindow 标记统计窗口
        ├── plot_variable_editor.py  # PlotVariableEditorDialog 变量编辑器
        ├── main_window_base_manager.py  # MainWindow 基础管理器
        ├── file_loader_manager.py  # 文件加载管理器
        ├── cursor_sync_manager.py  # 游标同步管理器
        ├── layout_manager.py    # 布局管理器
        ├── dialogs/
        │   ├── help.py          # HelpDialog 帮助文档
        │   ├── layout_input.py  # LayoutInputDialog 行列数配置
        │   ├── axis.py          # AxisDialog 坐标轴设置
        │   └── time_correction.py  # TimeCorrectionDialog 时间修正
        └── widgets/
            ├── __init__.py
            ├── base_manager.py  # 管理器基类
            ├── custom_viewbox.py # 信号化 CustomViewBox（10 个信号解耦 MainWindow）
            ├── plot_container.py # PlotContainerWidget 绘图容器
            ├── plot_ui_manager.py # PlotUIManager 绘图 UI 管理器
            ├── plot_data_manager.py # PlotDataManager 绘图数据管理器
            ├── axis_manager.py  # AxisManager 坐标轴管理器
            ├── cursor_manager.py # CursorManager 游标管理器
            ├── multi_curve_manager.py # MultiCurveManager 多曲线管理器
            ├── mark_region_manager.py # MarkRegionManager 标记区域管理器
            └── event_handler.py # EventHandler 事件处理器
```

## 快速开始

### 系统要求
- Python 3.12 或更高版本
- 支持的操作系统：Windows、macOS、Linux
- pyqtgraph 建议 0.14.0 以上

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/Melon793/csv_plot.git
   cd csv_plot
   ```

2. **安装依赖**（使用 uv）
   ```bash
   uv sync
   ```

   或使用 pip：
   ```bash
   pip install pyqt6 pyqtgraph pandas numpy asammdf charset-normalizer ujson
   ```

3. **运行程序**
   ```bash
   uv run csv_plot_pyqt6.py
   ```

### 打包为独立应用

项目提供了 `scripts/` 目录下的打包脚本，可直接运行：

```bash
# PyInstaller（单目录模式，启动快）
bash scripts/build_exe_pyinstaller

# Nuitka（编译为原生可执行文件，性能更高）
bash scripts/build_exe_nuitka
```

**PyInstaller 手动命令**

```bash
# 单目录模式（推荐）
pyinstaller csv_plot_pyqt6.py --onedir --name csv_plot_pyqt6 \
    --icon assets/icon.ico \
    --add-data "assets/icon.ico;assets" \
    --add-data "assets/icon.icns;assets" \
    --add-data "assets/icon.png;assets" \
    --add-data "README.md;." \
    --noconsole --clean --noconfirm
```

**Nuitka 手动命令**

```bash
# 单目录模式
nuitka --standalone --output-filename=csv_plot_pyqt6 \
    --enable-plugin=pyqt6 \
    --windows-icon-from-ico=assets/icon.ico \
    --include-data-file=assets/icon.ico=assets \
    --include-data-file=assets/icon.icns=assets \
    --include-data-file=assets/icon.png=assets \
    --include-data-file=README.md=. \
    csv_plot_pyqt6.py
```

> **提示**：PyInstaller 快速但体积大；Nuitka 编译为原生代码，运行性能和启动速度更优，首次编译约 5-15 分钟，建议开发调试用 PyInstaller，正式发布用 Nuitka。

## 使用指南

### 基本操作流程

1. **加载数据**
   - 点击「加载文件」按钮选择 CSV / MDF 文件
   - 或直接将文件拖拽到程序窗口

2. **浏览数据**
   - 在左侧变量列表中查看所有变量
   - 观察颜色指示了解数据质量
   - 双击变量名查看详细数据表格（也可通过右键菜单添加）

3. **绘制图表**
   - 从变量列表拖拽变量到绘图区域（按住 Shift 可拖入多个）
   - 双击绘图区域打开变量编辑器管理曲线
   - 使用鼠标滚轮缩放、左键拖拽平移
   - 右键菜单提供更多选项

4. **数据分析**
   - 使用标记功能选择感兴趣的数据区间
   - 查看标记统计窗口的详细信息

### MDF 文件特别说明

| MDF 特性 | 支持方式 |
|----------|---------|
| 多 Channel Group | 自动聚合，同名变量加 `_G{index}` 后缀区分 |
| 枚举类型通道 | 自动构建 `{int→text_label}` 映射，绘图和游标显示文本标签 |
| 时间序列 | 自动提取时间通道作为 X 轴（`t` / `time` / `timestamp`） |
| asammdf 兼容 | 支持 7.x 和 8.x 两种 API |

### 快捷键说明

| 操作 | 方式 | 说明 |
|------|------|------|
| 清除图表 | 双击中键 | 清除当前图表所有曲线 |
| 缩放视图 | 滚轮 | 以鼠标位置为中心缩放 X 轴 |
| 移动视图 | 左键拖拽 | 平移视图 |
| 轴缩放 | Ctrl/Shift + 滚轮 | 在坐标轴区域缩放特定轴 |
| 游标模式 | 右键菜单 | 1 自由 / 1 锚定 / 2 锚定 |
| 跳转数据 | 右键菜单 | 跳转到数据表格对应位置 |

## 技术栈

- **PyQt6** — 现代化图形界面框架
- **pyqtgraph** — 高性能科学绘图库
- **pandas** — 数据分析库
- **numpy** — 数值计算基础库
- **asammdf** — ASAM MDF 文件解析
- **charset-normalizer** — 字符编码检测
- **ujson** — 高性能 JSON 处理

## 支持与反馈

如果您在使用过程中遇到问题或有改进建议，欢迎：

- 提交 Issue: [GitHub Issues](https://github.com/Melon793/csv_plot/issues)
- 参与讨论: [GitHub Discussions](https://github.com/Melon793/csv_plot/discussions)
- 给项目点赞: [GitHub Star](https://github.com/Melon793/csv_plot)

## 许可证

本项目采用开源许可证，供学习与研究使用。欢迎自由修改和扩展。

---

**如果这个项目对您有帮助，请给我们一个 Star！**
