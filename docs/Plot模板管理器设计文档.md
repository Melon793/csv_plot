# Plot 配置模板管理器 - 开发设计文档

> 文档版本: 1.4
> 创建日期: 2024-01-20
> 状态: 设计中

---

## 目录

1. [功能概述](#1-功能概述)
2. [架构设计](#2-架构设计)
3. [数据结构](#3-数据结构)
4. [核心模块设计](#4-核心模块设计)
5. [API 接口规范](#5-api-接口规范)
6. [UI 设计](#6-ui-设计)
7. [文件存储规范](#7-文件存储规范)
8. [可行性与风险评估](#8-可行性与风险评估)
9. [实施计划](#9-实施计划)
10. [附录](#10-附录)

---

## 1. 功能概述

### 1.1 需求背景

当前应用支持多 Plot 布局、多曲线显示，但用户每次加载数据后需要手动重新配置 Plot 的布局和变量。需求设计一个**模板管理器**，实现：

1. **自动保存/加载**: 加载数据后自动恢复上次的工作状态
2. **模板管理**: 用户可保存、加载、导出、导入 Plot 配置模板
3. **智能匹配**: 当变量重合度不足时，放弃自动加载配置
4. **文件夹监控**: 模板管理器监控固定文件夹，用户可从外部导入

### 1.2 核心功能列表

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 模板保存 | 将当前 Plot 配置保存为模板文件 | P0 |
| 模板加载 | 从模板文件恢复 Plot 配置 | P0 |
| 模板列表 | 显示管理文件夹内所有模板 | P0 |
| 外部导入 | 将外部 YAML 复制到管理文件夹 | P0 |
| 外部导出 | 将模板复制到用户指定位置 | P1 |
| 模板重命名 | 修改模板名称 | P1 |
| 模板删除 | 删除管理文件夹内的模板 | P0 |
| 模板搜索 | 按名称搜索模板 | P2 |
| 自动恢复 | 数据加载后自动应用配置 | P0 |
| 智能匹配 | 变量重合度判断是否应用配置 | P0 |
| 文件夹监控 | 实时监控管理文件夹变化 | P1 |

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MainWindow                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        CSVPlotApplication                         │   │
│  │  ┌───────────────────┐         ┌────────────────────────────┐   │   │
│  │  │  PlotConfigManager │         │     TemplateManager        │   │   │
│  │  │    (业务协调层)     │         │       (模板管理层)          │   │   │
│  │  │                    │         │                            │   │   │
│  │  │ • 数据加载流程      │         │ • CRUD 操作                │   │   │
│  │  │ • 配置应用流程      │         │ • 文件夹监控                │   │   │
│  │  │ • 状态同步          │         │ • 导入/导出                  │   │   │
│  │  │                    │         │                            │   │   │
│  │  │  ┌──────────────┐ │         │  ┌──────────────────────┐  │   │   │
│  │  │  │ AutoSaveManager│ │         │  │  TemplateStorage     │  │   │   │
│  │  │  │  (自动保存)    │ │         │  │    (存储层)           │  │   │   │
│  │  │  └──────────────┘ │         │  │                      │  │   │   │
│  │  └───────────────────┘         │  │  • 文件读写           │  │   │   │
│  │                                 │  │  • 文件系统监控        │  │   │   │
│  │                                 │  │  • 缓存管理            │  │   │   │
│  │                                 │  └──────────────────────┘  │   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────────┐
                        │     ~/.config/CSVPlot/templates/    │
                        │     (固定模板存储文件夹)            │
                        │                                    │
                        │  ├── 01.yaml   (模板1)             │
                        │  ├── 02.yaml   (模板2)             │
                        │  └── ...                           │
                        └───────────────────────────────────┘
```

### 2.2 模块职责

| 模块 | 职责 | 依赖 |
|------|------|------|
| `PlotConfigManager` | 业务协调：保存/应用配置的入口 | TemplateManager, AutoSaveManager |
| `AutoSaveManager` | 自动保存/恢复：数据加载时的配置处理 | PlotSessionConfig |
| `TemplateManager` | 模板管理：CRUD、导入导出、搜索 | TemplateStorage, PlotSessionConfig |
| `TemplateStorage` | 存储层：文件操作、监控、缓存 | 文件系统 |
| `PlotSessionConfig` | 数据模型：配置的数据结构定义 | 无 |

### 2.3 信号/槽设计

```python
class TemplateManager(QObject):
    """模板管理器信号"""
    
    template_added = pyqtSignal(str)      # template_id
    template_removed = pyqtSignal(str)    # template_id  
    template_updated = pyqtSignal(str)    # template_id
    template_list_changed = pyqtSignal()  # 列表整体变化
    
class AutoSaveManager:
    """自动保存管理器"""
    
    config_match_result = pyqtSignal(bool, str)  # 是否应用, 原因
    config_applied = pyqtSignal()                 # 配置已应用
```

---

## 3. 数据结构

### 3.1 PlotSessionConfig

完整的 Plot 会话配置（用于单次工作状态和模板）。

```python
@dataclass
class PlotSessionConfig:
    """完整的 Plot 会话配置"""
    
    created_at: str = ""                       # 创建时间 ISO 格式
    
    # ===== 全局布局 =====
    layout_rows: int = 3                       # 布局行数
    layout_cols: int = 1                       # 布局列数
    
    # ===== 全局设置 =====
    time_factor: float = 1.0
    time_offset: float = 0.0
    
    # ===== 各 Plot 配置 =====
    plots: list["PlotConfig"] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        ...
    
    @classmethod
    def from_dict(cls, data: dict) -> "PlotSessionConfig":
        """从字典反序列化"""
        ...
```

### 3.2 PlotConfig

单个 Plot 的配置。`mode` 和 `y_name` 不存入配置，由模板应用时根据 `curves` 数量自动推断。

```python
@dataclass
class PlotConfig:
    """单个 Plot 的配置"""
    curves: list[str] = field(default_factory=list)  # 曲线变量名列表
```

**mode 推断规则（应用模板时）：**
- `len(curves) == 0` → 空 Plot，不绑定任何变量
- `len(curves) == 1` → single 模式，变量取 `curves[0]`
- `len(curves) >= 2` → multi 模式

**Plot 与布局位置的映射（保存/加载时）：**
`plots` 列表按 **row-major** 顺序对应布局网格中的位置：

```
布局 3 行 × 2 列:
  (0,0)  (0,1)   ← plots[0]  plots[1]
  (1,0)  (1,1)   ← plots[2]  plots[3]
  (2,0)  (2,1)   ← plots[4]  plots[5]
```

- **保存时**: 遍历所有可见 Plot，按 row-major 顺序依次收集
- **加载时**: 按列表顺序依次分配到布局位置，不绑定固定 ID
- 用户修改布局后重新保存时，列表顺序自动反映新布局，无需维护 index

### 3.3 TemplateMetadata

模板元数据（独立存储用于快速列表展示）。

```python
@dataclass
class TemplateMetadata:
    """模板元数据"""
    
    id: str                                   # 唯一标识 (8位 UUID)
    name: str                                 # 显示名称
    description: str = ""                     # 描述
    
    created_at: str = ""                       # 创建时间
    updated_at: str = ""                      # 更新时间
    
    source_file: Optional[str] = None         # 外部导入源文件路径
```

### 3.4 PlotTemplate

完整的模板结构。

```python
@dataclass
class PlotTemplate:
    """完整的模板"""
    
    metadata: TemplateMetadata                 # 元数据
    config: dict                              # PlotSessionConfig 字典形式
```

### 3.5 YAML 文件格式

```yaml
# ~/.config/CSVPlot/templates/abc123.yaml
# 用户可直接编辑此文件；支持 # 注释，引号可省略
metadata:
  id: abc123
  name: 发动机测试配置
  description: 用于发动机性能测试的标准配置
  created_at: "2024-01-15T10:30:00"
  updated_at: "2024-01-20T14:22:00"

config:
  created_at: "2024-01-15T10:30:00"
  layout_rows: 2
  layout_cols: 2
  time_factor: 0.001
  time_offset: 0.0
  plots:
    - curves:            # → 位置 (0,0)
        - rpm
        - speed         # 2 条曲线 → multi 模式
    - curves:            # → 位置 (0,1)
        - throttle      # 1 条曲线 → single 模式
    - curves:            # → 位置 (1,0)
        - torque
        - fuel
    - curves:            # → 位置 (1,1)
        - temp
```

---

## 4. 核心模块设计

### 4.1 TemplateStorage (存储层)

```python
class TemplateStorage:
    """
    模板存储层 - 负责文件系统的读写和监控
    
    设计要点:
    1. 所有操作都通过此类进行文件 IO
    2. 维护内存缓存减少 IO
    3. 文件名使用 UUID，名称存储在 metadata 中
    4. 使用 YAML 格式（PyYAML），用户可直接编辑模板文件
    """
    
    def __init__(self, storage_path: Path):
        self._storage_path = Path(storage_path)
        self._cache: dict[str, PlotTemplate] = {}
        self._watcher = QFileSystemWatcher()
    
    # ==================== 文件操作 ====================
    
    def scan_directory(self) -> list[str]:
        """扫描目录，返回所有模板 ID"""
        ...
    
    def read_template(self, template_id: str) -> Optional[PlotTemplate]:
        """读取指定模板"""
        ...
    
    def write_template(self, template: PlotTemplate) -> bool:
        """写入模板到文件"""
        ...
    
    def delete_template(self, template_id: str) -> bool:
        """删除模板文件"""
        ...
    
    def template_exists(self, template_id: str) -> bool:
        """检查模板是否存在"""
        ...
    
    # ==================== 文件系统监控 ====================
    
    def start_watching(self):
        """开始监控目录变化"""
        ...
    
    def stop_watching(self):
        """停止监控"""
        ...
    
    # ==================== 导入导出 ====================
    
    def import_from_external(self, external_path: Path) -> Optional[PlotTemplate]:
        """
        从外部文件导入（复制到管理文件夹）
        - 验证文件格式
        - 生成新 ID
        - 复制到管理文件夹
        """
        ...
    
    def export_to_external(self, template_id: str, target_path: Path) -> bool:
        """
        导出到外部位置（复制到目标路径）
        """
        ...
```

### 4.2 TemplateManager (业务层)

```python
class TemplateManager(QObject):
    """
    模板管理器 - 提供模板的完整业务逻辑
    
    职责:
    1. CRUD 操作
    2. 搜索和筛选
    3. 导入/导出处理
    4. 文件夹监控信号派发
    """
    
    # 信号
    template_added = pyqtSignal(str)
    template_removed = pyqtSignal(str)
    template_updated = pyqtSignal(str)
    template_list_changed = pyqtSignal()
    
    def __init__(self, storage_path: Optional[Path] = None):
        ...
    
    # ==================== 查询 ====================
    
    def get_all_templates(self) -> list[PlotTemplate]:
        """获取所有模板（按更新时间倒序）"""
        ...
    
    def get_template(self, template_id: str) -> Optional[PlotTemplate]:
        """根据 ID 获取模板"""
        ...
    
    def search(
        self,
        keyword: str = "",
        min_variables: int = 0
    ) -> list[PlotTemplate]:
        """搜索模板"""
        ...
    
    # ==================== CRUD ====================
    
    def save_template(
        self,
        config: PlotSessionConfig,
        name: str,
        description: str = "",
        template_id: Optional[str] = None
    ) -> PlotTemplate:
        """
        保存模板
        
        Args:
            config: Plot 配置
            name: 模板名称
            description: 描述
            template_id: 指定 ID（None 则新建）
        
        Returns:
            保存的模板对象
        
        Raises:
            ValueError: 名称已存在
            IOError: 文件写入失败
        """
        ...
    
    def rename_template(self, template_id: str, new_name: str) -> bool:
        """重命名模板"""
        ...
    
    def delete_template(self, template_id: str) -> bool:
        """删除模板"""
        ...
    
    # ==================== 导入导出 ====================
    
    def import_template(self, external_path: Path) -> Optional[PlotTemplate]:
        """
        从外部文件导入
        - 用户选择任意位置的文件
        - 复制到管理文件夹
        - 生成新 ID
        """
        ...
    
    def export_template(self, template_id: str, target_path: Path) -> bool:
        """
        导出模板到外部位置
        - 用户选择目标位置
        - 复制到目标位置
        """
        ...
    
    def duplicate_template(self, template_id: str, new_name: str) -> Optional[PlotTemplate]:
        """复制模板"""
        ...
    
    # ==================== 内部处理 ====================
    
    def _on_directory_changed(self):
        """目录变化处理"""
        ...
    
    def _on_file_changed(self, filepath: Path):
        """单个文件变化处理"""
        ...
    
    def _validate_template_data(self, data: dict) -> bool:
        """验证模板数据格式"""
        ...
```

### 4.3 AutoSaveManager (自动保存层)

```python
class AutoSaveManager:
    """
    自动保存管理器 - 处理数据加载时的配置恢复
    
    设计要点:
    1. 管理唯一的自动保存配置（auto_save.yaml）
    2. 智能匹配算法判断是否应用配置
    3. 与 TemplateManager 共享存储路径
    """
    
    # 匹配结果信号
    match_result = pyqtSignal(bool, str)  # 是否匹配, 原因
    
    # 最小匹配比例（可配置）
    MIN_MATCH_RATIO: float = 0.6
    
    def __init__(self, template_manager: TemplateManager):
        self._tm = template_manager
        self._settings = QSettings("CSVPlot", "AutoSave")
    
    # ==================== 自动保存开关 ====================
    
    def is_auto_save_enabled(self) -> bool:
        """是否启用自动保存"""
        return self._settings.value("auto_save_enabled", True, bool)
    
    def set_auto_save_enabled(self, enabled: bool):
        self._settings.setValue("auto_save_enabled", enabled)
    
    # ==================== 自动保存 ====================
    
    def auto_save(self, config: PlotSessionConfig):
        """
        自动保存当前配置
        - 数据加载时调用
        - 退出时调用
        """
        ...
    
    def load_auto_save(self) -> Optional[PlotSessionConfig]:
        """加载自动保存的配置"""
        ...
    
    # ==================== 智能匹配 ====================
    
    def should_apply_auto_save(self, current_vars: list[str]) -> tuple[bool, str]:
        """
        判断是否应应用自动保存的配置
        
        Returns:
            (是否应用, 原因说明)
        """
        config = self.load_auto_save()
        if not config:
            return False, "无自动保存的配置"
        
        config_vars = self._extract_variables(config)
        current_vars_set = set(current_vars)
        
        # 计算匹配度
        matched = len(config_vars & current_vars_set)
        total = max(len(config_vars), len(current_vars_set))
        ratio = matched / total if total > 0 else 0.0
        
        if ratio >= self.MIN_MATCH_RATIO:
            return True, f"匹配度 {ratio:.0%}"
        elif ratio > 0:
            return False, f"匹配度仅 {ratio:.0%}（需≥{self.MIN_MATCH_RATIO:.0%}）"
        else:
            return False, "无匹配变量"
    
    @staticmethod
    def _extract_variables(config: PlotSessionConfig) -> set[str]:
        """从 plots 中汇总所有变量名"""
        vars_set = set()
        for plot in config.plots:
            for v in plot.curves:
                vars_set.add(v)
        return vars_set
```

### 4.4 PlotConfigManager (协调层)

```python
class PlotConfigManager:
    """
    Plot 配置管理器 - 协调各模块的入口
    
    作为 MainWindow 和下层模块之间的桥梁
    """
    
    def __init__(self):
        self._tm = TemplateManager()
        self._asm = AutoSaveManager(self._tm)
    
    @property
    def template_manager(self) -> TemplateManager:
        return self._tm
    
    @property
    def auto_save_manager(self) -> AutoSaveManager:
        return self._asm
    
    # ==================== 配置导出 ====================
    
    def export_current_config(self, main_window) -> PlotSessionConfig:
        """
        从 MainWindow 导出当前状态为配置对象
        
        收集:
        - 布局信息
        - 各 Plot 的变量名列表（按 row-major 顺序）
        - 时间修正参数
        """
        ...
    
    # ==================== 配置应用 ====================
    
    def apply_config(self, main_window, config: PlotSessionConfig):
        """
        将配置应用到 MainWindow
        
        plots 列表按 row-major 顺序分配:
          plots[0] → 布局位置 (0,0), plots[1] → (0,1), ...
        """
        ...
    
    def apply_template(self, main_window, template_id: str) -> bool:
        """应用模板"""
        ...
    
    # ==================== 快捷操作 ====================
    
    def save_current_as_template(
        self,
        main_window,
        name: str,
        description: str = ""
    ) -> bool:
        """保存当前配置为模板"""
        ...
    
    def save_auto_save(self, main_window):
        """保存为自动恢复配置"""
        ...
    
    def try_auto_restore(self, main_window, current_vars: list[str]) -> bool:
        """
        尝试自动恢复配置
        
        Returns:
            是否成功应用配置
        """
        ...
```

---

## 5. API 接口规范

### 5.1 TemplateManager

```python
class TemplateManager:
    """模板管理器 API"""
    
    # ----- 查询 -----
    
    def get_all_templates(self) -> list[PlotTemplate]:
        """获取所有模板"""
    
    def get_template(self, template_id: str) -> Optional[PlotTemplate]:
        """获取单个模板"""
    
    def search(
        self,
        keyword: str = "",
        min_variables: int = 0
    ) -> list[PlotTemplate]:
        """搜索模板"""
    
    def exists(self, name: str) -> bool:
        """名称是否存在"""
    
    # ----- 写入 -----
    
    def save_template(
        self,
        config: PlotSessionConfig,
        name: str,
        description: str = "",
        template_id: Optional[str] = None
    ) -> PlotTemplate:
        """保存模板"""
    
    def rename_template(self, template_id: str, new_name: str) -> bool:
        """重命名"""
    
    def delete_template(self, template_id: str) -> bool:
        """删除"""
    
    def duplicate_template(
        self, 
        template_id: str, 
        new_name: str
    ) -> Optional[PlotTemplate]:
        """复制模板"""
    
    # ----- 导入导出 -----
    
    def import_template(self, external_path: Path) -> Optional[PlotTemplate]:
        """导入外部文件"""
    
    def export_template(
        self, 
        template_id: str, 
        target_path: Path
    ) -> bool:
        """导出到外部"""
```

### 5.2 AutoSaveManager

```python
class AutoSaveManager:
    """自动保存管理器 API"""
    
    # ----- 开关 -----
    
    def is_auto_save_enabled(self) -> bool:
        """是否启用"""
    
    def set_auto_save_enabled(self, enabled: bool):
        """设置启用状态"""
    
    # ----- 自动保存 -----
    
    def auto_save(self, config: PlotSessionConfig):
        """保存自动配置"""
    
    def load_auto_save(self) -> Optional[PlotSessionConfig]:
        """加载自动配置"""
    
    # ----- 匹配 -----
    
    def should_apply_auto_save(
        self, 
        current_vars: list[str]
    ) -> tuple[bool, str]:
        """判断是否应应用"""
```

---

## 6. UI 设计

### 6.1 菜单结构

```
菜单栏
├── 文件(&F)
│   ├── 打开...          Ctrl+O
│   └── 退出             Ctrl+Q
│
├── 视图(&V)
│   ├── 显示/隐藏绘图区    Ctrl+P
│   ├── 布局...          Ctrl+L
│   └── ─────────────
│   └── 标记区域         Ctrl+M
│
├── 配置(&C)
│   ├── ☑ 加载数据后自动恢复布局     ← QAction，复选框
│   ├── ─────────────────────
│   ├── 💾 保存当前布局为模板...     → 打开模板编辑器（新建模式）
│   └── 📋 管理模板...              → 打开模板管理器（统一入口）
│
└── 帮助(&H)
```

### 6.2 模板管理器对话框（加载 + 管理合并）

模板列表、搜索、CRUD、加载功能的统一入口。

```
┌──────────────────────────────────────────────────────────┐
│  📋 模板管理                                        [X]  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  🔍 搜索: [________________________]                     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 名称                │ 变量 │ Plot │ 更新时间       │  │
│  ├────────────────────────────────────────────────────┤  │
│  │ 发动机测试配置        │  8   │  4   │ 01-20 14:22   │  │
│  │ 底盘性能测试          │  12  │  3   │ 01-19 10:30   │  │
│  │ 高速工况测试          │  6   │  2   │ 01-18 16:45   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  选中: 发动机测试配置                                    │
│  描述: 用于发动机性能测试的标准配置                       │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ [📥 导入] [📤 导出] [✏️ 编辑] [📋 复制] [🗑️ 删除]     │
│                                    [📂 加载选中]  [关闭] │
└──────────────────────────────────────────────────────────┘
```

| 操作 | 行为 |
|------|------|
| 双击列表行 | 加载该模板到当前布局，关闭对话框 |
| `[加载选中]` 按钮 | 加载当前选中项并关闭 |
| `[编辑]` 按钮 | 打开模板编辑器（YAML + 预览），编辑已有模板 |
| `[复制]` 按钮 | 复制当前模板 → 打开编辑器（名称追加"(副本)"，新 ID） |
| `[导入]` 按钮 | 选择 `.yaml` 文件复制到管理文件夹 |
| `[导出]` 按钮 | 将选中模板导出到用户指定位置 |
| `[删除]` 按钮 | 确认后删除 |
| 搜索框输入 | 按名称实时过滤列表 |

**变量数 / Plot 数**: 打开对话框时解析 YAML 现场计算（几十个文件的总 IO 可忽略），不在 metadata 中冗余存储。

### 6.3 模板编辑器对话框（YAML 编辑 + 网格预览）

新建和编辑模板的统一界面，左侧 YAML 编辑器，右侧布局网格实时预览。

```
┌──────────────────────────────────────────────────────────────────┐
│  ✏️ 编辑模板 / 💾 新建模板                                   [X] │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  名称: [发动机测试配置___________________________]               │
│  描述: [用于发动机性能测试的标准配置_____________]               │
│                                                                  │
│  ┌──────────────────────────────┬──────────────────────────────┐ │
│  │  YAML 编辑器                 │  布局预览 (2×2)              │ │
│  │                              │                              │ │
│  │  layout_rows: 2              │  ┌─────────┬───────────┐    │ │
│  │  layout_cols: 2              │  │ rpm     │ throttle  │    │ │
│  │  time_factor: 0.001          │  │ speed   │           │    │ │
│  │  time_offset: 0.0            │  │         │           │    │ │
│  │  plots:                      │  ├─────────┼───────────┤    │ │
│  │    - curves:                 │  │ torque  │ temp      │    │ │
│  │        - rpm                 │  │ fuel    │           │    │ │
│  │        - speed               │  │         │           │    │ │
│  │    - curves:                 │  └─────────┴───────────┘    │ │
│  │        - throttle            │                              │ │
│  │    - curves:                 │  cells: 2×2                  │ │
│  │        - torque              │  vars: 5 | plots: 4          │ │
│  │        - fuel                │                              │ │
│  │    - curves:                 │                              │ │
│  │        - temp                │                              │ │
│  │                              │                              │ │
│  └──────────────────────────────┴──────────────────────────────┘ │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│              [保存]  [另存为...]  [取消]                         │
└──────────────────────────────────────────────────────────────────┘
```

**左侧 — YAML 文本编辑器:**
- `QPlainTextEdit`，简单的 YAML 语法高亮（关键字、注释、字符串）
- 保存前 `yaml.safe_load` 校验，失败时提示错误位置

**右侧 — 网格预览:**
- 根据 `layout_rows` × `layout_cols` 绘制网格
- 每个单元格渲染对应 `plots` 中的变量名：空单元格灰色占位；单变量显示名称 + "single" 标记；多变量逐行显示 + "multi" 标记
- 行高按可用高度均分（总可用高度 / layout_rows，减行间距）
- 底部统计 `cells: M×N | vars: N | plots: N`
- YAML 内容变化时实时重新解析并更新预览；解析失败显示 "⚠️ YAML 格式错误"

**三种打开模式:**

| 来源 | 标题 | YAML 来源 | 保存行为 |
|------|------|-----------|----------|
| 菜单 `💾 保存当前布局为模板...` | "新建模板" | 从当前布局填充 | 新建 |
| 管理器 `[编辑]` | "编辑模板" | 从选中模板加载 | 覆盖写 |
| 管理器 `[复制]` → 编辑 | "复制模板" | 从原模板加载，名称加"(副本)" | 新建 |

**按钮行为:**

| 按钮 | 行为 |
|------|------|
| `[保存]` | YAML 校验 → 写入模板文件（覆盖或新建） → 关闭 |
| `[另存为...]` | YAML 校验 → 弹名称输入 → 新 ID → 保存为新模板 → 关闭 |
| `[取消]` | 有未保存修改则确认后关闭 |

---

### 6.4 状态栏提示

> 同原 6.5，编号前移。

| 场景 | 状态栏提示 |
|------|-----------|
| 自动恢复成功 | ✅ 已自动恢复布局（匹配度 85%） |
| 自动恢复跳过（匹配度低） | ℹ️ 跳过自动恢复，匹配度仅 45% |
| 模板保存成功 | ✅ 模板已保存 |
| 导入成功 | ✅ 已导入配置 |
| 导入失败 | ❌ 导入失败：文件格式错误 |
| 名称冲突 | ⚠️ 模板名称已存在 |

---

## 7. 文件存储规范

### 7.1 目录结构

```
~/.config/CSVPlot/
├── config.ini              # QSettings 配置文件
│                           # [General]
│                           # auto_save_enabled=true
│                           #
│
└── templates/
    ├── 8a1b2c3d.yaml       # 模板文件（ID.yaml）
    ├── 9e4f5g6h.yaml
    └── ...
```

### 7.2 文件命名规则

- **文件名**: `{8位UUID}.yaml`
- **编码**: UTF-8
- **缩进**: 2 空格
- **换行**: LF

### 7.3 备份策略

```python
# 保存前备份
auto_save.yaml.backup  ← 上上次
auto_save.yaml        ← 上次

# 文件损坏时从 backup 恢复
```

---

## 8. 可行性与风险评估

### 8.1 技术可行性评估

| 方面 | 可行性 | 说明 |
|------|--------|------|
| 数据结构设计 | ✅ 高 | 基于现有代码的数据结构，易于适配 |
| 文件 IO | ✅ 高 | Python yaml (PyYAML) + QSettings 成熟稳定 |
| QFileSystemWatcher | ✅ 高 | PyQt6 内置，跨平台支持良好 |
| 配置应用 | ✅ 高 | 可复用现有 Plot 操作逻辑 |
| UI 集成 | ✅ 高 | 菜单 + 对话框模式，实现简单 |

### 8.2 风险矩阵

| 风险 | 等级 | 影响 | 缓解措施 |
|------|------|------|----------|
| **文件损坏/格式错误** | 中 | 用户模板丢失 | 1. 备份机制<br>2. 格式验证<br>3. 异常捕获 |
| **外部导入恶意文件** | 低 | 无安全隐患（只读操作） | 1. 验证 YAML 结构<br>2. 不执行动态代码 |
| **文件夹权限问题** | 中 | 无法保存模板 | 1. 启动时检查<br>2. 优雅降级到用户目录 |
| **版本不兼容** | 低 | 配置无法应用 | 1. 解析时忽略未知字段<br>2. 向前兼容处理 |
| **大量模板文件** | 低 | 扫描变慢 | 1. 缓存机制<br>2. 分页加载（如果>100个） |
| **多窗口并发写** | 中 | 配置互相覆盖 | 1. 使用锁机制<br>2. 最后写入胜出 |
| **QSettings 跨平台** | 低 | 配置丢失 | 1. 使用标准 Path<br>2. 测试各平台 |

### 8.3 兼容性考虑

```python
# YAML 解析时忽略未知字段即可向前兼容
class PlotSessionConfig:
    @classmethod
    def from_dict(cls, data: dict) -> "PlotSessionConfig":
        return cls(
            created_at=data.get("created_at", ""),
            layout_rows=data.get("layout_rows", 1),
            layout_cols=data.get("layout_cols", 1),
            time_factor=data.get("time_factor", 1.0),
            time_offset=data.get("time_offset", 0.0),
            plots=[PlotConfig(curves=p.get("curves", []))
                   for p in data.get("plots", [])],
        )
```

### 8.4 测试计划

| 测试类型 | 覆盖内容 |
|----------|----------|
| 单元测试 | `PlotConfig.from_dict/to_dict` 序列化 |
| 单元测试 | `TemplateManager.save/get/search` |
| 单元测试 | `AutoSaveManager.should_apply` 匹配算法 |
| 集成测试 | 保存→加载→应用完整流程 |
| 集成测试 | 外部导入→保存→导出循环 |
| 集成测试 | 文件夹监控触发 |
| UI 测试 | 对话框交互 |
| 边界测试 | 空模板、损坏 YAML、超长名称 |

---

## 9. 实施计划

### 9.1 Phase 1: 基础设施 (预计 1-2 天)

**目标**: 完成数据结构和存储层

```
src/core/
├── __init__.py
├── plot_config.py      # PlotSessionConfig, PlotConfig
├── template_models.py   # TemplateMetadata, PlotTemplate
└── storage.py           # TemplateStorage
```

**交付物**:
- [ ] `PlotSessionConfig` 数据类
- [ ] `PlotTemplate` 数据类
- [ ] `TemplateStorage` 文件读写
- [ ] 单元测试

### 9.2 Phase 2: 模板管理 (预计 2-3 天)

**目标**: 完成 TemplateManager 业务层

```
src/core/
└── template_manager.py  # TemplateManager
```

**交付物**:
- [ ] `TemplateManager` CRUD 操作
- [ ] 导入/导出功能
- [ ] 搜索功能
- [ ] 文件夹监控
- [ ] 相关单元测试

### 9.3 Phase 3: 自动保存 (预计 1-2 天)

**目标**: 完成 AutoSaveManager

```
src/core/
└── auto_save_manager.py  # AutoSaveManager
```

**交付物**:
- [ ] 自动保存/加载
- [ ] 智能匹配算法
- [ ] UI 开关集成
- [ ] 相关单元测试

### 9.4 Phase 4: 集成与 UI (预计 2-3 天)

**目标**: 集成到 MainWindow，添加 UI

```
src/ui/
├── dialogs/
│   ├── template_manager_dialog.py   # 模板管理器（列表 + CRUD + 加载）
│   └── template_editor_dialog.py    # 模板编辑器（YAML + 网格预览）
└── plot_config_manager.py  # 协调层
```

**交付物**:
- [ ] `PlotConfigManager` 协调类
- [ ] 菜单项和 QAction（简化后的）
- [ ] 模板管理器对话框
- [ ] 模板编辑器对话框（YAML 编辑 + 实时预览）
- [ ] 状态栏提示
- [ ] 集成测试

### 9.5 Phase 5: 测试与修复 (预计 1-2 天)

**目标**: 全面测试，修复问题

- [ ] 完整单元测试覆盖
- [ ] 集成测试
- [ ] 边界条件测试
- [ ] 文档更新

---

## 10. 附录

### 10.1 现有代码参考点

| 功能 | 文件位置 | 关键属性/方法 |
|------|----------|---------------|
| Plot 列表 | `layout_manager.py` | `self.plot_widgets` |
| 布局 | `layout_manager.py` | `_plot_row_current`, `_plot_col_current` |
| 行高 | `layout_manager.py` | `row_height_factors` |
| 单曲线 | `csv_plot_pyqt6.py` | `self.y_name`, `self.curve` |
| 多曲线 | `csv_plot_pyqt6.py` | `self.curves`, `self.is_multi_curve_mode` |
| 曲线颜色 | `csv_plot_pyqt6.py` | `self.curve_colors`, `self.current_color_index` |
| CurveInfo | `types.py` | `CurveInfo` dataclass |
| Mark Region | `layout_manager.py` | `saved_mark_range` |
| 时间修正 | `csv_plot_pyqt6.py` | `self.factor`, `self.offset` |

### 10.2 默认配置值

```python
# 默认值（用于配置为空时）
DEFAULT_LAYOUT_ROWS = 1
DEFAULT_LAYOUT_COLS = 1
MIN_MATCH_RATIO = 0.6
```

### 10.3 错误码定义

```python
class TemplateError(Exception):
    """模板相关异常基类"""
    pass

class TemplateNotFoundError(TemplateError):
    """模板不存在"""
    pass

class TemplateNameConflictError(TemplateError):
    """模板名称冲突"""
    pass

class TemplateValidationError(TemplateError):
    """模板验证失败"""
    pass

class TemplateStorageError(TemplateError):
    """存储操作失败"""
    pass
```

### 10.4 日志规范

```python
# 使用现有的 debug_log
debug_log("TemplateManager: loaded %d templates", len(templates))
debug_log("AutoSave: match ratio %.2f, should_apply=%s", ratio, should_apply)
debug_log("Template saved: id=%s, name=%s", tid, name)
```

---

## 变更记录

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| 1.4 | 2026-05-21 | 合并 UI：选择加载 + 管理器 → 统一模板管理器；保存 + 编辑 → 模板编辑器（YAML 编辑 + 网格实时预览）；简化菜单为 2 个入口 |
| 1.3 | 2026-05-21 | 移除 PlotConfig.index（plots 列表顺序 = 布局 row-major 位置）；移除 TemplateMetadata.variable_count/plot_count（可从 config 推导） |
| 1.2 | 2026-05-21 | 移除 variable_list（可从 plots 汇总）、data_file_hash（模板跨文件通用）；移除 PlotConfig.mode/y_name（由 curves 数量推断）；文件格式 JSON → YAML（支持注释、便于手动编辑） |
| 1.1 | 2026-05-21 | 简化设计：Plot 仅保存变量名列表，颜色/样式由 plot 自动分配；移除 cursor/stat region 信息；移除 row_heights、version、tags 字段 |
| 1.0 | 2024-01-20 | 初始版本 |

---

*文档生成工具: AI Assistant*
