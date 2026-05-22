# Plot 模板管理器 — 施工总结

> 分支: `dev-tmpltMgr`  
> 基于设计文档: `docs/Plot模板管理器设计文档.md`

---

## 一、新增文件清单

| 文件 | 说明 |
|------|------|
| `src/core/plot_config.py` | `PlotSessionConfig` / `PlotConfig` 数据结构 + 异常类 |
| `src/core/template_models.py` | `TemplateMetadata` / `PlotTemplate` 数据模型 |
| `src/core/storage.py` | `TemplateStorage` 存储层（文件 IO + 目录监控） |
| `src/core/template_manager.py` | `TemplateManager` 业务逻辑层 |
| `src/core/auto_save_manager.py` | `AutoSaveManager` 自动保存/恢复 |
| `src/ui/plot_config_manager.py` | `PlotConfigManager` 协调层（MainWindow 桥梁） |
| `src/ui/dialogs/__init__.py` | UI 对话框包 |
| `src/ui/dialogs/template_manager_dialog.py` | 模板管理器对话框 |
| `src/ui/dialogs/template_editor_dialog.py` | 模板编辑器对话框（YAML + 实时预览） |

## 二、修改文件清单

| 文件 | 改动 |
|------|------|
| `pyproject.toml` | 新增 `pyyaml>=6.0.0` 依赖 |
| `src/core/__init__.py` | 导出新增模块 |
| `csv_plot.py` | 集成 PlotConfigManager、添加"模板"二级菜单按钮、模板操作方法 |
| `src/ui/file_loader_manager.py` | `_post_load_actions` 中集成自动恢复逻辑 |

## 三、架构分层

```
┌────────────────────────────────┐
│        MainWindow (csv_plot.py)│
│  "模板" QToolButton (二级菜单)  │
└──────────────┬─────────────────┘
               │
┌──────────────▼─────────────────┐
│   PlotConfigManager (协调层)    │
│  export / apply / auto_restore  │
└───┬──────────────────┬─────────┘
    │                  │
┌───▼──────────┐  ┌────▼──────────────┐
│TemplateManager│  │ AutoSaveManager  │
│ CRUD / 搜索   │  │ 定时保存 / 匹配   │
└───┬──────────┘  └────┬──────────────┘
    │                  │
┌───▼──────────────────▼──────────────┐
│         TemplateStorage             │
│   文件 IO（YAML）+ QFileSystemWatcher│
└─────────────────────────────────────┘
```

## 四、核心功能

### 4.1 模板管理
- **保存为模板**: 当前布局 + 各 Plot 变量配置 → YAML 文件
- **加载模板**: 选择模板 → 恢复布局 + 变量绑定
- **模板列表**: 名称、变量数、Plot 数、更新时间，支持搜索
- **编辑模板**: YAML 编辑器 + 右侧网格实时预览
- **复制 / 删除 / 重命名**: 完整 CRUD
- **导入 / 导出**: 跨电脑分享 `.yaml` 文件

### 4.2 文件名策略
- 模板文件名 = `{安全化模板名称}.yaml`（例: `发动机_曲线.yaml`）
- UUID 仅作为内部 ID 存在 YAML `metadata.id` 字段
- 重命名自动删除旧文件

### 4.3 自动保存与恢复
- 数据加载后自动保存当前配置
- 下次加载同文件时，计算变量重合度（≥ 60% 则自动恢复）
- 可通过 QSettings 开关

### 4.4 UI 交互
- 顶部"模板"按钮 → 弹出二级菜单（保存为模板 / 模板管理器）
- 表格列宽按 3:1:1:3 比例自适应填充
- "保存为模板"对话框记忆上次名称/描述（会话内）
- 模板管理器内"新建"按钮 → 打开空白模板编辑器（预填默认配置）

## 五、异常体系

| 异常 | 场景 |
|------|------|
| `TemplateNotFoundError` | 删除/加载不存在的模板 |
| `TemplateNameConflictError` | 保存/重命名时名称重复 |
| `TemplateValidationError` | YAML 格式不符 |
| `TemplateStorageError` | 磁盘读写失败 |

## 六、存储位置

```
macOS: ~/Library/Application Support/CSVPlot/templates/
```

- 模板文件: `{名称}.yaml`
- 自动保存: `auto_save.yaml`

## 七、已修复的问题

1. **eventFilter 回调时 layout_manager 未初始化** — 移动 `PlotConfigManager` 初始化到 `LayoutManager` 之后 + eventFilter 守卫
2. **QWidget 未导入** — 补全 `template_editor_dialog.py` 导入
3. **空 `curves: []`** — `export_current_config` 仅导出有曲线的 Plot
4. **表格列宽不可调** — 改为 `_ProportionalTable` 3:1:1:3 自适应
