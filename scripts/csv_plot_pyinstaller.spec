# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 优化版 spec 文件
=============================
相比命令行参数方式, spec 文件提供更精细的打包控制:
- 明确的 hidden-imports (替代 --hidden-import)
- 排除未使用的标准库/第三方子模块 (替代多个 --exclude-module)
- 精确的数据文件映射
- 优化级别设置

使用方式:
    pyinstaller scripts/csv_plot_pyinstaller.spec
    或运行 scripts/build_exe_pyinstaller.bat (Windows)
    或运行 scripts/build_exe_pyinstaller (macOS/Linux)
"""

import sys
import os
from pathlib import Path

# ── 项目根 & 配置 ──────────────────────────────────────────────
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(SPECPATH)))  # noqa: F821
NAME = "csv_plot"
ENTRY = str(PROJECT_DIR / "csv_plot.py")
ICON_WIN = str(PROJECT_DIR / "assets" / "icon.ico")
ICON_MAC = str(PROJECT_DIR / "assets" / "icon.icns")
ICON_PNG = str(PROJECT_DIR / "assets" / "icon.png")

is_win = sys.platform == "win32"

# ── Analysis ───────────────────────────────────────────────────
a = Analysis(
    [ENTRY],
    pathex=[],
    binaries=[],
    datas=[
        (str(PROJECT_DIR / "assets" / "icon.ico"), "assets"),
        (str(PROJECT_DIR / "assets" / "icon.icns"), "assets"),
        (str(PROJECT_DIR / "assets" / "icon.png"), "assets"),
        (str(PROJECT_DIR / "README.md"), "."),
    ],
    hiddenimports=[
        # 本项目需要手动声明的隐藏导入
        "src",
        "src.utils",
        "src.utils.paths",
        "src.core",
        "src.core.config",
        "src.core.data_types",
        "src.core.scheduler",
        "src.core.font_cache",
        "src.core.logger",
        "src.core.auto_save_manager",
        "src.core.storage",
        "src.core.plot_config",
        "src.core.template_models",
        "src.core.template_manager",
        "src.data",
        "src.data.loader",
        "src.data.mdf_lazy_loader",
        "src.data.metadata",
        "src.ui",
        "src.ui.main_window_base_manager",
        "src.ui.file_loader_manager",
        "src.ui.layout_manager",
        "src.ui.splash_screen",
        "src.ui.cursor_sync_manager",
        "src.ui.drag_drop",
        "src.ui.variable_list",
        "src.ui.mark_stats",
        "src.ui.table_dialog",
        "src.ui.plot_variable_editor",
        "src.ui.plot_config_manager",
        "src.app",
        "src.app.plot_context",
        "src.ui.dialogs",
        "src.ui.dialogs.help",
        "src.ui.dialogs.layout_input",
        "src.ui.dialogs.axis",
        "src.ui.dialogs.time_correction",
        "src.ui.dialogs.log_window",
        "src.ui.dialogs.template_manager_dialog",
        "src.ui.dialogs.template_editor_dialog",
        "src.ui.widgets",
        "src.ui.widgets.custom_viewbox",
        "src.ui.widgets.plot_container",
        "src.ui.widgets.base_manager",
        "src.ui.widgets.plot_ui_manager",
        "src.ui.widgets.axis_manager",
        "src.ui.widgets.plot_data_manager",
        "src.ui.widgets.multi_curve_manager",
        "src.ui.widgets.cursor_manager",
        "src.ui.widgets.mark_region_manager",
        "src.ui.widgets.event_handler",
        "src.ui.widgets.log_viewer",
        # 第三方库中可能自动检测漏掉的
        "asammdf",
        "chardet",
        "charset_normalizer",
        "ujson",
        # PyQtGraph 内部依赖
        "pyqtgraph",
        "pyqtgraph.parametertree",
        "pyqtgraph.multiprocess",
        # pandas 内部依赖
        "pandas.io.formats.excel",
        "pandas.io.formats.csvs",
        "pandas.api.typing",
        "pandas.io.json",
        # numpy 内部依赖
        "numpy.core",
        "numpy.core._multiarray_umath",
        "numpy.core._dtype_ctypes",
        "numpy.random",
        # PySide6 内部依赖
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # ── 我们的工具链 (不会被打包但声明排除更安全) ──
        "nuitka",
        "pyinstaller",
        "pytest",
        # ── 明显未使用的标准库模块 ──
        "tkinter",
        "turtle",
        "unittest",
        "test",
        "doctest",
        "distutils",
        "setuptools",
        "pip",
        "wheel",
        "ensurepip",
        "idlelib",
        "lib2to3",
        "xmlrpc",
        "mailcap",
        "smtpd",
        "email.mime",
        "asyncio",
        "wsgiref",
        "cgi",
        "cgitb",
        "http.server",
        "socketserver",
        "webbrowser",
        # ── PySide6 中大概率未使用的子模块 ──
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebChannel",
        "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "PySide6.QtQml",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtBluetooth",
        "PySide6.QtNetwork",
        "PySide6.QtSql",
        "PySide6.QtSvg",
        "PySide6.QtSvgWidgets",
        "PySide6.QtTest",
        "PySide6.QtXml",
        "PySide6.QtHelp",
        "PySide6.QtLocation",
        "PySide6.QtPositioning",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtWebSockets",
        "PySide6.QtNfc",
        "PySide6.QtPrintSupport",
        "PySide6.QtDesigner",
        "PySide6.QtUiTools",
        "PySide6.QtAxContainer",
        "PySide6.QtTextToSpeech",
        "PySide6.QtDataVisualization",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DLogic",
        "PySide6.QtCharts",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtStateMachine",
        "PySide6.QtVirtualKeyboard",
        "PySide6.QtHttpServer",
        "PySide6.QtSpatialAudio",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ── PYZ (字节码归档) ─────────────────────────────────────────
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# ── EXE ────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,           # 剥离符号表, 减小体积
    upx=False,             # 禁用 UPX 压缩, 避免解压开销
    upx_exclude=[
        "vcruntime140.dll",
        "python3.dll",
        "python312.dll",
    ],
    console=not is_win,    # Windows 下无控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[ICON_WIN if is_win else ICON_MAC],
    runtime_tmpdir=None,
    uac_admin=False,
    uac_uiaccess=False,
)

# ── COLLECT (onedir 目录收集) ───────────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=False,
    upx_exclude=[],
    name=NAME,
)
