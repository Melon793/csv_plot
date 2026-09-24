import importlib.metadata
import importlib.util
import os
import sys
import shutil
import subprocess
import py_compile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENTRY_FILE = PROJECT_ROOT / "csv_plot.py"
OUTPUT_NAME = "csv_plot"
REPORT_FILE = "report.xml"
ASSETS_DIR = "assets"
README_FILE = "README.md"
ICON_FILE = "assets/icon.ico"


def get_asammdf_path():
    print("[Build] 正在定位 asammdf 的 site-packages 路径...")
    try:
        import asammdf
        asammdf_path = Path(os.path.dirname(asammdf.__file__))
        if not asammdf_path.exists():
            raise FileNotFoundError(f"asammdf 路径不存在: {asammdf_path}")
        print(f"[Build] asammdf 路径: {asammdf_path}")
        return asammdf_path
    except ImportError:
        print("[ERROR] 无法导入 asammdf，请确认当前 Python 环境已安装 asammdf。")
        sys.exit(1)


PYPI_TO_IMPORT = {
    "python-dateutil": "dateutil",
    "typing-extensions": "typing_extensions",
}

# asammdf 的真实运行期依赖（不含 extras）。
# 只有白名单内的包才会被 --include-* 强制打入，避免"环境中装了什么就打什么"。
ASAMMDF_RUNTIME_DEPS = {
    "canmatrix",
    "chardet",
    "deflate",
    "isal",
    "lxml",
    "lz4",
    "numexpr",
    "numpy",
    "pandas",
    "python-dateutil",
    "typing-extensions",
    "zstd",
}


def _resolve_import_name(pkg_name):
    pkg_name = pkg_name.split("[")[0].strip()
    if pkg_name in PYPI_TO_IMPORT:
        return PYPI_TO_IMPORT[pkg_name]
    try_name = pkg_name.replace("-", "_")
    return try_name


def _is_extra_requirement(req_line):
    """判断 requirement 行是否被 extras 限定，例如 pyqtgraph>=0.13.4; extra == "gui"。

    asammdf 的 extras（gui / export / export-matlab-v5 / plot / encryption /
    filesystem / symbolic-math）都不是运行期必需依赖。把它们当作无条件依赖
    会让 --include-package 把 pyqtgraph（含 examples/flowchart/opengl 等
    非绘图子包，约 25 MB 目标代码）、matplotlib、scipy 等整体打进产物。
    """
    marker = req_line.partition(";")[2]
    if not marker:
        return False
    # 只看第一个 "==" 之前的部分，可覆盖 "extra == ..." 与
    # "python_version < ... or extra == ..." 两种写法，同时不会误判
    # platform_machine 之类的环境标记。
    return "extra" in marker.split("==")[0]


def discover_asammdf_deps():
    print("[Build] 正在扫描 asammdf 的传递依赖...")
    try:
        raw_requires = importlib.metadata.requires("asammdf")
    except importlib.metadata.PackageNotFoundError:
        print("[WARN] 无法读取 asammdf 元数据，跳过传递依赖扫描。")
        return [], set()

    include_packages = []
    include_modules = []
    hidden_excludes = set()

    for req_line in raw_requires:
        if _is_extra_requirement(req_line):
            continue

        name = req_line.partition(";")[0].strip()
        pkg_name = name.partition(">=")[0].partition("<")[0].partition("==")[0].partition("~=")[0].strip()
        pkg_name = pkg_name.split("[")[0].strip()

        if pkg_name not in ASAMMDF_RUNTIME_DEPS:
            print(f"  [Build] 跳过非运行期依赖: {pkg_name}")
            continue

        import_name = _resolve_import_name(pkg_name)

        if pkg_name in ("chardet", "numexpr", "numpy", "pandas"):
            continue

        try:
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                continue
        except (ValueError, ModuleNotFoundError):
            continue

        if spec.submodule_search_locations is not None:
            include_packages.append(import_name)
        else:
            include_modules.append(import_name)

        if pkg_name == "lxml":
            hidden_excludes.add("lxml")

        print(f"  [Build] 已发现传递依赖: {import_name}")

    print(f"[Build] 共发现 {len(include_packages) + len(include_modules)} 个传递依赖")
    return include_packages, include_modules, hidden_excludes


def build_nuitka_cmd(include_packages, include_modules, hidden_excludes):
    pyside6_excludes = [
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtWebChannel",
        "PySide6.QtWebSockets",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtQuickControls2",
        "PySide6.QtQuickWidgets",
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DRender",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtGraphs",
        "PySide6.QtGraphsWidgets",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtSpatialAudio",
        "PySide6.QtBluetooth",
        "PySide6.QtNfc",
        "PySide6.QtSensors",
        "PySide6.QtSerialBus",
        "PySide6.QtSerialPort",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtHelp",
        "PySide6.QtLocation",
        "PySide6.QtPositioning",
        "PySide6.QtSql",
        "PySide6.QtDBus",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtStateMachine",
        "PySide6.QtTest",
        "PySide6.QtTextToSpeech",
        "PySide6.QtHttpServer",
        "PySide6.QtDesigner",
        "PySide6.QtUiTools",
    ]

    numpy_excludes = [
        "numpy.f2py",
        "numpy.polynomial",
        "numpy.fft",
        "numpy.testing",
    ]

    misc_excludes = [
        "nuitka",
        "pytest",
        "pyinstaller",
        "tkinter",
        "unittest",
        "test",
        "distutils",
        "setuptools",
        "pip",
        "wheel",
        "scipy",
        "PIL",
        "cv2",
        "matplotlib",
        "pandas.tests",
        "asammdf",

        # --- pyqtgraph 非绘图必需子包 ---
        # 运行时已验证：这些子包在"加载 CSV/MDF/Excel + 主窗口"路径上均未被 import
        # （console 仅在 pyqtgraph.dbg()/stack() 调试函数内导入，opengl 为已废弃
        # 渲染方案遗留）。examples 约 25 MB 目标代码，是本项最大收益来源。
        # 注意：multiprocess / imageview / exporters 会被 pyqtgraph 顶层或其控件
        # 真实导入，不可排除。
        # 另注意：排除 pyqtgraph.opengl 并不会解除对 PySide6.QtOpenGL 的依赖 ——
        # graphicsItems/PlotCurveItem 与 Qt/OpenGLHelpers 仍会动态导入它，
        # 故上面的 --include-module=PySide6.QtOpenGL* 不可删除。
        "pyqtgraph.examples",
        "pyqtgraph.flowchart",
        "pyqtgraph.opengl",
        "pyqtgraph.console",
        "pyqtgraph.jupyter",

        # --- canmatrix 仅保留类型与核心 ---
        # asammdf 只在模块顶层用到 canmatrix / canmatrix.canmatrix.CanMatrix /
        # canmatrix.formats（canmatrix/formats/__init__.py 会逐个尝试导入各格式，
        # 失败被 try/except ImportError 吞掉，因此剔除后不会报错）。
        # 本程序无 DBC/ARXML 加载入口，cli 与 tests 更无必要（cli 同时引入 click）。
        "canmatrix.cli",
        "canmatrix.tests",
        "canmatrix.formats.arxml",
        "canmatrix.formats.odx",
        "canmatrix.formats.xlsx",
        "canmatrix.formats.xls",
        "canmatrix.formats.xls_common",
        "canmatrix.formats.scapy",
        "canmatrix.formats.wireshark",
        "canmatrix.formats.fibex",
        "canmatrix.formats.kcd",
        "canmatrix.formats.ldf",
        "canmatrix.formats.sym",
        "canmatrix.formats.dbf",
        "canmatrix.formats.eds",
        "canmatrix.formats.yaml",

        # --- lxml 仅保留 etree ---
        # asammdf.serde 只使用 lxml.etree 解析 MDF 的 XML 块
        "lxml.html",
        "lxml.objectify",
        "lxml.isoschematron",
        "lxml.cssselect",
        "lxml.sax",
        "lxml.builder",
        "lxml.usedoctest",
        "lxml.doctestcompare",
        "lxml.ElementInclude",
        "lxml.includes",
        "lxml.pyclasslookup",

        # --- 测试与 CLI 入口 ---
        "numexpr.tests",
        "chardet.cli",
        "chardet.__main__",
        "dateutil.zoneinfo.rebuild",

        # --- 仅被懒加载引用的标准库网络栈 ---
        # 来源：src/core/logger.py 的 logging.handlers（smtplib/poplib/imaplib）、
        # pandas.io.common 与 numpy.lib._datasource 的 urllib.request。
        # 运行时已验证这些模块不会在应用启动/加载数据时被 import。
        # 注意：libcrypto 由 _hashlib 使用，不可随 _ssl 一并移除。
        # 注意：asyncio 不可排除 —— 惰性分支的 polars 在 import 期经
        # polars.expr.categorical → typing_extensions.deprecated 触发
        # `import asyncio.coroutines`，排除后 import polars 直接
        # ModuleNotFoundError（实测表现为「parquet 转换失败，回退内存加载」）。
        "ssl",
        "ftplib",
        "imaplib",
        "poplib",
        "smtplib",
        "http",
        "urllib.request",

        # --- 仅因 extras 解析缺陷被强制打入（全项目无 import packaging） ---
        "packaging",
    ]

    cmd = [sys.executable, "-m", "nuitka", "--standalone"]

    py_ver = sys.version_info[:2]
    if py_ver >= (3, 13):
        print(f"[Build] Python {py_ver[0]}.{py_ver[1]} 检测到，Python 3.13+ 不支持 --mingw64，将使用 --clang 编译器")
        print("[Build] 提示: Python 3.13+ 官方发行版已内置 Clang，无需额外安装")
        cmd += ["--clang"]
    else:
        print(f"[Build] Python {py_ver[0]}.{py_ver[1]} 检测到，将使用 --mingw64 编译器")
        if shutil.which("gcc"):
            gcc_version = subprocess.check_output(["gcc", "-dumpversion"], text=True).strip()
            print(f"[Build] GCC 版本: {gcc_version}")
        else:
            print("[Build] 未检测到 GCC，编译可能失败。")
        cmd += ["--mingw64"]

    cmd += [
        f"--report={REPORT_FILE}",
        f"--output-filename={OUTPUT_NAME}",
        "--enable-plugin=pyside6",
        "--include-package=src",
        "--include-module=src.ui.main_window",
        "--include-module=src.ui.widgets.plot_widget",
        "--include-module=src.core.curve_strategy",
        # 阶段二 4.2.1：chardet / numexpr 仅需顶层入口，避免 --include-package 整包强打
        # （chardet 各编码内存按需、numexpr.tests 已被 nofollow 排除）。
        "--include-module=chardet",
        "--include-module=chardet.universaldetector",
        "--include-module=numexpr",
        # 注：numexpr 是单模块包，evaluate 是包内函数名（不含独立子模块），
        # 故仅保留 --include-module=numexpr，不再追加 *.evaluate（会 FATAL 找不到）。
        "--include-module=numpy",
        "--include-module=pandas",
        # asammdf 是收尾阶段整包拷成 .pyc 的，不在 Nuitka 静态 import 图里。
        # v2_v3_blocks 顶层 `from getpass import getuser` 无人替它引用，
        # getpass（Nuitka 按需 stdlib）因此漏收 → MDF 加载报「No module named 'getpass'」。
        # 纯标准库小模块（约 8KB），显式 include。Windows v5 实测：MDF/Excel/CSV 全通过，+5.5KB。
        "--include-module=getpass",
        # polars 是 Rust 扩展：只给 include-package 可能漏掉扩展数据文件，
        # 两条都要（D11）；惰性 parquet 路径必需，Windows 产物已实测复核。
        "--include-package=polars",
        "--include-package-data=polars",
        "--include-module=charset_normalizer",
        "--include-module=ujson",
        # 必须显式 include：pyqtgraph 通过 importlib.import_module(f'{QT_LIB}.QtOpenGL')
        # 动态加载这两个模块（Qt/OpenGLHelpers.py、graphicsItems/PlotCurveItem.py、
        # widgets/RawImageWidget.py），Nuitka 静态分析发现不了。缺失时
        # import pyqtgraph 直接 ModuleNotFoundError，程序无法启动。
        "--include-module=PySide6.QtOpenGL",
        "--include-module=PySide6.QtOpenGLWidgets",
        f"--include-data-dir={ASSETS_DIR}={ASSETS_DIR}",
        f"--include-data-file={README_FILE}={README_FILE}",
        "--include-data-file=docs/help.md=docs/help.md",
        "--include-data-file=docs/template.yaml=docs/template.yaml",
        # 阶段二 4.2.2：移除 --follow-imports。Nuitka 只按静态可判定的 import 递归收集，
        # 可再省下 asyncio/http/ftplib/imaplib/poplib/smtplib/email/ssl 等约 382 个按懒加载
        # 拉入的模块。项目内函数体 import（import openpyxl、chardet/detect、pd.read_excel 引擎）
        # 仍为静态可见会被保留；唯一拼串导入 canmatrix.formats.* 已在 nofollow 排除且被
        # try/except ImportError 包住，src 侧无别的 importlib 动态加载（已核验）。
        # 注意：src 侧无动态 import 这一结论已在惰性分支复核（仅 __import__("PySide6")
        # 与 importlib.metadata）；但「静态可见」不等于「运行期一定够用」，
        # polars 的 asyncio 缺失就是反例，见 misc_excludes 中 asyncio 的注释。
        "--no-deployment-flag=excluded-module-usage",
        # 剥离 docstring / assert 常量：pandas、pyqtgraph、numpy、openpyxl
        # 的 docstring 体量极大，是 exe 体积的主要来源之一。
        "--python-flag=no_docstrings",
        "--python-flag=no_asserts",
        "--lto=yes",
        "--jobs=8",
        "--windows-console-mode=disable",
        f"--windows-icon-from-ico={ICON_FILE}",
    ]

    # —— 阶段二 4.2.1：对只需顶层入口的依赖改用细粒度 include-module，替代 --include-package 整包 ——
    # lxml：asammdf.serde 仅用 .etree 解析 MDF XML 块，只需 etree + _elementpath。
    # chardet / numexpr 已在上面 cmd 硬编码区细化为 include-module。
    # canmatrix 例外保留整包：canmatrix/__init__.py 顶层 `import canmatrix.formats`，
    # 且 asammdf.mdf 顶层 `from canmatrix import CanMatrix`，format 装载按需 try/except；
    # 改细粒度收益低、回归风险高。（其 cli/tests 及排除外的 formats 子项已由 nofollow 剔除。）
    fine_grained_modules = {
        "lxml": ["lxml.etree", "lxml._elementpath"],
    }
    for pkg in include_packages:
        if pkg in fine_grained_modules:
            for _mod in fine_grained_modules[pkg]:
                cmd.insert(-1, f"--include-module={_mod}")
            continue
        cmd.insert(-1, f"--include-package={pkg}")
    for mod in include_modules:
        cmd.insert(-1, f"--include-module={mod}")

    cmd.append(str(ENTRY_FILE))

    for mod in pyside6_excludes:
        cmd.insert(-1, f"--nofollow-import-to={mod}")

    for mod in numpy_excludes:
        cmd.insert(-1, f"--nofollow-import-to={mod}")

    for mod in misc_excludes:
        if mod in hidden_excludes:
            continue
        cmd.insert(-1, f"--nofollow-import-to={mod}")

    return cmd


def generate_build_info():
    """生成 src/_build_info.py，注入版本号与编译时间（由 _version.py 运行时读取）。"""
    print("[Build] 正在生成构建信息 (_build_info.py)...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_build_info.py")]
    )
    if result.returncode != 0:
        print("[ERROR] 构建信息生成失败!")
        sys.exit(result.returncode)


def cleanup_build_info():
    """构建结束后清理 src/_build_info.py，避免陈旧数据污染开发环境。"""
    target = PROJECT_ROOT / "src" / "_build_info.py"
    if target.exists():
        target.unlink()
        print("[Build] 已清理临时构建信息文件 src/_build_info.py")


def build():
    os.chdir(PROJECT_ROOT)

    generate_build_info()

    try:
        _run_build()
    finally:
        cleanup_build_info()


def _run_build():
    include_packages, include_modules, hidden_excludes = discover_asammdf_deps()

    cmd = build_nuitka_cmd(include_packages, include_modules, hidden_excludes)

    print("[Build] ========================================")
    print(f"[Build] 开始 Nuitka 编译: {OUTPUT_NAME}")
    print("[Build] ========================================")
    print(f"[Build] 命令: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print()
        print("[Build] ========================================")
        print("[ERROR] Nuitka 编译失败!")
        print("[Build] ========================================")
        sys.exit(result.returncode)

    print()
    print("[Build] ========================================")
    print("[Build] Nuitka 编译成功!")
    print("[Build] ========================================")

    dist_dir = PROJECT_ROOT / f"{OUTPUT_NAME}.dist"
    if not dist_dir.exists():
        print(f"[ERROR] 未找到输出目录: {dist_dir}")
        sys.exit(1)

    print(f"[Build] 输出目录: {dist_dir}")

    asammdf_src = get_asammdf_path()
    asammdf_dst = dist_dir / "asammdf"

    print(f"[Build] 开始复制 asammdf -> {asammdf_dst}")
    if asammdf_dst.exists():
        shutil.rmtree(asammdf_dst)
    shutil.copytree(asammdf_src, asammdf_dst)

    print("[Build] asammdf 复制完成")

    print("[Build] 正在将 asammdf 源码编译为 .pyc 字节码并清除 .py 源文件...")
    optimize = 2
    pyc_count = 0
    py_removed = 0
    pycache_dirs = []

    for root, dirs, files in os.walk(str(asammdf_dst)):
        for d in dirs:
            if d == "__pycache__":
                pycache_dirs.append(os.path.join(root, d))
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            src = os.path.join(root, f)
            dst = os.path.join(root, f + "c")
            try:
                py_compile.compile(src, cfile=dst, dfile=f, optimize=optimize, quiet=1)
                pyc_count += 1
            except py_compile.PyCompileError as e:
                print(f"  [WARN] 编译失败, 保留源文件: {src} ({e})")
                continue
            os.remove(src)
            py_removed += 1

    for d in pycache_dirs:
        shutil.rmtree(d, ignore_errors=True)

    print(f"[Build] .pyc 编译与清理完成 (编译 {pyc_count} 个, 清理 {py_removed} 个 .py, 清理 {len(pycache_dirs)} 个 __pycache__)")

    print()
    print("[Build] ========================================")
    print("[Build] 混合打包完成!")
    print(f"[Build] 输出: {dist_dir / f'{OUTPUT_NAME}.exe'}")
    print("[Build] ========================================")


if __name__ == "__main__":
    build()
