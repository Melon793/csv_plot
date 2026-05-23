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


def _resolve_import_name(pkg_name):
    pkg_name = pkg_name.split("[")[0].strip()
    if pkg_name in PYPI_TO_IMPORT:
        return PYPI_TO_IMPORT[pkg_name]
    try_name = pkg_name.replace("-", "_")
    return try_name


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
        name = req_line.partition(";")[0].strip()
        pkg_name = name.partition(">=")[0].partition("<")[0].partition("==")[0].partition("~=")[0].strip()
        pkg_name = pkg_name.split("[")[0].strip()
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
    ]

    cmd = [sys.executable, "-m", "nuitka", "--standalone"]

    if shutil.which("gcc"):
        gcc_version = subprocess.check_output(["gcc", "-dumpversion"], text=True).strip()
        print(f"[Build] GCC 版本: {gcc_version}")
    else:
        print("[Build] 未检测到 GCC，编译可能失败。")

    cmd += [
        "--mingw64",
        f"--report={REPORT_FILE}",
        f"--output-filename={OUTPUT_NAME}",
        "--enable-plugin=pyside6",
        "--include-module=PySide6.QtOpenGL",
        "--include-module=PySide6.QtOpenGLWidgets",
        "--include-package=src",
        "--include-package=numexpr",
        "--include-module=numpy",
        "--include-module=pandas",
        "--include-package=chardet",
        "--include-module=charset_normalizer",
        "--include-module=ujson",
        f"--include-data-dir={ASSETS_DIR}={ASSETS_DIR}",
        f"--include-data-file={README_FILE}={README_FILE}",
        "--follow-imports",
        "--no-deployment-flag=excluded-module-usage",
        "--lto=yes",
        "--jobs=8",
        "--windows-console-mode=attach",
        f"--windows-icon-from-ico={ICON_FILE}",
    ]

    for pkg in include_packages:
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


def build():
    os.chdir(PROJECT_ROOT)

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
    print(f"[Build] 混合打包完成!")
    print(f"[Build] 输出: {dist_dir / f'{OUTPUT_NAME}.exe'}")
    print("[Build] ========================================")


if __name__ == "__main__":
    build()
