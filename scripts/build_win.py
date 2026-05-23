import os
import sys
import shutil
import subprocess
import compileall
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


def build_nuitka_cmd():
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
        "numpy.random",
        "numpy.f2py",
        "numpy.ma",
        "numpy.polynomial",
        "numpy.linalg",
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
        "lxml",
        "PIL",
        "cv2",
        "matplotlib",
        "pandas.tests",
        "pandas.plotting",
        "asammdf",
    ]

    cmd = ["nuitka", "--standalone"]

    if shutil.which("gcc"):
        gcc_version = subprocess.check_output(["gcc", "-dumpversion"], text=True).strip()
        print(f"[Build] GCC 版本: {gcc_version}")
    else:
        print("[Build] 未检测到 GCC，编译可能失败。")

    cmd += [
        "--mingw64",
        "--report", REPORT_FILE,
        "--output-filename", OUTPUT_NAME,
        "--enable-plugin", "pyside6",
        "--include-module", "PySide6.QtOpenGL",
        "--include-module", "PySide6.QtOpenGLWidgets",
        "--include-package", "src",
        "--include-package", "numexpr",
        "--include-module", "numpy",
        "--include-module", "pandas",
        "--include-package", "chardet",
        "--include-module", "charset_normalizer",
        "--include-module", "ujson",
        "--include-data-dir", f"{ASSETS_DIR}={ASSETS_DIR}",
        "--include-data-file", f"{README_FILE}={README_FILE}",
        "--follow-imports",
        "--lto=yes",
        "--jobs=8",
        "--windows-console-mode=disable",
        "--windows-icon-from-ico", ICON_FILE,
        str(ENTRY_FILE),
    ]

    for mod in pyside6_excludes:
        cmd.insert(-1, f"--nofollow-import-to={mod}")

    for mod in numpy_excludes:
        cmd.insert(-1, f"--nofollow-import-to={mod}")

    for mod in misc_excludes:
        cmd.insert(-1, f"--nofollow-import-to={mod}")

    return cmd


def build():
    os.chdir(PROJECT_ROOT)

    cmd = build_nuitka_cmd()

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

    print("[Build] 正在将 asammdf 源码编译为 .pyc 字节码...")
    optimize = 2
    compileall.compile_dir(
        str(asammdf_dst),
        maxlevels=20,
        ddir=str(asammdf_dst),
        force=True,
        optimize=optimize,
        quiet=1,
    )
    print("[Build] .pyc 编译完成")

    print("[Build] 正在清除 asammdf 原始 .py 源码文件...")
    py_files_removed = 0
    for root, dirs, files in os.walk(str(asammdf_dst)):
        for f in files:
            if f.endswith(".py"):
                file_path = os.path.join(root, f)
                os.remove(file_path)
                py_files_removed += 1
    print(f"[Build] 已清除 {py_files_removed} 个 .py 源码文件")

    print()
    print("[Build] ========================================")
    print(f"[Build] 混合打包完成!")
    print(f"[Build] 输出: {dist_dir / f'{OUTPUT_NAME}.exe'}")
    print("[Build] ========================================")


if __name__ == "__main__":
    build()
