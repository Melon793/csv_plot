"""全局测试环境配置（Phase 1 测试基础设施）

职责：
1. 在任何 Qt / src import 之前设置 offscreen 平台与路径注入环境变量
2. 提供会话级 QApplication（qapp）
3. 提供 AppSettings 单例隔离 fixture
4. 按目录自动打 unit/component/e2e/perf marker
"""

import os
import sys
import tempfile

# ★ 必须在任何 Qt import 之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

# 会话级隔离根目录：配置 / 日志 / 模板全部重定向，避免污染真实用户环境
_TEST_ROOT = tempfile.mkdtemp(prefix="csv_plot_test_")
os.environ["CSV_PLOT_CONFIG_DIR"] = os.path.join(_TEST_ROOT, "config")
os.environ["CSV_PLOT_LOG_DIR"] = os.path.join(_TEST_ROOT, "logs")

# 保证项目根目录在 sys.path 中（src.* 可导入）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest


@pytest.fixture(scope="session")
def qapp():
    """会话级 QApplication（offscreen）。

    pytest-qt 的 qtbot 依赖已存在的 QApplication 实例，
    会话级作用域避免每个用例重复创建/销毁 QApplication。
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setApplicationName("csv-plot-test")
    app.setOrganizationName("CSVPlotTest")
    yield app


@pytest.fixture
def app_settings(qapp):
    """隔离的 AppSettings 单例：每个用例前后重置，指向测试配置目录。"""
    from src.core.settings import AppSettings

    AppSettings._reset_for_tests()
    settings = AppSettings()
    yield settings
    AppSettings._reset_for_tests()


def pytest_collection_modifyitems(items):
    """按测试文件所在目录自动打分层 marker。"""
    for item in items:
        path = str(item.fspath).replace(os.sep, "/")
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/component/" in path:
            item.add_marker(pytest.mark.component)
        elif "/e2e/" in path:
            item.add_marker(pytest.mark.e2e)
        elif "/perf/" in path:
            item.add_marker(pytest.mark.perf)
