"""component 测试共享夹具：可独立绘图的 plot widget"""


import pandas as pd
import pytest

from PySide6.QtWidgets import QWidget

from src.core.config import VAR_INFO_COPY_BTN_MARGIN, VAR_INFO_COPY_BTN_SIZE, widget_alive
from src.data.loader import FastDataLoader
from src.data.mdf_lazy_loader import MDFLazyLoader
from src.ui.dialogs.variable_info_dialog import VariableInfoDialog
from src.ui.table_dialog import DataTableDialog
from tests.component._vid_shared import FakeMainWindow, _wait_stats
from tests.fixtures.data_factory import write_csv, write_mdf
from tests.fixtures.waits import flush_deferred_deletes, settle


class FakePlotContext:
    """最小 PlotContext 替身（真实实现见 src/app/plot_context.py，依赖 MainWindow 服务）。

    覆盖绘图主路径访问的属性：value_cache / loader / _enum_text_maps /
    request_mark_stats_refresh，使 DraggableGraphicsLayoutWidget 可脱离
    MainWindow 独立执行 plot_variable / dropEvent。
    """

    def __init__(self):
        self.value_cache = {}
        self.loader = None
        self._enum_text_maps = {}
        # 清除绘图播报落点：记录 (label, curves)，用例断言"报了什么、报了几条"
        self.cleared_announces = []

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass

    def announce_cleared(self, label: str, curves: int) -> None:
        self.cleared_announces.append((label, curves))


class FakeLayoutManager:
    """layout_manager 替身：覆盖 drop/拖拽路径访问的方法（均 no-op）"""

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass

    def _hide_drag_indicator_for_plot(self, pw):
        pass

    def _show_drag_indicator_for_plot(self, pw, var_names, text=None):
        pass

    def _get_plot_container(self, pw):
        return None


class FakeHost(QWidget):
    """伪宿主窗口：提供 layout_manager 替身。

    plot_widget.dropEvent 末尾会调 self.window().layout_manager.
    request_mark_stats_refresh()，独立顶层 widget 缺少该属性会抛
    AttributeError（真实应用中 window() 为 MainWindow，不暴露）。
    """

    def __init__(self):
        super().__init__()
        self.layout_manager = FakeLayoutManager()


@pytest.fixture()
def plot_factory(qapp):
    """构造可独立绘图的 DraggableGraphicsLayoutWidget 工厂。

    每个生成的 widget 自动完成：
    1. 注入 FakePlotContext（否则绘图主路径触发 QMessageBox 阻塞，见
       tests/README.md 陷阱清单 #1）；
    2. setParent(FakeHost)，使 window().layout_manager 可解析；
    3. 登记到 created/hosts 列表，fixture 收尾时统一 deleteLater。

    df 缺省时注入 a/b/c 三列小型 DataFrame。
    """
    created = []
    hosts = []

    def _make(df: pd.DataFrame | None = None):
        from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget
        if df is None:
            df = pd.DataFrame({
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            })
        pw = DraggableGraphicsLayoutWidget({}, df)
        pw.plot_context = FakePlotContext()
        host = FakeHost()
        pw.setParent(host)  # window() 返回 host，提供 layout_manager 替身
        created.append(pw)
        hosts.append(host)
        return pw

    yield _make

    for pw in created:
        pw._is_being_destroyed = True
        pw.deleteLater()
    for host in hosts:
        host.deleteLater()


# --------------------------------------------------------------------------
# 变量信息弹窗（B3 拆分后的共享夹具）
# --------------------------------------------------------------------------
@pytest.fixture()
def env(qapp, tmp_path):
    """每个测试一套独立环境：合成 CSV + 替身主窗口 + 单例复位。

    复位必须放在 yield **之前**也执行一次：单例是类级状态，若上一个测试
    异常退出未清理，本测试会继承到脏页面。
    """
    VariableInfoDialog.reset_for_tests()
    # reset_for_tests 已 shutdown_worker（join 线程）并把窗口 deleteLater：
    # 这里只需让删除落地，并排空它留下的投递事件
    flush_deferred_deletes()

    path_a = write_csv(
        tmp_path / "a.csv",
        header=["time", "speed", "load", "note"],
        units=["s", "km/h", "%", "-"],
        rows=[
            ["0.0", "10.0", "20.0", "aa"],
            ["0.1", "20.0", "30.0", "bb"],
            ["0.2", "30.0", "40.0", "cc"],
        ],
    )
    # reload 用：刻意缺 load / note，用于验证"变量不在新数据中"的失效标注
    path_b = write_csv(
        tmp_path / "b.csv",
        header=["time", "speed"],
        units=["s", "km/h"],
        rows=[["0.0", "1.0"], ["0.1", "5.0"]],
    )

    loader_a = FastDataLoader(str(path_a), has_unit=True, sep=",")
    mw = FakeMainWindow(loader_a)

    class Env:
        pass

    e = Env()
    e.mw = mw
    e.loader_a = loader_a
    e.path_b = path_b
    e.tmp_path = tmp_path

    yield e

    VariableInfoDialog.reset_for_tests()
    flush_deferred_deletes()
    mw.loader = None
    for attr in ("var_stats_cache",):
        setattr(mw, attr, {})
    loader_a.release_memory()
    mw.deleteLater()
    flush_deferred_deletes()


@pytest.fixture()
def many_loader(qapp, tmp_path):
    """含 v1..v8 八列的合成 CSV，用于验证标签页上限截断。

    用真实 loader 而非测试替身：截断逻辑依赖 build_snapshot 与
    compute_stats 的真实返回，替身反而会把真正要测的集成面遮掉。
    """
    names = [f"v{i}" for i in range(1, 9)]
    path = write_csv(
        tmp_path / "many.csv",
        header=["time"] + names,
        units=["s"] + ["-"] * len(names),
        rows=[["0.0"] + [str(i + 1.0) for i in range(len(names))],
              ["0.1"] + [str(i + 2.0) for i in range(len(names))]],
    )
    loader = FastDataLoader(str(path), has_unit=True, sep=",")
    yield loader, names
    loader.release_memory()


@pytest.fixture()
def mdf_env(qapp, tmp_path):
    """一套带归属信息的合成 MDF4 环境（不复用 ``env``）。

    不复用是因为 ``env`` 走 CSV 路径、永远不可能是归属块的正面样本；
    两者各守一侧：本夹具验「MDF 该有的东西有」，``env`` 验「CSV 不应
    有的东西没有」。

    变量名直接用 ``ATTRIBUTION_*`` 常量而不是字面量：改名时只需改工厂，
    否则测试会因变量不存在而降级成“错误页也能通过断言”的空转。
    """
    VariableInfoDialog.reset_for_tests()
    flush_deferred_deletes()

    # 12 点与 unit 测试的 attr4_loader 一致；点数不影响归属提取（全部取自
    # 加载期已解析的块结构），只影响统计回填的数值
    path = write_mdf(
        tmp_path / "attr.mf4", version="4.10", n=12, with_attribution=True
    )
    loader = MDFLazyLoader(str(path))
    mw = FakeMainWindow(loader)

    class Env:
        pass

    e = Env()
    e.mw = mw
    e.loader = loader

    yield e

    VariableInfoDialog.reset_for_tests()
    flush_deferred_deletes()
    mw.loader = None
    loader.close()
    mw.deleteLater()
    flush_deferred_deletes()


@pytest.fixture()
def submit_spy(env):
    """返回一个函数：给当前单例的 worker.submit 装上记录器。

    不能在 popup **之前**装：弹窗本身就要靠 submit 提交首批任务。
    也不能靠 queue_size()==0 反推“未提交”：任务可能已经跑完，
    队列自然为空。
    """
    calls: list = []

    def install(dlg):
        original = dlg.worker.submit

        def spy(jobs):
            calls.extend(jobs)
            return original(jobs)

        dlg.worker.submit = spy
        return original

    yield install, calls


@pytest.fixture()
def page(env):
    """当前可见的单标签页（统计已回填），并卡住几何前置条件。

    只测当前可见页：QStackedWidget 下非当前页没参与布局，单元格矩形全是
    零尺寸，拿它测坐标等于空转。
    """
    dlg = VariableInfoDialog.popup(["speed"], parent=env.mw)
    assert _wait_stats(dlg, "speed")
    # offscreen 默认屏幕仅 800x600，超屏尺寸会被夹住，故取一个安全尺寸
    dlg.resize(700, 520)
    settle()

    pg = dlg._pages["speed"]
    reserved = VAR_INFO_COPY_BTN_SIZE + 2 * VAR_INFO_COPY_BTN_MARGIN
    # 前置条件而非功能断言：列宽不够时按钮热区会贴到单元格左边界外，
    # 后面的坐标就没有意义
    assert pg.tree.columnWidth(1) > 4 * reserved, f"值列过窄：{pg.tree.columnWidth(1)}"
    assert pg.tree.viewport().size().isValid()
    return dlg, pg


# --------------------------------------------------------------------------
# 变量数值表 tab 模式（B3 拆分后的共享夹具）
# --------------------------------------------------------------------------
@pytest.fixture()
def mdf_loader(qapp, tmp_path):
    """合成 MDF4：G0 Press_G0(bar) 12 点 @0.1s；G1 Press_G1(degC) 6 点 @0.2s。"""
    path = write_mdf(tmp_path / "tab.mf4", version="4.10", n=12)
    return MDFLazyLoader(str(path))


@pytest.fixture()
def tab_dialog(qapp, mdf_loader, monkeypatch):
    monkeypatch.setattr(DataTableDialog, "_instance", None)
    dlg = DataTableDialog()
    dlg._resolve_loader = lambda: mdf_loader
    yield dlg
    # closeEvent 现在会 deleteLater()：自关闭的用例（空表自动关闭）走到这里时
    # C++ 骨架可能已被销毁，收尾必须先问存活。
    if widget_alive(dlg):
        dlg.hide()
        dlg.deleteLater()
    flush_deferred_deletes()


# ---------- 锚点污染回归（线上反馈：切换后跳表末尾 / 定位后跳 0s） ----------
# 需要真实布局：show + 小窗口让 G0（12 行）的滚动条有非零范围，
# 否则 setValue 被钳到 0、发不出 valueChanged，测试退化成空转。

@pytest.fixture()
def shown_tab_dialog(tab_dialog):
    tab_dialog.resize(420, 200)
    tab_dialog.show()
    # show() 是同步调用，这里只需把入队的事件（布局/曝光）排空；后续用例
    # 各自在 _add_variable_to_tab 后再 settle()，布局会持续被推进
    settle(5)
    return tab_dialog
