"""枚举通道自动阶梯（v1 保守版）组件测试：三处构造点端到端。

覆盖 decide_initial_draw_mode 在真实绘图路径的接线：
- CSV（loader=None / 无 get_metadata）→ 直线，source=='auto'
- 枚举通道（get_metadata → is_enum=True）→ stepMode=='right'，
  draw_mode=='step-post'，source=='auto'（含批量与 plot_variable 路径）
- get_metadata 抛 KeyError / 时间通道 → 直线（防御）
- 用户手动切回直线 → source=='user'（apply_curve_draw_mode 钉住）
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.waits import settle

from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_SOURCE_AUTO,
    DRAW_MODE_SOURCE_USER,
    DRAW_MODE_STEP_POST,
)
from src.data.metadata import VarMetadata


def _enum_meta(name="v", **overrides) -> VarMetadata:
    defaults = dict(name=name, unit="", is_enum=True)
    defaults.update(overrides)
    return VarMetadata(**defaults)


def _mock_loader(metas: dict[str, VarMetadata] | None = None, *, raise_key=False):
    """带 get_metadata 的最小 loader 替身（模拟 MDF/BLF 通道类数据源）。"""
    metas = metas or {}

    def get_metadata(var_name: str):
        if raise_key:
            raise KeyError(var_name)
        return metas.get(var_name)

    return SimpleNamespace(get_metadata=get_metadata)


def _enum_df(n: int = 20) -> pd.DataFrame:
    # 档位型数据：1/2/3 轮换（码值离散），CSV 场景下不判定仍画直线
    rng = np.random.default_rng(7)
    return pd.DataFrame({"v": rng.choice([1.0, 2.0, 3.0], size=n)})


@pytest.fixture()
def csv_plot(plot_factory, qapp):
    """CSV-like 场景：FakePlotContext.loader=None（无通道元数据）。"""
    n = 20
    pw = plot_factory(_enum_df(n))
    pw.resize(800, 600)
    assert pw.add_variable_to_plot("v")
    settle()
    return pw


@pytest.fixture()
def enum_plot(plot_factory, qapp):
    """枚举通道场景：loader.get_metadata → is_enum=True。"""
    n = 20
    df = _enum_df(n)
    pw = plot_factory(df)
    pw.plot_context.loader = _mock_loader({"v": _enum_meta("v")})
    pw.resize(800, 600)
    assert pw.add_variable_to_plot("v")
    settle()
    return pw


# ---------------------------------------------------------------------------
# CSV / Excel：永不判定
# ---------------------------------------------------------------------------

def test_csv_stays_line(csv_plot):
    """CSV（无 get_metadata）：0/1/档位数据也保持直线（保守边界）。"""
    ci = csv_plot.curves["v"]
    assert ci.draw_mode == DRAW_MODE_AUTO
    assert ci.draw_mode_source == DRAW_MODE_SOURCE_AUTO
    assert ci.curve.opts["stepMode"] is None


def test_loader_without_get_metadata_stays_line(plot_factory, qapp):
    """loader 存在但没有 get_metadata（FastDataLoader 形状）→ 直线。"""
    pw = plot_factory(_enum_df())
    pw.plot_context.loader = SimpleNamespace()  # 仅有名字，无 get_metadata
    assert pw.add_variable_to_plot("v")
    settle()
    assert pw.curves["v"].curve.opts["stepMode"] is None


# ---------------------------------------------------------------------------
# 枚举通道：自动 step-post（三条构造路径）
# ---------------------------------------------------------------------------

def test_enum_channel_auto_steps(enum_plot):
    """add_variable_to_plot 路径：枚举通道 → stepMode=='right'。"""
    ci = enum_plot.curves["v"]
    assert ci.draw_mode == DRAW_MODE_STEP_POST
    assert ci.draw_mode_source == DRAW_MODE_SOURCE_AUTO
    assert ci.curve.opts["stepMode"] == "right"


def test_enum_channel_via_plot_variable(plot_factory, qapp):
    """plot_variable 首绘路径：枚举通道 → stepMode=='right'。"""
    pw = plot_factory(_enum_df())
    pw.plot_context.loader = _mock_loader({"v": _enum_meta("v")})
    pw.resize(800, 600)
    pw.plot_variable("v")
    settle()
    assert pw.curves["v"].curve.opts["stepMode"] == "right"
    assert pw.curves["v"].draw_mode == DRAW_MODE_STEP_POST


def test_enum_channel_batch_path(plot_factory, qapp):
    """批量拖入路径（_batch_adding）：枚举 → 阶梯，非枚举 → 直线。"""
    df = _enum_df(20)
    df["cont"] = np.linspace(0.0, 1.0, 20)  # 连续量对照
    pw = plot_factory(df)
    pw.plot_context.loader = _mock_loader({
        "v": _enum_meta("v"),
        "cont": VarMetadata(name="cont", unit="V"),
    })
    pw.resize(800, 600)
    pw._multi_curve_manager.add_variables_to_plot(["v", "cont"])
    settle()
    assert pw.curves["v"].curve.opts["stepMode"] == "right"
    assert pw.curves["cont"].curve.opts["stepMode"] is None


# ---------------------------------------------------------------------------
# 防御路径：异常与时间通道
# ---------------------------------------------------------------------------

def test_get_metadata_keyerror_stays_line(plot_factory, qapp):
    """get_metadata 抛 KeyError（数据源关闭）→ 直线，不影响加曲线。"""
    pw = plot_factory(_enum_df())
    pw.plot_context.loader = _mock_loader(raise_key=True)
    assert pw.add_variable_to_plot("v")
    settle()
    assert pw.curves["v"].curve.opts["stepMode"] is None


def test_time_channel_stays_line(plot_factory, qapp):
    """时间通道防御性否决：即使 is_enum 异常为真也直线。"""
    pw = plot_factory(_enum_df())
    pw.plot_context.loader = _mock_loader({
        "v": _enum_meta("v", is_time_channel=True),
    })
    assert pw.add_variable_to_plot("v")
    settle()
    assert pw.curves["v"].curve.opts["stepMode"] is None


# ---------------------------------------------------------------------------
# 用户钉住：手动选择写入 source=='user'
# ---------------------------------------------------------------------------

def test_user_override_pins_source(enum_plot):
    """用户手动切回直线 → source=='user'，draw_mode=='auto'。"""
    pw = enum_plot
    assert pw.curves["v"].draw_mode == DRAW_MODE_STEP_POST
    assert pw._multi_curve_manager.apply_curve_draw_mode("v", DRAW_MODE_AUTO)
    ci = pw.curves["v"]
    assert ci.draw_mode == DRAW_MODE_AUTO
    assert ci.draw_mode_source == DRAW_MODE_SOURCE_USER
    assert ci.curve.opts["stepMode"] is None


def test_user_switch_to_step_on_csv(csv_plot):
    """CSV 曲线手动选阶梯 → 正常生效（手动路径不受自动判定限制）。"""
    pw = csv_plot
    assert pw._multi_curve_manager.apply_curve_draw_mode("v", DRAW_MODE_STEP_POST)
    assert pw.curves["v"].curve.opts["stepMode"] == "right"
    assert pw.curves["v"].draw_mode_source == DRAW_MODE_SOURCE_USER


# ---------------------------------------------------------------------------
# reload：用户钉住的 draw_mode 跨 reload 保留（replots_after_loading 路径）
# ---------------------------------------------------------------------------

from collections import OrderedDict  # noqa: E402

from PySide6.QtWidgets import QPushButton, QWidget  # noqa: E402

from src.ui.cursor_sync_manager import CursorSyncManager  # noqa: E402
from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget  # noqa: E402


class EnumMutableLoader:
    """通道类 loader 替身：df 可替换 + get_metadata 返回枚举/连续元数据。"""

    LOADER_TYPE = "mdf"

    def __init__(self, df: pd.DataFrame, metas: dict[str, VarMetadata]):
        self.df = df
        self.metas = metas
        self.datalength = int(len(df))
        self.var_names = list(df.columns)
        self.df_validity = {v: 1 for v in df.columns}
        self.units = {}
        self.time_channels_info = {}
        self.time_column_name = None
        self.time_axis_label = "Time"

    def get_metadata(self, var_name: str):
        return self.metas.get(var_name)

    def get_series(self, var_name: str):
        return self.df[var_name]

    @property
    def global_time_range(self) -> tuple[float, float]:
        return (0.0, float(self.datalength))


class _ReloadHost(QWidget):
    """最小宿主：覆盖 replots_after_loading 链路访问的成员。"""

    def __init__(self):
        super().__init__()
        self.cursor_btn = QPushButton()
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.setChecked(True)
        self.cursor_values_hidden = False
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []
        self._is_loading_new_data = False
        self._is_syncing_crosshair = False
        self._pending_crosshair_x = None
        self._crosshair_update_timer = None
        self.plot_widgets = []
        self.loader = None
        self.value_cache = OrderedDict()
        self._enum_text_maps = {}
        self._baseline_density = 0.0
        self._global_max_density = 0.0
        self.cursor_sync_manager = None
        self.mark_stats_window = None
        self.status_messages: list[str] = []

    def _show_status_message(self, message: str):
        self.status_messages.append(message)


class _ReloadCtx:
    def __init__(self, host, loader):
        self._host = host
        self.loader = loader
        self.plot_widgets = []
        self.cleared_announces = []

    @property
    def value_cache(self):
        return self._host.value_cache

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class _Container:
    def __init__(self, pw):
        self.plot_widget = pw

    def isVisible(self):
        return True


@pytest.fixture()
def reload_env(qapp):
    """单页 + 通道类 loader（v 枚举 / c 连续）+ 真实 CursorSyncManager。"""
    df = _enum_df(30)
    df["c"] = np.linspace(0.0, 1.0, 30)
    metas = {"v": _enum_meta("v"), "c": VarMetadata(name="c", unit="V")}
    loader = EnumMutableLoader(df, metas)

    host = _ReloadHost()
    ctx = _ReloadCtx(host, loader)
    pw = DraggableGraphicsLayoutWidget({}, df)
    pw.plot_context = ctx
    pw.setParent(host)
    pw.resize(700, 260)
    pw.show()

    container = _Container(pw)
    host.plot_widgets.append(container)
    ctx.plot_widgets = list(host.plot_widgets)
    host.loader = loader

    manager = CursorSyncManager(host)
    host.cursor_sync_manager = manager

    # 初始曲线：v（枚举，自动 step-post）+ c（连续，直线）
    assert pw.add_variable_to_plot("v")
    assert pw.add_variable_to_plot("c")
    settle()

    def reload_with(columns: dict[str, list]):
        new_df = pd.DataFrame(columns)
        loader.df = new_df
        loader.datalength = len(new_df)
        loader.var_names = list(new_df.columns)
        loader.df_validity = {v: 1 for v in new_df.columns}
        for c in host.plot_widgets:
            c.plot_widget.data = new_df
        manager.replots_after_loading()

    yield host, pw, reload_with

    pw._is_being_destroyed = True
    pw.deleteLater()
    host.deleteLater()


def test_reload_keeps_user_pinned_step(reload_env):
    """用户钉住 step-post → reload → 阶梯与 'user' 来源都保留。"""
    host, pw, reload_with = reload_env
    assert pw._multi_curve_manager.apply_curve_draw_mode("c", DRAW_MODE_STEP_POST)
    assert pw.curves["c"].draw_mode_source == DRAW_MODE_SOURCE_USER

    reload_with({"v": list(_enum_df(30)["v"]), "c": list(np.linspace(0.0, 1.0, 30))})

    assert "c" in pw.curves
    ci = pw.curves["c"]
    assert ci.draw_mode == DRAW_MODE_STEP_POST
    assert ci.draw_mode_source == DRAW_MODE_SOURCE_USER
    assert ci.curve.opts["stepMode"] == "right"


def test_reload_keeps_user_pinned_line_on_enum(reload_env):
    """枚举通道用户改回直线 → reload → 不被自动判定翻回阶梯。"""
    host, pw, reload_with = reload_env
    assert pw.curves["v"].draw_mode == DRAW_MODE_STEP_POST  # 自动判定生效
    assert pw._multi_curve_manager.apply_curve_draw_mode("v", DRAW_MODE_AUTO)

    reload_with({"v": list(_enum_df(30)["v"]), "c": list(np.linspace(0.0, 1.0, 30))})

    ci = pw.curves["v"]
    assert ci.draw_mode == DRAW_MODE_AUTO
    assert ci.draw_mode_source == DRAW_MODE_SOURCE_USER
    assert ci.curve.opts["stepMode"] is None


def test_reload_redecides_auto_source(reload_env):
    """'auto' 来源的自动阶梯 → reload → 按元数据重新判定，仍 step-post。"""
    host, pw, reload_with = reload_env
    assert pw.curves["v"].draw_mode == DRAW_MODE_STEP_POST
    assert pw.curves["v"].draw_mode_source == DRAW_MODE_SOURCE_AUTO

    reload_with({"v": list(_enum_df(30)["v"]), "c": list(np.linspace(0.0, 1.0, 30))})

    ci = pw.curves["v"]
    assert ci.draw_mode == DRAW_MODE_STEP_POST
    assert ci.draw_mode_source == DRAW_MODE_SOURCE_AUTO
    assert ci.curve.opts["stepMode"] == "right"


def test_reload_continuous_stays_line(reload_env):
    """连续通道 reload 后仍直线（自动判定与 reload 均不改）。"""
    host, pw, reload_with = reload_env
    reload_with({"v": list(_enum_df(30)["v"]), "c": list(np.linspace(0.0, 1.0, 30))})
    assert pw.curves["c"].draw_mode == DRAW_MODE_AUTO
    assert pw.curves["c"].curve.opts["stepMode"] is None


# ---------------------------------------------------------------------------
# 状态栏播报：交互入口播报，quiet 程序化路径静默
# ---------------------------------------------------------------------------

class _StatusHost(QWidget):
    """最小宿主：收集 _show_status_message 播报。"""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def _show_status_message(self, message: str):
        self.messages.append(message)


def _attach_host(pw):
    host = _StatusHost()
    pw.setParent(host)
    pw._announce_host = host  # 保活引用：host 被 GC 会连带删除 pw
    return host


def test_announce_single_enum(plot_factory, qapp):
    """单条添加枚举通道 → 播报 1 条（无中间态说明）。"""
    pw = plot_factory(_enum_df())
    pw.plot_context.loader = _mock_loader({"v": _enum_meta("v")})
    host = _attach_host(pw)
    assert pw.add_variable_to_plot("v")
    settle()
    assert any("枚举" in m and "v" in m for m in host.messages)


def test_no_announce_for_continuous(plot_factory, qapp):
    """连续通道添加 → 不播报。"""
    df = _enum_df()
    df["c"] = np.linspace(0.0, 1.0, 20)
    pw = plot_factory(df)
    pw.plot_context.loader = _mock_loader({
        "v": _enum_meta("v"), "c": VarMetadata(name="c", unit="V"),
    })
    host = _attach_host(pw)
    assert pw.add_variable_to_plot("c")
    settle()
    assert host.messages == []


def test_announce_batch_summary(plot_factory, qapp):
    """批量拖入（v 枚举 + c 连续）→ 只播报 1 条汇总，不逐条刷屏。"""
    df = _enum_df()
    df["c"] = np.linspace(0.0, 1.0, 20)
    pw = plot_factory(df)
    pw.plot_context.loader = _mock_loader({
        "v": _enum_meta("v"), "c": VarMetadata(name="c", unit="V"),
    })
    host = _attach_host(pw)
    pw._multi_curve_manager.add_variables_to_plot(["v", "c"])
    settle()
    step_msgs = [m for m in host.messages if "枚举" in m]
    assert len(step_msgs) == 1
    assert "1 个枚举通道" in step_msgs[0]


def test_no_announce_on_reload(reload_env):
    """reload（quiet 路径）→ 不播报枚举阶梯。"""
    host, pw, reload_with = reload_env
    assert pw.curves["v"].draw_mode == DRAW_MODE_STEP_POST
    host.status_messages.clear()
    reload_with({"v": list(_enum_df(30)["v"]), "c": list(np.linspace(0.0, 1.0, 30))})
    assert host.status_messages == []


# ---------------------------------------------------------------------------
# 修复回归：_recreate_curve 转发层 / 首绘播报 / 批量重复误播报
# ---------------------------------------------------------------------------

def test_recreate_curve_via_pw_forwarder(reload_env):
    """_recreate_curve 经 pw 转发层调用（announce_step 透传）不再 TypeError，
    曲线对象失效重建成功，且用户钉住的绘制方式跨重建保留。"""
    host, pw, reload_with = reload_env
    # 用户钉住 c 为 step-post
    assert pw._multi_curve_manager.apply_curve_draw_mode("c", DRAW_MODE_STEP_POST)
    old_curve = pw.curves["c"].curve

    # 模拟曲线对象失效（脱离 scene）后走 _recreate_curve（经 pw 转发层）
    try:
        pw.plot_item.removeItem(old_curve)
    except Exception:
        pass
    pw._multi_curve_manager._recreate_curve("c")

    # 重建成功（未被 except Exception 静默吞掉）
    assert pw.curves["c"].curve is not None
    assert pw.curves["c"].curve is not old_curve
    # 钉住保留
    ci = pw.curves["c"]
    assert ci.draw_mode == DRAW_MODE_STEP_POST
    assert ci.draw_mode_source == DRAW_MODE_SOURCE_USER
    assert ci.curve.opts["stepMode"] == "right"


def test_first_plot_announces_enum(plot_factory, qapp):
    """空图首绘（plot_variable 直接构造路径）枚举通道 → 播报。"""
    pw = plot_factory(_enum_df())
    pw.plot_context.loader = _mock_loader({"v": _enum_meta("v")})
    host = _attach_host(pw)
    pw.plot_variable("v")  # 空图首绘不走 add_variable_to_plot
    settle()
    assert any("枚举" in m for m in host.messages)


def test_batch_duplicate_enum_not_recounted(plot_factory, qapp, monkeypatch):
    """批量拖入含已在图上的枚举曲线 → 重复项不计入汇总播报。"""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    df = _enum_df()
    df["c"] = np.linspace(0.0, 1.0, 20)
    pw = plot_factory(df)
    pw.plot_context.loader = _mock_loader({
        "v": _enum_meta("v"), "c": VarMetadata(name="c", unit="V"),
    })
    host = _attach_host(pw)
    # v 已在图上（auto step-post）
    assert pw.add_variable_to_plot("v")
    host.messages.clear()
    # 批量再拖 v + c：只有 c 是新建（直线，不播报）
    pw._multi_curve_manager.add_variables_to_plot(["v", "c"])
    settle()
    assert host.messages == []
