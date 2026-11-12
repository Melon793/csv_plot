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
