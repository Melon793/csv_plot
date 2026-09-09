"""legend 拖拽功能 component 测试（由 tmp/test_legend_drag_units.py 转正）。

覆盖：
1. parse_anchor_var_name：正常名/含空格/中文/URL 特殊字符/空串/非法 scheme
2. build_legend_var_mimedata：text/plain 可被 parse_var_names_from_mimedata 还原
3. _build_legend_html：锚点格式不变（curve:/// + URL 编码），与解析函数闭环
4. remove_variables_from_plot：全部存在/部分存在/全部不存在/删空（不 emit、返回值正确）
   + remove_variable_from_plot 单量 API 的条件 emit 行为
5. 拖拽指示器文案 _build_indicator_text
"""

import re

import numpy as np
import pandas as pd
import pytest

from src.ui.drag_drop import (
    LEGEND_MIME_FORMAT,
    build_legend_var_mimedata,
    parse_anchor_var_name,
    parse_var_names_from_mimedata,
)


# ---------------------------------------------------------------------------
# 1. parse_anchor_var_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "href,expected",
    [
        ("curve:///simple_name", "simple_name"),
        ("curve:///name%20with%20spaces", "name with spaces"),
        ("curve:///%E4%B8%AD%E6%96%87%E5%8F%98%E9%87%8F", "中文变量"),
        ("curve:///a%2Fb%23c%25d", "a/b#c%d"),
        ("curve:///UPPER_Case_123", "UPPER_Case_123"),
    ],
)
def test_parse_anchor_var_name_valid(href, expected):
    assert parse_anchor_var_name(href) == expected


@pytest.mark.parametrize(
    "href",
    [
        "",
        None,
        "curve:///",          # 空变量名
        "http:///foo",        # 非法 scheme
        "curve://foo",        # 缺少三斜杠（host 形式）
        "plain_text",
    ],
)
def test_parse_anchor_var_name_invalid(href):
    assert parse_anchor_var_name(href) is None


# ---------------------------------------------------------------------------
# 2. build_legend_var_mimedata 兼容性
# ---------------------------------------------------------------------------

def test_build_legend_var_mimedata_roundtrip():
    source = object()  # 任意对象，仅取 id
    names = ["var_A", "中文 变量", "b/c"]
    mime = build_legend_var_mimedata(names, source)
    # text/plain 兼容层可被现有解析函数还原
    assert parse_var_names_from_mimedata(mime) == names
    # 自定义格式携带源 plot id
    assert mime.hasFormat(LEGEND_MIME_FORMAT)
    assert int(bytes(mime.data(LEGEND_MIME_FORMAT)).decode()) == id(source)


# ---------------------------------------------------------------------------
# 3. _build_legend_html 锚点格式闭环
# ---------------------------------------------------------------------------

class _FakePW:
    """仅供 _build_legend_html 读取的最小伪 plot（curves + units）"""


def test_build_legend_html_anchor_roundtrip():
    from types import SimpleNamespace

    from src.core.data_types import CurveInfo
    from src.ui.widgets.multi_curve_manager import MultiCurveManager

    pw = _FakePW()
    pw.units = {"var A": "rpm", "中文变量": ""}
    pw.curves = {
        "var A": CurveInfo(
            var_name="var A", curve=None,
            x_data=np.arange(3.0), y_data=np.arange(3.0),
            color="blue", visible=True,
        ),
        "中文变量": CurveInfo(
            var_name="中文变量", curve=None,
            x_data=np.arange(3.0), y_data=np.arange(3.0),
            color="red", visible=False,
        ),
    }
    mgr = MultiCurveManager(SimpleNamespace(pw=pw))
    html = mgr._build_legend_html()

    # 锚点协议不变：curve:///URL编码变量名
    hrefs = re.findall(r"href='([^']+)'", html)
    assert len(hrefs) == 2
    assert all(h.startswith("curve:///") for h in hrefs)
    # 解析闭环：href → 变量名与 curves 键一致
    assert [parse_anchor_var_name(h) for h in hrefs] == list(pw.curves.keys())


# ---------------------------------------------------------------------------
# 4. remove_variables_from_plot / remove_variable_from_plot
# ---------------------------------------------------------------------------

@pytest.fixture()
def plot_widget(qapp):
    from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    pw = DraggableGraphicsLayoutWidget({}, df)
    yield pw
    pw._is_being_destroyed = True
    pw.deleteLater()


def _inject_curve(pw, name):
    """直接注入 CurveInfo（curve=None，绕过真实绘制），模拟已有曲线"""
    from src.core.data_types import CurveInfo
    pw.curves[name] = CurveInfo(
        var_name=name, curve=None,
        x_data=np.arange(3.0), y_data=np.arange(3.0),
    )


def test_remove_variables_all_exist(plot_widget):
    pw = plot_widget
    for name in ("a", "b", "c"):
        _inject_curve(pw, name)
    removed = pw.remove_variables_from_plot(["a", "b", "c"])
    assert removed == ["a", "b", "c"]
    assert pw.curves == {}
    assert pw.current_color_index == 0


def test_remove_variables_partial_and_none(plot_widget):
    pw = plot_widget
    _inject_curve(pw, "a")
    removed = pw.remove_variables_from_plot(["a", "ghost"])
    assert removed == ["a"]
    removed = pw.remove_variables_from_plot(["ghost1", "ghost2"])
    assert removed == []


def test_remove_variables_no_emit(plot_widget):
    pw = plot_widget
    _inject_curve(pw, "a")
    _inject_curve(pw, "b")
    fired = []
    pw.curves_changed.connect(lambda: fired.append(1))
    # 批量 API 不 emit（编辑器路径防信号回环）
    pw.remove_variables_from_plot(["a"])
    assert fired == []
    pw.remove_variables_from_plot(["b"])  # 删空也不 emit（由调用方显式触发）
    assert fired == []


def test_remove_variable_single_emit(plot_widget):
    pw = plot_widget
    _inject_curve(pw, "a")
    _inject_curve(pw, "b")
    fired = []
    pw.curves_changed.connect(lambda: fired.append(1))
    # 单量 API（legend 移动拖拽专用）：删除成功时 emit
    assert pw.remove_variable_from_plot("a") is True
    assert fired == [1]
    # 不存在的变量：返回 False 且不 emit
    assert pw.remove_variable_from_plot("ghost") is False
    assert fired == [1]
    # emit_changed=False 时不 emit
    assert pw.remove_variable_from_plot("b", emit_changed=False) is True
    assert fired == [1]
    assert pw.curves == {}


# ---------------------------------------------------------------------------
# 5. 统一指示器文案
# ---------------------------------------------------------------------------

def test_indicator_text_normal(plot_widget):
    """正常态：释放以{action}"""
    pw = plot_widget
    assert pw._build_indicator_text(["x"], "添加", False) == "释放以添加"
    assert pw._build_indicator_text(["x"], "替换", False) == "释放以替换"
    assert pw._build_indicator_text(["x"], "复制", False) == "释放以复制"
    assert pw._build_indicator_text(["x"], "移动", False) == "释放以移动"


def test_indicator_text_already_exists(plot_widget):
    """已存在态：变量已存在 / 变量已存在，释放以{action}"""
    pw = plot_widget
    # 添加/复制遇重复：仅显示"变量已存在"（无后续动作）
    assert pw._build_indicator_text(["x"], "添加", True) == "变量已存在"
    assert pw._build_indicator_text(["x"], "复制", True) == "变量已存在"
    # 替换/移动遇重复：显示"变量已存在，释放以{action}"（有后续动作）
    assert pw._build_indicator_text(["x"], "替换", True) == "变量已存在，释放以替换"
    assert pw._build_indicator_text(["x"], "移动", True) == "变量已存在，释放以移动"
