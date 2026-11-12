"""parse_legend_anchor 协议层单测（图例锚点按 scheme 分发）。

协议（施工方案 §2.1）：
- curve:///NAME → ("toggle", NAME)：原显隐锚点，失效占位条目同协议；
- color:///NAME → ("color", NAME)：色块改色锚点（图例快速改色）。
非法/空/空变量名 → None。parse_anchor_var_name 原样保留，不受影响。
"""

import pytest

from src.ui.drag_drop import parse_anchor_var_name, parse_legend_anchor


@pytest.mark.parametrize(
    "href,expected",
    [
        ("curve:///simple_name", ("toggle", "simple_name")),
        ("curve:///name%20with%20spaces", ("toggle", "name with spaces")),
        ("curve:///%E4%B8%AD%E6%96%87%E5%8F%98%E9%87%8F", ("toggle", "中文变量")),
        ("color:///simple_name", ("color", "simple_name")),
        ("color:///a%2Fb%23c", ("color", "a/b#c")),
    ],
)
def test_parse_legend_anchor_valid(href, expected):
    assert parse_legend_anchor(href) == expected


@pytest.mark.parametrize(
    "href",
    [
        "",
        None,
        "curve:///",     # 空变量名
        "color:///",     # 空变量名
        "http:///foo",   # 非法 scheme
        "color://foo",   # 缺少三斜杠（host 形式）
        "plain_text",
    ],
)
def test_parse_legend_anchor_invalid(href):
    assert parse_legend_anchor(href) is None


def test_parse_anchor_var_name_unchanged():
    """既有协议零影响：curve:/// 仍解析出变量名；color:/// 不被其接受。"""
    assert parse_anchor_var_name("curve:///a") == "a"
    assert parse_anchor_var_name("color:///a") is None
