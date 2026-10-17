"""标注数据模型单元测试（零 Qt 依赖）。

模型层是「用户手工产物不许被一条坏记录毁掉」这条原则的唯一落点：
所有非法输入都必须被**收拢或丢弃**，绝不抛异常。本文件把每条收拢规则
钉成一条断言，让将来放宽/收紧校验时能看出是"有意改的"。

配套：
- 图元级行为见 tests/component/test_annotation_manager.py
- 模板向后兼容见 tests/unit/core/test_plot_config_annotations.py
"""

from __future__ import annotations

import math

import pytest

from src.core.annotation_models import (
    ANNOTATION_KINDS,
    DEFAULT_ANCHOR,
    DEFAULT_FILL_ALPHA,
    DEFAULT_FONT_SIZE,
    DEFAULT_STROKE,
    DEFAULT_STROKE_WIDTH,
    FILL_ALPHA_RANGE,
    FONT_SIZE_RANGE,
    MAX_POLYLINE_POINTS,
    MAX_TEXT_LENGTH,
    STROKE_WIDTH_RANGE,
    AnnotationItem,
    minimum_points,
    new_annotation_id,
    safe_color,
    safe_int,
)


def _payload(**overrides) -> dict:
    """一条合法的矩形标注载荷，逐项被 overrides 覆盖"""
    data = {
        "id": "abcd1234",
        "kind": "rect",
        "points": [[0.0, 0.0], [1.0, 2.0]],
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# id 生成
# ---------------------------------------------------------------------------

class TestNewAnnotationId:
    def test_is_eight_hex_chars(self):
        aid = new_annotation_id()
        assert len(aid) == 8
        int(aid, 16)  # 非十六进制会抛 ValueError

    def test_is_unique_enough(self):
        assert len({new_annotation_id() for _ in range(200)}) == 200


# ---------------------------------------------------------------------------
# 字段收拢器
# ---------------------------------------------------------------------------

class TestSafeColor:
    @pytest.mark.parametrize("value", ["#f00", "#E24B4A", "#80E24B4A"])
    def test_accepts_hex_forms(self, value):
        assert safe_color(value) == value

    def test_strips_surrounding_space(self):
        assert safe_color("  #E24B4A ") == "#E24B4A"

    @pytest.mark.parametrize("value", ["red", "rgb(1,2,3)", "#12", "#12345", "", None, 42])
    def test_falls_back_on_non_hex(self, value):
        # 不接受颜色名：Qt 认识的颜色名集合随版本变化，拼错还会静默变黑
        assert safe_color(value) == DEFAULT_STROKE

    def test_custom_fallback_is_respected(self):
        assert safe_color("bogus", "#000000") == "#000000"

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_rejects_non_finite(self, value):
        # NaN/inf 会被 Qt 当成未定义颜色，渲染结果不可预期
        assert safe_color(value) == DEFAULT_STROKE


class TestSafeInt:
    def test_clamps_both_ends(self):
        low, high = STROKE_WIDTH_RANGE
        assert safe_int(-99, DEFAULT_STROKE_WIDTH, low, high) == low
        assert safe_int(9999, DEFAULT_STROKE_WIDTH, low, high) == high

    def test_accepts_float_and_truncates(self):
        assert safe_int(3.9, 2, 1, 8) == 3

    @pytest.mark.parametrize("value", ["3", None, [1], {}, float("nan"), float("inf")])
    def test_falls_back_on_non_numeric(self, value):
        assert safe_int(value, 2, 1, 8) == 2

    def test_bool_is_not_an_int(self):
        # bool 是 int 的子类；放行会让 True 静默变成线宽 1
        assert safe_int(True, DEFAULT_STROKE_WIDTH, *STROKE_WIDTH_RANGE) == DEFAULT_STROKE_WIDTH


# ---------------------------------------------------------------------------
# 反序列化：非法输入一律返回 None
# ---------------------------------------------------------------------------

class TestFromDictRejects:
    @pytest.mark.parametrize("raw", [None, [], "text", 42, ()])
    def test_non_dict(self, raw):
        assert AnnotationItem.from_dict(raw) is None

    @pytest.mark.parametrize("kind", ["", "arrowhead", "RECT", None, 1])
    def test_unknown_kind(self, kind):
        assert AnnotationItem.from_dict(_payload(kind=kind)) is None

    @pytest.mark.parametrize("points", [None, "0,0", 5, [(0, 0)]])
    def test_points_not_a_list_of_pairs(self, points):
        assert AnnotationItem.from_dict(_payload(points=points)) is None

    def test_pair_with_wrong_arity(self):
        assert AnnotationItem.from_dict(_payload(points=[[0.0, 0.0, 1.0], [1.0, 1.0]])) is None

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "x"])
    def test_non_finite_coordinate_kills_whole_record(self, bad):
        # 刻意不做"只丢坏点"：点的下标带语义，悄悄丢一个会把矩形变成另一个矩形
        assert AnnotationItem.from_dict(_payload(points=[[0.0, 0.0], [bad, 1.0]])) is None

    def test_too_few_points_for_kind(self):
        assert AnnotationItem.from_dict(_payload(kind="text", points=[])) is None
        assert AnnotationItem.from_dict(_payload(kind="rect", points=[[0.0, 0.0]])) is None
        assert AnnotationItem.from_dict(_payload(kind="polyline", points=[[0.0, 0.0]])) is None


class TestFromDictAccepts:
    @pytest.mark.parametrize("kind", ANNOTATION_KINDS)
    def test_every_whitelisted_kind(self, kind):
        points = [[0.0, 1.0]] if kind == "text" else [[0.0, 1.0], [2.0, 3.0]]
        model = AnnotationItem.from_dict(_payload(kind=kind, points=points))
        assert model is not None and model.kind == kind

    def test_extra_points_truncated_for_fixed_arity_kinds(self):
        model = AnnotationItem.from_dict(_payload(kind="rect", points=[[0.0, 0.0], [1.0, 1.0], [9.0, 9.0]]))
        assert model is not None
        assert model.points == [(0.0, 0.0), (1.0, 1.0)]

    def test_polyline_keeps_all_points(self):
        pts = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]
        model = AnnotationItem.from_dict(_payload(kind="polyline", points=pts))
        assert model is not None
        assert len(model.points) == 4

    def test_polyline_is_truncated_at_the_cap(self):
        """折点可以由用户在段上点击无限追加，模板 points 更是外部输入

        不设上限等于让一份坏模板把场景图元数撑爆；超出的点按顺序截断（保留前 N 个），
        与固定点数 kind 的"多给就丢"同一条规则。
        """
        pts = [[float(i), float(i)] for i in range(MAX_POLYLINE_POINTS + 20)]
        model = AnnotationItem.from_dict(_payload(kind="polyline", points=pts))
        assert model is not None
        assert len(model.points) == MAX_POLYLINE_POINTS
        assert model.points[0] == (0.0, 0.0) and model.points[-1] == (float(MAX_POLYLINE_POINTS - 1),) * 2

    def test_polyline_at_the_cap_is_kept_whole(self):
        """边界自己不能少一个：截断条件必须是 > 而不是 >= """
        pts = [[float(i), 0.0] for i in range(MAX_POLYLINE_POINTS)]
        model = AnnotationItem.from_dict(_payload(kind="polyline", points=pts))
        assert model is not None and len(model.points) == MAX_POLYLINE_POINTS

    def test_int_coordinates_become_float(self):
        model = AnnotationItem.from_dict(_payload(points=[[0, 0], [1, 2]]))
        assert model is not None
        assert all(isinstance(v, float) for p in model.points for v in p)

    def test_tuple_points_accepted(self):
        model = AnnotationItem.from_dict(_payload(points=((0.0, 0.0), (1.0, 1.0))))
        assert model is not None and model.points == [(0.0, 0.0), (1.0, 1.0)]


class TestFromDictFieldFallbacks:
    @pytest.mark.parametrize("raw_id", [None, "", "   ", 42, []])
    def test_missing_id_is_generated(self, raw_id):
        model = AnnotationItem.from_dict(_payload(id=raw_id))
        assert model is not None
        assert len(model.id) == 8 and model.id.strip()

    def test_existing_id_is_kept(self):
        model = AnnotationItem.from_dict(_payload(id="deadbeef"))
        assert model is not None and model.id == "deadbeef"

    def test_text_truncated_to_limit(self):
        model = AnnotationItem.from_dict(_payload(kind="text", points=[[0.0, 0.0]], text="字" * 900))
        assert model is not None and len(model.text) == MAX_TEXT_LENGTH

    @pytest.mark.parametrize("raw", [None, 42, ["a"]])
    def test_non_string_text_becomes_empty(self, raw):
        model = AnnotationItem.from_dict(_payload(kind="text", points=[[0.0, 0.0]], text=raw))
        assert model is not None and model.text == ""

    def test_stroke_width_font_size_alpha_are_clamped(self):
        model = AnnotationItem.from_dict(
            _payload(stroke_width=999, font_size=-3, fill_alpha=999)
        )
        assert model is not None
        assert model.stroke_width == STROKE_WIDTH_RANGE[1]
        assert model.font_size == FONT_SIZE_RANGE[0]
        assert model.fill_alpha == FILL_ALPHA_RANGE[1]

    def test_stroke_width_zero_is_kept_for_borderless_text(self):
        """线宽 0 = 文字无边框：负值收拢到 0，正零保留（模板往返不丢语义）"""
        assert AnnotationItem.from_dict(_payload(stroke_width=0)).stroke_width == 0
        assert AnnotationItem.from_dict(_payload(stroke_width=-1)).stroke_width == 0

    def test_style_defaults_when_keys_absent(self):
        model = AnnotationItem.from_dict(_payload())
        assert model is not None
        assert model.stroke == DEFAULT_STROKE
        assert model.stroke_width == DEFAULT_STROKE_WIDTH
        assert model.fill_alpha == DEFAULT_FILL_ALPHA
        assert model.font_size == DEFAULT_FONT_SIZE
        assert model.fill is None
        assert model.visible is True

    def test_fill_absent_or_blank_means_no_fill(self):
        assert AnnotationItem.from_dict(_payload(fill="")).fill is None
        assert AnnotationItem.from_dict(_payload(fill="   ")).fill is None
        assert AnnotationItem.from_dict(_payload()).fill is None

    def test_fill_bad_color_falls_back_not_dropped(self):
        # 有填充意图但色值写错：回退成默认描边色，而不是静默变成"无填充"
        model = AnnotationItem.from_dict(_payload(fill="chartreuse"))
        assert model is not None and model.fill == DEFAULT_STROKE

    @pytest.mark.parametrize("raw,expected", [("viewport", "viewport"), ("data", "data"), ("bogus", DEFAULT_ANCHOR), (None, DEFAULT_ANCHOR)])
    def test_anchor_falls_back(self, raw, expected):
        model = AnnotationItem.from_dict(_payload(anchor=raw))
        assert model is not None and model.anchor == expected

    @pytest.mark.parametrize("raw,expected", [(False, False), (True, True), ("no", True), (0, True), (None, True)])
    def test_visible_only_accepts_real_bool(self, raw, expected):
        model = AnnotationItem.from_dict(_payload(visible=raw))
        assert model is not None and model.visible is expected


# ---------------------------------------------------------------------------
# 序列化往返
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _all_kinds(self) -> list[AnnotationItem]:
        return [
            AnnotationItem(id="t1", kind="text", points=[(1.5, -2.5)], text="工况 A 合格"),
            AnnotationItem(id="t2", kind="rect", points=[(0.0, 0.0), (3.0, 4.0)], fill="#00FF00", fill_alpha=80),
            AnnotationItem(id="t3", kind="ellipse", points=[(-1.0, -1.0), (1.0, 1.0)]),
            AnnotationItem(id="t4", kind="line", points=[(0.0, 0.0), (5.0, 5.0)], stroke_width=4),
            AnnotationItem(id="t5", kind="arrow", points=[(0.0, 0.0), (5.0, 5.0)], stroke="#123456"),
            AnnotationItem(id="t6", kind="polyline", points=[(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)]),
        ]

    def test_to_dict_from_dict_is_idempotent(self):
        for model in self._all_kinds():
            dumped = model.to_dict()
            restored = AnnotationItem.from_dict(dumped)
            assert restored is not None, model.kind
            assert restored.to_dict() == dumped, model.kind

    def test_dump_is_plain_json_types(self):
        import json

        dumped = [m.to_dict() for m in self._all_kinds()]
        # 出现 numpy / Qt 类型就会在这里炸掉，而不是在写 YAML 时才炸
        json.dumps(dumped)

    def test_points_serialized_as_lists_not_tuples(self):
        dumped = AnnotationItem.from_dict(_payload()).to_dict()
        assert all(isinstance(p, list) for p in dumped["points"])

    def test_tuple_input_normalises_to_same_dict(self):
        a = AnnotationItem.from_dict(_payload(points=((0.0, 0.0), (1.0, 1.0))))
        b = AnnotationItem.from_dict(_payload(points=[[0.0, 0.0], [1.0, 1.0]]))
        assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# 自检与几何辅助
# ---------------------------------------------------------------------------

class TestIsValid:
    def test_valid_model(self):
        assert AnnotationItem(id="x", kind="rect", points=[(0.0, 0.0), (1.0, 1.0)]).is_valid()

    def test_rejects_unknown_kind(self):
        assert not AnnotationItem(id="x", kind="hexagon", points=[(0.0, 0.0)]).is_valid()

    @pytest.mark.parametrize("bad_id", ["", "   "])
    def test_rejects_blank_id(self, bad_id):
        assert not AnnotationItem(id=bad_id, kind="rect", points=[(0.0, 0.0), (1.0, 1.0)]).is_valid()

    def test_rejects_too_many_points_for_kind(self):
        assert not AnnotationItem(id="x", kind="text", points=[(0.0, 0.0), (1.0, 1.0)]).is_valid()

    def test_rejects_non_finite_point(self):
        assert not AnnotationItem(id="x", kind="line", points=[(0.0, 0.0), (math.nan, 1.0)]).is_valid()

    def test_rejects_polyline_over_the_cap(self):
        pts = [(float(i), 0.0) for i in range(MAX_POLYLINE_POINTS + 1)]
        assert not AnnotationItem(id="x", kind="polyline", points=pts).is_valid()

    def test_accepts_polyline_with_four_points(self):
        """折线点数在区间内一律合法（不是"只能 2 点"）"""
        pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 0.0), (3.0, 1.0)]
        assert AnnotationItem(id="x", kind="polyline", points=pts).is_valid()


class TestMinimumPoints:
    """UI 侧回写用它挡"图元中间态读到的点太少"

    折线重建手柄的间隙会读到过少的点，写进模型会让整条标注变非法、
    进而让整份模板加载失败，所以下限必须是公开可查的。
    """

    @pytest.mark.parametrize("kind", ANNOTATION_KINDS)
    def test_covers_every_whitelisted_kind(self, kind):
        assert minimum_points(kind) >= 1

    def test_text_needs_one_point(self):
        assert minimum_points("text") == 1

    @pytest.mark.parametrize("kind", ["rect", "ellipse", "line", "arrow", "polyline"])
    def test_two_point_kinds_need_two(self, kind):
        assert minimum_points(kind) == 2

    def test_unknown_kind_falls_back_to_one(self):
        """未知 kind 不该让调用方拿到 0（那会把"点数为 0"当成合法中间态）"""
        assert minimum_points("hexagon") == 1
        assert minimum_points("") == 1


class TestBounds:
    def test_text_has_no_box(self):
        assert AnnotationItem(id="x", kind="text", points=[(1.0, 2.0)]).bounds() is None

    def test_two_point_kind_returns_normalised_box(self):
        # 对角点顺序颠倒也要给出 (xmin, ymin, xmax, ymax)
        model = AnnotationItem(id="x", kind="rect", points=[(3.0, 4.0), (1.0, 2.0)])
        assert model.bounds() == (1.0, 2.0, 3.0, 4.0)

    def test_polyline_box_covers_all_points(self):
        model = AnnotationItem(id="x", kind="polyline", points=[(0.0, 5.0), (2.0, -1.0), (-3.0, 2.0)])
        assert model.bounds() == (-3.0, -1.0, 2.0, 5.0)


class TestMoveBy:
    def test_shifts_every_point(self):
        model = AnnotationItem(id="x", kind="polyline", points=[(0.0, 0.0), (1.0, 1.0)])
        model.move_by(2.5, -1.5)
        assert model.points == [(2.5, -1.5), (3.5, -0.5)]


class TestSummary:
    def test_text_summary_shows_quoted_body(self):
        text = AnnotationItem(id="x", kind="text", points=[(0.0, 0.0)], text="异常段")
        assert '文字　"异常段"' == text.summary()

    def test_long_text_is_elided(self):
        text = AnnotationItem(id="x", kind="text", points=[(0.0, 0.0)], text="字" * 50)
        # 正文截到 24 字 + 省略号（注意"文字"标签本身也含"字"，不能数总字数）
        assert text.summary() == '文字　"' + "字" * 24 + '…"'

    def test_blank_text_is_marked(self):
        assert "(空)" in AnnotationItem(id="x", kind="text", points=[(0.0, 0.0)], text="   ").summary()

    def test_multiline_text_flattened(self):
        summary = AnnotationItem(id="x", kind="text", points=[(0.0, 0.0)], text="一\n二").summary()
        assert "\n" not in summary and "一 二" in summary

    def test_shape_summary_contains_both_corners(self):
        model = AnnotationItem(id="x", kind="rect", points=[(0.0, 0.0), (2.0, 3.0)])
        summary = model.summary()
        assert summary.startswith("矩形") and "0" in summary and "3" in summary
