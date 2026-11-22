"""R5：换算后 Y 轴条件救援（事件式判据 / 合并播报 / D24 冻结）。

真机缺陷：编辑 factor/offset 后曲线被换算整体移出当前视野，用户看到
「没反应」。判据必须是事件式的——换算前在 X 窗口内可见、换算后与 Y 视野
完全无交集才救援；其余情况视野一律不动（不抢用户缩放）。
"""

import numpy as np
import pytest

from tests.fixtures.waits import settle


def _add(pw, name="a", y=(1.0, 2.0, 3.0)):
    x = np.arange(len(y), dtype=float)
    assert pw.add_variable_to_plot(
        name, x_values=x, y_values=np.asarray(y, float)
    )
    return pw.curves[name]


def _status_spy(pw):
    msgs = []
    pw.window()._show_status_message = msgs.append
    return msgs


def _rescue_calls(pw, monkeypatch):
    calls = []
    original = pw.auto_y_in_x_range
    monkeypatch.setattr(
        pw, "auto_y_in_x_range", lambda: (calls.append(1), original())[1]
    )
    return calls


def _view_x(pw):
    return tuple(pw.view_box.viewRange()[0])


def _view_y(pw):
    return tuple(pw.view_box.viewRange()[1])


def _y_limits(pw):
    return pw.view_box.state["limits"]["yLimits"]


def _suffix_hits(msgs):
    return [m for m in msgs if "已适配 Y" in m]


def _winner_window(pw, name="a"):
    """把 X 视野收到 1..2：窗口内只剩 y=[2,3]，视野仍是 [0.8,3.2]。"""
    pw.view_box.setXRange(1.0, 2.0, padding=0)
    assert name in pw.curves


class TestRescuePositive:
    def test_transform_out_of_window_rescues_and_keeps_x(self, plot_factory, monkeypatch):
        """G 正例：前可见 → 后被裁 → 救援覆盖后置数据，X 逐值不变。"""
        pw = plot_factory()
        _add(pw)
        _winner_window(pw)
        msgs = _status_spy(pw)
        x_before = _view_x(pw)
        calls = _rescue_calls(pw, monkeypatch)

        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)

        lo, hi = _view_y(pw)
        assert lo <= 4.0 and hi >= 6.0, f"救援未覆盖窗口内后置数据 [4,6]: {lo, hi}"
        assert x_before == _view_x(pw), "救援改动了 X 视野"
        assert calls == [1]
        hits = _suffix_hits(msgs)
        assert len(hits) == 1, f"救援播报应合并为一条: {msgs}"
        assert len(msgs) == 1, f"不得二次播报: {msgs}"
        assert "已修改 a 换算" in hits[0]

    def test_rescue_freezes_y_after_update(self, plot_factory):
        """D24：救援同步落实后冻结 Y，排队重算不会再把视野拉回全曲线。"""
        pw = plot_factory()
        _add(pw)
        _winner_window(pw)
        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)
        rescued = _view_y(pw)
        assert pw.view_box.state["autoRange"][1] is False

        settle(5)

        assert _view_y(pw) == pytest.approx(rescued), "救援视野被排队重算覆盖"
        # 再模拟一帧绘制：paint 路径会消费 _autoRangeNeedsUpdate
        pw.view_box.prepareForPaint()
        assert _view_y(pw) == pytest.approx(rescued)
        assert pw.view_box._autoRangeNeedsUpdate is False
        assert pw.view_box.state["autoRange"][1] is False

    def test_second_transform_does_not_auto_scale(self, plot_factory):
        """粘滞回归：救援后再次换算，视野不因残留 autoRange 跳回全量。"""
        pw = plot_factory()
        _add(pw)
        _winner_window(pw)
        pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)
        before = _view_y(pw)

        assert pw._multi_curve_manager.apply_curve_transform("a", 3.0, 0.0)

        settle(5)
        assert _view_y(pw) == pytest.approx(before)
        assert pw.view_box.state["autoRange"][1] is False

    def test_flat_curve_leaving_view_is_rescued(self, plot_factory):
        """平直曲线：换前点在视野内、换后出界 → 救援（§4 平直正例）。"""
        pw = plot_factory()
        _add(pw, y=(5.0, 5.0))
        msgs = _status_spy(pw)

        assert pw._multi_curve_manager.apply_curve_transform("a", 1.0, 100.0)

        lo, hi = _view_y(pw)
        assert lo <= 105.0 <= hi, f"平直点 105 未被救援进视野: {lo, hi}"
        assert len(_suffix_hits(msgs)) == 1

    def test_multi_curve_rescue_uses_visible_union(self, plot_factory):
        """D23：执行口径与右键一致（可见曲线并集），被编辑曲线必须可见。"""
        pw = plot_factory()
        _add(pw, "a", (1.0, 2.0, 3.0))
        _add(pw, "b", (0.5, 1.0, 1.5))
        pw.view_box.setXRange(1.0, 2.0, padding=0)

        assert pw._multi_curve_manager.apply_curve_transform("a", 1.0, 1000.0)

        lo, hi = _view_y(pw)
        assert lo <= 1.5, f"并集未覆盖兄弟曲线 b: {lo, hi}"
        assert hi >= 1001.0, f"并集未覆盖被编辑曲线 a: {lo, hi}"


class TestRescueNegative:
    def test_same_value_noop(self, plot_factory):
        pw = plot_factory()
        _add(pw)
        msgs = _status_spy(pw)
        x_before, y_before = _view_x(pw), _view_y(pw)
        assert pw._multi_curve_manager.apply_curve_transform("a", 1.0, 0.0) is False
        assert _view_x(pw) == x_before
        assert _view_y(pw) == y_before
        assert msgs == []

    def test_still_visible_after_transform_no_rescue(self, plot_factory, monkeypatch):
        """A 负例：新 limits 把视野夹到数据旁，换算后仍可见 → 不救。"""
        pw = plot_factory()
        _add(pw)
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)

        assert pw._multi_curve_manager.apply_curve_transform("a", 1.0, 1000.0)

        assert calls == [], "换算后仍可见却触发了救援"
        assert _suffix_hits(msgs) == []
        assert len(msgs) == 1

    def test_sibling_keeps_view_in_place(self, plot_factory, monkeypatch):
        """E 负例：兄弟曲线撑住 limits，小位移仍在视野内 → view 逐值不变。"""
        pw = plot_factory()
        _add(pw, "a", (1.0, 2.0, 3.0))
        _add(pw, "b", (1.0, 500.0, 1000.0))
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)
        before = _view_y(pw)

        assert pw._multi_curve_manager.apply_curve_transform("a", 1.0, 1000.0)

        assert calls == []
        assert _view_y(pw) == pytest.approx(before)
        assert _suffix_hits(msgs) == []

    def test_not_visible_before_transform_no_rescue(self, plot_factory, monkeypatch):
        """F 负例：用户缩放在别处（换算前就不可见）→ 不抢缩放。"""
        pw = plot_factory()
        _add(pw, "a", (1.0, 2.0, 3.0))
        _add(pw, "b", (1.0, 500.0, 1000.0))
        pw.view_box.setLimits(yMin=-2000, yMax=2000)
        pw.view_box.setYRange(380.0, 620.0)
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)
        zoomed = _view_y(pw)

        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)

        assert calls == []
        assert _view_y(pw) == pytest.approx(zoomed)
        assert _suffix_hits(msgs) == []

    def test_flat_curve_already_out_of_view_no_rescue(self, plot_factory, monkeypatch):
        """平直负例：换前点已不在视野 → 不救。"""
        pw = plot_factory()
        _add(pw, y=(5.0, 5.0))
        pw.view_box.setLimits(yMin=-1000, yMax=1000)
        pw.view_box.setYRange(100.0, 200.0)
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)

        assert pw._multi_curve_manager.apply_curve_transform("a", 1.0, 100.0)

        assert calls == []
        assert _suffix_hits(msgs) == []

    def test_hidden_curve_not_rescued(self, plot_factory, monkeypatch):
        """隐藏曲线不参与判定：改换算不救、不动视野。"""
        pw = plot_factory()
        _add(pw, "a", (1.0, 2.0, 3.0))
        _add(pw, "b", (40.0, 50.0, 60.0))
        _winner_window(pw)
        pw._multi_curve_manager.apply_curve_visibility("a", False)
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)
        before = _view_y(pw)

        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)

        assert calls == []
        assert _view_y(pw) == pytest.approx(before)
        assert _suffix_hits(msgs) == []

    def test_batch_adding_not_rescued(self, plot_factory, monkeypatch):
        pw = plot_factory()
        _add(pw)
        _winner_window(pw)
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)
        before = _view_y(pw)

        pw._batch_adding = True
        try:
            assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)
        finally:
            pw._batch_adding = False

        assert calls == []
        assert _view_y(pw) == pytest.approx(before)
        assert msgs == []  # 批量窗口内本来就不播报

    def test_allow_y_rescue_false_not_rescued(self, plot_factory, monkeypatch):
        """单条搬运（batch=False）必须显式关闭救援（P1-5）。"""
        pw = plot_factory()
        _add(pw)
        _winner_window(pw)
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)
        before = _view_y(pw)

        assert pw._multi_curve_manager.apply_curve_transform(
            "a", 2.0, 0.0, allow_y_rescue=False
        )

        assert calls == []
        assert _view_y(pw) == pytest.approx(before)
        assert _suffix_hits(msgs) == []
        assert len(msgs) == 1

    def test_empty_window_no_rescue_no_error(self, plot_factory, monkeypatch):
        """空窗（X 视野在数据外）：不救、不抛异常。"""
        pw = plot_factory()
        _add(pw)
        pw.view_box.setXRange(100.0, 200.0)
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)

        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)

        assert calls == []
        assert _suffix_hits(msgs) == []

    def test_user_auto_follow_not_overridden(self, plot_factory, monkeypatch):
        """已开启 Ctrl+Y 跟随时不救援：下一帧系统自行适配，救反而关掉跟随。"""
        pw = plot_factory()
        _add(pw)
        pw.auto_y_in_x_range()
        pw.view_box.setXRange(1.0, 2.0, padding=0)
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)

        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)

        assert calls == []
        assert pw.view_box.state["autoRange"][1], "救援静默关掉了用户的 Y 跟随"
        assert _suffix_hits(msgs) == []

        # 模拟一帧绘制：跟随确实把视野带到窗口内新数据上
        pw.view_box.prepareForPaint()
        lo, hi = _view_y(pw)
        assert lo <= 4.0 and hi >= 6.0, "跟随开启时下一帧未自动适配"

    def test_reset_direction_clamped_view_no_rescue(self, plot_factory):
        """反向（1000→1）：limits 夹紧已把视野带回，不额外救援。"""
        pw = plot_factory()
        _add(pw)
        _winner_window(pw)
        assert pw._multi_curve_manager.apply_curve_transform("a", 1000.0, 0.0)
        msgs = _status_spy(pw)

        assert pw._multi_curve_manager.apply_curve_transform("a", 1.0, 0.0)

        lo, hi = _view_y(pw)
        assert lo <= 1.0 and hi >= 3.0, f"重置换算后视野未覆盖原始数据: {lo, hi}"
        assert any("已重置" in m for m in msgs)
        assert _suffix_hits(msgs) == []

    def test_all_nan_window_no_rescue(self, plot_factory, monkeypatch):
        """窗口内全是 NaN：不救、不抛异常。"""
        pw = plot_factory()
        _add(pw, y=(1.0, np.nan, 3.0))
        # 只把 NaN 点收进窗口（其余点在窗外）
        pw.view_box.setXRange(0.9, 1.1, padding=0)
        msgs = _status_spy(pw)
        calls = _rescue_calls(pw, monkeypatch)

        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)

        assert calls == []
        assert _suffix_hits(msgs) == []


class TestTransferDoesNotRescue:
    def test_apply_transfer_style_passes_rescue_off(self, plot_factory, monkeypatch):
        """搬运落地换算时显式 allow_y_rescue=False（单条搬运 batch=False 也走）。"""
        pw = plot_factory()
        _add(pw)
        _winner_window(pw)
        captured = {}
        mcm = pw._multi_curve_manager
        original = mcm.apply_curve_transform

        def spy(name, factor, offset, **kwargs):
            captured["kwargs"] = kwargs
            return original(name, factor, offset, **kwargs)

        monkeypatch.setattr(mcm, "apply_curve_transform", spy)

        from src.ui.drag_drop import CurveStyleSnapshot

        src = pw.curves["a"]
        src.factor, src.offset = 2.0, 0.0
        snap = CurveStyleSnapshot.from_curve_info(src)
        src.factor, src.offset = 1.0, 0.0  # 目标曲线回到未换算，快照待落地
        pw._apply_transfer_style("a", snap)

        assert captured.get("kwargs", {}).get("allow_y_rescue") is False