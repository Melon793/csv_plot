"""P1-21：标记区统计的「最近点」下标改走 O(log n) 的 searchsorted。

`_get_mark_stats_multi_curve` 原来每条曲线做两次 `np.argmin(np.abs(x_data - v))`：
先把 `abs(x - v)` 整份物化再全量扫描，百万点一对实测 ≈ 2.5 ms，拖一次标记区
（`sigRegionChanged` → `update_mark_stats`）在 12 条曲线上就是 ≈ 30 ms。

换成 `_nearest_index` 的前提是**语义一字不差**，所以判据分两层：
① `_nearest_index` 与 `argmin` 逐点等价（含重复 x、平手、越界等会改变下标的分支）；
② 整条统计链路在两种实现下产出的 `MarkStatEntry` 字段完全相同。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ui.widgets import mark_region_manager as mrm
from src.ui.widgets.mark_region_manager import _nearest_index


def _legacy(x_data, value):
    """修复前的写法：全量物化 + 扫描，返回首个最小值下标。"""
    return int(np.argmin(np.abs(x_data - value)))


def _ascending_arrays(rng):
    """一批升序数组，特意含重复值、单元素、越界目标会命中的边界形态。"""
    return [
        np.linspace(0.0, 1000.0, 1001),
        np.sort(rng.random(500) * 100.0),
        np.repeat(np.arange(50, dtype=float), 3),
        np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 5.0, 5.0]),
        np.array([5.0]),
        np.array([0.0, 2.0]),
        np.zeros(10),
        np.linspace(-50.0, 50.0, 101),
        np.linspace(0.0, 100.0, 201).astype(np.float32),
        np.arange(1_500_000_000, 1_500_002_000, dtype=np.float64),
    ]


class TestNearestIndexMatchesArgmin:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_identical_index_on_every_shape(self, seed):
        rng = np.random.default_rng(seed)
        mismatches = []
        for x in _ascending_arrays(rng):
            targets = np.linspace(x.min() - 10, x.max() + 10, 300)
            targets = np.concatenate([targets, [0.0, -1e9, 1e9], x[:3], x[-3:]])
            for v in targets:
                a, b = _legacy(x, v), _nearest_index(x, float(v))
                if a != b:
                    mismatches.append((float(v), a, b))
        assert mismatches == []

    def test_duplicate_x_takes_the_first_occurrence(self):
        """重复 x 段里 y 各不相同，下标必须落在段首（与 argmin 的「首个最小值」一致）。

        这条是 `_nearest_index` 里第二次 searchsorted 回退存在的理由：只取
        `idx-1` 会落到 2 而不是 0，y1 就从 100 变成 300，slope 跟着错。
        """
        x = np.array([0.0, 0.0, 0.0, 9.0])
        assert _nearest_index(x, 0.4) == 0
        assert _nearest_index(x, 0.4) == _legacy(x, 0.4)
        assert _nearest_index(x, 8.6) == 3, "右侧更近时不回退，且本就是该值首次出现"

    def test_tie_takes_the_left_point(self):
        x = np.array([0.0, 2.0, 4.0])
        assert _nearest_index(x, 3.0) == 1 == _legacy(x, 3.0)

    def test_targets_beyond_the_ends_clamp(self):
        x = np.array([10.0, 10.0, 20.0, 30.0, 30.0])
        assert _nearest_index(x, -1e9) == 0, "首值有重复时回到其首次出现处"
        assert _nearest_index(x, 1e9) == 3, "末值有重复时回到其首次出现处"
        for v in (-1e9, 1e9, 100.0, -1.0):
            assert _nearest_index(x, v) == _legacy(x, v)


def _arm_stats(plot_factory, x_data, y_data):
    """造一个「单曲线 + 已绘图」的现场，直接调统计入口。"""
    pw = plot_factory()
    assert pw.plot_variable("a") is True
    ci = pw.curves["a"]
    ci.x_data = np.asarray(x_data, dtype=np.float64)
    ci.y_data = np.asarray(y_data, dtype=np.float64)
    ci.update_x_range()
    ci.visible = True
    return pw


def _entry_dict(entry):
    """统计条目转可比较字典：NaN 字段（空区域）按「两边都 NaN」判等。"""
    out = {}
    for key, value in vars(entry).items():
        out[key] = "<nan>" if value != value else value
    return out


class TestMarkStatsParity:
    def test_entries_identical_to_the_argmin_implementation(self, plot_factory, monkeypatch):
        """端到端等价：两种下标实现跑同一条链路，MarkStatEntry 字段逐个相同。"""
        rng = np.random.default_rng(7)
        x = np.sort(rng.integers(0, 400, 400).astype(float))  # 升序 + 大量重复
        y = rng.random(400)
        pw = _arm_stats(plot_factory, x, y)

        regions = [(5.0, 300.0), (0.0, 0.0), (-10.0, 500.0), (12.0, 12.5)]
        new_entries = [pw._mark_region_manager._get_mark_stats_multi_curve(*r) for r in regions]

        monkeypatch.setattr(mrm, "_nearest_index", _legacy)
        old_entries = [
            pw._mark_region_manager._get_mark_stats_multi_curve(*r) for r in regions
        ]

        for (lo, hi), fresh, stale in zip(regions, new_entries, old_entries):
            assert fresh is not None and stale is not None, f"区域 ({lo}, {hi}) 未产出统计"
            assert len(fresh) == len(stale) == 1
            assert _entry_dict(fresh[0]) == _entry_dict(stale[0]), (
                f"区域 ({lo}, {hi}) 统计结果发生漂移"
            )

    def test_no_full_scan_left_on_the_stats_path(self, plot_factory, monkeypatch):
        """统计链路不得再出现 np.argmin：一次都不许调。

        反向对照是上一条用例（同一条链路在 `_legacy` 下能正常出结果），
        本条只钉「没有全量扫描」，两者合起来才是「更快且没变」。
        """
        pw = _arm_stats(plot_factory, np.linspace(0, 100, 1001), np.zeros(1001))

        calls = []
        real_argmin = np.argmin

        def spy(*args, **kwargs):
            calls.append(args)
            return real_argmin(*args, **kwargs)

        monkeypatch.setattr(np, "argmin", spy)
        entries = pw._mark_region_manager._get_mark_stats_multi_curve(20.0, 80.0)

        assert entries is not None
        assert calls == [], f"统计路径仍在做全量扫描: {calls}"
