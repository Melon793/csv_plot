"""PAV 列内垂直堆叠护栏（纯数学单元测试）。

``CursorManager._pav_stack`` 用等渗回归直解替换原贪心单向下推链：
最小化 Σ(y−ideal)²，约束相邻标签不重叠，簇围绕中心上下对称分裂。

另含两条结构性断言：混合密度权重删除后代码中不再存在「像素半径 vs
数据差」比较，因此算法对输入量纲（整体缩放）保持不变。
"""

from __future__ import annotations

import random

import pytest

from src.ui.widgets.cursor_manager import CursorManager

H = 20.0
GAP = 4.0
STEP = H + GAP
TOP = 0.0
BOTTOM = 400.0


def _items(ideals: list[float]) -> list[dict]:
    return [{"ideal": v} for v in ideals]


def _clamp(v: float, top: float, bottom: float, h: float) -> float:
    lo, hi = top + h / 2, bottom - h / 2
    if lo > hi:
        return (top + bottom) / 2
    return max(lo, min(hi, v))


def _greedy_stack(ideals: list[float]) -> list[float]:
    """现行贪心链的基线复刻（混合项 + 单向下推 + 正/反扫描）。"""
    n = len(ideals)
    if n == 0:
        return []
    half = H / 2
    if n == 1:
        return [_clamp(ideals[0], TOP, BOTTOM, H)]

    spacing = (BOTTOM - TOP) / n
    out: list[float] = []
    prev_bottom = TOP + half
    for i, ideal in enumerate(ideals):
        radius = H * 3
        density = sum(1 for v in ideals if abs(v - ideal) <= radius)
        w_ideal = max(0.2, 1.0 - density * 0.15)
        uniform = TOP + spacing * (i + 0.5)
        target = w_ideal * ideal + (1 - w_ideal) * uniform
        max_offset = min(H * (0.5 + n * 0.1), H * 2)
        target = max(ideal - max_offset, min(ideal + max_offset, target))
        target = max(target, prev_bottom + GAP + half)
        target = _clamp(target, TOP, BOTTOM, H)
        out.append(target)
        prev_bottom = target + half

    for i in range(1, n):
        prev_bottom = out[i - 1] + half
        if out[i] - half < prev_bottom + GAP:
            out[i] = min(prev_bottom + GAP + half, BOTTOM - half)
    for i in range(n - 2, -1, -1):
        next_top = out[i + 1] - half
        if out[i] + half > next_top - GAP:
            out[i] = max(next_top - GAP - half, TOP + half)
    return out


# ---------- ① 单调最小间距 ----------

def test_pav_monotonic_with_min_spacing():
    ideals = [30.0, 34.0, 37.0, 36.0, 200.0, 201.0, 360.0]
    y = CursorManager._pav_stack(_items(ideals), TOP, BOTTOM, H, GAP)

    assert len(y) == len(ideals)
    for i in range(1, len(y)):
        assert y[i] - y[i - 1] >= STEP - 1e-9, f"第 {i} 个标签间距不足"
    for v in y:
        assert TOP + H / 2 - 1e-9 <= v <= BOTTOM - H / 2 + 1e-9


# ---------- ② 簇对称分裂 ----------

def test_pav_cluster_splits_symmetrically():
    # 5 个理想位重合于视图中心：应围绕中心上下对称展开
    y = CursorManager._pav_stack(_items([200.0] * 5), TOP, BOTTOM, H, GAP)

    assert y[2] == pytest.approx(200.0)
    assert [v - 200.0 for v in y] == pytest.approx(
        [-2 * STEP, -STEP, 0.0, STEP, 2 * STEP]
    )
    # 上移总量 == 下移总量
    assert pytest.approx(-(y[0] - 200.0) - (y[1] - 200.0)) == (
        (y[3] - 200.0) + (y[4] - 200.0)
    )


# ---------- ③ 边界 clamp + 倒挂防御 + 有界推送 ----------

def test_pav_boundary_clamp_and_bounded_push():
    # 全部贴近上边缘：正推后仍在视口内且保持最小间距
    top_y = CursorManager._pav_stack(_items([0.0] * 4), TOP, BOTTOM, H, GAP)
    assert top_y == pytest.approx([10.0, 34.0, 58.0, 82.0])
    assert top_y[0] >= TOP + H / 2 - 1e-9

    # 全部贴近下边缘：反推后仍在视口内
    bottom_y = CursorManager._pav_stack(_items([400.0] * 4), TOP, BOTTOM, H, GAP)
    assert bottom_y == pytest.approx([318.0, 342.0, 366.0, 390.0])
    assert bottom_y[-1] <= BOTTOM - H / 2 + 1e-9

    # 极矮图（区间倒挂）：单标签回退列中心，不推出界
    inverted = CursorManager._pav_stack(_items([50.0]), 0.0, 10.0, H, GAP)
    assert inverted == pytest.approx([5.0])


# ---------- ④ n=0 / n=1 ----------

def test_pav_empty_and_single():
    assert CursorManager._pav_stack([], TOP, BOTTOM, H, GAP) == []

    assert CursorManager._pav_stack(_items([50.0]), TOP, BOTTOM, H, GAP) == (
        pytest.approx([50.0])
    )
    # 单标签同样受边界 clamp 约束
    assert CursorManager._pav_stack(_items([-5.0]), TOP, BOTTOM, H, GAP) == (
        pytest.approx([10.0])
    )
    assert CursorManager._pav_stack(_items([500.0]), TOP, BOTTOM, H, GAP) == (
        pytest.approx([390.0])
    )


# ---------- ⑤ 随机分布：PAV 最大位移 ≤ 贪心（50 组固定种子） ----------

def test_pav_max_displacement_not_worse_than_greedy():
    rng = random.Random(20261115)
    for case in range(50):
        n = rng.randint(2, 12)
        ideals = sorted(rng.uniform(-40.0, 440.0) for _ in range(n))
        pav = CursorManager._pav_stack(_items(ideals), TOP, BOTTOM, H, GAP)
        greedy = _greedy_stack(ideals)

        pav_max = max(abs(v - ideal) for v, ideal in zip(pav, ideals))
        greedy_max = max(abs(v - ideal) for v, ideal in zip(greedy, ideals))
        assert pav_max <= greedy_max + 1e-9, (
            f"case {case}: PAV 最大位移 {pav_max:.2f} > 贪心 {greedy_max:.2f}"
        )


# ---------- ⑥ 幂等性与结构性（合并顺序 / 量纲缩放） ----------

def _pav_reference_left_merge(ideals: list[float]) -> list[float]:
    """参考实现：每次从左到右合并首个违序块（验证合并顺序无关）。"""
    blocks = [[ideal - i * STEP, 1, i] for i, ideal in enumerate(ideals)]
    changed = True
    while changed:
        changed = False
        for k in range(len(blocks) - 1):
            a, b = blocks[k], blocks[k + 1]
            if b[0] / b[1] < a[0] / a[1] - 1e-9:
                a[0] += b[0]
                a[1] += b[1]
                blocks.pop(k + 1)
                changed = True
                break

    y = [0.0] * len(ideals)
    for block_sum, count, start in blocks:
        mean = block_sum / count
        for i in range(start, start + int(count)):
            y[i] = mean + i * STEP
    return y


def test_pav_idempotent_and_structure_preserved():
    # 远离边界（不触发 clamp），可逐值对照参考实现
    ideals = [112.0, 115.0, 115.5, 150.0, 280.0, 282.0, 283.0]

    first = CursorManager._pav_stack(_items(ideals), TOP, BOTTOM, H, GAP)
    second = CursorManager._pav_stack(_items(ideals), TOP, BOTTOM, H, GAP)
    assert first == second, "同输入两次结果必须一致"

    # 违序块的合并结果与合并顺序无关（内部均值不受合并路径影响）
    assert first == pytest.approx(_pav_reference_left_merge(ideals))

    # 量纲缩放不变性：整体缩放 k 倍，输出同步缩放 k 倍
    # （原混合项以像素半径比较数据差，会在此处产生量纲漂移）
    k = 1e6
    scaled = CursorManager._pav_stack(
        _items([v * k for v in ideals]), TOP * k, BOTTOM * k, H * k, GAP * k
    )
    assert scaled == pytest.approx([v * k for v in first])