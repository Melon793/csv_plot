"""ViewBox state 兼容性工具

封装 pyqtgraph ViewBox 内部 state 的直接修改操作，
提供运行时探测 + 安全降级，避免 pyqtgraph 版本升级破坏。

设计：
- 首次使用时校验 state key 是否存在（'autoVisibleOnly', 'autoRange'）
- 缓存校验结果，后续直接走对应路径，无额外开销
- 高效路径：直接修改 state（无 sigRangeChanged 触发）
- 降级路径：blockSignals + setAutoVisible + 清除 _autoRangeNeedsUpdate
  （抑制反馈循环，但接受单次 paint 的轻微开销）

为何不锁定 pyqtgraph 版本：
- 直接修改 state 是性能关键路径（每次 wheel 交互都走）
- 用 blockSignals + API 调用会触发 _autoRangeNeedsUpdate + prepareForPaint
  → 延迟 sigRangeChanged → 反馈循环
- 所以优先用 state 直接修改，仅在 key 不存在时降级
"""

from __future__ import annotations

from src.core.logger import get_logger

logger = get_logger(__name__)

# 模块级缓存：None=未校验, True=可用直接修改, False=需降级
_state_keys_available: bool | None = None


def _verify_state_keys(view_box) -> bool:
    """校验 pyqtgraph ViewBox state 是否包含所需 key

    只在首次调用时执行，结果缓存到模块级变量。
    """
    global _state_keys_available
    if _state_keys_available is not None:
        return _state_keys_available

    try:
        state = view_box.state
        _ = state['autoVisibleOnly']
        _ = state['autoRange']
        _state_keys_available = True
    except (KeyError, TypeError, AttributeError):
        import pyqtgraph
        logger.warning(
            "[COMPAT] pyqtgraph %s: ViewBox state keys changed "
            "('autoVisibleOnly' or 'autoRange' missing). "
            "Y autoVisible optimization will use API fallback path "
            "(may have minor feedback loop risk).",
            getattr(pyqtgraph, '__version__', 'unknown'),
        )
        _state_keys_available = False
    return _state_keys_available


def disable_y_autovisible(view_box) -> bool:
    """禁用 ViewBox 的 Y autoVisible

    优先直接修改 state['autoVisibleOnly'][1]=False（高效，无 sigRangeChanged）；
    降级用 blockSignals + setAutoVisible + 清除 _autoRangeNeedsUpdate。

    Args:
        view_box: pyqtgraph ViewBox 实例

    Returns:
        True=使用了高效路径, False=使用了降级路径
    """
    if _verify_state_keys(view_box):
        try:
            view_box.state['autoVisibleOnly'][1] = False
            return True
        except (KeyError, TypeError, IndexError):
            pass  # 落到降级路径

    # 降级路径：blockSignals + API 调用 + 清除延迟更新标志
    was_blocked = view_box.blockSignals(True)
    try:
        view_box.setAutoVisible(x=False, y=False)
        # 清除 _autoRangeNeedsUpdate，避免下次 paint 时触发延迟 sigRangeChanged
        # 这是反馈循环的根因；用 hasattr 保护，避免私有属性改名的风险
        if hasattr(view_box, '_autoRangeNeedsUpdate'):
            view_box._autoRangeNeedsUpdate = False
    finally:
        view_box.blockSignals(was_blocked)
    return False


def disable_y_autorange(view_box) -> bool:
    """禁用 ViewBox 的 Y autoRange[1]（交互期 Y 冻结用）

    仅禁用 autoVisibleOnly 不足以冻结 Y 轴：autoVisibleOnly 只控制
    auto-range 的数据源，不禁用 auto-range 本身；必须同时禁用
    autoRange[1] 才能阻止 prepareForPaint → updateAutoRange 重算 Y
    （见 tmp/yrange_jitter_after_phase2_analysis.md 根因 #3）。

    优先直接修改 state['autoRange'][1]=False（高效，无副作用）；
    降级用 blockSignals + enableAutoRange(YAxis, False)。

    Returns:
        True=使用了高效路径, False=使用了降级路径
    """
    if _verify_state_keys(view_box):
        try:
            view_box.state['autoRange'][1] = False
            return True
        except (KeyError, TypeError, IndexError):
            pass  # 落到降级路径

    # 降级路径：先清除挂起的 _autoRangeNeedsUpdate 再禁用，
    # 避免 enableAutoRange(False) 的 "one last auto-range"
    # （ViewBox.py L886-887）在禁用瞬间触发一次 updateAutoRange → Y 跳变
    was_blocked = view_box.blockSignals(True)
    try:
        if hasattr(view_box, '_autoRangeNeedsUpdate'):
            view_box._autoRangeNeedsUpdate = False
        view_box.enableAutoRange(axis=view_box.YAxis, enable=False)
    finally:
        view_box.blockSignals(was_blocked)
    return False


def restore_y_autorange(view_box) -> bool:
    """恢复 ViewBox 的 Y autoRange[1]=True

    优先直接修改 state['autoRange'][1]=True；
    降级用 enableAutoRange(Y) + 清除 _autoRangeNeedsUpdate。

    Returns:
        True=使用了高效路径, False=使用了降级路径
    """
    if _verify_state_keys(view_box):
        try:
            view_box.state['autoRange'][1] = True
            return True
        except (KeyError, TypeError, IndexError):
            pass

    # 降级路径：blockSignals + API 调用
    was_blocked = view_box.blockSignals(True)
    try:
        view_box.enableAutoRange(axis=view_box.YAxis, enable=True)
        if hasattr(view_box, '_autoRangeNeedsUpdate'):
            view_box._autoRangeNeedsUpdate = False
    finally:
        view_box.blockSignals(was_blocked)
    return False


def restore_y_autovisible(view_box) -> bool:
    """恢复 ViewBox 的 Y autoVisibleOnly[1]=True

    Returns:
        True=使用了高效路径, False=使用了降级路径
    """
    if _verify_state_keys(view_box):
        try:
            view_box.state['autoVisibleOnly'][1] = True
            return True
        except (KeyError, TypeError, IndexError):
            pass

    # 降级路径：blockSignals + API 调用
    was_blocked = view_box.blockSignals(True)
    try:
        view_box.setAutoVisible(x=False, y=True)
        if hasattr(view_box, '_autoRangeNeedsUpdate'):
            view_box._autoRangeNeedsUpdate = False
    finally:
        view_box.blockSignals(was_blocked)
    return False


def reset_compatibility_cache() -> None:
    """重置校验缓存（仅用于测试）

    正常运行时无需调用：首次校验后结果固定，
    pyqtgraph 版本不会在运行时变化。
    """
    global _state_keys_available
    _state_keys_available = None
