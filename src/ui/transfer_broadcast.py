"""曲线跨页搬运的状态栏播报：所有兄弟路径共用同一套文案。

设计口径（tmp/2026-11-13/curve-transfer-preview-plan.md §7）：
- 入区（剪切/复制/源销毁拒绝）、出区（移动/复制）、判重跳过、批量失败、
  plot drop 源销毁降级、加载拒绝、stale 拦截 —— 全部收口到本模块；
- 禁止各入口自拼文案，保证同一动作在任何入口播报同规格。
"""

from __future__ import annotations

from typing import Any, Iterable

_JOIN = "、"


def _names_text(names: Iterable[str]) -> str:
    return _JOIN.join(str(name) for name in names)


def build_transfer_message(
    action: str,
    names: Iterable[str] = (),
    *,
    src_label: str = "",
    dst_label: str = "",
    skipped: Iterable[str] = (),
    failed: Iterable[str] = (),
    degraded: bool = False,
    updated: int = 0,
) -> str:
    """拼装搬运播报文案。

    Args:
        action: ``cut`` 剪切入区 / ``shelf_copy`` 复制入区 /
            ``move`` 出区移出 / ``copy`` 出区复制
        names: 成功搬运的变量名
        src_label: 来源徽章（"页1·图2" / "暂存区" / "变量列表"）
        dst_label: 目标徽章（出区时必填）
        skipped: 目标已存在而跳过的变量
        failed: 数据校验失败、未搬运的变量
        degraded: 源绘图已关闭（plot drop 端按复制降级）
        updated: 同名暂存条目被更新的数量
    """
    names = list(names)
    skipped = list(skipped)
    failed = list(failed)
    parts: list[str] = []

    if action == "cut":
        text = f"已剪切 {_names_text(names)}" if names else "未搬运"
        if names:
            if src_label:
                text += f"（{src_label}）"
            text += "→ 暂存区，请在目标图拖出"
        parts.append(text)
    elif action == "shelf_copy":
        text = f"已复制 {_names_text(names)}" if names else "未搬运"
        if names:
            if src_label:
                text += f"（{src_label}）"
            text += "→ 暂存区"
        parts.append(text)
    elif action == "move":
        if names:
            parts.append(
                f"已移动 {_names_text(names)}："
                f"{src_label or '暂存区'} → {dst_label}"
            )
        else:
            parts.append("未搬运")
    elif action == "copy":
        if names:
            parts.append(
                f"已复制 {_names_text(names)}："
                f"{src_label or '暂存区'} → {dst_label}"
            )
        else:
            parts.append("未搬运")
    elif names:
        # 容错：未知 action 不吞掉成功信息
        parts.append(f"已搬运 {_names_text(names)}")

    if skipped:
        parts.append(
            f"已跳过 {len(skipped)} 个已存在变量：{_names_text(skipped)}"
        )
    if failed:
        parts.append(f"{_names_text(failed)} 数据校验失败，未搬运")
    if degraded:
        parts.append("源绘图已关闭，已按复制处理")
    if updated:
        parts.append(f"已更新 {updated} 条同名暂存")
    return "；".join(part for part in parts if part)


def build_return_message(
    groups: Iterable[tuple[str, Iterable[str], bool]],
    *,
    skipped: Iterable[str] = (),
    failed: Iterable[str] = (),
    origin_closed: Iterable[str] = (),
    no_origin: Iterable[str] = (),
) -> str:
    """拼装「退回原图」播报。

    Args:
        groups: ``(目标徽章, 退回成功的变量名, 该图当前是否被布局隐藏)``
        skipped: 原图已有同名、未退回的变量（条目保留）
        failed: 数据校验失败、未退回的变量
        origin_closed: 原绘图已关闭/已移除的变量（条目保留）
        no_origin: 复制/变量列表入区，无原图可退回的变量

    全句仍用「；」连接，与 ``build_transfer_message`` 同口径。
    """
    parts: list[str] = []
    success_parts: list[str] = []
    for label, names, hidden in groups:
        names = list(names)
        if not names:
            continue
        text = f"已退回 {_names_text(names)}：暂存区 → {label or '原绘图'}"
        if hidden:
            text += "（该子图当前被布局隐藏）"
        success_parts.append(text)

    skipped = list(skipped)
    failed = list(failed)
    origin_closed = list(origin_closed)
    no_origin = list(no_origin)
    if not (success_parts or skipped or failed or origin_closed or no_origin):
        return ""

    parts.extend(success_parts)
    if not success_parts:
        parts.append("未退回")
    if skipped:
        parts.append(f"原图已有 {_names_text(skipped)}，条目已保留")
    if failed:
        parts.append(f"{_names_text(failed)} 数据校验失败，未退回")
    if origin_closed:
        parts.append(f"{_names_text(origin_closed)} 的原绘图已关闭，未退回（条目已保留）")
    if no_origin:
        parts.append(f"{_names_text(no_origin)} 无原绘图来源，仍留在暂存区")
    return "；".join(part for part in parts if part)


def broadcast_return(
    host: Any,
    groups: Iterable[tuple[str, Iterable[str], bool]],
    **kwargs: Any,
) -> str:
    """统一播报「退回原图」（无 ``_show_status_message`` 时静默）。"""
    message = build_return_message(groups, **kwargs)
    if not message:
        return ""
    show = getattr(host, "_show_status_message", None)
    if callable(show):
        show(message)
    return message


def broadcast_transfer(
    host: Any,
    action: str,
    names: Iterable[str] = (),
    **kwargs: Any,
) -> str:
    """统一播报入口：host 为 MainWindow（无 ``_show_status_message`` 时静默）。

    Returns:
        实际播报的文案（无文案时为空串），供测试与日志复用。
    """
    message = build_transfer_message(action, names, **kwargs)
    if not message:
        return ""
    show = getattr(host, "_show_status_message", None)
    if callable(show):
        show(message)
    return message


def broadcast_text(host: Any, message: str) -> str:
    """非搬运结果的单句播报（源销毁降级 / 加载拒绝 / stale 拦截）。"""
    if not message:
        return ""
    show = getattr(host, "_show_status_message", None)
    if callable(show):
        show(message)
    return message


def build_selection_message(action: str, name: str = "") -> str:
    """选择手势播报文案（D9 唯一来源，供各入口与测试复用）。

    未知 action 返回空串 = 不播报（调用点据此静默）。
    """
    if action == "select":
        return f"已选中：{name}"
    if action == "deselect":
        return f"已取消选中：{name}"
    if action == "clear":
        return "已清除选择"
    return ""


def announce_selection(pw: Any, action: str, name: str = "") -> str:
    """手势层薄封装：取 host + 文案 + 播报，返回实际文案（空串=未播报）。

    只在**用户手势**调用点使用；系统路径（setHtml 重放 / 占位清选 /
    reload prune）一律静默。host 无 ``_show_status_message`` 时
    ``broadcast_text`` 静默跳过但仍返回原文案，独立 component 测试安全。
    """
    msg = build_selection_message(action, name)
    if not msg:
        return ""
    window = getattr(pw, "window", None)
    host = window() if callable(window) else None
    return broadcast_text(host, msg)


def source_closed_message(names: Iterable[str]) -> str:
    return f"源绘图已关闭，无法暂存 {_names_text(names)}"


def stale_message(names: Iterable[str]) -> str:
    return f"{_names_text(names)} 在当前数据中不存在，已置灰"


def loading_reject_message() -> str:
    return "数据加载中，暂不可搬运变量"
