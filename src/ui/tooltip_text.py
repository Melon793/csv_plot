"""变量名悬停提示文案（多处共用，避免格式漂移）"""

from __future__ import annotations

import html

from src.ui import theme

# loader 在缺单位时写 "-"，历史/异常数据可能留下空串，两者都按"无单位"处理
NO_UNIT = ("", "-")

UNIT_PX = 11  # 单位是副信息：比名称小一档，视觉上退到第二层


def var_tooltip(name, unit=None) -> str:
    """构造变量名悬停提示（名称加粗为第一层，单位为次级灰字）。

    显式 `<qt>` 富文本 + 转义：变量名里的成对尖括号会被 QToolTip 当标签吃掉，
    `&` 也可能被实体化，转义后必须包 `<qt>` 才能保证按富文本还原原文。
    """
    text = f"<b>{html.escape(str(name))}</b>"
    u = (unit or "").strip()
    if u and u not in NO_UNIT:
        text += (
            f"<br><span style='{theme.muted_text(UNIT_PX)}'>"
            f"单位: {html.escape(u)}</span>"
        )
    return f"<qt>{text}</qt>"