"""破坏性动作的状态栏回执文案：同一动作的所有入口共用同一句。"""


def page_closed(title: str) -> str:
    return f"已关闭「{title}」"


def pages_closed(count: int) -> str:
    return f"已关闭 {count} 个绘图页"


def shelf_entries_removed(count: int, name: str = "") -> str:
    if count == 1 and name:
        return f"已删除暂存条目「{name}」"
    return f"已删除 {count} 个暂存条目"


def shelf_cleared(count: int) -> str:
    return f"已清空暂存区 · {count} 个条目"
