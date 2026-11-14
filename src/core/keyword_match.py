"""关键词匹配：子串 + `*` / `?` 通配符

变量列表过滤（MyTableWidget.hide_non_matching）与绘图变量编辑器搜索栏
（VariableSearchBar._filter_and_sort）共用，保证两个入口语义一致。

语义：
- 关键词不含通配符 → 保持原有纯子串匹配，行为零变化
- `*` → 任意长度（含 0 个字符）
- `?` → 恰好一个字符（不区分是否为换行）
- 其余字符一律按字面量处理：变量名里的 `(`、`.`、`[` 无需转义；
  文本自身含 `*` / `?` 时也按普通字符对待（由通配符去匹配它们）
- 匹配位置不锚定：名字任意位置命中即算命中（与原有子串语义一致）

实现说明：不使用正则。`*a*a*a*b` 这类模式在 re 里会产生灾难性回溯；
这里用经典双指针 glob（最坏 O(len(text)*len(pattern))，无指数回溯），
字面量逐字符比较，天然不需要转义。

匹配对象必须是已 lower() 的文本；调用方在逐行循环外准备好，
避免大列表里每个关键词重复做一次 lower()。
"""

from __future__ import annotations

_WILDCARDS = "*?"


def _glob_match(text: str, pattern: str) -> bool:
    """整串通配匹配（经典双指针，支持 `*` / `?`）"""
    ti = pi = 0
    star = -1
    match = 0
    n, m = len(text), len(pattern)
    while ti < n:
        if pi < m and pattern[pi] == "*":
            # `*` 必须先于字面量判断：文本自身含 `*` 时也要按通配处理，
            # 否则会把模式的 `*` 当普通字符配对，丢掉回溯点
            star = pi
            match = ti
            pi += 1
        elif pi < m and (pattern[pi] == "?" or pattern[pi] == text[ti]):
            ti += 1
            pi += 1
        elif star != -1:
            # 回溯到最近的 `*`，让它多吃一个字符
            pi = star + 1
            match += 1
            ti = match
        else:
            return False
    while pi < m and pattern[pi] == "*":
        pi += 1
    return pi == m


def _window_match(text: str, keyword: str) -> bool:
    """只含 `?` 的关键词：滑动窗口，`?` 恰好占一个字符"""
    n, m = len(text), len(keyword)
    if m > n:
        return False
    for start in range(n - m + 1):
        for offset in range(m):
            ch = keyword[offset]
            if ch != "?" and ch != text[start + offset]:
                break
        else:
            return True
    return False


class KeywordMatcher:
    """预编译的关键词匹配器：构造一次，循环内逐行调用 matches_lower()"""

    __slots__ = ("keyword", "_has_wildcard")

    def __init__(self, keyword: str):
        self.keyword = keyword.lower()
        self._has_wildcard = any(ch in self.keyword for ch in _WILDCARDS)

    def matches_lower(self, text_lower: str) -> bool:
        """text_lower 必须已 lower()；返回关键词是否命中"""
        if not self._has_wildcard:
            # 无通配符：纯子串路径，与改造前完全一致
            return self.keyword in text_lower
        if "*" in self.keyword:
            # 前后各补一个 `*`：把「整串匹配」变成「任意位置包含」
            return _glob_match(text_lower, f"*{self.keyword}*")
        return _window_match(text_lower, self.keyword)
