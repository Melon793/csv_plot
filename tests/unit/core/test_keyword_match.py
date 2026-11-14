"""关键词通配匹配器单元测试（纯逻辑，无 Qt）。

锁住语义边界：
- 不含通配符的关键词保持原子串匹配
- `*` 任意长度（含空）、`?` 恰好一个字符
- 其余字符一律字面量：`(`、`.`、`[` 不会被当正则语法
- 多段 `*` 不产生指数回溯（对抗性模式应当在毫秒内结束）
"""

from __future__ import annotations

from src.core.keyword_match import KeywordMatcher


def _hit(keyword: str, name: str) -> bool:
    return KeywordMatcher(keyword).matches_lower(name.lower())


def test_plain_keyword_keeps_substring_semantics():
    assert _hit("RPM", "engine_rpm_1")  # 大小写不敏感
    assert _hit("rpm_1", "engine_rpm_10")  # 子串命中即可，不要求整串
    assert not _hit("rpm", "temp_coolant")


def test_empty_keyword_matches_anything():
    assert _hit("", "engine_rpm")
    assert _hit("", "")


def test_star_matches_any_length():
    assert _hit("demo__*_msg", "demo__rl_w_msg")
    assert _hit("demo__*_msg", "demo___msg")  # `*` 可以是 0 个字符
    assert _hit("*w_msg", "demo__rl_w_msg")  # 位置不锚定
    assert not _hit("demo__*_msg", "demo__rl_w_sig")


def test_bare_star_and_question():
    assert _hit("*", "")  # 纯 `*` 命中一切（含空名）
    assert _hit("*", "anything")
    assert _hit("?", "a")
    assert not _hit("?", "")  # 纯 `?` 要求恰好一个字符
    assert _hit("?", "\n")  # 不区分是否为换行


def test_question_matches_exactly_one_char():
    assert _hit("demo__r?_w_msg", "demo__rl_w_msg")
    assert not _hit("demo__r?_w_msg", "demo__rll_w_msg")
    assert not _hit("demo__r?_w_msg", "demo__r_w_msg")


def test_regex_special_chars_are_literal_when_wildcard_present():
    assert _hit("temp(ENG01).*oil", "temp(eng01).oil")
    assert not _hit("temp(ENG01).*oil", "tempENG01xoil")
    assert _hit("sig[1]*", "sig[1]_value")
    assert not _hit("sig[1]*", "sig1_value")


def test_regex_special_chars_are_literal_without_wildcard():
    assert _hit("sig[1]", "sig[1]_value")
    assert not _hit("sig[1]", "sig1_value")


def test_text_containing_wildcard_chars_is_matched_by_wildcard():
    """文本自身含 `*` 时，模式的 `*` 仍必须是通配符（不能按字面配对丢回溯点）"""
    assert _hit("k*g", "kg*m/s^2")
    assert _hit("*n", "n*m")
    assert _hit("a*b", "a*b")


def test_adversarial_pattern_does_not_blow_up():
    """多段 `*` 的经典回溯陷阱模式应快速返回 False"""
    assert not _hit("*a*a*a*a*a*a*a*a*b", "a" * 120)
    assert _hit("*a*a*a*a*a*a*a*a*b", "a" * 120 + "b")


def test_no_match():
    assert not _hit("demo__*_msg", "engine_rpm")
