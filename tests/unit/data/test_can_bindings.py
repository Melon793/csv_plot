"""``CanBindings`` 单测：1 基转换、校验、restrict、判等（评审 F8/F9）。"""

from __future__ import annotations

import pytest

from src.data.can_bindings import (
    CanBindings,
    format_bus_channel,
    is_dir_writable,
    load_yaml,
    save_yaml,
    sidecar_config_path,
)


def test_empty_is_unbound():
    cfg = CanBindings.empty()
    assert cfg.pool == ()
    assert cfg.normalized_loader_bindings() == {}
    assert cfg.validate() == []


def test_format_bus_channel_is_one_based_and_uses_alias():
    assert format_bus_channel(0) == "CAN1"
    assert format_bus_channel(1, "车身CAN") == "CAN2 车身CAN"
    assert format_bus_channel(1, "   ") == "CAN2"


def test_from_paths_and_loader_bindings_shape():
    cfg = CanBindings.from_paths(
        ["a.dbc", "b.arxml"],
        by_channel={0: ["a.dbc"], 1: []},
        alias={0: "动力"},
    )
    assert cfg.pool == ("a.dbc", "b.arxml")
    # 空绑定 = 未绑定（用全局池），不传给 loader
    assert cfg.normalized_loader_bindings() == {0: frozenset({"a.dbc"})}
    assert cfg.alias_map() == {0: "动力"}


def test_validate_reports_duplicates_and_pool_external_bindings():
    cfg = CanBindings.from_paths(
        ["a.dbc", "a.dbc"],
        by_channel={0: ["x.dbc"]},
    )
    issues = cfg.validate()
    assert any("重复" in s for s in issues)
    assert any("池外" in s for s in issues)


def test_restrict_to_pool_drops_removed_files_and_keeps_aliases(tmp_path):
    a = str(tmp_path / "a.dbc")
    b = str(tmp_path / "b.dbc")
    cfg = CanBindings.from_paths(
        [a, b], by_channel={0: [a, b], 1: [b]}, alias={0: "动力", 1: "车身"}
    )

    kept = cfg.restrict_to_pool([a])

    assert kept.pool == (a,)
    assert kept.by_channel == {0: (a,)}
    # 别名与绑定解耦：通道 1 的绑定文件被移除，别名仍保留（评审 P0）
    assert kept.alias == {0: "动力", 1: "车身"}


def test_restrict_to_pool_keeps_alias_for_a_channel_that_was_never_bound(tmp_path):
    a = str(tmp_path / "a.dbc")
    cfg = CanBindings.from_paths([a], alias={2: "底盘"})

    assert cfg.restrict_to_pool([a]).alias == {2: "底盘"}


def test_yaml_roundtrip_uses_one_based_channel_keys():
    cfg = CanBindings.from_paths(
        ["p.dbc"], by_channel={0: ["p.dbc"]}, alias={0: "动力"}
    )

    data = cfg.to_yaml_dict()

    assert data["schema"] == "canview/1"
    assert data["channels"] == {"1": {"alias": "动力", "databases": ["p.dbc"]}}
    assert CanBindings.from_yaml_dict(data) == cfg


def test_from_yaml_rejects_non_positive_channel_keys():
    with pytest.raises(ValueError):
        CanBindings.from_yaml_dict({"databases": [], "channels": {"0": {}}})
    with pytest.raises(ValueError):
        CanBindings.from_yaml_dict({"databases": [], "channels": {"x": {}}})


def test_equality_covers_pool_order_bindings_and_alias():
    """F 路径判等必须比整份配置（评审 F9）：只改绑定不得被判「未变化」。"""
    base = dict(by_channel={0: ["p.dbc"]}, alias={0: "动力"})
    a = CanBindings.from_paths(["p.dbc", "q.dbc"], **base)

    assert a == CanBindings.from_paths(["p.dbc", "q.dbc"], **base)
    assert a != CanBindings.from_paths(["q.dbc", "p.dbc"], **base)
    assert a != CanBindings.from_paths(
        ["p.dbc", "q.dbc"], by_channel={1: ["p.dbc"]}, alias={0: "动力"}
    )
    assert a != CanBindings.from_paths(
        ["p.dbc", "q.dbc"], by_channel={0: ["p.dbc"]}, alias={0: "车身"}
    )


# ---------------------------------------------------------------------------
# 显式配置文件 IO（评审 F11：相对路径 / 只读盘 / 坏文件）
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip_with_relative_paths(tmp_path):
    db = tmp_path / "db" / "p.dbc"
    db.parent.mkdir()
    db.write_text("x", encoding="utf-8")
    cfg = CanBindings.from_paths(
        ["db/p.dbc"], by_channel={0: ["db/p.dbc"]}, alias={0: "动力"}
    )
    yaml_path = tmp_path / "conf" / "x.canview.yaml"

    assert save_yaml(cfg, str(yaml_path)) == []

    loaded, issues = load_yaml(str(yaml_path), blf_path=str(tmp_path / "log.blf"))
    assert issues == []
    # 相对路径解析成绝对路径（先相对 YAML 目录、再相对日志目录）
    assert loaded.pool == (str(db.resolve()),)
    assert loaded.by_channel == {0: (str(db.resolve()),)}
    assert loaded.alias == {0: "动力"}


def test_load_resolves_relative_to_the_blf_directory(tmp_path):
    logdir = tmp_path / "logs"
    (logdir / "db").mkdir(parents=True)
    db = logdir / "db" / "q.dbc"
    db.write_text("x", encoding="utf-8")
    cfgdir = tmp_path / "conf"
    cfgdir.mkdir()
    save_yaml(
        CanBindings.from_paths(["db/q.dbc"]), str(cfgdir / "c.canview.yaml")
    )

    loaded, issues = load_yaml(
        str(cfgdir / "c.canview.yaml"), blf_path=str(logdir / "a.blf")
    )

    assert issues == []
    assert loaded.pool == (str(db.resolve()),)


def test_missing_database_is_reported_not_raised(tmp_path):
    yaml_path = tmp_path / "c.canview.yaml"
    save_yaml(CanBindings.from_paths(["nope.dbc"]), str(yaml_path))

    loaded, issues = load_yaml(str(yaml_path), blf_path=str(tmp_path / "x.blf"))

    assert loaded.pool  # 仍保留候选路径，交给 UI 列红
    assert any("数据库不存在" in s for s in issues)


def test_bad_yaml_and_future_schema_return_issues(tmp_path):
    bad = tmp_path / "bad.canview.yaml"
    bad.write_text("::: [", encoding="utf-8")
    cfg, issues = load_yaml(str(bad))
    assert cfg == CanBindings.empty()
    assert issues

    future = tmp_path / "future.canview.yaml"
    future.write_text(
        "schema: canview/9\ndatabases: [a.dbc]\nchannels: {}\n", encoding="utf-8"
    )
    cfg, issues = load_yaml(str(future))
    assert cfg == CanBindings.empty()
    assert any("不受支持" in s for s in issues)


def test_sidecar_path_and_writability_probe(tmp_path, monkeypatch):
    log = tmp_path / "log.blf"
    assert sidecar_config_path(str(log)) == str(tmp_path / "log.canview.yaml")
    assert is_dir_writable(str(tmp_path)) is True

    # 平台中性：Windows 的 chmod 不产生只读目录，直接让探针写入失败
    import src.data.can_bindings as cb

    def _denied(*_args, **_kwargs):
        raise OSError("denied")

    monkeypatch.setattr(cb.tempfile, "mkstemp", _denied)
    assert is_dir_writable(str(tmp_path)) is False

def test_yaml_roundtrip_keeps_alias_only_channels():
    """第二轮评审 P0：别名-only 的通道必须落盘，且读回不能被当成空绑定。"""
    cfg = CanBindings.from_paths(
        ["p.dbc"], by_channel={0: ["p.dbc"]}, alias={1: "车身CAN"}
    )

    data = cfg.to_yaml_dict()

    assert data["channels"]["2"] == {"databases": [], "alias": "车身CAN"}
    loaded = CanBindings.from_yaml_dict(data)
    assert loaded.alias == {1: "车身CAN"}
    assert loaded.by_channel == {0: ("p.dbc",)}
    assert loaded == cfg


# ---------------------------------------------------------------------------
# P1-2（第 5 轮）：合法 YAML 但错误形状 / 非 UTF-8 → issues 降级，不抛异常
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path, text, name="bad.yaml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("text", [
    "schema: canview/1\ndatabases: []\nchannels: abc\n",          # 标量 channels
    "schema: canview/1\ndatabases: []\nchannels:\n  '1': oops\n",  # 标量条目
    "schema: canview/1\ndatabases: []\nchannels: [a]\n",          # 列表 channels
    "schema: canview/1\ndatabases: 5\nchannels: {}\n",            # int databases
])
def test_load_yaml_malformed_shapes_never_raise(tmp_path, text):
    cfg, issues = load_yaml(_write_yaml(tmp_path, text))
    assert cfg.pool == ()
    assert issues, "坏形状必须给出 issue 而不是静默"


def test_load_yaml_scalar_databases_does_not_iterate_chars(tmp_path):
    """databases 写成单字符串时不得逐字符迭代出伪库池（会覆盖好的记忆）。"""
    cfg, issues = load_yaml(_write_yaml(
        tmp_path, "schema: canview/1\ndatabases: p.dbc\nchannels: {}\n"
    ))
    assert cfg.pool == ()
    assert any("列表" in i for i in issues)


def test_load_yaml_non_str_alias_is_reported_not_stringified(tmp_path):
    """YAML 1.1 未加引号的 yes 解析成 bool；不得 str() 成 'True' 别名。"""
    cfg, issues = load_yaml(_write_yaml(
        tmp_path,
        "schema: canview/1\ndatabases: []\nchannels:\n  '1':\n    alias: yes\n",
    ))
    assert cfg.alias == {}
    assert any("alias" in i for i in issues)


def test_load_yaml_non_utf8_file_returns_issues(tmp_path):
    p = tmp_path / "gbk.yaml"
    p.write_bytes(
        "schema: canview/1\ndatabases: []\nchannels:\n  '1':\n    alias: 动力\n"
        .encode("gbk")
    )
    cfg, issues = load_yaml(str(p))
    assert cfg.pool == ()
    assert any("UTF-8" in i for i in issues)


@pytest.mark.parametrize("data", [
    {"databases": [], "channels": [1]},
    {"databases": [], "channels": {"1": "oops"}},
    {"databases": 5, "channels": {}},
    {"databases": "p.dbc", "channels": {}},
    {"databases": [], "channels": {"1": {"alias": True}}},
    {"databases": [], "channels": {"1": {"databases": "p.dbc"}}},
])
def test_from_yaml_dict_bad_shapes_raise_valueerror(data):
    """按文件记忆路径只捕 ValueError：形状非法必须抛 ValueError 而非 AttributeError。"""
    with pytest.raises(ValueError):
        CanBindings.from_yaml_dict(data)


def test_from_yaml_dict_valid_roundtrip_unchanged():
    """防御不得误伤合法配置（含别名-only 通道与 1 基转换）。"""
    cfg = CanBindings.from_paths(
        ["p.dbc", "q.dbc"], by_channel={0: ["q.dbc"]}, alias={1: "动力CAN"}
    )
    assert CanBindings.from_yaml_dict(cfg.to_yaml_dict()) == cfg


def test_load_yaml_deep_nesting_returns_issues(tmp_path):
    """评审 B1：病态深层嵌套是合法 YAML 语法，RecursionError 不得穿透契约。"""
    cfg, issues = load_yaml(_write_yaml(tmp_path, "[" * 3000 + "]" * 3000))
    assert cfg.pool == ()
    assert issues


@pytest.mark.parametrize("data", [
    5,                                                    # 顶层非映射（N1）
    {"databases": ""},                                    # falsy 标量（N2）
    {"databases": 0},
    {"databases": False},
    {"channels": []},                                     # falsy 非映射（N2）
    {"databases": [123], "channels": {}},                 # 非 str 池条目（N3）
    {"databases": [], "channels": {"1": {"databases": [123]}}},
])
def test_from_yaml_dict_more_bad_shapes_raise_valueerror(data):
    with pytest.raises(ValueError):
        CanBindings.from_yaml_dict(data)


def test_from_yaml_dict_none_entry_and_mixed_keys_are_tolerated():
    """正面护栏：None 条目与 int/str 混合 1 基键是合法手编辑形态，不得误伤。"""
    cfg = CanBindings.from_yaml_dict({
        "databases": ["p.dbc"],
        "channels": {"1": None, 2: {"alias": "动力CAN"}},
    })
    assert cfg.pool == ("p.dbc",)
    assert cfg.alias == {1: "动力CAN"}
    assert cfg.by_channel == {}


@pytest.mark.parametrize("text", [
    "schema: canview/1\ndatabases: 2020-99-99\nchannels: {}\n",   # 越界月份
    "schema: canview/1\ndatabases: []\nchannels:\n  '1':\n    alias: 2020-13-01\n",
    "schema: canview/1\ndatabases: 2020-01-01T25:00:00\nchannels: {}\n",
])
def test_load_yaml_invalid_timestamp_scalar_returns_issues(tmp_path, text):
    """复审 BL-1：PyYAML 时间戳构造抛裸 ValueError，不得穿透契约。"""
    cfg, issues = load_yaml(_write_yaml(tmp_path, text))
    assert cfg.pool == ()
    assert issues


def test_load_yaml_huge_integer_returns_issues(tmp_path):
    """复审 BL-1：超 CPython int 转换上限（4300 位）的数字标量同样降级。"""
    cfg, issues = load_yaml(_write_yaml(
        tmp_path, "schema: canview/1\ndatabases: " + "9" * 5000 + "\nchannels: {}\n"
    ))
    assert cfg.pool == ()
    assert issues


def test_load_yaml_schema_alias_bomb_message_is_bounded(tmp_path):
    """复审 BL-2：schema 位别名炸弹的 repr 指数展开不得进错误消息（冻结面）。"""
    lines = ["x: &l0 [a,a,a,a,a,a,a,a,a]"]
    for i in range(1, 6):
        refs = ",".join(["*l%d" % (i - 1)] * 9)
        lines.append("x%d: &l%d [%s]" % (i, i, refs))
    lines.append("schema: *l5")
    lines.append("databases: []")
    lines.append("channels: {}")
    cfg, issues = load_yaml(_write_yaml(tmp_path, "\n".join(lines) + "\n"))
    assert cfg.pool == ()
    assert issues and "不受支持" in issues[0]
    assert len(issues[0]) < 200, "错误消息必须只含类型名/截断字符串，不得展开嵌套"


def test_load_yaml_channel_falsy_databases_reports_issue(tmp_path):
    """复审 NB-1：通道层 falsy 标量与顶层同口径——报 issue 而非静默吞掉。"""
    cfg, issues = load_yaml(_write_yaml(
        tmp_path,
        "schema: canview/1\ndatabases: []\nchannels:\n  '1':\n    databases: 0\n",
    ))
    assert cfg.by_channel == {}
    assert any("databases" in i for i in issues)


def test_load_yaml_empty_string_db_entry_is_dropped(tmp_path):
    """复审 NB-3：空字符串库条目不得被解析成配置目录本身混进池。"""
    cfg, issues = load_yaml(_write_yaml(
        tmp_path, "schema: canview/1\ndatabases: ['']\nchannels: {}\n"
    ))
    assert cfg.pool == ()
