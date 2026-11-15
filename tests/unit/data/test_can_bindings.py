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


def test_restrict_to_pool_drops_removed_files_and_empty_channels(tmp_path):
    a = str(tmp_path / "a.dbc")
    b = str(tmp_path / "b.dbc")
    cfg = CanBindings.from_paths(
        [a, b], by_channel={0: [a, b], 1: [b]}, alias={0: "动力", 1: "车身"}
    )

    kept = cfg.restrict_to_pool([a])

    assert kept.pool == (a,)
    assert kept.by_channel == {0: (a,)}
    # 通道 1 的绑定文件被移除 → 绑定清空 → 别名一起清掉（不留孤立别名）
    assert kept.alias == {0: "动力"}


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