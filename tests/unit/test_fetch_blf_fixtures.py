from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import fetch_blf_fixtures as fixtures


def _setup_manifest(tmp_path: Path, monkeypatch, names: tuple[str, ...]) -> dict:
    dest = tmp_path / "bus_data"
    dest.mkdir()
    manifest_path = dest / "manifest.json"
    files = {
        name: {
            "url": f"https://example.invalid/{name}",
            "sha256": fixtures._sha256(f"locked:{name}".encode()),
        }
        for name in names
    }
    payload = {"files": files, "licenses": {}}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    specs = tuple(
        (name, "Demo/source", name, "MIT", "test fixture") for name in names
    )
    monkeypatch.setattr(fixtures, "DEST", dest)
    monkeypatch.setattr(fixtures, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fixtures, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(fixtures, "LICENSE_DIR", dest / "licenses")
    monkeypatch.setattr(fixtures, "_RAW_SPECS", specs)
    real_build = fixtures.build
    monkeypatch.setattr(
        fixtures, "build", lambda **kwargs: real_build(specs, **kwargs)
    )
    return payload


def _snapshot(dest: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(dest)): path.read_bytes()
        for path in dest.rglob("*")
        if path.is_file()
    }


def test_check_is_local_read_only_and_reports_bad_files(tmp_path, monkeypatch, capsys):
    payload = _setup_manifest(
        tmp_path, monkeypatch, ("good.blf", "drift.blf", "missing.blf")
    )
    dest = fixtures.DEST
    (dest / "good.blf").write_bytes(b"locked:good.blf")
    (dest / "drift.blf").write_bytes(b"changed")
    before = _snapshot(dest)
    network_calls: list[str] = []

    def forbidden_fetch(url, **kwargs):
        network_calls.append(url)
        return b"locked:missing.blf"

    def forbidden_licenses(**kwargs):
        raise AssertionError("--check must not fetch licenses")

    monkeypatch.setattr(fixtures, "_fetch", forbidden_fetch)
    monkeypatch.setattr(fixtures, "fetch_licenses", forbidden_licenses)

    result = fixtures.main(["--check"])

    assert result != 0
    assert network_calls == []
    assert _snapshot(dest) == before
    assert json.loads(fixtures.MANIFEST_PATH.read_text(encoding="utf-8")) == payload
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["mode"] == "check"
    assert report["fetched"] == 0
    assert report["ok"] == ["good.blf"]
    assert report["bad"] == ["drift.blf", "missing.blf"]


def test_check_rejects_incomplete_manifest(tmp_path, monkeypatch, capsys):
    _setup_manifest(tmp_path, monkeypatch, ("good.blf",))
    (fixtures.DEST / "good.blf").write_bytes(b"locked:good.blf")
    monkeypatch.setattr(
        fixtures, "_RAW_SPECS",
        fixtures._RAW_SPECS + (("missing.blf", "Demo/source", "missing.blf", "MIT", "test"),),
    )

    assert fixtures.main(["--check"]) != 0
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["requested"] == 2
    assert report["ok"] == ["good.blf"]
    assert report["bad"] == ["missing.blf"]


def test_check_rejects_malformed_entries(tmp_path, monkeypatch, capsys):
    payload = _setup_manifest(tmp_path, monkeypatch, ("good.blf",))
    (fixtures.DEST / "good.blf").write_bytes(b"locked:good.blf")
    payload["files"]["broken.blf"] = None
    payload["files"]["empty-hash.blf"] = {"sha256": ""}
    fixtures.MANIFEST_PATH.write_text(json.dumps(payload), encoding="utf-8")

    assert fixtures.main(["--check"]) != 0
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["ok"] == ["good.blf"]
    assert report["bad"] == ["broken.blf", "empty-hash.blf"]


def test_check_rejects_non_mapping_manifest(tmp_path, monkeypatch, capsys):
    _setup_manifest(tmp_path, monkeypatch, ("good.blf",))
    fixtures.MANIFEST_PATH.write_text(
        json.dumps({"files": ["broken"], "licenses": {}}), encoding="utf-8"
    )

    assert fixtures.main(["--check"]) != 0
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["requested"] == 1
    assert report["ok"] == []
    assert report["bad"] == ["good.blf"]


@pytest.mark.parametrize("payload", [[], None])
def test_check_rejects_non_mapping_top_level(
    tmp_path, monkeypatch, capsys, payload
):
    _setup_manifest(tmp_path, monkeypatch, ("good.blf",))
    fixtures.MANIFEST_PATH.write_text(json.dumps(payload), encoding="utf-8")

    assert fixtures.main(["--check"]) != 0
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["requested"] == 1
    assert report["ok"] == []
    assert report["bad"] == ["good.blf"]


def test_check_success_is_read_only(tmp_path, monkeypatch, capsys):
    _setup_manifest(tmp_path, monkeypatch, ("good.blf",))
    (fixtures.DEST / "good.blf").write_bytes(b"locked:good.blf")
    before = _snapshot(fixtures.DEST)

    def forbidden(*args, **kwargs):
        raise AssertionError("--check must not access the network")

    monkeypatch.setattr(fixtures, "_fetch", forbidden)
    monkeypatch.setattr(fixtures, "fetch_licenses", forbidden)

    assert fixtures.main(["--check"]) == 0
    assert _snapshot(fixtures.DEST) == before
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["ok"] == ["good.blf"]
    assert report["bad"] == []


def test_check_without_manifest_fails_without_creating_files(tmp_path, monkeypatch):
    dest = tmp_path / "absent"
    monkeypatch.setattr(fixtures, "DEST", dest)
    monkeypatch.setattr(fixtures, "MANIFEST_PATH", dest / "manifest.json")
    monkeypatch.setattr(fixtures, "LICENSE_DIR", dest / "licenses")
    monkeypatch.setattr(fixtures, "_RAW_SPECS", (("missing.blf", "Demo/source", "missing.blf", "MIT", "test"),))

    def forbidden(*args, **kwargs):
        raise AssertionError("--check must not download or write")

    monkeypatch.setattr(fixtures, "_fetch", forbidden)
    monkeypatch.setattr(fixtures, "build", forbidden)
    monkeypatch.setattr(fixtures, "fetch_licenses", forbidden)

    assert fixtures.main(["--check"]) != 0
    assert not dest.exists()


def test_default_rejects_changed_locked_content(tmp_path, monkeypatch):
    payload = _setup_manifest(tmp_path, monkeypatch, ("a.blf",))
    local = fixtures.DEST / "a.blf"
    local.write_bytes(b"locked:a.blf")
    before = fixtures.MANIFEST_PATH.read_bytes()
    calls: list[str] = []

    def changed(url, **kwargs):
        calls.append(url)
        return b"changed"

    def forbidden_candidates(*args, **kwargs):
        raise AssertionError("default fetch must use the locked URL")

    monkeypatch.setattr(fixtures, "_fetch", changed)
    monkeypatch.setattr(fixtures, "_fetch_first", forbidden_candidates)

    result = fixtures.main(["--skip-licenses"])

    assert result != 0
    assert calls == ["https://example.invalid/a.blf"]
    assert fixtures.MANIFEST_PATH.read_bytes() == before
    assert json.loads(before) == payload
    assert local.read_bytes() == b"locked:a.blf"


def test_default_partial_failure_keeps_manifest_and_fails(tmp_path, monkeypatch):
    _setup_manifest(tmp_path, monkeypatch, ("good.blf", "bad.blf"))
    before = fixtures.MANIFEST_PATH.read_bytes()
    calls: list[str] = []

    def fetch(url, **kwargs):
        calls.append(url)
        if url.endswith("bad.blf"):
            raise OSError("offline")
        return b"locked:good.blf"

    def forbidden_candidates(*args, **kwargs):
        raise AssertionError("default fetch must use locked URLs")

    monkeypatch.setattr(fixtures, "_fetch", fetch)
    monkeypatch.setattr(fixtures, "_fetch_first", forbidden_candidates)

    result = fixtures.main(["--skip-licenses"])

    assert result != 0
    assert sorted(calls) == [
        "https://example.invalid/bad.blf",
        "https://example.invalid/good.blf",
    ]
    assert fixtures.MANIFEST_PATH.read_bytes() == before
    assert (fixtures.DEST / "good.blf").read_bytes() == b"locked:good.blf"
    assert not (fixtures.DEST / "bad.blf").exists()


def test_default_success_keeps_locked_manifest(tmp_path, monkeypatch):
    _setup_manifest(tmp_path, monkeypatch, ("a.blf",))
    before = fixtures.MANIFEST_PATH.read_bytes()
    calls: list[str] = []

    def fetch(url, **kwargs):
        calls.append(url)
        return b"locked:a.blf"

    def forbidden_candidates(*args, **kwargs):
        raise AssertionError("default fetch must use the locked URL")

    monkeypatch.setattr(fixtures, "_fetch", fetch)
    monkeypatch.setattr(fixtures, "_fetch_first", forbidden_candidates)

    assert fixtures.main(["--skip-licenses"]) == 0
    assert calls == ["https://example.invalid/a.blf"]
    assert fixtures.MANIFEST_PATH.read_bytes() == before
    assert (fixtures.DEST / "a.blf").read_bytes() == b"locked:a.blf"


def test_default_without_manifest_requires_explicit_lock(tmp_path, monkeypatch):
    dest = tmp_path / "bus_data"
    dest.mkdir()
    monkeypatch.setattr(fixtures, "DEST", dest)
    monkeypatch.setattr(fixtures, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fixtures, "MANIFEST_PATH", dest / "manifest.json")
    monkeypatch.setattr(fixtures, "LICENSE_DIR", dest / "licenses")
    monkeypatch.setattr(
        fixtures, "_RAW_SPECS",
        (("a.blf", "Demo/source", "a.blf", "MIT", "test"),),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("missing lock must fail before network or writes")

    monkeypatch.setattr(fixtures, "fetch_licenses", forbidden)
    monkeypatch.setattr(fixtures, "build", forbidden)

    assert fixtures.main([]) != 0
    assert not fixtures.MANIFEST_PATH.exists()


def test_explicit_lock_partial_failure_keeps_manifest(tmp_path, monkeypatch):
    _setup_manifest(tmp_path, monkeypatch, ("good.blf", "bad.blf"))
    before = fixtures.MANIFEST_PATH.read_bytes()
    monkeypatch.setattr(
        fixtures, "_fetch_first",
        lambda urls, **kwargs: (
            None if urls[0].endswith("bad.blf")
            else (urls[0], b"locked:good.blf")
        ),
    )

    assert fixtures.main(["--lock", "--skip-licenses"]) != 0
    assert fixtures.MANIFEST_PATH.read_bytes() == before


def test_explicit_lock_can_refresh_manifest(tmp_path, monkeypatch):
    _setup_manifest(tmp_path, monkeypatch, ("a.blf",))
    monkeypatch.setattr(
        fixtures, "_fetch_first",
        lambda *args, **kwargs: ("https://example.invalid/new/a.blf", b"new-content"),
    )

    assert fixtures.main(["--lock", "--skip-licenses"]) == 0
    entry = json.loads(fixtures.MANIFEST_PATH.read_text())["files"]["a.blf"]
    assert entry["url"] == "https://example.invalid/new/a.blf"
    assert entry["sha256"] == fixtures._sha256(b"new-content")
