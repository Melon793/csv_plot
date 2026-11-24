from __future__ import annotations

import json
from pathlib import Path

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
    monkeypatch.setattr(fixtures, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(fixtures, "LICENSE_DIR", dest / "licenses")
    monkeypatch.setattr(fixtures, "_RAW_SPECS", specs)
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
