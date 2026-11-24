from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("script_name", "module"),
    [
        ("build_exe_nuitka", "nuitka"),
        ("build_exe_pyinstaller", "PyInstaller"),
    ],
)
def test_posix_wrapper_uses_selected_python_module(tmp_path, script_name, module):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "src").mkdir()
    shutil.copy2(ROOT / "scripts" / script_name, scripts / script_name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv.log"
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$UV_LOG"\n')
    uv.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", UV_LOG=str(log))

    result = subprocess.run(
        ["bash", str(scripts / script_name)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert f"run python -m {module} " in " ".join(calls)


def test_windows_wrapper_uses_selected_python_module():
    script = (ROOT / "scripts" / "build_exe_pyinstaller.bat").read_text()
    assert "!PYTHON! -m PyInstaller --noconfirm --clean" in script
    assert "\npyinstaller --noconfirm" not in script
