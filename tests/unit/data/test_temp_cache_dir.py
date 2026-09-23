"""TempCacheDir（临时目录创建/清扫/atexit/回退链）单元测试（P2）。

D17 三重保险逐条覆盖 + 回退路径两种触发条件各有单测。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import src.data.temp_cache_dir as tcd
from src.data.temp_cache_dir import TempCacheDir


def _find_dead_pid() -> int:
    """找一个当前不在进程表里的 PID（探测式，找不到则 skip）。"""
    for pid in range(30_000, 90_000):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except PermissionError:
            continue
    pytest.skip("未找到空闲 PID")


def _dead_child_pid() -> int:
    """起一个立即退出的子进程，返回其（已死亡）PID。"""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.fixture()
def fake_base(tmp_path, monkeypatch):
    """把 base_dir 指到 tmp_path，隔离真实缓存目录。"""
    base = tmp_path / "lazy_tmp"
    monkeypatch.setattr(TempCacheDir, "base_dir", staticmethod(lambda: base))
    base.mkdir(parents=True, exist_ok=True)
    return base


class TestCreateAndCleanup:
    def test_create_makes_unique_dirs_per_instance(self, fake_base):
        d1 = TempCacheDir.create()
        d2 = TempCacheDir.create()
        assert Path(d1.path()).is_dir()
        assert Path(d2.path()).is_dir()
        assert d1.path() != d2.path()
        assert re.fullmatch(r"\d+_[0-9a-f]{8,}", Path(d1.path()).name)

    def test_cleanup_removes_dir_and_is_idempotent(self, fake_base):
        d = TempCacheDir.create()
        p = Path(d.path())
        (p / "data.parquet").write_bytes(b"x")
        d.cleanup()
        assert not p.exists()
        d.cleanup()  # 幂等，不抛
        assert not p.exists()


class TestSweepGuards:
    """三重保险逐条。"""

    def test_guard2_name_pattern_skips_non_matching(self, fake_base):
        # 不匹配 ^\d+_[0-9a-f]{8,}$ 的目录/文件一律保留
        dead = _find_dead_pid()
        keep_dirs = ["notes", "123_zzzzzzzz", f"{dead}_ZZZZZZZZ", "12_abcd123"]
        for name in keep_dirs:
            (fake_base / name).mkdir()
        (fake_base / "readme.txt").write_text("x")  # 文件也不动
        removed = TempCacheDir.sweep_stale()
        assert removed == 0
        for name in keep_dirs:
            assert (fake_base / name).exists()
        assert (fake_base / "readme.txt").exists()

    def test_guard3_dead_pid_removed(self, fake_base):
        dead = _dead_child_pid()
        stale = fake_base / f"{dead}_{os.urandom(4).hex()}"
        stale.mkdir()
        removed = TempCacheDir.sweep_stale()
        assert removed == 1
        assert not stale.exists()

    def test_guard3_live_pid_kept(self, fake_base):
        live = fake_base / f"{os.getpid()}_{os.urandom(4).hex()}"
        live.mkdir()
        removed = TempCacheDir.sweep_stale()
        assert removed == 0
        assert live.exists()

    def test_guard3_stale_mtime_removed_even_if_pid_alive(self, fake_base):
        # PID 复用防线：目录 PID 活着但 mtime 超 72h → 删
        live = fake_base / f"{os.getpid()}_{os.urandom(4).hex()}"
        live.mkdir()
        old = time.time() - 73 * 3600
        os.utime(live, (old, old))
        assert TempCacheDir.sweep_stale() == 1
        assert not live.exists()

    def test_guard3_live_pid_25h_mtime_kept(self, fake_base):
        # 72h 阈值：活 PID + 挂机一天（含隔夜）→ 保留，不误删
        live = fake_base / f"{os.getpid()}_{os.urandom(4).hex()}"
        live.mkdir()
        old = time.time() - 25 * 3600
        os.utime(live, (old, old))
        assert TempCacheDir.sweep_stale() == 0
        assert live.exists()

    def test_guard3_fresh_live_pid_dir_with_recent_mtime_kept(self, fake_base):
        # 活 PID + 新 mtime：双条件都不满足删除 → 保留
        live = fake_base / f"{os.getpid()}_{os.urandom(4).hex()}"
        live.mkdir()
        assert TempCacheDir.sweep_stale() == 0
        assert live.exists()

    def test_guard1_only_scans_top_level(self, fake_base):
        # sweep 只扫 base 这一层：嵌套在普通目录里的「合法名」目录不会被碰
        dead = _find_dead_pid()
        nested_parent = fake_base / "other_app"
        nested = nested_parent / f"{dead}_{os.urandom(4).hex()}"
        nested.mkdir(parents=True)
        assert TempCacheDir.sweep_stale() == 0
        assert nested.exists()

    def test_sweep_on_missing_base_returns_zero(self, tmp_path):
        assert TempCacheDir.sweep_stale(base=tmp_path / "nope") == 0


class TestHeartbeat:
    """touch 心跳：让 72h 陈旧判据只对真残留生效。"""

    def test_touch_refreshes_mtime(self, fake_base):
        d = TempCacheDir.create()
        p = Path(d.path())
        old = time.time() - 25 * 3600
        os.utime(p, (old, old))
        d.touch()
        assert tcd._mtime_age_s(p) < 60

    def test_touch_cleaned_instance_is_noop(self, fake_base):
        d = TempCacheDir.create()
        d.cleanup()
        d.touch()  # 不抛

    def test_touch_missing_dir_swallows_oserror(self, fake_base):
        d = TempCacheDir.create()
        shutil.rmtree(d.path())
        d.touch()  # 目录被外部删除，心跳失败无妨

    def test_touch_all_live_refreshes_registry(self, fake_base):
        d1 = TempCacheDir.create()
        d2 = TempCacheDir.create()
        p1, p2 = Path(d1.path()), Path(d2.path())
        old = time.time() - 25 * 3600
        os.utime(p1, (old, old))
        os.utime(p2, (old, old))
        TempCacheDir.touch_all_live()
        assert tcd._mtime_age_s(p1) < 60
        assert tcd._mtime_age_s(p2) < 60

    def test_heartbeat_keeps_sweep_from_deleting(self, fake_base):
        # 端到端口径：活 PID + 心跳刷新 mtime → sweep 保留
        d = TempCacheDir.create()
        old = time.time() - 25 * 3600
        os.utime(d.path(), (old, old))
        TempCacheDir.touch_all_live()
        assert TempCacheDir.sweep_stale() == 0


class TestFallbackChain:
    def test_fallback_when_location_empty(self, tmp_path, monkeypatch):
        # 触发条件一：writableLocation 返回空串
        monkeypatch.setattr(tcd, "_cache_location", lambda: "")
        monkeypatch.setattr(tcd.tempfile, "gettempdir", lambda: str(tmp_path))
        base = TempCacheDir.base_dir()
        assert base == tmp_path / "CSVPlot_lazy"
        assert base.is_dir()

    def test_fallback_when_mkdir_fails(self, tmp_path, monkeypatch):
        # 触发条件二：主路径 mkdir 抛 PermissionError（真实走 _try_mkdir
        # 的捕获分支，而非绕过它）
        monkeypatch.setattr(tcd, "_cache_location", lambda: str(tmp_path / "qtcache"))
        real_mkdir = Path.mkdir

        def _fail_lazy_tmp(self, *args, **kwargs):
            if "lazy_tmp" in str(self):
                raise PermissionError("read-only")
            return real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", _fail_lazy_tmp)
        monkeypatch.setattr(tcd.tempfile, "gettempdir", lambda: str(tmp_path))
        base = TempCacheDir.base_dir()
        assert base == tmp_path / "CSVPlot_lazy"
        assert base.is_dir()

    def test_primary_used_when_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tcd, "_cache_location", lambda: str(tmp_path))
        base = TempCacheDir.base_dir()
        assert base == tmp_path / "CSVPlot" / "lazy_tmp"
        assert base.is_dir()


class TestPidAlive:
    """_pid_alive 探测语义。

    现场事故：os.kill(pid, 0) 在 Windows Python 3.12 是 TerminateProcess，
    分身窗口启动清扫时会把前一个 app 进程杀掉（或探测报错导致误删其
    临时目录）。win32 分支改 GetExitCodeProcess 后，这些用例用假
    kernel32 钉死五条路径（macOS 无法跑真实 Windows 语义）。
    """

    def test_dispatch_to_win32_probe(self, monkeypatch):
        calls = []
        monkeypatch.setattr(tcd.sys, "platform", "win32")
        monkeypatch.setattr(tcd, "_win32_pid_alive", lambda pid: calls.append(pid) or True)
        assert tcd._pid_alive(1234) is True
        assert calls == [1234]

    def test_posix_branch_unchanged(self, monkeypatch):
        # 非 win32 仍走 os.kill 信号 0（本用例进程自身必然存活）
        monkeypatch.setattr(tcd.sys, "platform", "darwin")
        assert tcd._pid_alive(os.getpid()) is True
        dead = _dead_child_pid()
        assert tcd._pid_alive(dead) is False

    def test_win32_alive_still_active(self, monkeypatch):
        fake = _FakeKernel32(open_result=4242, exit_code=259)
        monkeypatch.setattr(tcd, "_win32_api", lambda: fake)
        assert tcd._win32_pid_alive(4242) is True
        assert fake.closed == 1

    def test_win32_dead_exit_code_zero(self, monkeypatch):
        fake = _FakeKernel32(open_result=4242, exit_code=0)
        monkeypatch.setattr(tcd, "_win32_api", lambda: fake)
        assert tcd._win32_pid_alive(4242) is False
        assert fake.closed == 1

    def test_win32_access_denied_treated_alive(self, monkeypatch):
        # ERROR_ACCESS_DENIED(5)：进程存在但不可查询 → 判活（宁可漏删）
        fake = _FakeKernel32(open_result=0)
        monkeypatch.setattr(tcd, "_win32_api", lambda: fake)
        import ctypes

        # get_last_error 仅 Windows 构建存在，macOS 上注入
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
        assert tcd._win32_pid_alive(4242) is True

    def test_win32_no_such_process_treated_dead(self, monkeypatch):
        # ERROR_INVALID_PARAMETER(87)：进程不存在 → 判死
        fake = _FakeKernel32(open_result=0)
        monkeypatch.setattr(tcd, "_win32_api", lambda: fake)
        import ctypes

        monkeypatch.setattr(ctypes, "get_last_error", lambda: 87, raising=False)
        assert tcd._win32_pid_alive(4242) is False

    def test_win32_get_exit_code_fails_conservative_alive(self, monkeypatch):
        fake = _FakeKernel32(open_result=4242, get_exit_ok=0)
        monkeypatch.setattr(tcd, "_win32_api", lambda: fake)
        assert tcd._win32_pid_alive(4242) is True
        assert fake.closed == 1

    def test_win32_queries_with_limited_information(self, monkeypatch):
        # 只允许查询限权访问（0x1000），任何情况下不得请求终止权限
        fake = _FakeKernel32(open_result=4242, exit_code=259)
        monkeypatch.setattr(tcd, "_win32_api", lambda: fake)
        tcd._win32_pid_alive(4242)
        assert fake.opened[0] == 0x1000


class _FakeKernel32:
    """假 kernel32：只实现探测用到的三个 API（含句柄关闭计数）。"""

    def __init__(self, open_result=0, exit_code=259, get_exit_ok=1):
        self.open_result = open_result
        self.exit_code = exit_code
        self.get_exit_ok = get_exit_ok
        self.closed = 0
        self.opened = None

    def OpenProcess(self, access, inherit, pid):
        self.opened = (access, inherit, pid)
        return self.open_result

    def GetExitCodeProcess(self, handle, code_ref):
        if self.get_exit_ok:
            code_ref._obj.value = self.exit_code
        return self.get_exit_ok

    def CloseHandle(self, handle):
        self.closed += 1
        return 1


class TestAtexit:
    def test_atexit_cleanup_removes_live_instances(self, fake_base, monkeypatch):
        TempCacheDir.register_atexit()
        d = TempCacheDir.create()  # 持引用防止 WeakSet 提前回收
        p = Path(d.path())
        assert p.is_dir()
        TempCacheDir._atexit_cleanup()
        assert not p.exists()

    def test_register_atexit_idempotent(self, monkeypatch):
        # 全局 flag 已被其他用例置位的场景下再调一次也不重复注册
        import src.data.temp_cache_dir as mod

        monkeypatch.setattr(mod, "_atexit_registered", True)
        TempCacheDir.register_atexit()  # 不应抛异常或重复注册

    def test_register_atexit_registers_once(self, monkeypatch):
        import src.data.temp_cache_dir as mod

        calls: list[str] = []

        def fake_register(func, *a, **kw):
            calls.append("registered")

        monkeypatch.setattr(mod.atexit, "register", fake_register)
        monkeypatch.setattr(mod, "_atexit_registered", False)
        TempCacheDir.register_atexit()
        TempCacheDir.register_atexit()
        assert calls == ["registered"]
        monkeypatch.setattr(mod, "_atexit_registered", True, raising=False)


if __name__ == "__main__":
    pytest.main([__file__])
