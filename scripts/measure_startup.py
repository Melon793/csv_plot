#!/usr/bin/env python3
"""Measure packaged CSV Plot startup on Windows without third-party packages.

The monitor is deliberately external to the app under test.  It records process
creation -> first splash, splash -> main window, graceful shutdown, and splash
pixel-change cadence using GDI screen capture.  Internal stage telemetry is
picked up when the optimized build writes CSV_PLOT_STARTUP_METRICS.

Example:
    python scripts/measure_startup.py dist/csv_plot/csv_plot.exe --runs 6 \
        --splash-delay 3 --output tmp/startup-optimized.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zlib
from ctypes import wintypes
from pathlib import Path


if sys.platform != "win32":
    raise SystemExit("This benchmark currently supports Windows only.")

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE_V2
except (AttributeError, OSError):
    pass

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
_user32.EnumWindows.restype = wintypes.BOOL
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.IsWindowVisible.restype = wintypes.BOOL
_user32.IsWindowEnabled.argtypes = [wintypes.HWND]
_user32.IsWindowEnabled.restype = wintypes.BOOL
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowTextW.restype = ctypes.c_int
_user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
_user32.GetWindowRect.restype = wintypes.BOOL
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.PostMessageW.restype = wintypes.BOOL

_user32.GetDC.argtypes = [wintypes.HWND]
_user32.GetDC.restype = wintypes.HDC
_user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
_user32.ReleaseDC.restype = ctypes.c_int
_gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
_gdi32.CreateCompatibleDC.restype = wintypes.HDC
_gdi32.DeleteDC.argtypes = [wintypes.HDC]
_gdi32.DeleteDC.restype = wintypes.BOOL
_gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
_gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
_gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_gdi32.DeleteObject.restype = wintypes.BOOL
_gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
_gdi32.SelectObject.restype = wintypes.HGDIOBJ
_gdi32.BitBlt.argtypes = [
    wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
]
_gdi32.BitBlt.restype = wintypes.BOOL


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


_gdi32.GetDIBits.argtypes = [
    wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
    ctypes.c_void_p, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
]
_gdi32.GetDIBits.restype = ctypes.c_int

_WM_CLOSE = 0x0010
_SRCCOPY = 0x00CC0020
_DIB_RGB_COLORS = 0
_BI_RGB = 0


def visible_windows() -> list[tuple[int, str, tuple[int, int, int, int], int]]:
    found: list[tuple[int, str, tuple[int, int, int, int], int]] = []

    def callback(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextW(hwnd, None, 0)
        if length <= 0:
            title = ""
        else:
            buf = ctypes.create_unicode_buffer(length + 1)
            _user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
        rect = wintypes.RECT()
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            found.append((int(hwnd), title, (rect.left, rect.top, rect.right, rect.bottom), int(pid.value)))
        return True

    _user32.EnumWindows(_EnumWindowsProc(callback), 0)
    return found


def find_main_window(pid: int) -> tuple[int, str] | None:
    candidates: list[tuple[int, str, int]] = []
    for hwnd, title, rect, window_pid in visible_windows():
        if window_pid != pid or not _user32.IsWindowEnabled(hwnd):
            continue
        width, height = rect[2] - rect[0], rect[3] - rect[1]
        if title.startswith("CSV Plot") or (width >= 800 and height >= 500):
            candidates.append((width * height, hwnd, title or "CSV Plot (untitled top-level window)"))
    if not candidates:
        return None
    _area, hwnd, title = max(candidates)
    return hwnd, title


def find_splash_window(pid: int) -> tuple[int, tuple[int, int, int, int]] | None:
    for hwnd, title, rect, window_pid in visible_windows():
        if window_pid != pid or title:
            continue
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        # Allow DPI scaling while rejecting ordinary hidden/zero-size widgets.
        if 350 <= width <= 800 and 180 <= height <= 500:
            ratio = width / max(1, height)
            if abs(ratio - (500 / 280)) < 0.08:
                return hwnd, rect
    return None


class SplashCapture(threading.Thread):
    """Capture the splash rectangle and retain timestamps of pixel changes."""

    def __init__(self, hwnd: int, rect: tuple[int, int, int, int], interval_s: float = 0.005):
        super().__init__(daemon=True)
        self.hwnd = hwnd
        self.rect = rect
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.started = threading.Event()
        self.capture_times: list[float] = []
        self.changed_times: list[float] = []
        self.paint_estimate_ms: list[float] = []
        self.error: str | None = None

    def run(self) -> None:
        left, top, right, bottom = self.rect
        width, height = right - left, bottom - top
        screen_dc = mem_dc = None
        bitmap = old_bitmap = None
        buffer = None
        try:
            screen_dc = _user32.GetDC(0)
            mem_dc = _gdi32.CreateCompatibleDC(screen_dc)
            bitmap = _gdi32.CreateCompatibleBitmap(screen_dc, width, height)
            old_bitmap = _gdi32.SelectObject(mem_dc, bitmap)
            info = BITMAPINFO()
            info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            info.bmiHeader.biWidth = width
            info.bmiHeader.biHeight = -height  # top-down DIB
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            info.bmiHeader.biCompression = _BI_RGB
            info.bmiHeader.biSizeImage = width * height * 4
            buffer = ctypes.create_string_buffer(info.bmiHeader.biSizeImage)
            previous_crc = None
            self.started.set()
            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                if not _user32.IsWindowVisible(self.hwnd):
                    break
                if not _gdi32.BitBlt(
                    mem_dc, 0, 0, width, height, screen_dc, left, top, _SRCCOPY
                ):
                    self.error = "BitBlt failed"
                    break
                copied = _gdi32.GetDIBits(
                    mem_dc, bitmap, 0, height, buffer, ctypes.byref(info), _DIB_RGB_COLORS
                )
                if copied != height:
                    self.error = f"GetDIBits copied {copied}/{height} rows"
                    break
                now = time.perf_counter()
                crc = zlib.crc32(buffer.raw)
                if previous_crc is None or crc != previous_crc:
                    self.changed_times.append(now)
                previous_crc = crc
                self.capture_times.append(now)
                self.paint_estimate_ms.append((now - t0) * 1000)
                delay = self.interval_s - (time.perf_counter() - t0)
                if delay > 0:
                    self.stop_event.wait(delay)
        except Exception as exc:  # diagnostic tool must not hide app failures
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.started.set()
            if old_bitmap is not None and mem_dc is not None:
                _gdi32.SelectObject(mem_dc, old_bitmap)
            if bitmap is not None:
                _gdi32.DeleteObject(bitmap)
            if mem_dc is not None:
                _gdi32.DeleteDC(mem_dc)
            if screen_dc is not None:
                _user32.ReleaseDC(0, screen_dc)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.95)))
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _deltas(values: list[float]) -> list[float]:
    return [(b - a) * 1000 for a, b in zip(values, values[1:]) if b >= a]


def _read_metrics(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def measure_once(
    exe: Path,
    run_index: int,
    root: Path,
    config_dir: Path,
    splash_delay: int | None,
    timeout_s: float,
    capture_interval_ms: float,
    no_splash: bool = False,
) -> dict:
    run_dir = root / f"run-{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "startup-metrics.json"
    env = os.environ.copy()
    env["CSV_PLOT_CONFIG_DIR"] = str(config_dir)
    env["CSV_PLOT_STARTUP_METRICS"] = str(metrics_path)
    env["QT_QPA_PLATFORM"] = "windows"

    args = [str(exe)]
    if splash_delay is not None:
        args.append(f"--splash-delay={splash_delay}")
    if no_splash:
        args.append("--no-splash")
    launched = time.perf_counter()
    process = subprocess.Popen(
        args,
        cwd=str(run_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    splash_time = main_time = None
    splash_hwnd = None
    capture: SplashCapture | None = None
    main_hwnd = None
    result: dict = {
        "run": run_index,
        "config_mode": "warm" if run_index else "cold",
        "pid": process.pid,
        "args": args,
        "config_dir": str(config_dir),
        "returncode": None,
    }

    try:
        while time.perf_counter() - launched < timeout_s:
            if process.poll() is not None:
                result["error"] = f"process exited early with {process.returncode}"
                break
            now = time.perf_counter()
            if splash_time is None:
                found = find_splash_window(process.pid)
                if found is not None:
                    splash_hwnd, rect = found
                    splash_time = now
                    capture = SplashCapture(splash_hwnd, rect, capture_interval_ms / 1000)
                    capture.start()
                    capture.started.wait(1.0)
            main = find_main_window(process.pid)
            if main is not None:
                main_hwnd, main_title = main
                main_time = now
                break
            time.sleep(0.002)

        if capture is not None:
            capture.stop_event.set()
            capture.join(timeout=2.0)

        result.update({
            "process_to_splash_ms": None if splash_time is None else (splash_time - launched) * 1000,
            "splash_to_main_ms": None if splash_time is None or main_time is None else (main_time - splash_time) * 1000,
            "process_to_main_ms": None if main_time is None else (main_time - launched) * 1000,
            "main_title": main_title if main_time is not None else None,
            "main_hwnd": main_hwnd,
        })

        if capture is not None:
            result["animation"] = {
                "capture_interval_ms": _stats(capture.paint_estimate_ms),
                "capture_loop_gap_ms": _stats(_deltas(capture.capture_times)),
                "changed_frame_gap_ms": _stats(_deltas(capture.changed_times)),
                "changed_frames": len(capture.changed_times),
                "captured_frames": len(capture.capture_times),
                "capture_error": capture.error,
                "note": f"External GDI sampling every {capture_interval_ms:g} ms; changed-frame gaps include monitor and capture overhead.",
            }
        if main_hwnd is not None:
            _user32.PostMessageW(main_hwnd, _WM_CLOSE, 0, 0)
        try:
            result["returncode"] = process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            result["close_error"] = "graceful close timed out after 10s"
            process.terminate()
            try:
                result["returncode"] = process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                result["returncode"] = process.wait(timeout=3.0)
        # Telemetry is flushed by app shutdown (atexit / event-loop finally).
        result["internal_metrics"] = _read_metrics(metrics_path)
    finally:
        if capture is not None and capture.is_alive():
            capture.stop_event.set()
            capture.join(timeout=1.0)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3.0)
    return result


def summarize(runs: list[dict]) -> dict:
    def metric_stats(name: str) -> dict:
        return _stats([r[name] for r in runs if isinstance(r.get(name), (int, float))])

    animation_gaps = [
        gap
        for r in runs
        for gap in (
            [x for x in [r.get("animation", {}).get("changed_frame_gap_ms", {}).get("max")] if isinstance(x, (int, float))]
        )
    ]
    return {
        "runs": len(runs),
        "process_to_splash_ms": metric_stats("process_to_splash_ms"),
        "splash_to_main_ms": metric_stats("splash_to_main_ms"),
        "process_to_main_ms": metric_stats("process_to_main_ms"),
        "max_changed_frame_gap_ms": _stats(animation_gaps),
        "successful_runs": sum(1 for r in runs if r.get("main_title") and r.get("returncode") == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument(
        "--splash-delay",
        type=int,
        default=None,
        help="Force minimum splash seconds; omit to use the application's default 800 ms.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--capture-interval-ms", type=float, default=5.0)
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="Launch with --no-splash for a functional smoke test.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exe = args.exe.resolve()
    if not exe.is_file():
        parser.error(f"executable not found: {exe}")
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.no_splash and args.splash_delay is not None:
        parser.error("--no-splash cannot be combined with --splash-delay")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="csv-plot-startup-", dir=str(args.output.parent)))
    config_dir = root / "config"
    config_dir.mkdir()
    runs = [
        measure_once(
            exe,
            index,
            root,
            config_dir,
            args.splash_delay,
            args.timeout,
            args.capture_interval_ms,
            args.no_splash,
        )
        for index in range(args.runs)
    ]
    report = {
        "tool": "scripts/measure_startup.py",
        "executable": str(exe),
        "python": sys.version,
        "platform": sys.platform,
        "splash_delay_seconds": args.splash_delay,
        "no_splash": args.no_splash,
        "runs": runs,
        "summary": summarize(runs),
        "artifacts_dir": str(root),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Full report: {args.output}")
    return 0 if report["summary"]["successful_runs"] == args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
