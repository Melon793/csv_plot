"""``tests/fixtures/bus_data/`` 的真实第三方夹具访问层。

这里刻意**不硬编码任何哈希**：期望值全在 ``bus_data/manifest.json`` 里，由
``scripts/fetch_blf_fixtures.py --lock`` 生成。于是「换一份上游文件」这件事
只需要重跑脚本，不需要改测试代码；而测试代码负责的是「本地这份是不是 manifest
里锁定的那一份」。

设计取舍：

* 读不到文件时抛 :class:`BusDataMissing`，调用方一律 ``pytest.skip`` 而不是
  ``fail`` —— 这些夹具靠网络下载，CI 无外网是**预期**场景，不能把整套测试
  拖红（合成夹具才是主路径）。
* 本地文件已存在但 sha256 与 manifest 不符时抛 :class:`BusDataCorrupted`，
  调用方必须 ``pytest.fail``。真值被改动过还继续断言，得到的绿灯毫无意义。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BUS_DATA_DIR = Path(__file__).resolve().parent / "bus_data"
MANIFEST_PATH = BUS_DATA_DIR / "manifest.json"

# 单个夹具的最大可接受体积：这批样本按设计都 < 200 KB，真超了说明下载脚本
# 抓错了东西（比如把整个仓库归档拖下来），不该被当成夹具读进内存。
_MAX_BYTES = 2 * 1024 * 1024


class BusDataMissing(Exception):
    """夹具未下载。应当 skip，而不是 fail。"""


class BusDataCorrupted(Exception):
    """夹具与 manifest 锁定的 sha256 不符。必须 fail。"""


class _Cache:
    """manifest 的进程内缓存。

    做成显式类而不是模块级 dict + ``global``：单测可以直接读
    ``_CACHE.entries is None`` 来证明缓存命中，不必依赖 ``importlib.reload``
    去重置模块状态（而那个重置本身就是缓存没做对的信号）。
    """

    def __init__(self) -> None:
        self.entries: dict[str, dict] | None = None


_CACHE = _Cache()


def load_manifest(*, force_reload: bool = False) -> dict[str, dict]:
    """读取 ``manifest.json`` 的 ``files`` 段。

    ``force_reload=True`` 供单测绕过进程内缓存 —— 缓存本身是刻意的（二十几个
    用例各解析一次 13 KB JSON 纯属浪费），但它同时意味着「测试期间外部改写了
    manifest 却看不到」，所以留一个明确的口子而不是靠 reload 模块。
    """
    if _CACHE.entries is not None and not force_reload:
        return _CACHE.entries
    if not MANIFEST_PATH.is_file():
        raise BusDataMissing("manifest 不存在，请先跑 scripts/fetch_blf_fixtures.py")
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = {str(k): dict(v) for k, v in (payload.get("files") or {}).items()}
    # 键一律过一遍名字白名单：清单里被塞一个 ``../../etc/passwd`` 的话，
    # 后面的 BUS_DATA_DIR / name 会直接逃出夹具目录。
    for name in files:
        if Path(name).name != name or not name:
            raise BusDataCorrupted(f"manifest 含非法文件名: {name!r}")
    _CACHE.entries = files
    return files


def available() -> list[str]:
    """已下载且哈希正确的夹具名（供参数化用）。"""
    out = []
    try:
        files = load_manifest()
    except (BusDataMissing, BusDataCorrupted, OSError, ValueError):
        return out
    for name, entry in files.items():
        path = BUS_DATA_DIR / name
        if not path.is_file():
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        if digest == str(entry.get("sha256") or ""):
            out.append(name)
    return out


def path_for(name: str) -> Path:
    """返回夹具绝对路径，必要时下载/校验。

    Raises:
        BusDataMissing: 文件不存在且 manifest 也不在（调用方应 skip）。
        BusDataCorrupted: 文件存在但哈希与 manifest 不符（调用方应 fail）。
    """
    files = load_manifest()
    entry = files.get(name)
    if entry is None:
        raise BusDataMissing(f"{name} 不在 manifest 中")
    path = BUS_DATA_DIR / name
    want = str(entry.get("sha256") or "")
    if not path.is_file():
        raise BusDataMissing(f"{name} 未下载（跑 scripts/fetch_blf_fixtures.py）")
    data = path.read_bytes()
    if len(data) > _MAX_BYTES:
        raise BusDataCorrupted(f"{name} 体积 {len(data)} 超出夹具应有量级")
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise BusDataCorrupted(
            f"{name} sha256 不符：{got[:16]}… != manifest 的 {want[:16]}…"
        )
    return path


def require(name: str) -> Path:
    """``path_for`` 的宽容版：缺失时直接 ``pytest.skip``。

    哈希不符**不**降级成 skip —— 那等于把被篡改的样本当真值继续跑。
    """
    import pytest

    try:
        return path_for(name)
    except BusDataMissing as e:
        pytest.skip(str(e))
    except BusDataCorrupted as e:
        pytest.fail(str(e))
