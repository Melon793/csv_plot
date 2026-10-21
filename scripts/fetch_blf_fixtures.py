#!/usr/bin/env python3
"""下载 BLF / DBC / ARXML 第三方公开夹具，并按 sha256 校验。

为什么需要这个脚本：本项目 v1 的验收样本里有 4 类公开真实文件（见
``docs`` 中「BLF 文件特别说明」），它们不入库（许可各异、且 CLAUDE.md 明确
不许把第三方夹具放进运行时代码），CI 无网络时用 ``pytest.skip`` 降级。

用法::

    uv run python scripts/fetch_blf_fixtures.py            # 下载 + 校验
    uv run python scripts/fetch_blf_fixtures.py --lock     # 首次：把实测 sha256 写回 manifest
    uv run python scripts/fetch_blf_fixtures.py --check    # 只校验本地已有文件，不下载

manifest 落在 ``tests/fixtures/bus_data/manifest.json``，逐条记录
``url / sha256 / bytes / license / purpose``。``--check`` 与 ``fetch_report.json``
一起构成「本机已核对」的证据链：``tests/`` 里的真实夹具用例读的是这份 manifest，
文件缺失或哈希不符就 skip，而不是让整套测试红掉。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "tests" / "fixtures" / "bus_data"
MANIFEST_PATH = DEST / "manifest.json"
LICENSE_DIR = DEST / "licenses"

USER_AGENT = "CSV_Plot_PySide/fixture-fetcher (+https://github.com/CSV_Plot_PySide)"

# 分支名兜底：各仓库默认分支不统一（main / master / develop），逐个试
_BRANCHES = ("main", "master", "develop")

# (逻辑名, 仓库, 仓库内路径, 许可, 用途)。逻辑名即落盘文件名，带来源前缀，
# 避免不同仓库的同名文件互相覆盖，也让测试里的引用能一眼看出出处。
_RAW_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    # --- hardbyte/python-can：BLF 二进制层的每一种 CAN 对象类型 -------------
    ("pcan_test_CanMessage.blf", "hardbyte/python-can",
     "test/data/test_CanMessage.blf", "LGPL-3.0-or-later",
     "经典 CAN 报文基线样本"),
    ("pcan_test_CanMessage2.blf", "hardbyte/python-can",
     "test/data/test_CanMessage2.blf", "LGPL-3.0-or-later",
     "经典 CAN 报文（第二种产出工具）"),
    ("pcan_test_CanFdMessage.blf", "hardbyte/python-can",
     "test/data/test_CanFdMessage.blf", "LGPL-3.0-or-later",
     "CAN FD 报文（8 B 码率切换）"),
    ("pcan_test_CanFdMessage64.blf", "hardbyte/python-can",
     "test/data/test_CanFdMessage64.blf", "LGPL-3.0-or-later",
     "CAN FD 报文（64 B payload，决定桶 stride 上限）"),
    ("pcan_test_CanErrorFrameExt.blf", "hardbyte/python-can",
     "test/data/test_CanErrorFrameExt.blf", "LGPL-3.0-or-later",
     "扩展错误帧：扫描期必须跳过、不产生采样点"),
    ("pcan_issue_1905.blf", "hardbyte/python-can",
     "test/data/issue_1905.blf", "LGPL-3.0-or-later",
     "历史回归：valid_bytes 大于实际数据长度"),
    # --- mrafatpanah/blf-viewer：多总线 + 真实数据库的配对样例 --------------
    ("blfviewer_two-channel.blf", "mrafatpanah/blf-viewer",
     "src/test/fixtures/two-channel.blf", "MIT",
     "双通道 BLF：group = (bus 通道, CAN ID) 分桶的直接证据"),
    ("blfviewer_two-channel.dbc", "mrafatpanah/blf-viewer",
     "src/test/fixtures/two-channel.dbc", "MIT",
     "与上一行配对的真实 DBC：端到端「导入 → 选库 → 出信号」闭环"),
    # --- cantools/cantools：数据库兼容性矩阵 + 交叉验证真值 -----------------
    ("ct_multiplex.dbc", "cantools/cantools",
     "tests/files/dbc/multiplex.dbc", "MIT",
     "多路复用：先解 mux 选择器再按值掩码，非本 mux 值填 NaN"),
    ("ct_floating_point.dbc", "cantools/cantools",
     "tests/files/dbc/floating_point.dbc", "MIT",
     "IEEE float32/64 信号的 view 解码与大小端"),
    ("ct_choices.dbc", "cantools/cantools",
     "tests/files/dbc/choices.dbc", "MIT",
     "VAL_ 枚举表 → is_enum + enum_map"),
    ("ct_big_numbers.dbc", "cantools/cantools",
     "tests/files/dbc/big_numbers.dbc", "MIT",
     "> 32 bit 大整数位段的向量化边界"),
    ("ct_cp1252.dbc", "cantools/cantools",
     "tests/files/dbc/cp1252.dbc", "MIT",
     "cp1252 注释编码：单文件解析失败只 warning 不中断"),
    ("ct_bus_comment.dbc", "cantools/cantools",
     "tests/files/dbc/bus_comment.dbc", "MIT",
     "总线级注释（非报文注释），canmatrix 属性读取兜底"),
    ("ct_system-3.2.3.arxml", "cantools/cantools",
     "tests/files/arxml/system-3.2.3.arxml", "MIT",
     "AUTOSAR 3.2 ARXML 导入"),
    ("ct_system-4.2.arxml", "cantools/cantools",
     "tests/files/arxml/system-4.2.arxml", "MIT",
     "AUTOSAR 4.2 ARXML 导入（主路径）"),
    ("ct_ecu-extract-4.2.arxml", "cantools/cantools",
     "tests/files/arxml/ecu-extract-4.2.arxml", "MIT",
     "ECU extract：单对象而非 cluster 的返回形态"),
    ("ct_system-float-values.arxml", "cantools/cantools",
     "tests/files/arxml/system-float-values.arxml", "MIT",
     "ARXML 侧浮点 + 物理值上下限"),
    # --- zariiii9003/vblf：逐对象头/边界样本（作者即 asammdf 维护者）--------
    ("vblf_CAN_MESSAGE.lobj", "zariiii9003/vblf",
     "tests/data/CAN_MESSAGE.lobj", "MIT", "单对象：经典 CAN"),
    ("vblf_CAN_MESSAGE2.lobj", "zariiii9003/vblf",
     "tests/data/CAN_MESSAGE2.lobj", "MIT", "单对象：经典 CAN（第二种）"),
    ("vblf_CAN_FD_MESSAGE.lobj", "zariiii9003/vblf",
     "tests/data/CAN_FD_MESSAGE.lobj", "MIT", "单对象：CAN FD"),
    ("vblf_CAN_FD_MESSAGE_64.lobj", "zariiii9003/vblf",
     "tests/data/CAN_FD_MESSAGE_64.lobj", "MIT", "单对象：CAN FD 64 B"),
    ("vblf_CAN_ERROR_EXT.lobj", "zariiii9003/vblf",
     "tests/data/CAN_ERROR_EXT.lobj", "MIT", "单对象：扩展错误帧"),
    ("vblf_LOG_CONTAINER.lobj", "zariiii9003/vblf",
     "tests/data/LOG_CONTAINER.lobj", "MIT", "单对象：日志容器（v1 明确不解析）"),
    ("vblf_ENV_DOUBLE.lobj", "zariiii9003/vblf",
     "tests/data/ENV_DOUBLE.lobj", "MIT", "单对象：ENV 变量（v1 明确不解析）"),
    ("vblf_SYS_VARIABLE__DOUBLE.lobj", "zariiii9003/vblf",
     "tests/data/SYS_VARIABLE__DOUBLE.lobj", "MIT",
     "单对象：SYS 变量（v1 明确不解析）"),
    ("vblf_APP_TEXT.lobj", "zariiii9003/vblf",
     "tests/data/APP_TEXT.lobj", "MIT", "单对象：APP 文本（v1 明确不解析）"),
)

# 各仓库 LICENSE 文件；核对再分发条款后原样存档，一并作为打包 legal/ 的来源
_LICENSES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("python-can", "hardbyte/python-can", ("LICENSE.txt", "LICENSE")),
    ("blf-viewer", "mrafatpanah/blf-viewer", ("LICENSE", "LICENSE.md")),
    ("cantools", "cantools/cantools", ("LICENSE", "LICENSE.md")),
    ("vblf", "zariiii9003/vblf", ("LICENSE", "LICENSE.md")),
    # canmatrix 不在本仓库下载夹具（它的 tests/ 样本直接取 .venv 里已装好的
    # 那一份），但它的 LICENSE 仍须归档：它是 can_database.py 的唯一依赖。
    # 仓库在 ebroecker/canmatrix（BSD-3-Clause），旧名 codezoon/ 已 404。
    ("canmatrix", "ebroecker/canmatrix",
     ("LICENSE", "LICENSE.txt", "LICENSE.md")),
)

# 部分夹具不在默认分支上，允许按顺序试多个 ref（``--lock`` 会记录命中的 URL）
_EXTRA_REFS: dict[str, tuple[str, ...]] = {}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ssl_context(*, verify: bool) -> ssl.SSLContext:
    """给受限企业网络留一个显式逃生口，**默认绝不静默降级校验**。

    ``--insecure`` 只应在维护者本机、且随后立刻跑 ``--check``（不带该开关）
    重新校验时使用；CI 永不传该开关 —— 它没有交互能力，宁可 skip 夹具用例
    也不能拿被篡改的样本当真值。
    """
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _candidate_urls(repo: str, rel: str) -> list[str]:
    refs = list(_EXTRA_REFS.get(f"{repo}:{rel}", ())) + [
        f"https://raw.githubusercontent.com/{repo}/{b}/{rel}" for b in _BRANCHES
    ]
    return refs


def _fetch(url: str, *, timeout: float, verify_ssl: bool = True) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout,
                                context=_ssl_context(verify=verify_ssl)) as resp:
        return resp.read()


def _fetch_first(urls: list[str], *, timeout: float,
                 verify_ssl: bool = True) -> tuple[str, bytes] | None:
    """按顺序试多个 URL，返回第一个成功的 ``(url, 内容)``。

    **仅用于首次 ``--lock``**：仓库可能改默认分支，这里做的是 ref 兜底而非
    主机兜底，候选表只有 ``_candidate_urls`` 一个来源，不存在被重定向到任意
    主机的 SSRF 面。校验阶段（已有 manifest）不走本函数，只用锁定的 URL。
    """
    last_error: Exception | None = None
    for url in urls:
        try:
            return url, _fetch(url, timeout=timeout, verify_ssl=verify_ssl)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):  # 分支名不对，换下一个
                last_error = e
                continue
            last_error = e
            return None
        except Exception as e:  # 网络故障不重试其它分支，避免把超时放大 N 倍
            last_error = e
            break
    if last_error is not None:
        print(f"  ! {urls[0]} → {type(last_error).__name__}: {last_error}")
    return None


def _load_existing_manifest() -> dict:
    if MANIFEST_PATH.is_file():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"manifest 读取失败，按空表处理: {e}")
    return {}


def build(specs=_RAW_SPECS, *, timeout: float = 30.0,
          verify_ssl: bool = True) -> dict:
    """下载全部夹具并生成 manifest 条目。"""
    urls_map = {name: _candidate_urls(repo, rel) for name, repo, rel, _, _ in specs}
    meta_map = {name: (repo, rel, lic, purpose)
                for name, repo, rel, lic, purpose in specs}

    def download(entry: tuple[str, str, str, str, str]):
        name = entry[0]
        got = _fetch_first(urls_map[name], timeout=timeout, verify_ssl=verify_ssl)
        if got is None:
            return None
        url, data = got
        return name, data, url

    manifest: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(download, specs):
            if item is None:
                continue
            name, data, url = item
            repo, rel, lic, purpose = meta_map[name]
            manifest[name] = {
                "url": url,
                "repo": repo,
                "repo_path": rel,
                "license": lic,
                "purpose": purpose,
                "bytes": len(data),
                "sha256": _sha256(data),
            }
            (DEST / name).write_bytes(data)

    return manifest


def fetch_licenses(*, timeout: float = 30.0, verify_ssl: bool = True) -> dict:
    """存档各来源仓库的 LICENSE 全文，供许可核对与打包 legal/ 使用。"""
    LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for logical, repo, names in _LICENSES:
        for fname in names:
            got = _fetch_first(_candidate_urls(repo, fname), timeout=timeout,
                               verify_ssl=verify_ssl)
            if got is None:
                continue
            url, data = got
            dest = LICENSE_DIR / f"{logical}-{fname}"
            dest.write_bytes(data)
            out[logical] = {
                "file": str(dest.relative_to(REPO_ROOT)),
                "url": url,
                "sha256": _sha256(data),
            }
            break
        else:
            print(f"  ! 未取到 {repo} 的 LICENSE（请人工核对再分发条款）")
    return out


def check_only(expected: dict, *, timeout: float = 30.0,
               verify_ssl: bool = True) -> tuple[list[str], list[str]]:
    """校验模式：只按 manifest 里锁定的 URL 下载/比对，不猜分支、不回写。

    返回 ``(ok, bad)``。任何「文件存在但哈希不符」都必须判失败并删除，
    让下次 ``--lock`` 重新取 —— 静默沿用被改动的夹具等于把真值交出去。
    """
    ok: list[str] = []
    bad: list[str] = []
    for name, entry in sorted(expected.items()):
        path = DEST / name
        want = str(entry.get("sha256") or "")
        if path.is_file():
            got = _sha256(path.read_bytes())
            if got == want:
                ok.append(name)
                continue
            print(f"  ✖ 哈希不符，删除待重取: {name} ({got[:12]}… != {want[:12]}…)")
            path.unlink(missing_ok=True)
        url = entry.get("url")
        if not url:
            bad.append(name)
            continue
        try:
            data = _fetch(url, timeout=timeout, verify_ssl=verify_ssl)
        except Exception as e:
            print(f"  ! 下载失败 {name}: {type(e).__name__}: {e}")
            bad.append(name)
            continue
        got = _sha256(data)
        if got != want:
            print(f"  ✖ 远端内容已变，拒绝写入: {name}")
            bad.append(name)
            continue
        path.write_bytes(data)
        ok.append(name)
    return ok, bad


def write_manifest(manifest: dict, licenses: dict) -> None:
    payload = {
        "_comment": (
            "第三方公开夹具清单。由 scripts/fetch_blf_fixtures.py --lock 生成，"
            "请勿手改 sha256。测试用例读这份表：文件缺失或哈希不符即 skip。"
        ),
        "files": manifest,
        "licenses": licenses,
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_report(manifest: dict, licenses: dict, *, mode: str) -> None:
    total = sum(int(e.get("bytes", 0)) for e in manifest.values())
    report = {
        "mode": mode,
        "requested": len(_RAW_SPECS),
        "fetched": len(manifest),
        "total_bytes": total,
        "missing": sorted({n for n, _, _, _, _ in _RAW_SPECS} - set(manifest)),
        "licenses": {k: v["file"] for k, v in licenses.items()},
        "python": sys.version.split()[0],
    }
    (DEST / "fetch_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", action="store_true",
                        help="首次生成/刷新 manifest：下载后把实测 sha256 写回")
    parser.add_argument("--check", action="store_true",
                        help="按既有 manifest 校验（不猜分支、不回写）")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--skip-licenses", action="store_true")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="跳过 TLS 证书校验，仅供受限网络的本机 bootstrap；CI 禁用",
    )
    args = parser.parse_args(argv)

    if args.check and args.lock:
        parser.error("--check 与 --lock 互斥")

    DEST.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_manifest()
    verify_ssl = not args.insecure
    if args.insecure:
        print("⚠ --insecure：已跳过 TLS 证书校验，取回的内容不可视为已验证")

    licenses = existing.get("licenses") or {}
    if not args.skip_licenses:
        licenses = fetch_licenses(timeout=args.timeout, verify_ssl=verify_ssl) or licenses

    if args.check and existing.get("files"):
        ok, bad = check_only(existing["files"], timeout=args.timeout,
                             verify_ssl=verify_ssl)
        manifest = existing["files"]
        mode = "check"
        print(f"校验：{len(ok)} 通过 / {len(bad)} 缺失或失败")
    else:
        manifest = build(timeout=args.timeout, verify_ssl=verify_ssl)
        if not manifest:
            print("✖ 一个文件都没取到（无网络？），manifest 保持原样")
            return 1
        mode = "lock" if args.lock else "fetch"
        write_manifest(manifest, licenses)
        print(f"已写入 {MANIFEST_PATH.relative_to(REPO_ROOT)}（{len(manifest)} 条）")

    write_report(manifest, licenses, mode=mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
