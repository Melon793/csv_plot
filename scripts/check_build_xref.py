#!/usr/bin/env python3
"""发布前静态交叉引用护栏：查「裸 .pyc 拷贝」包的真实运行期依赖是否进了产物。

背景：build_win.py 在收尾阶段用 shutil.copytree 把 asammdf 整包拷成 .pyc 塞进
dist。这些 .pyc 不在 Nuitka 的静态 import 图里，Nuitka 从未分析过它们的 import，
于是它们模块级引用的标准库/三方库不会被 --follow-imports 收集。历史故障：
asammdf/blocks/v2_v3_blocks.py 顶层 `from getpass import getuser`，getpass 漏收，
MDF 加载报「No module named 'getpass'」，且回退链的 except Exception 把它吞成一条
WARNING —— 光看「程序没崩」发现不了。

做法：把「被扫描包的全部模块级 import」与「Nuitka report.xml 的收录清单」求差，
列出 report.xml 里没有、又不是内置的模块。秒级完成，无需再等 30 分钟构建撞错。

用法（在 Windows 构建目录、有 report.xml 时）：
    python scripts/check_build_xref.py --report report.xml
    python scripts/check_build_xref.py --report report.xml --package asammdf --package canmatrix
    python scripts/check_build_xref.py --imports-only            # 只列外部模块级 import，不比对产物

退出码：0 = 无缺口；1 = 发现疑似缺失（可作 CI / 发布前门禁）。
"""

import argparse
import ast
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# Nuitka standalone 一定内置、report.xml 可能不单列的模块，避免误报为缺失。
_ALWAYS_PRESENT = {
    "sys", "builtins", "_frozen_importlib", "_frozen_importlib_external",
    "importlib", "marshal", "posix", "nt", "winreg", "time", "errno",
    "io", "abc", "stat", " codecs", "_codecs", "encodings",
}


def _module_root(name):
    return name.split(".")[0]


# 包内这些顶层子目录（GUI / 命令行 app）不在 `import <pkg>` 核心链上，
# 只被它们引用的依赖视为可选，默认放行（否则每次裸跑都误报 exit 1）。
_OPTIONAL_DIRS = {"gui", "app"}


def _is_optional_site(rel):
    """rel 形如 'gui/dialogs/x.py' 或 'blocks/y.py'；判断引用点是否在可选子树。"""
    return rel.split("/")[0] in _OPTIONAL_DIRS


def _scan_package_imports(pkg_name):
    """返回 {外部模块名: [(引用文件相对路径, lineno), ...]}，只收集模块级（顶层）import。"""
    try:
        spec = importlib.util.find_spec(pkg_name)
    except (ValueError, ModuleNotFoundError) as e:
        print(f"[ERROR] 无法定位包 {pkg_name}: {e}")
        sys.exit(2)
    if not spec or not spec.submodule_search_locations:
        print(f"[ERROR] {pkg_name} 不是包（无 submodule_search_locations）")
        sys.exit(2)

    pkg_root = _module_root(pkg_name)
    imports = {}

    for base in spec.submodule_search_locations:
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
                except SyntaxError:
                    continue
                rel = os.path.relpath(path, base).replace(os.sep, "/")
                for node in tree.body:  # 只看顶层，不看函数体
                    targets = []
                    if isinstance(node, ast.Import):
                        targets = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        if node.level and node.level > 0:
                            continue  # 相对导入：包内引用，跳过
                        if node.module:
                            targets = [node.module]
                    for mod in targets:
                        if _module_root(mod) == pkg_root:
                            continue  # 引用自己，跳过
                        imports.setdefault(mod, []).append((rel, node.lineno))
    return imports


def _bundle_modules_from_report(report_path):
    """从 Nuitka report.xml 收集已收录模块名（含包与其子模块）。"""
    present = set()
    root = ET.parse(report_path).getroot()
    for el in root.iter("module"):
        name = el.get("name")
        if name:
            present.add(name)
    return present


def _is_covered(mod, present, builtins):
    root = _module_root(mod)
    if mod in builtins or root in builtins or root in _ALWAYS_PRESENT:
        return True
    # 模块本体、或作为某个已收录包的子模块被收录
    for p in present:
        if p == mod or mod.startswith(p + ".") or p == root or p.startswith(root + "."):
            # p==root 说明该包的顶层已在产物里；子模块按名匹配
            if p == mod or p == root:
                return True
    # 兜底：report 里存在同名/父子关系即视为覆盖
    return any(p == mod or p.startswith(mod + ".") or mod.startswith(p + ".") for p in present)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", help="Nuitka report.xml 路径（产物收录清单真值）")
    ap.add_argument("--package", action="append", default=None,
                    help="被扫描的裸拷贝包名，可重复（默认 asammdf）")
    ap.add_argument("--imports-only", action="store_true",
                    help="只打印模块级外部 import，不比对 report")
    ap.add_argument("--whitelist", default="",
                    help="逗号分隔，额外确认无需进产物的模块名")
    ap.add_argument("--include-optional", action="store_true",
                    help="连仅被 <pkg>/gui|app 引用（不在 import 链上）的缺口也按失败处理")
    args = ap.parse_args()

    pkgs = args.package or ["asammdf"]
    builtins = set(sys.builtin_module_names)
    wl = {w.strip() for w in args.whitelist.split(",") if w.strip()}

    all_imports = {}
    for pkg in pkgs:
        for mod, locs in _scan_package_imports(pkg).items():
            all_imports.setdefault(mod, []).extend((pkg, rel, ln) for rel, ln in locs)

    ext = {m: locs for m, locs in all_imports.items() if _module_root(m) not in wl}

    if args.imports_only:
        print(f"[imports-only] {', '.join(pkgs)} 的模块级外部 import 共 {len(ext)} 个：")
        for m in sorted(ext):
            print(f"  {m}")
        return 0

    if not args.report:
        print("[ERROR] 需要 --report <report.xml>，或用 --imports-only。")
        return 2
    if not Path(args.report).exists():
        print(f"[ERROR] 找不到 report: {args.report}")
        return 2

    present = _bundle_modules_from_report(args.report)
    missing = {m: locs for m, locs in ext.items() if not _is_covered(m, present, builtins)}

    def _fmt(mod):
        return "\n".join(f"      ← {pkg}/{rel}:{ln}" for pkg, rel, ln in missing[mod][:3])

    def _core_only(mod):
        # 只要有任何一处引用不在 gui/ 或 app/ 顶层，就是核心链缺口
        return any(not _is_optional_site(rel) for _pkg, rel, _ln in missing[mod])

    core = {m for m in missing if _core_only(m)}
    optional = set(missing) - core
    if args.include_optional:
        core |= optional
        optional = set()

    print(f"[xref] 扫描包: {', '.join(pkgs)}  |  report 收录模块 {len(present)} 个")
    print(f"[xref] 模块级外部 import {len(ext)} 个，report 未收录 {len(missing)} 个"
          f"（核心链 {len(core)} / 仅可选子包 {len(optional)}）")

    if optional:
        print("\nℹ 已默认放行（仅被 <pkg>/gui|app 引用，不在 import 链上，加 --include-optional 可强制报错）：")
        for m in sorted(optional):
            print(f"  {m}")
            print(_fmt(m))

    if not core:
        print("\n✓ 无核心缺口：所有 import 链上的模块级依赖都能在产物里找到。")
        return 0

    print("\n⚠ 核心链缺口（需在 build_win.py 里 --include-module 显式收，或确认可忽略后加 --whitelist）：")
    for m in sorted(core):
        print(f"  {m}")
        print(_fmt(m))
    return 1


if __name__ == "__main__":
    sys.exit(main())
