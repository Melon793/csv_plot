"""``CanBindings`` —— BLF 分析的数据库池 + 通道绑定配置对象。

职责（只做数据与口径，不碰 Qt / numpy / 文件 IO）：

1. 承载「有序库池 + 每通道绑定的库子集 + 通道别名」这一份配置；
2. **1 基 ↔ 0 基转换的唯一单点**：YAML / UI 用 1 基（与 CANoe、help.md 对齐），
   内部桶键、``CanDatabaseSet.bindings``、``BLFLazyLoader.bindings`` 全用 0 基；
3. ``__eq__`` 即 F 路径「答案是否变化」的判等依据（池顺序 + 绑定 + 别名）；
4. ``format_bus_channel`` 是显示口径的第二个单点（另一个是 YAML/UI 的 1 基）。

调用纪律：**不要把本对象当 list 用**（``list(cfg)`` / 迭代 / 直接传给旧接口）——
取池一律 ``cfg.pool``，取 loader 入参一律 ``cfg.normalized_loader_bindings()``。
``list(db_paths or [])`` 对配置对象会 TypeError，这是评审 F8 点名的坑。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

SCHEMA_VERSION = "canview/1"
SIDECAR_SUFFIX = ".canview.yaml"


def _key(path: str) -> str:
    """绑定子集判定用的路径键：与 ``can_database._path_key`` 同一口径。"""
    return os.path.normcase(os.path.abspath(str(path)))


def format_bus_channel(channel: int, alias: Optional[str] = None) -> str:
    """0 基内部通道 → 用户可见标签：``CAN1``，有别名时 ``CAN1 动力CAN``。"""
    label = f"CAN{int(channel) + 1}"
    alias = (alias or "").strip()
    return f"{label} {alias}" if alias else label


@dataclass(frozen=True)
class CanBindings:
    """不可变配置。字段全部按「可直接写进 YAML」的原始路径保存（便于显示）。"""

    pool: tuple[str, ...] = ()
    by_channel: dict[int, tuple[str, ...]] = field(default_factory=dict)
    alias: dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> "CanBindings":
        return cls()

    @classmethod
    def from_paths(
        cls,
        pool: Iterable[str],
        by_channel: Optional[Mapping[int, Iterable[str]]] = None,
        alias: Optional[Mapping[int, str]] = None,
    ) -> "CanBindings":
        """从原始路径构造；**不做去重/校验**（``validate()`` 单独报告问题）。"""
        return cls(
            pool=tuple(str(p) for p in (pool or ()) if str(p)),
            by_channel={
                int(ch): tuple(str(p) for p in files if str(p))
                for ch, files in (by_channel or {}).items()
            },
            alias={int(ch): str(a) for ch, a in (alias or {}).items() if str(a).strip()},
        )

    # ------------------------------------------------------------------
    # 与 loader / 旧接口的桥
    # ------------------------------------------------------------------

    @property
    def pool_list(self) -> list[str]:
        """旧接口要的 ``list[str]`` 副本（避免把 tuple 传给出入参是 list 的代码）。"""
        return list(self.pool)

    def normalized_loader_bindings(self) -> dict[int, frozenset[str]]:
        """``BLFLazyLoader(bindings=...)`` 的形状；空绑定返回空 dict（= 未绑定）。

        ``frozenset`` 内保留原始路径字符串——规范化由 ``CanDatabaseSet`` 统一做，
        本对象不抢先改写用户看到/保存的写法。
        """
        return {
            ch: frozenset(files)
            for ch, files in self.by_channel.items()
            if files
        }

    def alias_map(self) -> dict[int, str]:
        return dict(self.alias)

    def restrict_to_pool(self, pool: Iterable[str]) -> "CanBindings":
        """池变化后的保守收敛：只保留仍属于新池的绑定文件。

        用于「重载复用 loader 状态」这条路径。绑定因此**变空**的通道退回
        未绑定（= 用全局池），而不是留下指向不存在文件的死引用。

        **别名与绑定解耦**：别名是展示层属性，即使通道退回未绑定也必须保留
        （评审 P0：此前只保留仍有绑定的通道，重载会把 `CAN2 车身CAN` 静默
        降级成 `CAN2`，并经成功回写持久化，用户无法察觉）。
        """
        allowed = {_key(p) for p in (pool or ())}
        kept: dict[int, tuple[str, ...]] = {}
        for ch, files in self.by_channel.items():
            subset = tuple(p for p in files if _key(p) in allowed)
            if subset:
                kept[ch] = subset
        return CanBindings(
            pool=tuple(str(p) for p in (pool or ()) if str(p)),
            by_channel=kept,
            alias=dict(self.alias),
        )

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """返回问题列表（空 = 合法）。**不抛异常**：坏配置不打断导入。"""
        issues: list[str] = []
        seen: set[str] = set()
        for path in self.pool:
            key = _key(path)
            if key in seen:
                issues.append(f"重复的数据库：{path}")
            seen.add(key)
        for ch, files in self.by_channel.items():
            if ch < 0:
                issues.append(f"非法通道号：{ch}（内部 0 基，必须 ≥ 0）")
            for path in files:
                if _key(path) not in seen:
                    issues.append(f"通道 {format_bus_channel(ch)} 绑定了池外文件：{path}")
        return issues

    # ------------------------------------------------------------------
    # YAML 形状（1 基只在这一层出现）
    # ------------------------------------------------------------------

    def to_yaml_dict(self) -> dict:
        # 别名与绑定解耦：别名-only 的通道也必须落盘，否则「只改通道名不绑库」
        # 在持久化边界静默丢失（第二轮评审 P0）。无绑定的通道写出空列表，
        # 读回时 from_yaml_dict 不会把它当成「绑定到空集」。
        channels: dict[str, dict] = {}
        for ch in sorted(set(self.by_channel) | set(self.alias)):
            entry: dict = {"databases": list(self.by_channel.get(ch, ()))}
            if self.alias.get(ch):
                entry["alias"] = self.alias[ch]
            channels[str(ch + 1)] = entry
        return {
            "schema": SCHEMA_VERSION,
            "databases": list(self.pool),
            "channels": channels,
        }

    @classmethod
    def from_yaml_dict(cls, data: Mapping) -> "CanBindings":
        """读 YAML dict；键一律按 1 基通道理解（写回时同样 1 基）。"""
        by_channel: dict[int, tuple[str, ...]] = {}
        alias: dict[int, str] = {}
        for raw_ch, entry in (data.get("channels") or {}).items():
            try:
                ch = int(raw_ch) - 1
            except (TypeError, ValueError):
                raise ValueError(f"非法通道键：{raw_ch!r}（应为 1 基整数）") from None
            if ch < 0:
                raise ValueError(f"非法通道键：{raw_ch!r}（应为 ≥ 1）")
            entry = entry or {}
            dbs = tuple(str(p) for p in (entry.get("databases") or ()))
            if dbs:
                # 空列表是「只有别名」的通道，不能当成绑定到空集（= 全部排除）
                by_channel[ch] = dbs
            if entry.get("alias"):
                alias[ch] = str(entry["alias"])
        return cls.from_paths(data.get("databases") or (), by_channel, alias)


# ---------------------------------------------------------------------------
# 文件 IO（显式配置）：旁车 YAML / 用户配置目录
# ---------------------------------------------------------------------------

def sidecar_config_path(blf_path: str) -> str:
    """日志旁的约定文件名：``<日志名>.canview.yaml``。"""
    return os.path.splitext(os.path.abspath(str(blf_path)))[0] + SIDECAR_SUFFIX


def is_dir_writable(folder: str) -> bool:
    """写入探针：只读盘 / UNC 上 os.access 不可靠，实际建删一个临时文件。"""
    try:
        os.makedirs(folder, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".canview-", dir=folder)
        os.close(fd)
        os.unlink(probe)
        return True
    except OSError:
        return False


def _resolve_db_path(raw: str, bases: list[str],
                     memo: dict[str, str]) -> str:
    """库路径解析：相对 YAML 目录 → 相对日志目录 → 绝对兜底。

    存在的相对路径解析成绝对路径（后续匹配/显示都稳定）；都不存在时保留
    首选基目录下的候选路径，交给调用方作为 issue 列红。
    """
    key = str(raw)
    if key in memo:
        return memo[key]
    candidate = Path(key)
    if candidate.is_absolute():
        resolved = str(candidate)
    else:
        resolved = ""
        for base in bases:
            try:
                probe = Path(base) / key
            except (TypeError, ValueError):
                continue
            if probe.exists():
                resolved = str(probe.resolve())
                break
        if not resolved:
            resolved = str(Path(bases[0]) / key) if bases else key
    memo[key] = resolved
    return resolved


def load_yaml(path: str, blf_path: Optional[str] = None) -> tuple[CanBindings, list[str]]:
    """读取显式配置。**不抛异常**：坏文件 / 版本不符 → ``(empty, issues)``。"""
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        return CanBindings.empty(), [f"配置文件无法读取：{exc}"]
    except yaml.YAMLError as exc:
        return CanBindings.empty(), [f"配置文件不是有效 YAML：{exc}"]
    if not isinstance(data, dict):
        return CanBindings.empty(), ["配置文件顶层必须是映射（schema / databases / channels）"]

    issues: list[str] = []
    schema = data.get("schema")
    if schema != SCHEMA_VERSION:
        if schema is None:
            issues.append(f"配置缺少 schema，已按 {SCHEMA_VERSION} 读取")
        else:
            return CanBindings.empty(), [
                f"配置版本 {schema!r} 不受支持（需要 {SCHEMA_VERSION}）"
            ]

    bases: list[str] = []
    for folder in (
        os.path.dirname(os.path.abspath(path)),
        os.path.dirname(os.path.abspath(blf_path)) if blf_path else "",
    ):
        if folder and folder not in bases:
            bases.append(folder)

    memo: dict[str, str] = {}
    resolved = dict(data)
    resolved.pop("schema", None)
    resolved["databases"] = [
        _resolve_db_path(p, bases, memo) for p in (data.get("databases") or [])
    ]
    channels: dict = {}
    for raw_ch, entry in (data.get("channels") or {}).items():
        entry = entry or {}
        item: dict = {
            "databases": [
                _resolve_db_path(p, bases, memo) for p in (entry.get("databases") or [])
            ]
        }
        if entry.get("alias"):
            item["alias"] = entry["alias"]
        channels[raw_ch] = item
    resolved["channels"] = channels

    try:
        cfg = CanBindings.from_yaml_dict(resolved)
    except ValueError as exc:
        return CanBindings.empty(), [str(exc)]

    for db_path in cfg.pool:
        if not os.path.exists(db_path):
            issues.append(f"数据库不存在：{db_path}")
    issues.extend(cfg.validate())
    return cfg, issues


def save_yaml(cfg: CanBindings, path: str) -> list[str]:
    """原子写显式配置（临时文件 + os.replace）。返回问题列表（空 = 成功）。"""
    import yaml

    folder = os.path.dirname(os.path.abspath(path)) or "."
    tmp_path = ""
    try:
        os.makedirs(folder, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".canview-", suffix=".yaml", dir=folder)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                cfg.to_yaml_dict(), handle, allow_unicode=True, sort_keys=False
            )
        os.replace(tmp_path, path)
        tmp_path = ""
        return []
    except OSError as exc:
        return [f"配置写入失败：{exc}"]
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass