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
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

SCHEMA_VERSION = "canview/1"


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

        用于「对话框当前还改不了绑定」的过渡路径（笔 2–3）：用户若在池里
        移除了某文件，对应绑定随之删除；绑定因此**变空**的通道退回未绑定
        （= 用全局池），而不是留下一个指向不存在文件的死引用。
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
            alias={ch: a for ch, a in self.alias.items() if ch in kept},
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
        channels: dict[str, dict] = {}
        for ch in sorted(self.by_channel):
            entry: dict = {"databases": list(self.by_channel[ch])}
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
            dbs = entry.get("databases")
            by_channel[ch] = tuple(str(p) for p in (dbs or ()))
            if entry.get("alias"):
                alias[ch] = str(entry["alias"])
        return cls.from_paths(data.get("databases") or (), by_channel, alias)