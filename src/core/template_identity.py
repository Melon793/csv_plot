"""模板跨文件身份匹配引擎（方案 A，D3/D4/D9/D10/D12）。

dat/mdf/blf 同名通道加载时自动加 ``_G{gi}`` 后缀，后缀被原样写进模板；
换一笔测量数据组/桶序变化 → 后缀漂移 → 模板匹配失败或同名误配。本模块
提供纯函数 :func:`resolve_identities`：套用前把模板里的旧显示名重映射到
当前数据的显示名，loader 命名策略零改动（D3）。

级联（D4：回退只到"唯一候选"为止，多候选不猜）：

- **L0** 显示名恒等命中。模板有身份条目时做 D9 一致性校验：
  original_name 必须相等；双方单位均非空时必须相等；双方指纹均非空时
  按 D12 矛盾否决口径比对。任一不一致 → 降 L1 重解析。
  旧档无身份条目 → L0 维持现状（残余风险：无指纹文件同名换内容不可检测）。
- **L1** 身份指纹唯一命中（D12）：MDF = 双方均非空的 device/ecu 字段必须
  全等 + 至少一对非空相等（矛盾否决）；BLF = (frame, can_id, extended,
  bus_channel) 四元组全等。
- **L2** 旧档名字启发式（D10）：查找键 = 名字本身 + 剥 ``_G\\d+$`` 后缀名；
  loader 侧双索引 = original_name（裸名）+ BLF 专属 ``frame.original_name``
  （限定基名，loader 构造，不做点号拆分）。任一键在索引中恰好 1 个候选
  即命中；多候选一律 L3。
- **L3** 保持原名（套用时因变量缺失降级为占位）。
- **D9 矛盾 fail-closed（M-1 方案乙，用户拍板）**：有身份条目、L0 名字命中
  但一致性校验检出矛盾，且 L1/L2 无法唯一重映射时，该曲线在 resolved
  config 中从 curves 移入 invalid_curves（灰色占位，不绘制），名字进
  demoted 名单。**仅**此路径降占位：旧档无身份 L0、纯缺失 L3、冲突被
  占用 L3 一律不改。demoted 由调用方并入失配口径（merge_demoted），
  ratio 不得虚高，三播报路径同规格披露。

契约：

- ``check_template_match`` 签名与返回形状零改动（D12）；调用方通过
  ``resolve_identity_snapshot`` 取得 resolved 配置与统计投影，匹配、套用和
  播报复用同一快照，不再在套用入口二次解析。
- 模板文件不因套用改写（D6）：产出深拷贝，原 config 不动。
- loader 为 None / 无 get_channel_info（CSV/Excel）→ 原样返回、remap/
  demoted 空（D8：CSV 路径不启用回退，行为与现状一致）。
- 冲突守卫：先跨权威配置预留字面曲线名，再整批解析真实 curves，最后解析
  invalid；不同模板名争抢同一当前名时第二个退 L3 + log warning。同一模板名
  在多个子图重复出现是正常资产，共享同一解析结果。
- v2 只解析权威 ``pages``；顶层 ``plots`` 是兼容回退视图，套用结束后从
  第 1 页同步，不参与身份分配和统计。

残余风险（披露）：
旧档无身份条目时 L0 同名换内容不可检测（D9 无法校验，维持现状）；
D9 矛盾后 L2 唯一候选（含兄弟候选被 claim 过滤后）回选自身仍照画——
拍板排除项（解释 #1，唯一候选不猜的反面：唯一即信）；
L3 保持原名不 claim；单格重复 resolved name 直接拒绝，避免同名资产两侧
共存或元数据互相覆盖；同名异身份分歧
remap 按不同模板名计数（终审 m-3，符合 D5 规格，已有 warning）。
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from src.core.logger import get_logger
from src.core.plot_config import PlotSessionConfig

logger = get_logger(__name__)

_SUFFIX_RE = re.compile(r"_G\d+$")

#: asammdf 写入合成/脚本通道时注入的伪影 source.name（评审 MJ-9 探针实证），
#: 采集侧视为空，不得当成 ECU 指纹。
_ASAMMDF_ARTIFACT_SOURCE = "Channel inserted by Python Script"

#: 单位空值口径：MDF/BLF  loader 对无单位通道落 "-" 或 ""。
_EMPTY_UNITS = ("", "-")


@dataclass(frozen=True)
class ResolvedIdentity:
    """一次身份解析的配置与统计投影。"""

    config: PlotSessionConfig
    remap: dict[str, str]
    demoted: list[str]


def _norm_unit(unit) -> str:
    """单位归一：非 str / 空白 / '-' 一律视为空。"""
    if not isinstance(unit, str):
        return ""
    u = unit.strip()
    return "" if u in _EMPTY_UNITS else u


def identity_from_channel_info(info: dict) -> dict | None:
    """从 loader.get_channel_info 的返回 dict 提取身份条目。

    采集侧（导出后处理）与引擎侧（当前数据指纹）共用，保证两侧口径一致。
    无法构成有效身份（缺 original_name / BLF 四元组不全）→ None。
    kind 判定：带 ``can`` 子 dict → blf；带 ``source`` 子 dict → mdf；
    其余（未来 loader）→ None，不采集。
    """
    if not isinstance(info, dict):
        return None
    meta = info.get("meta")
    original_name = getattr(meta, "original_name", "")
    if not isinstance(original_name, str) or not original_name.strip():
        return None
    can = info.get("can")
    if isinstance(can, dict):
        frame = can.get("message_name")
        can_id = can.get("can_id")
        extended = can.get("is_extended")
        bus = can.get("bus_channel_index")
        if (
            not isinstance(frame, str) or not frame.strip()
            or not isinstance(can_id, int) or isinstance(can_id, bool)
            or not isinstance(extended, bool)
            or not isinstance(bus, int) or isinstance(bus, bool)
        ):
            return None
        entry = {
            "original_name": original_name,
            "kind": "blf",
            "frame": frame,
            "can_id": can_id,
            "extended": extended,
            "bus_channel": bus,
        }
    elif isinstance(info.get("source"), dict):
        entry = {"original_name": original_name, "kind": "mdf"}
        source = info["source"]
        device = source.get("path")
        ecu = source.get("name")
        # device=SI.path、ecu=SI.name（刻意不复用 mdf_attribution 名字尾缀
        # 设备逻辑，MN-7）；asammdf 伪影 source.name 视为空（MJ-9）。
        if isinstance(device, str) and device.strip():
            entry["device"] = device
        if (
            isinstance(ecu, str) and ecu.strip()
            and ecu != _ASAMMDF_ARTIFACT_SOURCE
        ):
            entry["ecu"] = ecu
    else:
        return None
    unit = _norm_unit(getattr(meta, "unit", ""))
    if unit:
        entry["unit"] = unit
    return entry


def _mdf_fingerprint_match(template: dict, current: dict) -> bool:
    """D12 MDF L1：双方均非空字段必须全等 + 至少一对非空相等（矛盾否决）。"""
    positive = False
    for fld in ("device", "ecu"):
        t, c = template.get(fld), current.get(fld)
        if t and c:
            if t != c:
                return False
            positive = True
    return positive


def _blf_fingerprint_match(template: dict, current: dict) -> bool:
    """D12 BLF L1：四元组全等。模板侧四元组由 parse 保证齐全。"""
    return all(
        template.get(fld) == current.get(fld)
        for fld in ("frame", "can_id", "extended", "bus_channel")
    )


def _d9_consistent(template: dict, current_info: dict) -> bool:
    """D9：L0 恒等命中时的一致性校验（模板有身份条目才调用）。

    original_name 必须相等；双方单位均非空时必须相等；双方指纹均非空时
    按 D12 矛盾否决口径比对（MDF 逐字段矛盾否决；BLF 四元组全等）。
    """
    meta = current_info.get("meta") if isinstance(current_info, dict) else None
    if getattr(meta, "original_name", None) != template.get("original_name"):
        return False
    t_unit = _norm_unit(template.get("unit"))
    c_unit = _norm_unit(getattr(meta, "unit", ""))
    if t_unit and c_unit and t_unit != c_unit:
        return False
    current = identity_from_channel_info(current_info)
    if current is None:
        # 当前侧指纹不完整（理论上 MDF/BLF 不会）：只做 name/unit 校验
        return True
    if template.get("kind") != current.get("kind"):
        return False
    if template["kind"] == "mdf":
        for fld in ("device", "ecu"):
            t, c = template.get(fld), current.get(fld)
            if t and c and t != c:
                return False
        return True
    return _blf_fingerprint_match(template, current)


class _Resolver:
    """单次 resolve 的解析器：惰性建索引 + memo + 先到先得占用表。"""

    def __init__(self, get_channel_info, var_names: list[str]):
        self._gci = get_channel_info
        self._var_names = var_names
        self._current_set = set(var_names)
        self._info_cache: dict = {}
        self._by_original: dict[str, list[str]] | None = None
        self._by_qualified: dict[str, list[str]] = {}
        self._claimed: dict[str, str] = {}  # 当前名 → 模板名
        self._memo: dict = {}

    def _info(self, name: str):
        if name not in self._info_cache:
            try:
                self._info_cache[name] = self._gci(name)
            except (KeyError, AttributeError):
                self._info_cache[name] = None
        return self._info_cache[name]

    def _build_index(self) -> None:
        """D10 双索引：original_name（裸名）+ BLF 限定基名。只在首次需要
        L1/L2 时建一次（全命中 L0 的套用零额外开销）。"""
        if self._by_original is not None:
            return
        by_original: dict[str, list[str]] = {}
        for name in self._var_names:
            info = self._info(name)
            if info is None:
                continue
            meta = info.get("meta")
            on = getattr(meta, "original_name", "")
            if not isinstance(on, str) or not on:
                continue
            by_original.setdefault(on, []).append(name)
            can = info.get("can")
            if isinstance(can, dict):
                frame = can.get("message_name")
                if isinstance(frame, str) and frame:
                    self._by_qualified.setdefault(f"{frame}.{on}", []).append(name)
        self._by_original = by_original

    def _claim(self, target: str, template_name: str) -> bool:
        owner = self._claimed.get(target)
        if owner is None or owner == template_name:
            self._claimed[target] = template_name
            return True
        return False

    def reserve(self, name: str) -> None:
        self._claimed.setdefault(name, name)

    def release(self, name: str) -> None:
        if self._claimed.get(name) == name:
            self._claimed.pop(name, None)

    def _l1_candidates(self, name: str, identity: dict, *, respect_claim: bool = True) -> list[str]:
        self._build_index()
        kind = identity["kind"]
        out = []
        for cand in self._by_original.get(identity["original_name"], []):
            info = self._info(cand)
            current = identity_from_channel_info(info) if info else None
            if current is None or current.get("kind") != kind:
                continue
            if kind == "mdf":
                ok = _mdf_fingerprint_match(identity, current)
            else:
                ok = _blf_fingerprint_match(identity, current)
            if ok and (not respect_claim or self._claimed.get(cand) in (None, name)):
                out.append(cand)
        return out

    def _l2_candidates(self, key: str, name: str, *, respect_claim: bool = True) -> list[str]:
        self._build_index()
        seen: dict[str, None] = {}
        for index in (self._by_original, self._by_qualified):
            for cand in index.get(key, []):
                if not respect_claim or self._claimed.get(cand) in (None, name):
                    seen.setdefault(cand, None)
        return list(seen)

    def resolve_curve(
        self, name: str, identity: dict | None, blocked_names: set[str] | None = None
    ) -> tuple[str, bool]:
        """curves 侧解析：返回 (当前显示名, 是否 D9 矛盾降占位)。

        降占位仅限「有身份条目 + L0 命中但 D9 矛盾 + L1/L2 无法唯一
        重映射」路径（M-1 方案乙）；L2 唯一候选回选自身不算降占位。
        """
        blocked = frozenset(blocked_names or ())
        memo_key = (
            name, tuple(sorted(identity.items())) if identity else None, blocked,
        )
        if memo_key not in self._memo:
            self._memo[memo_key] = self._resolve_uncached(name, identity, blocked)
        return self._memo[memo_key]

    def resolve_name(self, name: str, identity: dict | None) -> str:
        """invalid_curves 侧解析：只换名，不降占位（已是占位）。"""
        return self.resolve_curve(name, identity)[0]

    def _resolve_uncached(
        self, name: str, identity: dict | None, blocked: frozenset[str]
    ) -> tuple[str, bool]:
        conflict = False
        d9_failed = False
        # L0：显示名恒等（+D9 校验）。同名被别的模板名占用时也走回退。
        if name in self._current_set:
            if name not in blocked and self._claimed.get(name) in (None, name):
                if identity is None:
                    self._claim(name, name)
                    return name, False
                info = self._info(name)
                if info is not None and _d9_consistent(identity, info):
                    self._claim(name, name)
                    return name, False
                if info is not None:
                    d9_failed = True
                # info 缺失（loader 异常态）无法校验 ≠ 检出矛盾，不降占位
                logger.info("L0 一致性校验未通过，降级重解析: %s", name)
            else:
                conflict = True
        # L1：身份指纹唯一命中（D12）
        if identity is not None:
            cands = [
                cand for cand in self._l1_candidates(name, identity)
                if cand not in blocked
            ]
            if len(cands) == 1:
                self._claim(cands[0], name)
                return cands[0], False
            if not cands and any(
                cand not in blocked
                for cand in self._l1_candidates(name, identity, respect_claim=False)
            ):
                conflict = True
        # L2：查找键唯一候选（D10）
        for key in (name, _SUFFIX_RE.sub("", name)):
            cands = [
                cand for cand in self._l2_candidates(key, name)
                if cand not in blocked
            ]
            if len(cands) == 1:
                self._claim(cands[0], name)
                return cands[0], False
            if not cands and any(
                cand not in blocked
                for cand in self._l2_candidates(key, name, respect_claim=False)
            ):
                conflict = True
        # L3：保持原名（套用时因缺失降级为占位）
        if conflict:
            # 终审 m-1：§3.2 明文要求冲突退 L3 必须可观测
            logger.warning(
                "身份重映射冲突，退 L3（目标已被其他模板名占用）: %s", name
            )
        if d9_failed:
            # M-1 方案乙：D9 矛盾且无法唯一重映射 → 降占位不绘制。
            # 冲突路径（d9_failed=False）与纯缺失路径不在此列。
            logger.warning("身份矛盾且无法唯一重映射，降占位（fail-closed）: %s", name)
        return name, d9_failed


def _iter_unique_plots(config: PlotSessionConfig):
    """只遍历权威视图：v2 pages，v1 回退 plots。"""
    for page in config.pages:
        yield from page.plots
    if not config.pages:
        yield from config.plots


def _rename_metadata(values, local: dict[str, str]):
    if not local:
        return values
    values = values or {}
    targets = set(local.values())
    renamed = {
        key: value
        for key, value in values.items()
        if key not in local and key not in targets
    }
    for key, target in local.items():
        if key in values:
            renamed[target] = values[key]
    return renamed


def resolve_identity_snapshot(config: PlotSessionConfig, loader) -> ResolvedIdentity:
    """一次解析身份并返回配置、重映射统计与降占位名单。"""
    if loader is None:
        return ResolvedIdentity(config, {}, [])
    gci = getattr(loader, "get_channel_info", None)
    var_names = getattr(loader, "var_names", None)
    if not callable(gci) or var_names is None:
        return ResolvedIdentity(config, {}, [])

    resolver = _Resolver(gci, list(var_names))
    resolved = copy.deepcopy(config)
    plots = list(_iter_unique_plots(resolved))
    remap: dict[str, str] = {}
    demoted: list[str] = []

    for plot in plots:
        identities = plot.curve_identities or {}
        for name in plot.curves:
            resolver.reserve(name)

    def _record_remap(name: str, new: str) -> None:
        if new != name:
            prev = remap.get(name)
            if prev is not None and prev != new:
                logger.warning("身份解析分歧: %s → %s / %s", name, prev, new)
            else:
                remap[name] = new
                logger.info("身份重映射: %s → %s", name, new)

    curve_results: dict[int, tuple[list[str], list[str], dict[str, str]]] = {}
    for plot in plots:
        identities = plot.curve_identities or {}
        local: dict[str, str] = {}
        new_curves: list[str] = []
        demoted_here: list[str] = []
        for name in plot.curves:
            new, dem = resolver.resolve_curve(
                name, identities.get(name), set(plot.invalid_curves)
            )
            _record_remap(name, new)
            if new != name:
                local[name] = new
            if dem or new != name:
                resolver.release(name)
            if new in new_curves:
                logger.warning("单格重复 resolved name，跳过曲线: %s", new)
                continue
            if dem:
                if new not in demoted_here:
                    demoted_here.append(new)
            else:
                new_curves.append(new)
        curve_results[id(plot)] = (new_curves, demoted_here, local)

    for plot in plots:
        identities = plot.curve_identities or {}
        new_curves, demoted_here, local = curve_results[id(plot)]
        new_invalid: list[str] = []
        for name in plot.invalid_curves:
            new = resolver.resolve_name(name, identities.get(name))
            _record_remap(name, new)
            if new != name:
                local[name] = new
            if new not in new_curves and new not in new_invalid:
                new_invalid.append(new)
        for name in demoted_here:
            if name not in new_curves and name not in new_invalid:
                new_invalid.append(name)
            if name not in demoted:
                demoted.append(name)
        plot.curves = new_curves
        plot.invalid_curves = new_invalid
        plot.curve_settings = _rename_metadata(plot.curve_settings, local)
        plot.curve_identities = _rename_metadata(identities, local)

    if resolved.pages:
        fallback = resolved.pages[0].plots
        if len(resolved.plots) != len(fallback) or any(
            left is not right for left, right in zip(resolved.plots, fallback)
        ):
            resolved.plots = copy.deepcopy(fallback)
    drawn = {name for plot in plots for name in plot.curves}
    demoted = [name for name in demoted if name not in drawn]
    return ResolvedIdentity(resolved, remap, demoted)


def resolve_identities(
    config: PlotSessionConfig, loader
) -> tuple[PlotSessionConfig, dict[str, str], list[str]]:
    """兼容旧三元组接口：返回 (resolved_config, remap, demoted)。"""
    snapshot = resolve_identity_snapshot(config, loader)
    return snapshot.config, snapshot.remap, snapshot.demoted


def merge_demoted(
    matched: set[str], unmatched: set[str], demoted: list[str]
) -> tuple[float, set[str], set[str]]:
    """demoted 并入失配口径（M-1 方案乙简报 §2，ratio 不得虚高）。

    check_template_match 只数 curves，降占位后分母会缩；调用方用本助手
    把 demoted 并入分母与缺失集合：返回 (ratio, matched, unmatched∪demoted)。
    demoted 为空时与 check_template_match 原值逐字一致（零回归）。
    """
    d = set(demoted)
    total = matched | unmatched | d
    ratio = len(matched) / len(total) if total else 1.0
    return ratio, matched, unmatched | d