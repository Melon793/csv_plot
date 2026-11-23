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

契约：

- ``check_template_match`` 签名与返回形状零改动（D12）；resolve 前置到
  各调用方，apply_config 入口再 resolve 一次（幂等：已 resolve 的 config
  再 resolve 全走 L0 恒等，remap 为空）。
- 模板文件不因套用改写（D6）：产出深拷贝，原 config 不动。
- loader 为 None / 无 get_channel_info（CSV/Excel）→ 原样返回、remap 空
  （D8：CSV 路径不启用回退，行为与现状一致）。
- 冲突守卫：两**不同**模板名映射同一当前名 → 先到先得（遍历序固定
  pages→plots→curves 列表序，MN-3），第二个退 L3 + log warning。
  同一模板名在多个子图重复出现是正常资产，共享同一解析结果。
- from_dict 的顶层 plots 与 pages[0].plots 是独立对象、导出式构造共享
  对象（评审 MJ-5）：深拷贝后按对象身份去重遍历两份，同名同身份条目
  经 memo 共享解析结果，保证两份结构同步换名。
"""

from __future__ import annotations

import copy
import re

from src.core.logger import get_logger
from src.core.plot_config import PlotSessionConfig

logger = get_logger(__name__)

_SUFFIX_RE = re.compile(r"_G\d+$")

#: asammdf 写入合成/脚本通道时注入的伪影 source.name（评审 MJ-9 探针实证），
#: 采集侧视为空，不得当成 ECU 指纹。
_ASAMMDF_ARTIFACT_SOURCE = "Channel inserted by Python Script"

#: 单位空值口径：MDF/BLF  loader 对无单位通道落 "-" 或 ""。
_EMPTY_UNITS = ("", "-")


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

    def _l1_candidates(self, name: str, identity: dict) -> list[str]:
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
            if ok and self._claimed.get(cand) in (None, name):
                out.append(cand)
        return out

    def _l2_candidates(self, key: str, name: str) -> list[str]:
        self._build_index()
        seen: dict[str, None] = {}
        for index in (self._by_original, self._by_qualified):
            for cand in index.get(key, []):
                if self._claimed.get(cand) in (None, name):
                    seen.setdefault(cand, None)
        return list(seen)

    def resolve_name(self, name: str, identity: dict | None) -> str:
        """返回模板名对应的当前显示名；L3 时保持原名。"""
        memo_key = (name, tuple(sorted(identity.items())) if identity else None)
        if memo_key in self._memo:
            return self._memo[memo_key]
        result = self._resolve_uncached(name, identity)
        self._memo[memo_key] = result
        return result

    def _resolve_uncached(self, name: str, identity: dict | None) -> str:
        # L0：显示名恒等（+D9 校验）。同名被别的模板名占用时也走回退。
        if name in self._current_set and self._claimed.get(name) in (None, name):
            if identity is None:
                self._claim(name, name)
                return name
            info = self._info(name)
            if info is not None and _d9_consistent(identity, info):
                self._claim(name, name)
                return name
            logger.info("L0 一致性校验未通过，降级重解析: %s", name)
        # L1：身份指纹唯一命中（D12）
        if identity is not None:
            cands = self._l1_candidates(name, identity)
            if len(cands) == 1:
                self._claim(cands[0], name)
                return cands[0]
        # L2：查找键唯一候选（D10）
        for key in (name, _SUFFIX_RE.sub("", name)):
            cands = self._l2_candidates(key, name)
            if len(cands) == 1:
                self._claim(cands[0], name)
                return cands[0]
        # L3：保持原名（套用时因缺失降级为占位）
        return name


def _iter_unique_plots(config: PlotSessionConfig):
    """pages→plots→（顶层）plots 列表序，按对象身份去重（MN-3/MJ-5）。

    导出式构造顶层与 pages[0] 共享对象：去重后只处理一次；from_dict
    独立对象：两份都处理，同名同身份经 resolver memo 得到相同结果。
    """
    seen: set[int] = set()
    for page in config.pages:
        for plot in page.plots:
            if id(plot) not in seen:
                seen.add(id(plot))
                yield plot
    for plot in config.plots:
        if id(plot) not in seen:
            seen.add(id(plot))
            yield plot


def resolve_identities(
    config: PlotSessionConfig, loader
) -> tuple[PlotSessionConfig, dict[str, str]]:
    """把模板配置里的旧显示名重映射到当前数据（纯函数，D6 深拷贝）。

    返回 (resolved_config, remap)；remap 只记真实改名（旧名 != 新名）。
    loader 为 None / 无 get_channel_info / 无 var_names → 原样返回。
    """
    if loader is None:
        return config, {}
    gci = getattr(loader, "get_channel_info", None)
    var_names = getattr(loader, "var_names", None)
    if not callable(gci) or var_names is None:
        return config, {}

    resolver = _Resolver(gci, list(var_names))
    resolved = copy.deepcopy(config)
    remap: dict[str, str] = {}

    def _rename(name: str, identities: dict) -> str:
        new = resolver.resolve_name(name, identities.get(name))
        if new != name:
            prev = remap.get(name)
            if prev is not None and prev != new:
                # 同名曲线在不同子图带不同身份条目且解析分歧：保留首个
                logger.warning("身份解析分歧: %s → %s / %s", name, prev, new)
            else:
                remap[name] = new
                logger.info("身份重映射: %s → %s", name, new)
        return new

    for plot in _iter_unique_plots(resolved):
        identities = plot.curve_identities or {}
        local: dict[str, str] = {}
        new_curves = []
        for name in plot.curves:
            new = _rename(name, identities)
            if new != name:
                local[name] = new
            new_curves.append(new)
        new_invalid = []
        for name in plot.invalid_curves:
            new = _rename(name, identities)
            if new != name:
                local[name] = new
            new_invalid.append(new)
        plot.curves = new_curves
        plot.invalid_curves = new_invalid
        if local:
            # 键同步只按本子图实际改名表：孤儿条目（用户手改曲线名留下的
            # 身份/设置键）保留不删（D11），引擎按精确键查，孤儿无害。
            plot.curve_settings = {
                local.get(k, k): v for k, v in (plot.curve_settings or {}).items()
            }
            plot.curve_identities = {
                local.get(k, k): v for k, v in identities.items()
            }
    return resolved, remap