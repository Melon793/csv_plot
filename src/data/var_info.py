"""变量信息提取层 —— 与具体 Loader 解耦的统一信息模型。

设计要点：

1. **零 Qt 依赖**：本模块只产出纯数据结构与 Markdown 字符串，可脱离 GUI
   做单元测试。
2. **元数据零磁盘 I/O 且 O(1)**：MDF 路径的信息全部来自加载期已解析的内存块
   结构（实测六组全属性访问约 99 μs）；表格（CSV/Excel/DAT）路径只读 pandas
   的 dtype / categories / 长度等元数据，**不物化列数据**。两条路径因此都
   可直接在 UI 线程同步调用，窗口打开即有内容。
3. **统计与元数据分离**：统计需读磁盘（实测 18.8 ms / 428k 点，为元数据的
   190 倍），由 ``compute_stats()`` 单独提供，交给后台线程调用；结果由
   ``main_window.var_stats_cache`` 缓存，快照本身不持有统计（避免两份真相）。
"""

from dataclasses import dataclass, field
from math import isnan
from typing import Callable, Optional

import numpy as np

from src.core.config import VAR_INFO_ATTRIBUTION_MAX_LEN
from src.core.logger import get_logger
from src.data import mdf_attribution as attr
from src.data.metadata import (
    CONST,
    INVALID,
    UNKNOWN,
    VALID,
    is_mdf3_version,
    is_range_text_conversion,
)

logger = get_logger(__name__)

# 有效性 → 中文标签，与 NoHoverDelegate 的绿/橙/红色块语义一一对应
_VALIDITY_LABELS = {
    VALID: "有效",
    CONST: "常量",
    INVALID: "无效",
    UNKNOWN: "未知",
}

# numpy dtype.kind 中属于"可统计数值"的类别：布尔/整数/无符号/浮点/复数
_NUMERIC_KINDS = "biufc"
# 不可绘图也不可统计的类别：字节串/Unicode 串/对象
_STRING_KINDS = "SUO"


def _effective_numpy_dtype(dtype) -> np.dtype:
    """把 pandas dtype 归一为 numpy dtype；扩展 dtype 降级为 object。

    pandas 的字符串列 dtype 可能是 ``StringDtype`` 这类**扩展 dtype**，
    ``np.dtype()`` 对它直接抛 ``TypeError``（实测 pandas 3.x 下
    ``Categorical.categories.dtype`` 即为 ``StringDtype``）；而
    ``Series.to_numpy()`` 对这类列物化出的就是 object 数组。这里取同样的
    口径，但只读 dtype 元数据、不触碰数据本身。
    """
    try:
        return np.dtype(dtype)
    except TypeError:
        return np.dtype("O")


@dataclass(slots=True)
class VarStats:
    """单个变量的统计特征。

    ``std`` / ``nan_count`` / ``inf_count`` / ``finite_count`` 与 min/max/mean 是
    同一次单趟遍历的副产物，因此整体缓存（拆分成只存这四个字段会增加复杂度
    而无收益）。单条约 150 字节，256 条上限合计约 38 KB。

    三个计数的口径必须严格与字段名一致：``nan_count`` 只数 NaN，``inf_count``
    只数 Inf，``finite_count`` 只数有限值，且 ``nan_count + inf_count +
    finite_count == 样本总数``。两条统计路径（``_stats_from_array`` 与
    ``_stats_mdf``）必须给出相同的数字，否则同一概念在 CSV 与 MDF 下答案不同。
    """

    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    nan_count: int = 0
    inf_count: int = 0
    finite_count: int = 0
    computed: bool = False
    error: str = ""
    # 缓存失效令牌：等于提交任务时的 main_window._data_version。
    # reload 后版本递增，旧条目即使残留在缓存里也会因校验失败而被忽略。
    generation: int = 0
    # 诊断字段：该结果是否取自缓存（而非本会话现算）。缓存中的母本恒为
    # False，命中时由 UI 层复制副本并置 True（R4：不得原地改写缓存对象，
    # 否则母本被污染）。D′ 之后 UI 不再显示「（缓存）」后缀 —— 该后缀
    # 既被用户误读为"数值可能是旧的"，又会污染 Markdown 导出的数值字段。
    from_cache: bool = False
    # 任务在分块边界被取消。取消不是有效统计结果：缓存层必须拒绝此类
    # 条目（缺-5），否则「已取消」会被下一次命中当作终态展示，且
    # add_variables 只对 stats is None 的页面提交重算，用户将永远卡在
    # 「已取消」直到手动点「刷新统计」。
    cancelled: bool = False
    # 结果有效但需附带说明（如 MDF 通道组声明的点数多于实际可读点数，
    # 统计只覆盖了前 n 个样本）。刻意与 ``error`` 分开：``computed=True``
    # 与 ``error`` 非空并存会让 _on_stats_ready 的 R2 守卫与 stats_to_rows
    # 的有效性判断语义相混；note 只追加一行提示，不参与任何判断。
    note: str = ""


@dataclass(slots=True)
class VarInfoSnapshot:
    """变量元数据快照，**刻意不含统计结果**。

    统计由 ``main_window.var_stats_cache`` 单独管理；快照重建零磁盘 I/O 且
    O(1)（MDF 路径实测约 99 μs），缓存它反而会引入 sections（含 comment
    与枚举表）陈旧的风险与额外的 reload 清理负担。
    """

    name: str
    source_kind: str = ""
    original_name: str = ""
    dtype: str = ""
    length: int = 0
    unit: str = "-"
    validity: int = UNKNOWN
    is_numeric: bool = True
    is_enum: bool = False
    # 整列都是空值（pandas 推断为 0 个 categories 的 category dtype）。
    # 单独标记而非并入 is_numeric，是因为两者面向用户的解释完全不同：
    # 非数值列是"这列是文本"，全空列是"这列没数据"。
    all_empty: bool = False
    # 有序的「分组标题 → [(键, 值)]」，UI 直接遍历渲染，不感知数据源类型
    sections: dict = field(default_factory=dict)
    generation: int = 0


# ---------------------------------------------------------------------------
# 格式化辅助
# ---------------------------------------------------------------------------


def validity_label(validity: int) -> str:
    return _VALIDITY_LABELS.get(validity, f"未知({validity})")


def format_size(num_bytes) -> str:
    try:
        n = float(num_bytes)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return "-"


def _fmt(value, suffix: str = "", dash: str = "-") -> str:
    """把块属性值格式化为可显示字符串：None/空白 → dash。

    bytes 在 loader 的 _block_attrs 中已解码，此处再兜一层以覆盖直接传入
    块对象的场景。
    """
    if value is None:
        return dash
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, str):
        s = value.strip()
        return s if s else dash
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        # 不用 ``value != value`` 的 NaN 惯用式：虽然等价，但看上去像笔误
        if isnan(value):
            return dash
        return f"{value:.6g}{suffix}"
    return f"{value}{suffix}"


def _fmt_rate(hz) -> str:
    """采样率带"平均"注记：该值由组首尾时间戳推算，变速采样会失真。"""
    if hz is None:
        return "-"
    return f"{hz:.4g} Hz（平均，按组首尾时间戳推算）"


def _fmt_raster(seconds) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.6g} s（标称 {1.0 / seconds:.4g} Hz）" if seconds > 0 else f"{seconds:.6g} s"


def _constant_label(table_name: str, mdf_version: Optional[str], value, prefix: str) -> str:
    """按 MDF 版本从 asammdf 常量表取可读标签。

    两版常量表语义完全不同（实测 ct=11 在 MDF4 是 BITFIELD、MDF3 是 TABX；
    ct=7 在 MDF4 是 TABX、MDF3 是 EXPO），因此必须按版本取表。

    延迟 import asammdf：展示常量仅在打开信息页时才需要，避免 CSV-only
    场景在导入本模块时就拉起 asammdf（项目有启动性能优化历史）。
    """
    if value is None:
        return "-"
    try:
        from asammdf.blocks import v2_v3_constants as v3c
        from asammdf.blocks import v4_constants as v4c

        module = v3c if is_mdf3_version(mdf_version) else v4c
        table = getattr(module, table_name, None) or {}
    except Exception:
        logger.debug("加载 asammdf 常量表 %s 失败", table_name, exc_info=True)
        return f"{prefix}{value}"
    if not table:
        # 该版本无此常量表（如 MDF3 无 SYNC_TYPE_TO_STRING），回退数值
        return f"{prefix}{value}"
    label = table.get(value)
    return str(label) if label is not None else f"UNKNOWN({value})"


def conversion_label(mdf_version: Optional[str], ct) -> str:
    return _constant_label("CONVERSION_TYPE_TO_STRING", mdf_version, ct, "CT=")


# channel_type_label / sync_type_label / source_type_label / bus_type_label
# 四个包装函数已随 block 合并移除（它们唯一的调用点在被删掉的
# 「通道 (CNBLOCK)」与「源信息 (SBLOCK)」行里）。版本感知的取表机制
# 仍由 _constant_label 保留，并继续被 conversion_label 使用。


# ---------------------------------------------------------------------------
# 快照构建（零磁盘 I/O）
# ---------------------------------------------------------------------------


def build_snapshot(loader, var_name: str, generation: int = 0) -> VarInfoSnapshot:
    """构建变量元数据快照：单一入口，按 ``LOADER_TYPE`` 分派。

    全程零磁盘 I/O。变量不存在时抛 ``KeyError``，由调用方决定提示方式。
    """
    kind = getattr(loader, "LOADER_TYPE", "") or ""
    if kind == "mdf":
        return _from_mdf(loader, var_name, generation)
    return _from_tabular(loader, var_name, generation, kind or "tabular")


def _from_tabular(loader, var_name, generation, kind) -> VarInfoSnapshot:
    df = getattr(loader, "df", None)
    if df is None or var_name not in df.columns:
        raise KeyError(f"变量 '{var_name}' 不存在")

    series = df[var_name]
    # 全程只读 pandas 元数据，刻意**不**调 series.to_numpy()：物化对 category
    # 列会生成完整的 object 数组（实测 5M 行文本 categories 18.98 ms、全空列
    # 6.23 ms），对 float 列则还要再付一次 np.isnan 全列扫描（10M 行 1.86 ms）。
    # 这些都是 O(N)，会打破"快照可在 UI 线程同步调用"的前提 —— 批量打开
    # VAR_INFO_MAX_TABS=50 个标签页时实测累计 95 ms 以上的 UI 冻结。
    length = len(series)
    s_dtype = series.dtype

    all_empty = False
    if str(s_dtype) == "category":
        # 整列为空的 CSV/Excel 列被 pandas 推断为 category dtype 且 categories
        # 为空，dtype 层面的 "category" 完全丢失了"这列其实全是空值"这一事实。
        # 这里做 O(1) 判定（只读 categories 长度）：刻意不用
        # pd.isna(series).all()，后者在千万行的列上约需 50 ms。
        try:
            categories = series.cat.categories
            all_empty = len(categories) == 0
        except Exception:
            logger.debug("判定 %s 是否为全空列失败", var_name, exc_info=True)
            categories = None
            all_empty = False
        # categories 为空时其 dtype 取决于 pandas 的推断（实测真实 CSV 为
        # object，而对全 NaN 的 float 列显式 astype('category') 会得到
        # float64），一律按 object 处理：否则"全空列"会被判成数值列，白白
        # 提交一个注定返回"全部为 NaN/Inf"的统计任务，且 all_empty 的专属
        # 文案永远出不来
        dtype = (
            np.dtype("O")
            if all_empty or categories is None
            else _effective_numpy_dtype(categories.dtype)
        )
    else:
        dtype = _effective_numpy_dtype(s_dtype)

    is_numeric = dtype.kind in _NUMERIC_KINDS

    validity = (getattr(loader, "df_validity", None) or {}).get(var_name, UNKNOWN)
    unit = (getattr(loader, "units", None) or {}).get(var_name, "-") or "-"
    time_formats = getattr(loader, "time_channels_info", None) or {}
    time_fmt = time_formats.get(var_name, "")

    snap = VarInfoSnapshot(
        name=var_name,
        source_kind=kind,
        original_name=var_name,
        dtype=str(dtype),
        length=length,
        unit=unit,
        validity=validity,
        is_numeric=is_numeric,
        is_enum=False,
        all_empty=all_empty,
        generation=generation,
    )

    # NaN 计数刻意**不放进快照**：它本质是统计量而非元数据，算它必须全列
    # 扫描（O(N)，正是上面要规避的开销），而 compute_stats 已在后台线程算过
    # 一遍并以「NaN 数」行呈现。在 UI 线程再算一遍除了阻塞主线程，还会造成
    # 同一棵树里「NaN 数量」与「NaN 数」两个标签并存、且 MDF 快照从来不
    # 提供该值的跨格式不一致（非浮点列更是恒报 0，对含 NaN 的文本列属于
    # 误导）。移除后 NaN 数从"打开即有"变成"统计回填后才有"，与
    # min/max/mean 的行为一致，用户心智模型反而更统一。
    column_rows = [
        ("是否时间格式列", "是" if var_name in time_formats else "否"),
    ]
    if time_fmt:
        column_rows.append(("时间格式", str(time_fmt)))
    if all_empty:
        column_rows.append(
            ("说明", f"该列全部为空值（共 {length} 行），"
                     "无有效样本（不支持统计与绘图）")
        )
    elif not is_numeric:
        column_rows.append(("说明", f"非数值列（{dtype}），不支持统计与绘图"))

    snap.sections = {
        "基本信息": [
            ("变量名", var_name),
            ("数据类型", str(dtype)),
            ("数据点总数", f"{length}"),
            ("单位", unit),
            ("有效性", validity_label(validity)),
        ],
        "列信息": column_rows,
        "文件信息": _tabular_file_rows(loader),
    }
    return snap


def _tabular_file_rows(loader) -> list:
    rows = [("文件路径", _fmt(getattr(loader, "path", None)))]
    size = getattr(loader, "file_size", None)
    if size is None:
        try:
            import os

            p = getattr(loader, "path", None)
            size = os.path.getsize(p) if p and os.path.exists(p) else None
        except Exception:
            size = None
    rows.append(("文件大小", format_size(size) if size else "-"))
    rows.append(("行数", _fmt(getattr(loader, "row_count", None))))
    rows.append(("列数", _fmt(getattr(loader, "column_count", None))))
    rows.append(("时间轴标签", _fmt(getattr(loader, "time_axis_label", None))))
    return rows


def _one_line(text: str) -> str:
    r"""把多行注释压成单行展示。

    信息页行高固定（VAR_INFO_ROW_HEIGHT），嵌换行符会被裁剪成看不到内容
    的黑盒；而实测 v4 合成文件与部分 CANape 注释确实带换行（如
    ``'E2E校验\nDataID ='``），故统一用 " / " 拼接。
    """
    return " / ".join(part.strip() for part in (text or "").splitlines() if part.strip())


def _clip(text: str, max_len: int = 0) -> str:
    """归属/注释行的展示截断；全值由信息页 tooltip 承担。"""
    text = (text or "").strip()
    limit = max_len or VAR_INFO_ATTRIBUTION_MAX_LEN
    if limit and len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _hd_attribution_rows(header_comment, max_len: int = 0) -> list[tuple[str, str]]:
    """试验级归属（数据库 / 试验 / 工作空间 / 设备清单 / 程序描述 / WP / RP）。

    实测 5/5 个 CANape/INCA 文件的 HD 注释都带这一段 ``Key: Value``，
    对定位“这条曲线来自哪套标定量”比整段文件注释有用得多。
    """
    parsed = attr.parse_hd_comment(header_comment)
    rows = []
    for raw_key, label in attr.HD_ATTRIBUTION_KEYS:
        value = parsed.get(raw_key, "")
        if value:
            # 设备清单可达十几个名字（实测 'VCU,ECU,CAN-Monitoring:1,…'），
            # 不截断会把「值」列撑宽到统计行看不见
            rows.append(
                (
                    f"{label}（{raw_key}）",
                    _clip(_one_line(attr.repair_text(value)), max_len),
                )
            )
    return rows


def _from_mdf(loader, var_name, generation) -> VarInfoSnapshot:
    # get_channel_info 全部取自加载期已解析的内存块结构，零磁盘 I/O
    info = loader.get_channel_info(var_name)
    meta = info["meta"]
    version = info["version"]
    ch = info["channel"]
    cg = info["channel_group"]
    conv = info["conversion"]
    header = info["header"]
    tb = info["time_base"]
    fi = info["file"]

    # 聚合改名场景（MDF 跨通道组重名会被改写为 Name_G{gi}）：
    # 同时显示聚合显示名与原始通道名，避免用户按原始名搜索不到变量。
    original = meta.original_name or var_name
    renamed = original != var_name

    # 「基本信息」由原先的五个块（基本信息 / 通道 CNBLOCK / 通道组 CGBLOCK /
    # 源信息 SBLOCK / 时间基准）合并而来，行序按用户实际关注顺序排列，
    # 而不再按 MDF 块结构排列 —— 用户查一个变量时想的是“它是什么、
    # 多少点、什么量程、什么时间跨度”，不是“CN 块里第几个字段”。
    #
    # 删掉的行全部经 data/ 下 7 个真实通道逐行核实，不是凭印象裁剪：
    #   源信息 (SBLOCK) —— 当时实测 7/7 通道均为单行“该通道无源信息块”
    #       占位。**该结论仅对 sampled MDF3 成立**：本次补测发现 MDF4 的 SI 块
    #       100% 有值（mf4 实测 name=ECU 节点 99.9%、path=设备 100%），故已以
    #       「归属信息」块的形式回归，且比原 6 行更能回答“这变量属于谁”
    #   同步类型 / 字节偏移 / 位偏移 / 采集名 / 采集源 —— 实测全为空
    #   循环数 (cycles_nr) —— 实测恒等于「数据点总数」，纯冗余
    #   master 通道名 —— 实测恒为 "time"
    #   块地址 / 通道组索引 / 通道索引 / 组内通道数 / 单条记录字节数
    #       —— 底层调试字段；组索引已含在下面的「名称改写原因」里
    # 两行看似删了其实无损：有效性在页头摘要与色块已展示（且
    # snapshot_to_markdown 的首行仍带它），转换类型在「转换规则 (CCBLOCK)」
    # 块已展示 —— 在树内重复只会让合并后的块更长。
    #
    # 标称采样间隔 / 有效采样率是刻意保留的（合并方案 2-B）：它们是全
    # 软件唯一展示点（grep 过 src/ 全部 .py），而实测存在标称 0.001 s
    # （1000 Hz）但有效仅 17.68 Hz 的严重偏差 —— 正是
    # VarMetadata.sampling_rate_hz 语义修复要在 UI 上体现的场景，删掉
    # 等于让那次修复彻底不可见。
    # 通道注释：实测 v4 真实文件直出整段 XML（<CNcomment …><TX>…</TX>…），
    # v3 的长文本在 comment 与 description 两处各一份、合成文件只有
    # description，且 CANape 写的中文被单字节解码成乱码 —— 三者统一交给
    # mdf_attribution 解析与回转。
    aux_text = attr.channel_aux_text(ch)

    basic = [
        ("变量名（聚合显示名）" if renamed else "变量名", var_name),
    ]
    if renamed:
        basic.append(("原始通道名", original))
        basic.append(
            (
                "名称改写原因",
                f"跨通道组重名（组 {meta.group_index}），已追加组序号后缀以便区分",
            )
        )
    basic += [
        ("单位", _fmt(meta.unit)),
        ("通道注释", _fmt(_one_line(aux_text))),
        ("记录 ID", _fmt(cg.get("record_id"))),
        ("组注释", _fmt(cg.get("comment"))),
        ("数据类型", _fmt(info["dtype"])),
        ("数据点总数", f"{meta.sample_count}"),
    ]
    if not meta.sample_count:
        # 不依赖统计结果就把空组说清：用户无需点"刷新统计"即可知道
        # 0 点是数据文件本身的特性，而非软件读取失败。
        basic.append(
            (
                "数据点总数说明",
                "所属通道组 cycles_nr=0 且无数据块，通常为采集时未写入的预留组",
            )
        )
    basic += [
        ("是否枚举", "是" if meta.is_enum else "否"),
        ("位宽 (bit_count)", _fmt(ch.get("bit_count"), " bit")),
        ("精度 (precision)", _fmt(ch.get("precision"))),
        ("下限 (lower_limit)", _fmt(ch.get("lower_limit"))),
        ("上限 (upper_limit)", _fmt(ch.get("upper_limit"))),
        ("标称采样间隔", _fmt_raster(tb.get("nominal_raster_s"))),
        ("有效采样率", _fmt_rate(tb.get("effective_rate_hz"))),
        ("起始时间戳", _fmt(tb.get("time_min"), " s")),
        ("结束时间戳", _fmt(tb.get("time_max"), " s")),
    ]
    if not info["is_numeric"]:
        basic.append(("说明", "该通道为字符串类型，不支持绘图与统计"))

    sections: dict[str, list[tuple[str, str]]] = {"基本信息": basic}

    # 归属信息紧跟基本信息；一行都没提取到时**不建空块**（D3），
    # 因此 CSV/Excel 路径与无归属信息的 MDF 变量完全不受影响。
    attribution_rows = attr.build_attribution_rows(info)
    if attribution_rows:
        sections["归属信息"] = attribution_rows

    sections["转换规则 (CCBLOCK)"] = _conversion_rows(version, conv, meta)

    # 文件注释同样不得直出 XML：先抽 TX + 乱码回转 + 截断，再把
    # 其中的 Key: Value 拆成独立行（见 _hd_attribution_rows）。
    hd_text = attr.repair_text(attr.extract_tx(header.get("comment")))
    file_rows = [
        ("MDF 版本", _fmt(version)),
        ("文件路径", _fmt(fi.get("path"))),
        ("文件大小", format_size(fi.get("size"))),
        ("通道组数", _fmt(fi.get("group_count"))),
        ("变量总数", _fmt(fi.get("var_count"))),
        ("作者", _fmt(header.get("author"))),
        ("部门", _fmt(header.get("department"))),
        ("项目", _fmt(header.get("project"))),
        ("主题", _fmt(header.get("subject"))),
        ("起始时间", _fmt(header.get("start_time_string"))),
        ("文件注释", _fmt(_clip(_one_line(hd_text)))),
    ]
    file_rows += _hd_attribution_rows(header.get("comment"))
    sections["文件信息 (HDBLOCK)"] = file_rows

    enum_map = info.get("enum_map") or meta.enum_map
    if enum_map:
        sections["枚举映射 (CCBLOCK)"] = _enum_rows(enum_map)

    return VarInfoSnapshot(
        name=var_name,
        source_kind="mdf",
        original_name=original,
        dtype=_fmt(info["dtype"], dash=""),
        length=int(meta.sample_count or 0),
        unit=_fmt(meta.unit),
        validity=meta.validity,
        is_numeric=bool(info["is_numeric"]),
        is_enum=bool(meta.is_enum),
        sections=sections,
        generation=generation,
    )


def _conversion_rows(version: Optional[str], conv: dict, meta) -> list:
    if not conv:
        return [("说明", "该通道无转换块 (CCBLOCK)，原始值即物理值")]

    ct = conv.get("conversion_type")
    rows = [
        ("转换类型", conversion_label(version, ct)),
        ("转换类型码", _fmt(ct)),
        ("物理单位", _fmt(conv.get("unit"))),
        ("转换名", _fmt(conv.get("name"))),
    ]

    # 线性转换（MDF4 的 a/b）与有理/公式转换（MDF3 的 P1..P7）按存在性展示
    if conv.get("a") is not None or conv.get("b") is not None:
        rows.append(("线性系数 a", _fmt(conv.get("a"))))
        rows.append(("线性系数 b", _fmt(conv.get("b"))))
    formula = conv.get("formula")
    if formula:
        rows.append(("公式", _fmt(formula)))
    params = [(k, conv.get(k)) for k in ("P1", "P2", "P3", "P4", "P5", "P6", "P7")]
    params = [(k, v) for k, v in params if v is not None]
    if params:
        rows.append(("有理/多项式系数", ", ".join(f"{k}={_fmt(v)}" for k, v in params)))

    ref_nr = conv.get("ref_param_nr")
    if ref_nr:
        rows.append(("枚举条目数 (ref_param_nr)", _fmt(ref_nr)))
    rows.append(("注释", _fmt(conv.get("comment"))))

    # 安全降级：is_enum 为真但文本表提取失败时，绘图仍走 raw=True 取码值
    # （避免拿到字符串数组导致崩溃），此处显式告知用户当前看到的是码值。
    if meta.is_enum and not meta.enum_map:
        if is_range_text_conversion(version, ct):
            # RTABX 按区间而非码值映射，与 dict[int, str] 契约不兼容（详见
            # extract_enum_map 的 docstring）。说“提取失败”会让用户以为
            # 解析出了 bug，实际上是尚不支持这种表结构。
            hint = (
                "范围文本表（RTABX）按区间而非码值映射，暂不支持展示；"
                "当前为原始码值"
            )
            rows.append(("提示", hint))
        else:
            rows.append(("提示", "文本表提取失败，当前展示原始码值"))
    return rows


def _enum_rows(enum_map: dict) -> list:
    from src.core.config import VAR_INFO_ENUM_DISPLAY_LIMIT

    rows = []
    limit = VAR_INFO_ENUM_DISPLAY_LIMIT
    for i, (k, v) in enumerate(sorted(enum_map.items())):
        if i >= limit:
            rows.append(("…", f"共 {len(enum_map)} 条，仅显示前 {limit} 条"))
            break
        rows.append((str(k), str(v)))
    return rows


# ---------------------------------------------------------------------------
# 统计计算（重活，供后台线程调用）
# ---------------------------------------------------------------------------


def compute_stats(
    loader,
    var_name: str,
    should_cancel: Optional[Callable[[str], bool]] = None,
) -> VarStats:
    """计算统计特征。**不写缓存** —— 缓存写入由 UI 层在收到信号时完成，
    避免子线程触碰 main_window 属性。
    """
    kind = getattr(loader, "LOADER_TYPE", "") or ""
    try:
        if kind == "mdf":
            return _stats_mdf(loader, var_name, should_cancel)
        return _stats_tabular(loader, var_name)
    except Exception as e:  # noqa: BLE001 - 后台线程须把任何异常转成可展示的错误
        logger.debug("计算 %s 统计失败", var_name, exc_info=True)
        return VarStats(error=f"{type(e).__name__}: {e}")


def _stats_tabular(loader, var_name) -> VarStats:
    df = getattr(loader, "df", None)
    if df is None or var_name not in df.columns:
        return VarStats(error="变量不存在或数据已释放")
    return _stats_from_array(df[var_name].to_numpy())


def _stats_from_array(a: np.ndarray) -> VarStats:
    """从内存数组计算统计（CSV / Excel 路径）。

    刻意**不做** ``astype(np.float64)`` 融合：实测 1000 万行 float32 下
    ``np.nanmin/nanmax/nanmean`` 三次遍历 9.75 ms，而 astype 单趟融合
    21.16 ms（慢 2.2 倍，其中 astype 独占 6.07 ms）。精度方面 numpy 采用
    pairwise summation，float32 均值的相对误差实测仅 3e-8，远小于 float32
    自身的表示精度 1.2e-7，因此 float64 累加是虚假收益。
    """
    if a.size == 0:
        return VarStats(error="数据为空")
    if a.dtype.kind not in _NUMERIC_KINDS:
        return VarStats(error=f"非数值类型（{a.dtype}），不适用统计")

    is_float = a.dtype.kind == "f"
    if is_float:
        # 三个计数各自独立、口径一致：nan_count 只数 NaN，inf_count 只数 Inf，
        # finite_count 只数有限值，三者之和恒等于 a.size。旧写法
        # finite_count = size - nan_count 把 Inf 算成了有限值（实测
        # [1, 2, inf, nan] 得 finite_count=3 而真值为 2）。
        nan_count = int(np.count_nonzero(np.isnan(a)))
        inf_count = int(np.count_nonzero(np.isinf(a)))
        finite_count = int(np.count_nonzero(np.isfinite(a)))
    else:
        # 整数/布尔列不可能出现 NaN 或 Inf，全部样本都有效
        nan_count = inf_count = 0
        finite_count = int(a.size)

    if finite_count == 0:
        # 必须显式判空，不能依赖 nanmin/nanmax 抛 ValueError：
        # numpy 2.4.6 实测对全 NaN 数组只发 RuntimeWarning（"All-NaN
        # slice encountered"）并**返回 nan**，于是四个统计量全为 nan 而
        # computed=True。后果：UI 的 min/max/mean/std 行显示 "nan"，
        # 且 _lookup_cache 把 computed=True 视为有效结果，这条 nan 会被
        # 写入缓存并长期复用。与 _stats_mdf 的 ``if n == 0`` 守卫对齐。
        return VarStats(
            nan_count=nan_count,
            inf_count=inf_count,
            error="全部为 NaN/Inf，无有效样本",
        )

    if inf_count:
        # 有 Inf 时必须先剔除再统计：nanmin/nanmax 会把 inf 当极值，nanmean
        # 得 inf，nanstd 因 inf-inf 得 nan（实测 [1, 2, inf, nan] 得
        # max=inf, mean=inf, std=nan），而 computed=True 会让这些脏值进缓存
        # 长期复用（与上面全 NaN 同源）。
        #
        # 对表格路径而言这是**纵深防御**：base_loader._postprocess_columns 在
        # 加载时已把 ±inf 统一清成 NaN（实测 CSV 写 "inf"/"-inf" 读回均为 nan），
        # 真实文件走不到这个分支。真正需要它的是 _stats_mdf——MDFLazyLoader
        # 不继承 BaseDataLoader，无此清理，inf 原样存活。两条路径仍必须口径
        # 一致，否则同一概念在 CSV 与 MDF 下给出不同答案。
        #
        # 只在真的存在 Inf 时才付这次布尔索引的代价：无 Inf 的常见路径
        # 保持原有零额外开销（本函数的性能取舍见 docstring）。
        a = a[np.isfinite(a)]

    try:
        with np.errstate(invalid="ignore"):
            mn = float(np.nanmin(a))
            mx = float(np.nanmax(a))
            mean = float(np.nanmean(a))
            std = float(np.nanstd(a))
    except ValueError:
        # 老版 numpy 的全 NaN 路径。上面的 finite_count 守卫已覆盖，此处
        # 仅作纵深防御，避免降级 numpy 时静默回归。
        return VarStats(
            nan_count=nan_count,
            inf_count=inf_count,
            error="全部为 NaN/Inf，无有效样本",
        )

    return VarStats(
        min=mn,
        max=mx,
        mean=mean,
        std=std,
        nan_count=nan_count,
        inf_count=inf_count,
        finite_count=finite_count,
        computed=True,
    )


def _stats_mdf(loader, var_name, should_cancel) -> VarStats:
    """分块流式统计 MDF 通道。

    恒定 ``raw=False`` 取物理值：绘图路径用 ``raw=is_enum``（枚举取码值配合
    文本标签），但统计必须基于物理值，否则 min/max/mean 得到的是无意义的
    枚举码。

    分块的两个目的：
    1. 把单次 ``_access_lock`` 持有时间压到 ≤20 ms，UI 线程并发绘图无停顿
    2. 每块之间检查取消标志，实现即时中断（关闭标签页 / reload）

    跨块累加器使用 float64：这与 ``_stats_from_array`` 的结论不矛盾 ——
    那里是单数组交给 numpy 内置的 pairwise summation，这里是跨多块手动累加，
    朴素 float32 累加会随块数线性放大误差。

    ``total`` 取自文件头的 ``cycles_nr``，**不可全信**（与上面 ``total <= 0``
    分支同根）：声明点数可能多于实际可读点数。因此循环按**实际返回量**
    推进 offset（按请求量推进会越过中间样本，在统计里挖掉一段连续数据），
    并在提前读到末尾时把截断事实如实写进 ``note``，而不是静默拿部分数据
    冒充完整结果。
    """
    from src.core.config import MDF_STATS_CHUNK_SIZE

    meta = loader.get_metadata(var_name)
    if meta is None:
        return VarStats(error=f"变量 '{var_name}' 不存在")

    total = int(meta.sample_count or 0)
    if total <= 0:
        # sample_count 依赖 CGBLOCK 的 cycles_nr。实测部分 MDF3 文件（如
        # Chery .dat 的 AI50 组）的 cycles_nr 在文件头里就是 0 且
        # data_blocks 为空，即采集时未写入数据的预留组。此时退回
        # 一次性读取做最后确认，并把结论如实告知用户。
        try:
            samples = loader.get_samples_chunked(var_name, 0, -1)
        except Exception as e:  # 含 loader 已关闭时的 KeyError
            return VarStats(error=str(e))
        arr = np.asarray(samples)
        if arr.size == 0:
            return VarStats(
                error="所属通道组无数据（cycles_nr=0 且无数据块），"
                      "通常为采集时未写入的预留组"
            )
        return _stats_from_array(arr)

    mn = np.inf
    mx = -np.inf
    total_sum = 0.0
    total_sumsq = 0.0
    n = 0
    nan_count = 0
    inf_count = 0
    offset = 0
    # 实际可读样本少于 cycles_nr 声明值时的截断位置（None = 未截断）
    truncated_at = None

    while offset < total:
        if should_cancel is not None and should_cancel(var_name):
            # cancelled 标记让缓存层识别并拒绝（缺-5）：worker 只 emit
            # 运行中的那条，此处返回值会经 _on_stats_ready 写缓存
            return VarStats(error="已取消", cancelled=True)

        count = min(MDF_STATS_CHUNK_SIZE, total - offset)
        try:
            samples = loader.get_samples_chunked(var_name, offset, count)
        except KeyError as e:
            # loader 已关闭（改进 I 保证统一抛 KeyError 而非 AttributeError）
            return VarStats(error=str(e))
        except Exception as e:
            return VarStats(error=f"读取失败: {e}")

        arr = np.asarray(samples)
        if arr.dtype.kind in _STRING_KINDS:
            # 枚举通道的物理值即文本标签（实测 MDF3 TABX 返回 |S14），
            # 对文本求 min/max/mean 无意义；引导用户去看枚举映射表。
            if getattr(meta, "is_enum", False):
                return VarStats(
                    error="枚举通道：物理值为文本标签（"
                          f"{arr.dtype}），不适用数值统计；"
                          "码值含义请见「枚举映射 (CCBLOCK)」"
                )
            return VarStats(error=f"字符串通道（{arr.dtype}），不适用统计")
        if arr.dtype.kind not in _NUMERIC_KINDS:
            return VarStats(error=f"非数值类型（{arr.dtype}），不适用统计")

        got = int(arr.size)
        if got == 0:
            # cycles_nr 声明的点数多于实际可读点数。不得静默 break 后当
            # 完整数据出结果：页面「基本信息」显示的 sample_count 是 total，
            # 而统计只覆盖了 offset 之前的样本，两者自相矛盾却均标为有效
            truncated_at = offset
            break

        f = arr.astype(np.float64, copy=False)
        bad = ~np.isfinite(f)
        bad_count = int(np.count_nonzero(bad))
        if bad_count:
            # NaN 与 Inf 分开计数：旧写法把 bad_count 全归给 nan_count，
            # 使 UI 显示的 "NaN 数" 大于真实 NaN 个数（实测 [1,2,inf,nan]
            # 得 nan_count=2 而真值为 1），用户无从判断数据到底出了什么。
            nan_count += int(np.count_nonzero(np.isnan(f)))
            inf_count += int(np.count_nonzero(np.isinf(f)))
            f = f[~bad]
        if f.size:
            mn = min(mn, float(f.min()))
            mx = max(mx, float(f.max()))
            total_sum += float(f.sum())
            total_sumsq += float((f * f).sum())
            n += int(f.size)

        # 按**实际返回量**推进：按请求量 count 推进会在 asammdf 返回不足
        # count 时越过中间 count-got 个样本，统计里凭空挖掉一段连续数据
        # （比尾部截断更糟：尾部至少前缀连续，这是中间挖洞）
        offset += got
        if got < count:
            # 已读到真实末尾，无需再试下一块
            truncated_at = offset
            break

    if n == 0:
        if truncated_at == 0:
            # 第一块就读不到数据：真实原因是"读不出样本"，不是"全是 NaN"。
            # 沿用旧文案会把文件头不可信误报成数据质量问题，用户会去查
            # 根本不存在的 NaN
            return VarStats(
                error=f"通道组无可读样本（文件头声明 {total} 点，实际 0 点）"
            )
        return VarStats(
            nan_count=nan_count,
            inf_count=inf_count,
            error="全部为 NaN/Inf，无有效样本",
        )

    mean = total_sum / n
    # 方差用 E[x²]-E[x]²，浮点舍入可能得到极小负数，需夹到 0
    variance = max(total_sumsq / n - mean * mean, 0.0)
    note = ""
    if truncated_at is not None:
        # 结果仍然有效，但适用范围小于页面宣称的 total，必须告知
        note = (
            f"仅统计到前 {truncated_at} 个样本"
            f"（通道组声明 {total} 个，其后无可读数据）"
        )
    return VarStats(
        min=float(mn),
        max=float(mx),
        mean=mean,
        std=float(np.sqrt(variance)),
        nan_count=nan_count,
        inf_count=inf_count,
        finite_count=n,
        computed=True,
        note=note,
    )


# ---------------------------------------------------------------------------
# Markdown 导出（改进 E）
# ---------------------------------------------------------------------------


def _fmt_stat(v, digits: int = 6) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{digits}g}"
    except (TypeError, ValueError):
        return str(v)


def stats_to_rows(stats: Optional[VarStats]) -> list:
    """把统计结果渲染为 [(指标, 值)]，未计算/出错时给出占位而不抛异常。"""
    if stats is None:
        return [("状态", "计算中…")]
    if not stats.computed:
        return [("状态", stats.error or "未计算")]
    # D′：缓存来源（from_cache）不再写入数值行。旧实现只给 min/max/mean
    # 三行加「（缓存）」后缀，一是同源同趟的标准差/计数行不一致，二是
    # snapshot_to_markdown 复用本函数，复制进报告会变成 `3.5（缓存）`
    # 破坏数值字段，三是用户极易读成"这个数可能是旧的"。
    rows = [
        ("最小值", _fmt_stat(stats.min)),
        ("最大值", _fmt_stat(stats.max)),
        ("平均值", _fmt_stat(stats.mean)),
        ("标准差", _fmt_stat(stats.std)),
        ("有效样本数", f"{stats.finite_count}"),
        ("NaN 数", f"{stats.nan_count}"),
    ]
    # Inf 单独成行且**仅在出现时**显示：绝大多数正常数据不该多一行噪声，
    # 而一旦有 Inf，用户需要立即知道为何“最大值不是那个 inf”。
    if stats.inf_count:
        rows.append(("Inf 数", f"{stats.inf_count}"))
    # note 放在最后：它是对上面数值**适用范围**的限制说明（如"仅统计到前
    # N 个样本"），用户读完数字再看到它才不会先入为主
    if stats.note:
        rows.append(("说明", stats.note))
    return rows


def snapshot_to_markdown(snap: VarInfoSnapshot, stats: Optional[VarStats] = None) -> str:
    """导出单个变量的 Markdown（便于粘贴到评审报告 / 缺陷单）。

    统计未回填时输出"计算中…"，任何时刻调用都不报错。
    """
    lines = [f"## 变量信息：{snap.name}", ""]
    lines.append(
        f"- 数据源: {snap.source_kind or '-'} | 单位: {snap.unit} | "
        f"点数: {snap.length} | dtype: {snap.dtype or '-'} | "
        f"有效性: {validity_label(snap.validity)}"
    )
    if snap.original_name and snap.original_name != snap.name:
        lines.append(f"- 原始通道名: {snap.original_name}")
    lines.append("")

    lines.append("### 统计特征")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    for k, v in stats_to_rows(stats):
        lines.append(f"| {k} | {_md_escape(v)} |")
    lines.append("")

    for title, rows in snap.sections.items():
        if not rows:
            continue
        lines.append(f"### {title}")
        lines.append("| 属性 | 值 |")
        lines.append("|---|---|")
        for k, v in rows:
            lines.append(f"| {k} | {_md_escape(v)} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def snapshots_to_markdown(items: list) -> str:
    """导出多个变量，以水平分隔线拼接。items 为 [(snapshot, stats), ...]。"""
    parts = [snapshot_to_markdown(snap, stats) for snap, stats in items]
    return "\n---\n\n".join(p.rstrip() + "\n" for p in parts)


def _md_escape(text) -> str:
    """转义 Markdown 表格中的破坏性字符：竖线与换行。"""
    s = str(text) if text is not None else "-"
    return s.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ")
