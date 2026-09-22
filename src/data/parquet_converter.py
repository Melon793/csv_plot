"""CSV/Excel → 临时 parquet 转换器（P3）。

两条链（§2.4.1 设计推论）：
- CSV：probe_csv_schema（P1 抽出的头部+200 行样本探测）→ 行数 oracle →
  polars 显式 schema 解析（S1–S5 语义复刻，§2.4.2）→ inf 清理 / 有效性
  规则表 / 枚举码值（D8）→ 写 parquet + meta.json。
- Excel：ExcelDataLoader 现有 calamine 链读成 pandas →（构造内部已完成
  _postprocess_columns）→ 逐列转 polars 写 parquet。不得用 polars 自己的
  Excel 读法（那会引入第二套类型推断，等价性无从保证）。

依赖声明（D11）：polars>=1.44.2,<2.0，函数内惰性导入——模块级 import
会让没装 polars 的环境（理论不存在，但打包裁剪可能）在 import 链上崩。
"""

from __future__ import annotations

import codecs
import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.core.logger import get_logger
from src.data.base_loader import BaseDataLoader
from src.data.loader import probe_csv_schema

logger = get_logger("data.parquet_converter")

# D8 枚举判据：category/文本列 n_unique <= 200 → 枚举列（Int32 码 +
# meta 标签表）。阈值依据：光标标签可读性 + meta.json 体积（最坏
# 888 列 × 200 条标签仍在数百 KB 量级）。
ENUM_LABEL_MAX = 200

# S1：pandas 默认 NA 集里不在 _NA_VALUES(30 项) 的两个 token（"<NA>" 与
# 小写 "null"）。补齐成 32 项后实测 48 个候选 token 两侧判 NA 31:31
# 完全一致（v2.1，§2.4.2 S1）。必须引用 _NA_VALUES 常量——抄一份将来必分叉。
_NULL_VALUES: list[str] = [*BaseDataLoader._NA_VALUES, "<NA>", "null"]

# row_group_size（v2.1 定向实测：200k 行 × 50 列 float32，30 次单列读中位数）：
#   默认(单组) 37.02MB / 0.51ms
#   16384      37.11MB / 0.47ms（最快）
#   262144     37.03MB / 0.81ms
#   1024       39.29MB / 2.39ms（慢 4.7×、体积 +6%）
# → 用小值是无条件劣化；取 16384（P3 又用合成大文件复核过，见施工报告）。
ROW_GROUP_SIZE = 16384

# 顺序扫描行数 oracle 的块大小（实测 582MB / 0.31s，成本可忽略）
_SCAN_CHUNK = 4 << 20

# 转码中间文件名（落在 outdir，绝不落源文件旁——源文件可能在只读网盘）
_TRANSCODED_NAME = "_transcoded_utf8.csv"


class ParquetConversionError(Exception):
    """转换失败（D10 回退链的触发信号：工厂捕获后清目录 + 回退内存 loader）。"""


@dataclass
class ConvertResult:
    """转换产物描述。columns 含全部列（含日期列，loader 侧按语义过滤）。"""

    parquet_path: str
    meta_path: str
    columns: list[str]
    rows: int
    elapsed_s: float
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def convert_to_parquet(
    src_path: str,
    *,
    desc_rows: int | None = 0,
    sep: str = ",",
    has_unit: bool | None = True,
    encoding: str | None = None,
    sheet_name: str | int = 0,
    is_excel: bool = False,
    outdir: str | Path,
    progress_cb=None,
) -> ConvertResult:
    """把 CSV/Excel 源文件转成 outdir/data.parquet + outdir/meta.json。

    Raises:
        ParquetConversionError: 任何转换期失败（行数守恒不过 / 坏行 /
            不可解码 / 磁盘错误）。调用方（loader_factory）按 D10 清理
            outdir 并回退内存 loader，本异常只记 warning 不进用户错误流。
    """
    start = time.perf_counter()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if is_excel:
        frame, meta, warnings = _convert_excel(
            src_path,
            desc_rows=desc_rows,
            has_unit=has_unit,
            sheet_name=sheet_name,
            outdir=outdir,
            progress_cb=progress_cb,
            warnings=warnings,
        )
    else:
        frame, meta, warnings = _convert_csv(
            src_path,
            desc_rows=desc_rows or 0,
            sep=sep,
            has_unit=has_unit,
            encoding=encoding,
            outdir=outdir,
            progress_cb=progress_cb,
            warnings=warnings,
        )

    _prog(progress_cb, 72)
    parquet_path = outdir / "data.parquet"
    frame.write_parquet(
        parquet_path,
        compression="zstd",
        statistics=True,
        row_group_size=ROW_GROUP_SIZE,
    )

    # D6 硬约束：立刻 del df 再做后续（这里后续只是写 meta，但保持同一
    # 顺序——polars 帧在写盘后不再需要）
    del frame
    gc.collect()

    _prog(progress_cb, 86)
    meta["source_fingerprint"] = _source_fingerprint(src_path)
    meta_path = outdir / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    elapsed = time.perf_counter() - start
    _prog(progress_cb, 95)
    return ConvertResult(
        parquet_path=str(parquet_path),
        meta_path=str(meta_path),
        columns=meta["var_names_all"],
        rows=meta["rows"],
        elapsed_s=elapsed,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# CSV 链
# ---------------------------------------------------------------------------


def _convert_csv(
    src_path: str,
    *,
    desc_rows: int,
    sep: str,
    has_unit: bool | None,
    encoding: str | None,
    outdir: Path,
    progress_cb,
    warnings: list[str],
):
    import polars as pl

    # 1. 头部探测 + 200 行样本（严禁构造 FastDataLoader——那会把整表
    #    读进内存，峰值收益当场归零，而且测试还可能是绿的）
    probe = probe_csv_schema(
        src_path, desc_rows=desc_rows, sep=sep, has_unit=has_unit, encoding=encoding
    )
    _prog(progress_cb, 10)

    # 2. 行数 oracle（S2 唯一静默分歧——空行——的防线）
    n_nonempty = _count_nonempty_lines(src_path)
    _prog(progress_cb, 18)

    # 3. 显式 schema（禁止 polars infer_schema：实测它把数值列判 Float64，
    #    常驻与体积双双劣化，正撞「常驻翻倍」坑）
    pl_dtypes = {"float32": pl.Float32, "float64": pl.Float64, "category": pl.String}
    schema: dict = {}
    for name in probe.var_names:
        if name in probe.date_formats and name not in probe.dtype_map:
            # D9：CSV 日期列不解析（do_parse_date=False 的既有事实），
            # 原样存字符串，date_formats 进 meta，由 UI 现算 x 轴
            schema[name] = pl.String
        else:
            schema[name] = pl_dtypes[probe.dtype_map.get(name, "category")]

    # 4. 解析（S5：转码是「抛错后重试」——常见路径非法字节只在单位行，
    #    skip_rows 已绕开，省一次全文件读写）
    frame = _read_csv_with_retry(
        pl, src_path, schema=schema, desc_rows=desc_rows, has_unit=probe.has_unit,
        sep=sep, outdir=outdir, encoding_used=probe.encoding_used,
    )
    _prog(progress_cb, 40)

    # 5. 行数守恒硬断言（S2：不等 → D10 回退）
    header_unit = 1 + desc_rows + (1 if probe.has_unit else 0)
    if frame.height != n_nonempty - header_unit:
        raise ParquetConversionError(
            f"行数守恒失败: polars={frame.height}, oracle={n_nonempty - header_unit} "
            f"(非空行 {n_nonempty} - 表头/单位/描述 {header_unit})；"
            "多半是空行/纯空格行（pandas 跳过、polars 读成全 null 行）"
        )

    # 6. inf→NaN 清理 + 有效性规则表 + 枚举码值
    frame, col_meta, enum_labels = _postprocess_pl_frame(
        pl,
        frame,
        var_names=probe.var_names,
        dtype_map=probe.dtype_map,
        date_formats=probe.date_formats,
        units=probe.units,
    )
    _prog(progress_cb, 60)

    meta = {
        "loader_type": "parquet",
        "is_excel": False,
        "rows": frame.height,
        "var_names_all": probe.var_names,
        "var_names": probe.var_names,  # time_column_name=None → 与全量相同
        "units": probe.units,
        "time_column_name": None,
        "date_formats": probe.date_formats,
        "columns": col_meta,
        "enum_labels": enum_labels,
    }
    return frame, meta, warnings


def _read_csv_with_retry(
    pl,
    src_path: str,
    *,
    schema: dict,
    desc_rows: int,
    has_unit: bool,
    sep: str,
    outdir: Path,
    encoding_used: str,
):
    """先直接解析；仅当 utf-8 解码失败才转码重试（S5）。

    严禁 encoding="utf8-lossy"（把 gb18030 中文变 U+FFFD 乱码）。
    转码产物只落 outdir（源文件可能在只读位置），用后立刻删。
    """
    skip_rows = (2 + desc_rows) if has_unit else (1 + desc_rows)

    def _read(path: str):
        # truncate_ragged_lines / ignore_errors 必须保持 False（S2 实测：
        # 打开会把「抛错=可检测」变成「静默分歧/整列错位」）
        return pl.read_csv(
            path,
            skip_rows=skip_rows,
            has_header=False,
            schema=schema,
            null_values=_NULL_VALUES,
            truncate_ragged_lines=False,
            ignore_errors=False,
            separator=sep,
        )

    try:
        return _read(src_path)
    except Exception as exc:  # noqa: BLE001 - 按 S5 只对 utf-8 错误转码重试
        msg = str(exc)
        if "utf-8" not in msg and "utf8" not in msg:
            raise
        # 数据区确有非 UTF-8 字节：按头部探测出的实际编码增量转码 →
        # 重试 → 删中间文件。转码编码用 probe.encoding_used（对整个头部
        # 验证过的编码），而不是对文件头再嗅探一次。
        trans_path = outdir / _TRANSCODED_NAME
        logger.info("polars 直解析遇非 UTF-8 字节，按 %s 转码重试", encoding_used)
        try:
            _transcode_to_utf8(src_path, trans_path, encoding_used)
        except UnicodeDecodeError:
            raise ParquetConversionError(
                f"源文件按 {encoding_used} 仍无法解码，转码失败"
            ) from exc
        try:
            return _read(str(trans_path))
        finally:
            try:
                trans_path.unlink(missing_ok=True)
            except OSError:
                pass


def _transcode_to_utf8(src: str | Path, dst: str | Path, encoding: str) -> None:
    """增量解码器流式转码（4MB 块，恒定内存）。"""
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            chunk = fin.read(_SCAN_CHUNK)
            if not chunk:
                fout.write(decoder.decode(b"", final=True).encode("utf-8"))
                break
            fout.write(decoder.decode(chunk).encode("utf-8"))


def _count_nonempty_lines(path: str) -> int:
    """顺序扫描数非空物理行数（S2 行数守恒 oracle）。

    空行 = strip 后为空（真空行与纯空格行，pandas 都跳过不计数）。
    块边界半行必须拼接（naive split 会在边界造出假空行，v2.1 实测踩过）。
    """
    count = 0
    tail = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_SCAN_CHUNK)
            if not chunk:
                break
            lines = (tail + chunk).split(b"\n")
            tail = lines.pop()  # 末段可能是半行，留给下一块
            for ln in lines:
                if ln.strip():
                    count += 1
    if tail.strip():
        count += 1  # 无结尾换行的最后一行
    return count


# ---------------------------------------------------------------------------
# polars 帧后处理（CSV 链与 Excel 链共用的文本/枚举规则）
# ---------------------------------------------------------------------------


def _postprocess_pl_frame(
    pl,
    frame,
    *,
    var_names: list[str],
    dtype_map: dict[str, str],
    date_formats: dict[str, str],
    units: dict[str, str],
):
    """inf 清理 + 有效性规则表（§2.1.1 八行）+ 枚举码值（D8）。

    返回 (新帧, 每列 meta, enum_labels)。CSV 链的 float 列 dtype 已由
    dtype_map 显式给出（probe 的 _evaluate_float32_safety 决策，与
    pandas read_csv(dtype=...) 同源），因此这里不再做降档判定——
    样本是前 200 行、全量 max_abs >= 样本 max_abs，样本判 float64 的
    列全量必然同样不满足降档条件。
    """
    col_meta: dict[str, dict] = {}
    enum_labels: dict[str, list[str]] = {}
    out_cols = []

    for name in var_names:
        s = frame.get_column(name)
        entry = {
            "unit": units.get(name, "-"),
            "is_enum": False,
        }
        # n_unique 统一取「非 null 值的不同值数」（pandas nunique 的
        # dropna=True 语义；polars 裸 n_unique() 会把 null 算一个，
        # 全空列会误报 1）
        n_unique = s.drop_nulls().n_unique()

        if name in date_formats and name not in dtype_map:
            # 日期列（D9）：String 原样，validity=1（_postprocess 的跳过规则）
            col_meta[name] = {
                **entry,
                "dtype_str": "object",
                "all_empty": s.len() == s.null_count(),
                "validity": 1,
                "n_unique": n_unique,
            }
            out_cols.append(s)
            continue

        dtype = dtype_map.get(name)
        if dtype in ("float32", "float64"):
            # inf → NaN(null) 清理是必需步骤（v2.1 升格：pandas 侧 1e400
            # 得 NaN、polars 得 inf，不清理逐位不等且 np.nanmax 把 inf
            # 当极值污染统计）。null_values 已把 NA token 判 null。
            if s.is_infinite().any():
                # pl.when(...).then(...).otherwise(...) 在 Series 上下文返回
                # Expr 而非 Series（后续 s.len() 会抛 "Expr is ambiguous"），
                # 必须经 frame.select 物化回 Series；null 值经 when 的 null
                # 分支原样保留
                s = frame.select(
                    pl.when(pl.col(name).is_finite())
                    .then(pl.col(name))
                    .otherwise(None)
                    .alias(name)
                ).to_series()
            col_meta[name] = {
                **entry,
                "dtype_str": dtype,
                "all_empty": s.len() == s.null_count(),
                "validity": _validity_float(pl, s),
                "n_unique": n_unique,
            }
            out_cols.append(s)
            continue

        # String 列（pandas 口径的 category）
        if n_unique == 0:
            # 全空列：pandas 给 category + 0 categories → meta 记
            # all_empty，parquet 存 String 全 null（S4）
            col_meta[name] = {
                **entry,
                "dtype_str": "category",
                "all_empty": True,
                "validity": -1,
                "n_unique": 0,
            }
            out_cols.append(s)
        elif n_unique <= ENUM_LABEL_MAX:
            # 枚举列（D8）：Int32 码值 + 同一次 cast 的类别表。
            # 码值与标签表同源持久化——不用 polars Categorical 直接存
            # （物理码序依赖全局字符串缓存，跨版本/跨进程语义会变）。
            # validity 记 1 是有意的正向差异（低基数文本列从「灰」变可选）。
            cat = s.cast(pl.Categorical)
            codes = cat.to_physical().cast(pl.Int32).alias(name)
            labels = cat.cat.get_categories().to_list()
            enum_labels[name] = labels
            col_meta[name] = {
                **entry,
                "dtype_str": "category",
                "all_empty": False,
                "validity": 1,
                "n_unique": n_unique,
                "is_enum": True,
            }
            out_cols.append(codes)
        else:
            # 高基数文本列：保持今天的行为（validity 复刻
            # pd.to_numeric(errors="raise") 的成败语义，不引入枚举）
            col_meta[name] = {
                **entry,
                "dtype_str": "category",
                "all_empty": False,
                "validity": _validity_text(pl, s),
                "n_unique": n_unique,
            }
            out_cols.append(s)

    return type(frame)(out_cols), col_meta, enum_labels


def _validity_float(pl, s) -> int:
    """float 列有效性（§2.1.1 规则表 float 三行）。

    与 np.nanmin/nanmax 同语义：null 忽略；全 null → -1；
    min == max → 0（-0.0 == 0.0 为 True，与 numpy 一致）；否则 1。
    """
    if s.len() == 0:
        return -1
    mn = s.min()
    mx = s.max()
    if mn is None and mx is None:
        return -1
    if mn == mx:
        return 0
    return 1


def _validity_text(pl, s) -> int:
    """高基数文本列有效性：复刻 pd.to_numeric(errors="raise") 语义。

    pandas 侧（base_loader.py:160-175）：to_numeric 失败 → -1；
    成功 → 有效值唯一 1 个 → 0，多个 → 1，空 → -1。
    polars 复刻：非 null 元素全部可转 Float64（cast 后无新 null）视为
    「to_numeric 成功」；否则 -1。
    """
    nn = s.drop_nulls()
    if nn.len() == 0:
        return -1
    casted = nn.cast(pl.Float64, strict=False)
    if casted.null_count() > 0:
        return -1  # 有不可转元素 → pandas 会 raise
    return 0 if casted.n_unique() == 1 else 1


# ---------------------------------------------------------------------------
# Excel 链
# ---------------------------------------------------------------------------


def _convert_excel(
    src_path: str,
    *,
    desc_rows: int | None,
    has_unit: bool | None,
    sheet_name: str | int,
    outdir: Path,
    progress_cb,
    warnings: list[str],
):
    """Excel 链：calamine 读成 pandas（ExcelDataLoader 现有链，构造内
    已完成 _postprocess_columns）→ 逐列转 polars。

    转换期峰值 = 整张 pandas 表的峰值（与今天加载同量级）；Excel 的
    收益在常驻内存，不在峰值（§2.4.1 / §9.1 修正口径）。
    """
    import polars as pl

    from src.data.excel_loader import ExcelDataLoader

    xl = ExcelDataLoader(
        src_path,
        sheet_name=sheet_name,
        desc_rows=desc_rows,
        has_unit=has_unit,
        _progress=progress_cb,
    )
    df = xl._df
    var_names = list(xl._var_names)
    validity_src = dict(xl._df_validity)
    date_formats = dict(getattr(xl, "date_formats", {}) or {})
    units = dict(xl._units)
    _prog(progress_cb, 40)

    col_meta: dict[str, dict] = {}
    enum_labels: dict[str, list[str]] = {}
    out_cols = []

    for name in var_names:
        s = df[name]
        ps, entry = _pandas_series_to_polars(pl, name, s)
        n_unique = int(s.nunique())
        if entry.pop("is_enum_candidate", False):
            # 低基数文本列 → 枚举码值化（D8，与 CSV 链同源规则）。
            # validity 记 1 是有意的正向差异（从「灰」变可选）。
            codes, labels = _enum_codes_labels(pl, ps)
            ps = codes
            enum_labels[name] = labels
            entry["is_enum"] = True
            entry["validity"] = 1
            entry["n_unique"] = len(labels)
        else:
            entry["is_enum"] = False
            entry["validity"] = validity_src.get(name, -1)
            entry["n_unique"] = n_unique
        entry["unit"] = units.get(name, "-")
        col_meta[name] = entry
        out_cols.append(ps)

    frame = pl.DataFrame(out_cols)
    rows = frame.height

    # 整个 pandas 帧与 loader 实例立刻释放（Excel 链峰值就在这里）
    del df, xl, out_cols
    gc.collect()

    meta = {
        "loader_type": "parquet",
        "is_excel": True,
        "rows": rows,
        "var_names_all": var_names,
        "var_names": var_names,  # time_column_name=None → 与全量相同
        "units": units,
        "time_column_name": None,
        "date_formats": date_formats,
        "columns": col_meta,
        "enum_labels": enum_labels,
    }
    return frame, meta, warnings


def _is_dt_series(s: pd.Series) -> bool:
    return bool(pd.api.types.is_datetime64_any_dtype(s))


def _enum_codes_labels(pl, s):
    """String 列 → (Int32 码值, 出现序类别表)。null 保持 null。"""
    cat = s.cast(pl.Categorical)
    codes = cat.to_physical().cast(pl.Int32)
    labels = cat.cat.get_categories().to_list()
    return codes, labels


def _pandas_series_to_polars(pl, name: str, s: pd.Series):
    """单列 pandas → (polars Series, meta 骨架)。dtype_str 记 pandas 口径。"""
    dtype_str = str(s.dtype)
    all_empty = bool(s.isna().all())

    if _is_dt_series(s):
        # D9：真 datetime。datetime64[s] 必须升采到 Datetime("ms") 存，
        # 读回 .astype("datetime64[s]") 还原（polars 没有 "s" 单位）。
        # pandas 3.0 的 numpy dtype 没有 .unit 属性（getattr 恒走默认值，
        # 实测 pandas 3.0.3），单位必须从 str(dtype) 解析：
        # "datetime64[ns]" / "datetime64[us, UTC]"（tz-aware 取逗号前段）。
        inner = dtype_str[dtype_str.index("[") + 1 : dtype_str.rindex("]")]
        unit = inner.split(",")[0].strip() or "ns"
        if unit in ("ns", "us", "ms"):
            pl_unit = unit
            values = s.to_numpy()
        else:  # "s" / "D" 等低精度 → 升采 ms（polars 不支持该 numpy 分辨率）
            pl_unit = "ms"
            values = s.astype("datetime64[ms]").to_numpy()
        # dtype= 必须显式给：构造器对 ns 数组默认降 us，后置 cast 无法还原
        ps = pl.Series(name, values, dtype=pl.Datetime(pl_unit))
        return ps, {
            "dtype_str": dtype_str,  # 如 datetime64[s]（读回还原的依据）
            "all_empty": all_empty,
            "is_enum": False,
        }

    if pd.api.types.is_float_dtype(s):
        pl_dt = pl.Float32 if dtype_str == "float32" else pl.Float64
        ps = pl.Series(name, s.to_numpy(), dtype=pl_dt)
        # inf → NaN 清理（与 CSV 链一致；Excel 数值单元格一般无 inf，
        # 但公式结果可能是）
        return ps, {"dtype_str": dtype_str, "all_empty": all_empty, "is_enum": False}

    if pd.api.types.is_integer_dtype(s):
        ps = pl.Series(name, s.to_numpy(), dtype=pl.Int64)
        return ps, {"dtype_str": dtype_str, "all_empty": all_empty, "is_enum": False}

    if pd.api.types.is_bool_dtype(s):
        ps = pl.Series(name, s.to_numpy(), dtype=pl.Boolean)
        return ps, {"dtype_str": dtype_str, "all_empty": all_empty, "is_enum": False}

    # 文本（object / category）：String 存。低基数的码值化由调用方做。
    # astype("string") 会把混合 object 列里的数字文本化——Excel 混合类型
    # 列是既有边角（今天 object dtype），等价面以标准文本/数值列为准。
    values = s.astype("string").to_numpy(dtype=object, na_value=None)
    ps = pl.Series(name, values, dtype=pl.String)
    entry = {"dtype_str": dtype_str, "all_empty": all_empty, "is_enum": False}
    n_unique = int(s.nunique())
    entry["is_enum_candidate"] = 0 < n_unique <= ENUM_LABEL_MAX
    return ps, entry


# ---------------------------------------------------------------------------
# 杂项
# ---------------------------------------------------------------------------


def _source_fingerprint(path: str) -> str:
    """size + mtime + 前 64KB sha1。仅作日志线索（D4：不做自动重转/失效判定）。"""
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(64 * 1024))
    return f"{st.st_size}:{int(st.st_mtime)}:{h.hexdigest()[:16]}"


def _prog(cb, value: int) -> None:
    if cb is not None:
        try:
            cb(value)
        except Exception:  # noqa: BLE001 - 进度回调失败不影响转换
            pass
