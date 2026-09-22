"""create_loader()：loader 构造的单一分派点（D18，§4.5）。

同步入口（file_loader_manager._load_sync）与异步入口（DataLoadThread._load）
共用本工厂，消除「两份分派逻辑必须永远同步演化」的漏改风险（R2）。

工厂只做分派，不包异常：两个调用点的异常语义不同（同步路弹 QMessageBox、
异步路 error.emit），各自现有的 try/except 保持不动。**唯一例外**是 D10
的转换回退链——转换期任何异常都清理临时目录、只记 warning，然后继续构造
内存 loader；回退后内存 loader 若也失败，其异常原样向上传播（用户看到的
仍是今天的错误文案，不被转换异常覆盖）。

D16：同步入口必须传 allow_lazy_convert=False。转换是 CPU 重活（实测
0.4-1.7s / 555MB），落在 GUI 线程就是整窗冻结；且 get_lazy_convert_min_mb()
已钳到 >= 2MB（同步/异步分流阈值），「同步路径做转换」在设计上不可达。

播报契约（Q6：状态栏播报、不弹窗）：转换失败回退后，工厂把原因挂在返回
的内存 loader 的 ``_lazy_fallback_reason`` 属性上；异步入口的 _on_load_done
读取它并经 file_loader_manager._lazy_fallback_message() 播报。转换只存在
于异步这一条入口（D16），播报天然同规格、文案函数只有一处。
"""

from __future__ import annotations

import os

from src.core.logger import get_logger

logger = get_logger("data.loader_factory")


def _lazy_convert_allowed(path: str) -> bool:
    """CSV 是否满足惰性转换的开关与大小阈值（D2）。"""
    from src.core.settings import AppSettings

    settings = AppSettings()
    if not settings.is_lazy_convert_enabled():
        return False
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    return size >= settings.get_lazy_convert_min_mb() * 1024 * 1024


def _try_create_parquet_loader(
    path: str,
    *,
    desc_rows: int | None,
    sep: str,
    has_unit: bool | None,
    encoding: str | None,
    progress,
):
    """尝试转换并构造 ParquetLazyLoader。

    Returns:
        (loader, None) 成功；(None, reason) 失败——reason 是一句话原因，
        由调用方挂到回退的内存 loader 上供状态栏播报（D10）。
    """
    from src.data.parquet_converter import convert_to_parquet
    from src.data.parquet_lazy_loader import ParquetLazyLoader
    from src.data.temp_cache_dir import TempCacheDir

    temp = None
    try:
        temp = TempCacheDir.create()
        # 进程级 atexit 兜底（幂等注册）：正常退出时清掉所有活动实例目录
        TempCacheDir.register_atexit()
        convert_to_parquet(
            path,
            desc_rows=desc_rows,
            sep=sep,
            has_unit=has_unit,
            encoding=encoding,
            is_excel=False,
            outdir=temp.path(),
            progress_cb=progress,
        )
        loader = ParquetLazyLoader(path, temp)
        logger.info(
            "惰性转换完成: %s (%.1f MB → %s)",
            os.path.basename(path),
            os.path.getsize(path) / 1024 / 1024,
            temp.path(),
        )
        return loader, None
    except Exception as e:  # noqa: BLE001 - D10：转换期任何异常都回退，不冒泡
        # 含 ParquetConversionError（契约内）与 polars 内部/写盘异常（B-04：
        # 契约外的转换产物错误也走回退；源文件本身的错误会在回退后的内存
        # loader 上再次抛出，归属正确）
        logger.warning("parquet 转换失败，回退内存加载: %s", e, exc_info=True)
        if temp is not None:
            temp.cleanup()
        return None, f"{type(e).__name__}: {e}"


def create_loader(
    path: str,
    *,
    desc_rows: int | None = 0,
    sep: str = ",",
    has_unit: bool | None = True,
    encoding: str | None = None,
    sheet_name: str | int | None = None,
    is_excel: bool = False,
    progress=None,
    allow_gc: bool = True,
    chunksize: int | None = None,
    allow_lazy_convert: bool = False,
):
    """按扩展名分派 loader；CSV 满足条件时走惰性转换路径。

    Args:
        path: 数据文件路径
        desc_rows / sep / has_unit / encoding / sheet_name: 解析参数，
            语义与今天的两个构造点一致
        is_excel: 同步入口从文件对话框拿到的显式 Excel 标记
        progress: 进度回调（异步入口传，同步入口为 None）
        allow_gc: FastDataLoader 的 GC 许可（异步 worker 必须 False）
        chunksize: FastDataLoader 分块大小（异步入口传 3600）
        allow_lazy_convert: 是否允许 CSV→parquet 惰性转换。**同步入口
            必须为 False（D16 硬性）**

    Returns:
        MDFLazyLoader / ExcelDataLoader / FastDataLoader / ParquetLazyLoader
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".mf4", ".mdf", ".dat"):
        from src.data.mdf_lazy_loader import MDFLazyLoader

        return MDFLazyLoader(path, _progress=progress)

    if ext in (".xlsx", ".xlsm") or is_excel:
        # Excel 链刻意不做惰性转换：真实 75MB xlsx 实测转换比内存加载慢
        # 4.3%、峰值高 14.8%（验收报告 B-03，触发硬停规则 2）。是否开闸
        # 待作者决策；决策前 Excel 一律走今天的 ExcelDataLoader。
        from src.data.excel_loader import ExcelDataLoader

        return ExcelDataLoader(
            path,
            sheet_name=sheet_name or 0,
            desc_rows=desc_rows,
            has_unit=has_unit,
            _progress=progress,
        )

    fallback_reason = None
    if allow_lazy_convert and _lazy_convert_allowed(path):
        loader, fallback_reason = _try_create_parquet_loader(
            path,
            desc_rows=desc_rows,
            sep=sep,
            has_unit=has_unit,
            encoding=encoding,
            progress=progress,
        )
        if loader is not None:
            return loader

    from src.data.loader import FastDataLoader

    mem_loader = FastDataLoader(
        path,
        desc_rows=desc_rows,
        sep=sep,
        has_unit=has_unit,
        encoding=encoding,
        chunksize=chunksize,
        _progress=progress,
        allow_gc=allow_gc,
    )
    if fallback_reason is not None:
        # 播报契约：_on_load_done 读取并经状态栏播报（不弹窗，Q6）
        mem_loader._lazy_fallback_reason = fallback_reason
    return mem_loader
