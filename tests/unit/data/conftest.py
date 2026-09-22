"""tests/unit/data 的共享夹具。

var_info 的用例拆成两个文件（mdf 侧 / 统计侧）后，两侧都要用的合成 loader
夹具集中到这里 —— 夹具只有放 conftest 才能免 import 共享。
"""


import pytest

from src.data.loader import FastDataLoader
from src.data.mdf_lazy_loader import MDFLazyLoader
from tests.fixtures.data_factory import write_csv, write_mdf


# --------------------------------------------------------------------------
# parquet 惰性 loader（P4 等价性/文案/统计测试共享）
# --------------------------------------------------------------------------
@pytest.fixture()
def lazy_parquet_factory(tmp_path):
    """转换源文件 → ParquetLazyLoader；测试结束自动 close（删临时目录）。"""
    from src.data.parquet_converter import convert_to_parquet
    from src.data.parquet_lazy_loader import ParquetLazyLoader
    from src.data.temp_cache_dir import TempCacheDir

    loaders: list = []

    def make(src_path, **conv_kwargs):
        temp = TempCacheDir.create()
        convert_to_parquet(str(src_path), outdir=temp.path(), **conv_kwargs)
        loader = ParquetLazyLoader(str(src_path), temp)
        loaders.append(loader)
        return loader

    yield make
    for ld in loaders:
        ld.close()


# --------------------------------------------------------------------------
# var_info 合成 loader（B3 拆分后的共享夹具）
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mdf4_loader(tmp_path_factory):
    path = write_mdf(
        tmp_path_factory.mktemp("vi4") / "syn4.dat",
        version="4.10",
        n=12,
        with_empty_group=True,
    )
    loader = MDFLazyLoader(str(path))
    yield loader
    loader.close()


@pytest.fixture(scope="module")
def mdf3_loader(tmp_path_factory):
    path = write_mdf(
        tmp_path_factory.mktemp("vi3") / "syn3.dat", version="3.30", n=12
    )
    loader = MDFLazyLoader(str(path))
    yield loader
    loader.close()


@pytest.fixture(scope="module")
def attr4_loader(tmp_path_factory):
    """带归属信息的 MDF4（SI 源块 + 报文组 + 层级显示名 + HD 试验注释）。"""
    path = write_mdf(
        tmp_path_factory.mktemp("attr4") / "syn4.dat",
        version="4.10",
        n=12,
        with_attribution=True,
    )
    loader = MDFLazyLoader(str(path))
    yield loader
    loader.close()


@pytest.fixture(scope="module")
def attr3_loader(tmp_path_factory):
    """带归属信息的 MDF3：无 SI 块，设备只能从通道名 '\\XCP:1' 后缀推断。"""
    path = write_mdf(
        tmp_path_factory.mktemp("attr3") / "syn3.dat",
        version="3.30",
        n=12,
        with_attribution=True,
    )
    loader = MDFLazyLoader(str(path))
    yield loader
    loader.close()


@pytest.fixture()
def csv_loader(tmp_path):
    path = write_csv(
        tmp_path / "s.csv",
        header=["time", "speed", "flag", "note"],
        units=["s", "km/h", "-", "-"],
        rows=[
            ["0.0", "10.0", "1", "a"],
            ["0.1", "nan", "1", "b"],
            ["0.2", "30.0", "1", "c"],
        ],
    )
    return FastDataLoader(str(path), has_unit=True, sep=",")
