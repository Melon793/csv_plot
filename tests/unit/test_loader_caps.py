"""loader 能力谓词 is_lazy_loader() 单元测试（P0）。

断言四个 loader 类的谓词取值：
- BaseDataLoader / FastDataLoader / ExcelDataLoader → False（后两者靠继承）
- MDFLazyLoader → True（类属性显式声明）

FastDataLoader / ExcelDataLoader 靠继承得到 False 这条断言是护栏：
防止将来有人给子类单独乱加 IS_LAZY 而不改造 UI 判据点（§2.4 表）。
"""

from __future__ import annotations

import pytest

from src.data.base_loader import BaseDataLoader
from src.data.excel_loader import ExcelDataLoader
from src.data.loader import FastDataLoader
from src.data.loader_caps import is_lazy_loader
from src.data.mdf_lazy_loader import MDFLazyLoader


class TestISLazyClassAttribute:
    def test_base_loader_is_not_lazy(self):
        assert BaseDataLoader.IS_LAZY is False

    def test_fast_loader_inherits_false(self):
        # 靠继承得到 False，子类自身未定义 IS_LAZY（防「将来乱加」的护栏）
        assert "IS_LAZY" not in FastDataLoader.__dict__
        assert FastDataLoader.IS_LAZY is False

    def test_excel_loader_inherits_false(self):
        assert "IS_LAZY" not in ExcelDataLoader.__dict__
        assert ExcelDataLoader.IS_LAZY is False

    def test_mdf_lazy_loader_is_lazy(self):
        assert MDFLazyLoader.IS_LAZY is True


class TestIsLazyLoaderPredicate:
    def test_base_loader_instance(self):
        assert is_lazy_loader(BaseDataLoader()) is False

    def test_fast_loader_instance(self, tmp_path):
        # 最小 CSV：表头 + 一行数据，构造真实 FastDataLoader
        csv = tmp_path / "s.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        loader = FastDataLoader(str(csv))
        assert is_lazy_loader(loader) is False

    def test_excel_loader_instance(self, tmp_path):
        from tests.fixtures.data_factory import write_xlsx

        xlsx = tmp_path / "s.xlsx"
        write_xlsx(xlsx, header=["a", "b"], rows=[[1, 3], [2, 4]])
        loader = ExcelDataLoader(str(xlsx))
        assert is_lazy_loader(loader) is False

    def test_mdf_loader_class_and_stub(self):
        # 不构造真实 MDF 文件，用类属性 + 最小桩验证谓词读取路径
        assert is_lazy_loader(MDFLazyLoader) is True

        class _Stub:
            IS_LAZY = True

        assert is_lazy_loader(_Stub()) is True

    def test_none_and_missing_attr_default_false(self):
        assert is_lazy_loader(None) is False

        class _NoAttr:
            pass

        # 未声明 IS_LAZY 的对象默认按内存型处理
        assert is_lazy_loader(_NoAttr()) is False


if __name__ == "__main__":
    pytest.main([__file__])
