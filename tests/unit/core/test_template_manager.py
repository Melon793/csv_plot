"""core.template_manager 单元测试：保存/冲突/删除/复制/搜索/导入导出。"""

from __future__ import annotations

import pytest

from src.core.plot_config import (
    PlotSessionConfig,
    TemplateNameConflictError,
    TemplateNotFoundError,
    TemplateValidationError,
)
from src.core.template_manager import TemplateManager


def _make_config(curves: list[str]) -> PlotSessionConfig:
    from src.core.plot_config import PlotConfig

    return PlotSessionConfig(plots=[PlotConfig(curves=curves)])


@pytest.fixture
def manager(qapp, tmp_path):
    return TemplateManager(storage_path=tmp_path / "templates")


class TestSaveTemplate:
    def test_save_new_emits_signal(self, manager, qtbot):
        with qtbot.waitSignal(manager.template_added, timeout=1000):
            tpl = manager.save_template(_make_config(["a"]), "tpl1")
        assert tpl.metadata.name == "tpl1"
        assert manager.exists("tpl1")

    def test_empty_name_rejected(self, manager):
        with pytest.raises(TemplateValidationError):
            manager.save_template(_make_config(["a"]), "   ")

    def test_duplicate_name_rejected(self, manager):
        manager.save_template(_make_config(["a"]), "dup")
        with pytest.raises(TemplateNameConflictError):
            manager.save_template(_make_config(["b"]), "dup")

    def test_update_existing(self, manager):
        tpl = manager.save_template(_make_config(["a"]), "orig", description="d1")
        updated = manager.save_template(
            _make_config(["a", "b"]), "orig", description="d2",
            template_id=tpl.metadata.id,
        )
        assert updated.metadata.description == "d2"
        assert len(manager.get_all_templates()) == 1

    def test_update_missing_id_raises(self, manager):
        with pytest.raises(TemplateNotFoundError):
            manager.save_template(
                _make_config(["a"]), "x", template_id="noexist"
            )


class TestDeleteAndDuplicate:
    def test_delete(self, manager):
        tpl = manager.save_template(_make_config(["a"]), "to_del")
        assert manager.delete_template(tpl.metadata.id) is True
        assert not manager.exists("to_del")

    def test_delete_missing_raises(self, manager):
        with pytest.raises(TemplateNotFoundError):
            manager.delete_template("noexist")

    def test_duplicate_creates_copy(self, manager):
        tpl = manager.save_template(_make_config(["a", "b"]), "src_tpl")
        dup = manager.duplicate_template(tpl.metadata.id, "copy_tpl")
        assert dup is not None
        assert dup.metadata.id != tpl.metadata.id
        assert dup.config == tpl.config
        assert len(manager.get_all_templates()) == 2


class TestSearch:
    def test_search_by_keyword(self, manager):
        manager.save_template(_make_config(["a"]), "alpha")
        manager.save_template(_make_config(["b"]), "beta")
        results = manager.search(keyword="alph")
        assert [t.metadata.name for t in results] == ["alpha"]

    def test_search_by_min_variables(self, manager):
        manager.save_template(_make_config(["a"]), "small")
        manager.save_template(_make_config(["a", "b", "c"]), "big")
        results = manager.search(min_variables=2)
        assert [t.metadata.name for t in results] == ["big"]


class TestImportExport:
    def test_export_import_roundtrip_with_name_dedup(self, manager, tmp_path):
        tpl = manager.save_template(_make_config(["a"]), "travel")
        export_path = tmp_path / "travel_export.yaml"
        assert manager.export_template(tpl.metadata.id, export_path) is True
        assert export_path.exists()

        # 同名导入 → 自动去重为 "travel (1)"
        imported = manager.import_template(export_path)
        assert imported is not None
        assert imported.metadata.name == "travel (1)"
        assert imported.metadata.id != tpl.metadata.id
        assert len(manager.get_all_templates()) == 2

    def test_export_missing_returns_false(self, manager, tmp_path):
        assert manager.export_template(
            "noexist", tmp_path / "out.yaml"
        ) is False
