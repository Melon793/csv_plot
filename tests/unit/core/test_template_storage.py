"""core.storage 单元测试：文件名安全化、模板读写/重命名/删除、数据校验。"""

from __future__ import annotations

import pytest

from src.core.plot_config import TemplateStorageError
from src.core.storage import TemplateStorage
from src.core.template_models import PlotTemplate, TemplateMetadata


def _make_template(template_id: str = "id00001", name: str = "demo") -> PlotTemplate:
    return PlotTemplate(
        metadata=TemplateMetadata(id=template_id, name=name),
        config={"plots": [{"curves": ["a", "b"]}]},
    )


class TestMakeFilename:
    def test_special_chars_replaced(self):
        # 特殊字符替换为 _ 后连续下划线被折叠
        assert TemplateStorage._make_filename('a/b\\c:d*e?"<>|f') == "a_b_c_d_e_f"

    def test_spaces_to_underscore_and_collapse(self):
        assert TemplateStorage._make_filename("my  template ") == "my_template"

    def test_blank_name_fallback(self):
        assert TemplateStorage._make_filename("   ") == "untitled"

    def test_length_truncated_to_120(self):
        assert len(TemplateStorage._make_filename("x" * 300)) == 120


class TestValidateTemplateData:
    def test_valid_data(self):
        data = _make_template().to_dict()
        assert TemplateStorage._validate_template_data(data) is True

    def test_missing_metadata(self):
        assert TemplateStorage._validate_template_data({"config": {}}) is False

    def test_missing_config(self):
        assert TemplateStorage._validate_template_data(
            {"metadata": {"id": "1", "name": "n"}}
        ) is False

    def test_metadata_missing_id(self):
        assert TemplateStorage._validate_template_data(
            {"metadata": {"name": "n"}, "config": {}}
        ) is False

    def test_non_dict_rejected(self):
        assert TemplateStorage._validate_template_data([]) is False


class TestTemplateStorage:
    @pytest.fixture
    def storage(self, qapp, tmp_path):
        return TemplateStorage(storage_path=tmp_path / "templates")

    def test_write_and_read_roundtrip(self, storage):
        tpl = _make_template()
        assert storage.write_template(tpl) is True
        restored = storage.read_template("id00001")
        assert restored is not None
        assert restored.metadata.name == "demo"
        assert restored.config == tpl.config

    def test_file_written_to_disk(self, storage, tmp_path):
        storage.write_template(_make_template(name="disk check"))
        assert (tmp_path / "templates" / "disk_check.yaml").exists()

    def test_read_unknown_id_returns_none(self, storage):
        assert storage.read_template("missing") is None

    def test_delete_removes_file_and_cache(self, storage, tmp_path):
        storage.write_template(_make_template())
        assert storage.delete_template("id00001") is True
        assert storage.read_template("id00001") is None
        assert not (tmp_path / "templates" / "demo.yaml").exists()

    def test_rename_removes_old_file(self, storage, tmp_path):
        tpl = _make_template(name="old_name")
        storage.write_template(tpl)
        tpl.metadata.name = "new_name"
        storage.write_template(tpl)
        assert not (tmp_path / "templates" / "old_name.yaml").exists()
        assert (tmp_path / "templates" / "new_name.yaml").exists()

    def test_name_conflict_raises(self, storage):
        storage.write_template(_make_template(template_id="id1", name="same"))
        other = _make_template(template_id="id2", name="same")
        with pytest.raises(TemplateStorageError):
            storage.write_template(other)

    def test_filename_collision_after_sanitize_raises(self, storage):
        """P1-K：不同名字安全化后落到同一文件名时必须拒绝，防止互相覆盖。"""
        storage.write_template(_make_template(template_id="id1", name="a:b"))
        other = _make_template(template_id="id2", name="a_b")
        with pytest.raises(TemplateStorageError):
            storage.write_template(other)
        # 被拒的模板不得进入缓存或磁盘
        assert storage.read_template("id2") is None

    def test_same_id_rewrite_of_colliding_name_ok(self, storage):
        """同一模板改名后重写自身（缓存里同 id）不应被碰撞检查误伤。"""
        tpl = _make_template(template_id="id1", name="a:b")
        storage.write_template(tpl)
        tpl.metadata.name = "a:b renamed"
        assert storage.write_template(tpl) is True
        assert storage.read_template("id1").metadata.name == "a:b renamed"

    def test_existing_untracked_file_on_disk_not_overwritten(self, storage, tmp_path):
        """B1：磁盘上已有但不在缓存的文件（如损坏 yaml）不得被静默覆盖。"""
        taken = tmp_path / "templates" / "taken.yaml"
        taken.write_text("not: [valid template", encoding="utf-8")
        with pytest.raises(TemplateStorageError):
            storage.write_template(
                _make_template(template_id="id9", name="taken")
            )
        assert taken.read_text(encoding="utf-8") == "not: [valid template"

    def test_case_only_rename_same_id_keeps_file(self, storage):
        """同 id 仅大小写改名：大小写不敏感文件系统上新旧路径同一文件，
        replace 后的旧文件 unlink 不得把它误删。"""
        tpl = _make_template(template_id="id1", name="a_b")
        storage.write_template(tpl)
        tpl.metadata.name = "A_B"
        assert storage.write_template(tpl) is True
        restored = storage.read_template("id1")
        assert restored is not None
        assert restored.metadata.name == "A_B"

    def test_case_variant_collision_raises_on_insensitive_fs(
        self, storage, tmp_path
    ):
        """B1："A_b" 与既有 "a:b"（消毒后 a_b）在大小写不敏感盘上同文件。"""
        probe = tmp_path / "case_probe.tmp"
        probe.write_text("x", encoding="utf-8")
        insensitive = (tmp_path / "CASE_PROBE.TMP").exists()
        probe.unlink()
        if not insensitive:
            pytest.skip("文件系统大小写敏感，无此碰撞路径")

        storage.write_template(_make_template(template_id="id1", name="a:b"))
        other = _make_template(template_id="id2", name="A_b")
        with pytest.raises(TemplateStorageError):
            storage.write_template(other)
        # 既有模板必须完好
        assert storage.read_template("id1") is not None

    def test_scan_directory_loads_existing_files(self, qapp, tmp_path):
        s1 = TemplateStorage(storage_path=tmp_path / "tpl")
        s1.write_template(_make_template())
        # 新实例扫描同一目录应发现已有模板
        s2 = TemplateStorage(storage_path=tmp_path / "tpl")
        assert "id00001" in [t.metadata.id for t in s2.get_all_templates()]

    def test_read_invalid_yaml_returns_none(self, storage, tmp_path):
        bad = tmp_path / "templates" / "bad.yaml"
        bad.write_text("not: [valid template", encoding="utf-8")
        assert storage.read_template_from_file(bad) is None
