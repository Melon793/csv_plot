"""P2-15：白名单外的文件要在入口被拦下，不能一路当成 CSV 去猜分隔符。

`_extract_file_extension` 认不出来时返回 `None`，而 `_load_file` 的 CSV 分支会拿
`file_ext[1:]` 去 `config_dict.json` 里查分隔符。实测旧行为（见 tmp/p2_15_red_before.py）
比报告说的更隐蔽：那个 `TypeError` 被同一段 `try/except Exception` 吞掉，用户看到的是一句
**「config_dict.json 读取失败」**（其实配置文件好得很），然后流程照旧往下走 ——
二进制内容恰能被 pandas 猜成 CSV 时就静默加载出错误数据，猜失败时由
`load_csv_file` 的兜底 except 调 `_release_old_data()` 把用户当前已加载的数据清掉。
"""

from __future__ import annotations

import json
import pytest

from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

import src.ui.file_loader_manager as flm_mod
from src.ui.file_loader_manager import FileLoaderManager, SUPPORTED_EXTENSIONS


class _FakeMainWindow(QWidget):
    """只覆盖 _load_file 早退路径会用到的成员。"""

    def __init__(self):
        super().__init__()
        self.load_btn = QPushButton(self)
        self.reload_btn = QPushButton(self)
        self._is_loading_new_data = False
        self.loader = None


@pytest.fixture()
def env(qapp, monkeypatch, tmp_path):
    mw = _FakeMainWindow()
    mw.load_btn.setEnabled(False)  # load_btn_click 进对话框前就是这么关掉的
    manager = FileLoaderManager(mw)

    recorded = []

    class _FakeBox:
        StandardButton = QMessageBox.StandardButton

        @staticmethod
        def warning(parent, title, text, *args, **kwargs):
            recorded.append((title, text))
            return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(flm_mod, "QMessageBox", _FakeBox)

    released = []
    monkeypatch.setattr(
        manager, "_release_old_data", lambda *a, **k: released.append(1)
    )

    # 报告点名的触发条件：exe/工作目录旁确实有 config_dict.json
    cfg = tmp_path / "config_dict.json"
    cfg.write_text(json.dumps({"csv": {"sep": ",", "skiprows": 0, "has_unit": True}}))
    monkeypatch.setattr(manager, "_resolve_config_path", lambda name: str(cfg))

    return manager, mw, recorded, released


def _touch(tmp_path, name):
    p = tmp_path / name
    p.write_text("a,b\n1,2\n")
    return str(p)


class TestUnsupportedExtensionRejected:
    def test_warns_and_returns_without_raising(self, env, tmp_path):
        manager, mw, recorded, _ = env
        path = _touch(tmp_path, "firmware.bin")

        manager._load_file(path)  # 修复前：误报「配置文件错误」后仍继续按 CSV 解析

        # 只此一条提示：既不能崩、也不能把好端端的 config_dict.json 报成读取失败
        assert [title for title, _ in recorded] == ["不支持的文件类型"]
        assert "firmware.bin" in recorded[0][1]
        assert "、".join(SUPPORTED_EXTENSIONS) in recorded[0][1], "提示要列出支持的后缀"
        assert mw.load_btn.isEnabled() is True, "被拦下后加载按钮必须恢复"

    def test_no_extension_and_all_files_selection(self, env, tmp_path):
        """文件对话框留了 All Files (*.*),无后缀名与陌生后缀同一条路径。"""
        manager, _, recorded, _ = env

        manager._load_file(_touch(tmp_path, "README"))
        manager._load_file(_touch(tmp_path, "plot.png"))

        assert [t for t, _ in recorded] == ["不支持的文件类型"] * 2

    def test_currently_loaded_data_survives_the_rejection(self, env, tmp_path):
        """拦下就到此为止：不能把用户已经加载好的数据清掉。"""
        manager, _, _, released = env

        manager._load_file(_touch(tmp_path, "notes.docx"))

        assert released == []


class TestGuardDoesNotOverReject:
    @pytest.mark.parametrize(
        "name",
        [
            "run.csv",
            "RUN.CSV",
            "log.txt",
            "drive.mf4",
            "book.xlsx",
            "data.mfile",
            "wave.t01",
            "split.csv.3",  # 分卷：扩展名在末尾序号之前
            "mdf.part.MDF.12",
        ],
    )
    def test_recognised_shapes_still_yield_an_extension(self, env, tmp_path, name):
        """守卫就是 `ext is None`，所以认得出后缀 = 不会被误拦。"""
        manager, _, _, _ = env

        assert manager._extract_file_extension(_touch(tmp_path, name)) in SUPPORTED_EXTENSIONS
