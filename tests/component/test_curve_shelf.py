"""曲线暂存区：协议 / 入区 / 去重 / stale / 几何 / 清空（offscreen）。

覆盖施工计划 §9.1 的入区与生命周期用例：
- shelf MIME roundtrip、注册表 id 不匹配 / 源销毁降级；
- legend 剪切入区（源延迟删除）/ Ctrl 复制（源保留）/ 源销毁拒绝；
- 变量列表入区默认样式 / 按名去重更新 / 缺名拒绝 / 加载拒绝；
- revalidate 置灰与恢复；清空 N>1 一次确认；
- 托盘 startDrag MIME（stale 不可拖）；几何 roundtrip（str 防御）。
"""

from types import SimpleNamespace

import pytest

from PySide6.QtCore import QByteArray, QPointF, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QMessageBox, QWidget

from src.core.data_types import (
    DRAW_MODE_SOURCE_AUTO,
    DRAW_MODE_STEP_PRE,
    SYMBOL_POLICY_ON,
)
from src.core.settings import AppSettings, ConfigKey
from src.ui.drag_drop import (
    LEGEND_MIME_FORMAT,
    SHELF_MIME_FORMAT,
    build_legend_var_mimedata,
    build_shelf_mimedata,
    build_var_mimedata,
    clear_active_legend_drag,
    get_active_shelf_drag_source,
    parse_shelf_mime_payload,
    set_active_legend_drag,
    set_active_shelf_drag,
)
from src.ui.widgets.curve_shelf import (
    CurveShelfDialog,
    CurveShelfWidget,
    ShelfEntry,
)
from tests.fixtures.waits import force_delete


# mime 保活：offscreen 下手工构造的 QDropEvent 不接管 mimeData 所有权
_keep_alive: list = []


class ShelfHost(QWidget):
    """最小宿主：加载闸门 + 播报记录器 + page_index。"""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []
        self.message_levels: list[str] = []
        self.loading = False

    def _show_status_message(self, message: str, level: str = "info") -> None:
        self.messages.append(message)
        self.message_levels.append(level)

    def is_data_loading(self) -> bool:
        return self.loading

    def reject_when_loading(self, action: str) -> bool:
        if not self.loading:
            return False
        self.messages.append(f"正在加载数据，请稍候再{action}")
        return True

    def page_index(self, page) -> int:
        return getattr(page, "index", -1)


class FakeContainer:
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget


class FakePage:
    def __init__(self, index: int, containers):
        self.index = index
        self.plot_widgets = list(containers)


@pytest.fixture()
def shelf(qapp):
    host = ShelfHost()
    widget = CurveShelfWidget(main_window=host)
    widget.show()
    yield widget, host
    widget.deleteLater()
    host.deleteLater()


def _drop_into_shelf(widget, mime, *, copy=False):
    mods = Qt.KeyboardModifier.AltModifier if copy else Qt.KeyboardModifier.NoModifier
    _keep_alive.append(mime)
    event = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        mods,
    )
    widget.dropEvent(event)


def _legend_drop(widget, source, name, *, copy=False):
    mime = build_legend_var_mimedata([name], source)
    set_active_legend_drag(source, [name])
    try:
        _drop_into_shelf(widget, mime, copy=copy)
    finally:
        clear_active_legend_drag()


# ---------------------------------------------------------------------------
# 协议 / 注册表
# ---------------------------------------------------------------------------


def test_shelf_mime_roundtrip_and_registry(shelf):
    widget, _ = shelf
    mime = build_shelf_mimedata(["k1", "k2"], ["a", "b"], widget)
    assert mime.hasFormat(LEGEND_MIME_FORMAT) is False
    assert mime.hasText() and mime.text() == "a;;b"
    payload = parse_shelf_mime_payload(mime)
    assert payload == {"id": id(widget), "keys": ["k1", "k2"]}

    set_active_shelf_drag(widget, ["k1", "k2"])
    try:
        assert get_active_shelf_drag_source(mime) is widget
        # id 不匹配：伪造的 MIME 不得反查到活托盘
        other = build_shelf_mimedata(["k1"], ["a"], object())
        assert get_active_shelf_drag_source(other) is None
    finally:
        from src.ui.drag_drop import clear_active_shelf_drag

        clear_active_shelf_drag()
    assert get_active_shelf_drag_source(mime) is None


def test_shelf_mime_bad_payload_returns_none(shelf):
    widget, _ = shelf
    mime = build_var_mimedata(["a"])
    assert parse_shelf_mime_payload(mime) is None
    mime.setData(SHELF_MIME_FORMAT, b"not-json")
    assert parse_shelf_mime_payload(mime) is None


# ---------------------------------------------------------------------------
# 入区：legend 剪切 / 复制 / 源销毁拒绝
# ---------------------------------------------------------------------------


def test_legend_cut_snapshots_and_removes_source(shelf, plot_factory, qapp):
    widget, host = shelf
    src = plot_factory()
    assert src.plot_variable("a")
    # 源样式：自定义色 + 隐藏 + 阶梯前 + 数据点常显
    src._multi_curve_manager.apply_curve_color("a", "#123456")
    src._multi_curve_manager.apply_curve_draw_mode("a", DRAW_MODE_STEP_PRE)
    src.curves["a"].draw_mode_source = DRAW_MODE_SOURCE_AUTO  # 模拟枚举自动判定
    src._multi_curve_manager.apply_curve_symbol_policy("a", SYMBOL_POLICY_ON)
    src._multi_curve_manager.toggle_curve_visibility_by_name("a")
    assert src.curves["a"].visible is False

    # 来源徽章：页1·图2（page_index=0，plot 位于第 2 格）
    src.page = FakePage(0, [FakeContainer(object()), FakeContainer(src)])

    _legend_drop(widget, src, "a")

    entry = widget.entry_for_name("a")
    assert entry is not None
    assert entry.color == "#123456"
    assert entry.visible is False
    assert entry.draw_mode == DRAW_MODE_STEP_PRE
    assert entry.draw_mode_source == DRAW_MODE_SOURCE_AUTO
    assert entry.symbol_policy == SYMBOL_POLICY_ON
    assert entry.origin_label == "页1·图2"
    assert entry.origin_plot() is src  # 剪切记来源：可退回
    assert any("已剪切 a（页1·图2）→ 暂存区" in m for m in host.messages)

    qapp.processEvents()  # 源端删除 singleShot(0) 落地
    assert "a" not in src.curves
    # 删源后台账刷新：不能停在入区瞬间的「原图已有」状态
    assert "可退回原图" in widget.item(0).toolTip()
    assert widget.returnable_entries([entry]) == [entry]


def test_legend_alt_copy_keeps_source(shelf, plot_factory, qapp):
    widget, host = shelf
    src = plot_factory()
    assert src.plot_variable("a")

    _legend_drop(widget, src, "a", copy=True)

    entry = widget.entry_for_name("a")
    assert entry is not None
    assert entry.origin_plot() is None  # 复制不记来源：原图仍有，无需退回
    qapp.processEvents()
    assert "a" in src.curves  # 复制语义：源保留
    assert any("已复制 a" in m for m in host.messages)


def test_legend_source_destroyed_rejected(shelf):
    widget, host = shelf
    # 独立 QWidget 作源：force_delete 后只留 Python 包装器，C++ 已析构
    src = QWidget()
    force_delete(src)

    mime = build_legend_var_mimedata(["a"], src)
    set_active_legend_drag(src, ["a"])
    try:
        _drop_into_shelf(widget, mime)
    finally:
        clear_active_legend_drag()

    assert widget.count() == 0
    assert any("源绘图已关闭，无法暂存 a" in m for m in host.messages)


def test_legend_missing_curve_rejected(shelf, plot_factory):
    widget, host = shelf
    src = plot_factory()
    assert src.plot_variable("a")
    _legend_drop(widget, src, "zz")
    assert widget.count() == 0
    assert any("曲线不可用" in m for m in host.messages)


# ---------------------------------------------------------------------------
# 入区：变量列表 / 去重 / 加载拒绝
# ---------------------------------------------------------------------------


def test_entry_survives_source_plot_destruction(shelf, qapp):
    """关页存活：条目快照自持，源绘图销毁后仍可拖出。"""
    widget, host = shelf
    source = QWidget()
    source.curves = {
        "a": SimpleNamespace(
            color="#101010",
            visible=True,
            draw_mode="auto",
            symbol_policy="auto",
            draw_mode_source="auto",
        )
    }
    removed = []
    source.remove_variable_from_plot = removed.append

    _legend_drop(widget, source, "a")
    qapp.processEvents()
    assert removed == ["a"]
    entry = widget.entry_for_name("a")
    assert entry is not None

    force_delete(source)  # 源绘图关闭

    assert widget.count() == 1
    assert widget.entry_for_name("a") is entry
    assert entry.color == "#101010"
    assert entry.stale is False


@pytest.mark.parametrize(
    "failure_mode", ["oserror", "runtime-error", "return-false"]
)
def test_delayed_source_delete_failure_keeps_one_copy_and_warns(
    shelf, qapp, failure_mode
):
    """延迟删源失败必须消除两侧重复；源仍在时保留唯一资产并明确 warn。"""
    widget, host = shelf
    source = QWidget()
    source.curves = {
        "a": SimpleNamespace(
            color="#101010",
            visible=True,
            draw_mode="auto",
            symbol_policy="auto",
            draw_mode_source="auto",
            display_name="合成曲线",
            factor=2.0,
            offset=3.0,
        )
    }
    attempted: list[str] = []

    def fail_remove(name):
        attempted.append(name)
        if failure_mode == "oserror":
            raise OSError("synthetic delete failure")
        if failure_mode == "runtime-error":
            raise RuntimeError("synthetic live-source delete failure")
        return False

    source.remove_variable_from_plot = fail_remove
    entry = ShelfEntry.from_curve_info(
        "a", source.curves["a"], origin_label="页1·图1", origin_plot=source
    )
    widget.add_entry(entry)
    widget._schedule_source_remove(source, ["a"])

    qapp.processEvents()

    assert attempted == ["a"]
    source_has = "a" in source.curves
    shelf_has = widget.entry_for_name("a") is not None
    assert source_has != shelf_has, "删源失败后不得在源绘图和暂存区各留一份"
    assert source_has or shelf_has, "唯一资产不能丢失"
    assert any(
        "a" in message and ("失败" in message or "无法" in message)
        for message in host.messages
    ), host.messages
    assert any(
        level in ("warn", "error")
        for message, level in zip(host.messages, host.message_levels)
        if "a" in message and ("失败" in message or "无法" in message)
    ), list(zip(host.messages, host.message_levels))


def test_duplicate_cut_from_another_plot_updates_origin(shelf, plot_factory, qapp):
    """同名二次剪入必须换来源：否则退回旧图且播报看起来完全正常。"""
    widget, host = shelf
    first = plot_factory()
    second = plot_factory()
    assert first.plot_variable("a")
    assert second.plot_variable("a")
    first.page = FakePage(0, [FakeContainer(first)])
    second.page = FakePage(0, [FakeContainer(second)])

    _legend_drop(widget, first, "a")
    qapp.processEvents()
    _legend_drop(widget, second, "a")

    assert widget.count() == 1
    entry = widget.entry_for_name("a")
    assert entry is not None
    assert entry.origin_plot() is second
    assert entry.origin_label == "页1·图1"
    assert any("已更新 1 条同名暂存" in m for m in host.messages)


def test_copy_intake_overwrites_cut_origin_with_none(shelf, plot_factory, qapp):
    """剪入后被复制入区覆盖 → 来源清空（最近一次入区动作决定语义）。"""
    widget, host = shelf
    cut_src = plot_factory()
    copy_src = plot_factory()
    assert cut_src.plot_variable("a")
    assert copy_src.plot_variable("a")
    cut_src.page = FakePage(0, [FakeContainer(cut_src)])
    copy_src.page = FakePage(0, [FakeContainer(copy_src)])

    _legend_drop(widget, cut_src, "a")
    qapp.processEvents()
    _legend_drop(widget, copy_src, "a", copy=True)

    entry = widget.entry_for_name("a")
    assert entry is not None
    assert entry.origin_plot() is None
    assert any("已更新 1 条同名暂存" in m for m in host.messages)


def test_origin_plot_none_after_source_destroyed(shelf, qapp):
    """来源绘图关闭后条目仍在，但退回来源解析为 None。"""
    widget, _ = shelf
    source = QWidget()
    source.curves = {
        "a": SimpleNamespace(
            color="#101010",
            visible=True,
            draw_mode="auto",
            symbol_policy="auto",
            draw_mode_source="auto",
        )
    }
    source.remove_variable_from_plot = lambda name: None
    source.page = FakePage(0, [FakeContainer(source)])

    _legend_drop(widget, source, "a")
    qapp.processEvents()
    entry = widget.entry_for_name("a")
    assert entry is not None and entry.origin_plot() is source

    force_delete(source)

    assert entry.origin_plot() is None
    assert widget.count() == 1
    assert entry.origin_label == "页1·图1"


def test_unweakrefable_origin_degrades_to_none():
    ci = SimpleNamespace(color="", visible=True)
    entry = ShelfEntry.from_curve_info("a", ci, origin_plot=object())
    assert entry.origin_plot_ref is None
    assert entry.origin_plot() is None


def test_resolve_origin_rejects_dead_or_detached_sources(shelf, plot_factory):
    widget, _ = shelf
    src = plot_factory()
    assert src.plot_variable("a")
    entry = ShelfEntry.from_curve_info(
        "a", src.curves["a"], origin_label="页1·图1", origin_plot=src
    )

    # 页已从宿主摘掉（ShelfHost.page_index 读 FakePage.index）
    src.page = FakePage(-1, [FakeContainer(src)])
    assert widget._resolve_origin(entry) == (None, "")

    # 图已不在该页容器列表里
    src.page = FakePage(0, [FakeContainer(object())])
    assert widget._resolve_origin(entry) == (None, "")

    # 销毁窗口期
    src.page = FakePage(0, [FakeContainer(src)])
    src._is_being_destroyed = True
    assert widget._resolve_origin(entry) == (None, "")

    # 正常路径
    src._is_being_destroyed = False
    assert widget._resolve_origin(entry) == (src, "页1·图1")


def test_variable_list_intake_default_style(shelf):
    widget, host = shelf
    _drop_into_shelf(widget, build_var_mimedata(["c"]))
    entry = widget.entry_for_name("c")
    assert entry is not None
    assert entry.origin_label == "变量列表"
    assert entry.color == ""
    assert entry.draw_mode == "auto"
    assert any("已复制 c（变量列表）→ 暂存区" in m for m in host.messages)


def test_duplicate_name_updates_in_place(shelf):
    widget, host = shelf
    _drop_into_shelf(widget, build_var_mimedata(["c"]))
    _drop_into_shelf(widget, build_var_mimedata(["c"]))
    assert widget.count() == 1
    assert any("已更新 1 条同名暂存" in m for m in host.messages)


def test_duplicate_snapshot_refreshes_metadata_without_replacing_entry(shelf, plot_factory):
    """R8：同名重入刷新显示名/换算，key、来源和其他样式必须留在原条目。"""
    widget, _host = shelf
    source = plot_factory()
    assert source.plot_variable("a")
    source._multi_curve_manager.apply_curve_color("a", "#123456")
    source._multi_curve_manager.apply_curve_draw_mode("a", DRAW_MODE_STEP_PRE)
    source._multi_curve_manager.apply_curve_symbol_policy("a", SYMBOL_POLICY_ON)
    source._multi_curve_manager.toggle_curve_visibility_by_name("a")
    source.curves["a"].display_name = "旧显示名"
    source.curves["a"].factor = 2.0
    source.curves["a"].offset = 3.0
    source.page = FakePage(0, [FakeContainer(source)])

    first = ShelfEntry.from_curve_info(
        "a", source.curves["a"], origin_label="页1·图1", origin_plot=source
    )
    assert widget.add_entry(first) == "added"

    source.curves["a"].display_name = "新显示名"
    source.curves["a"].factor = 4.0
    source.curves["a"].offset = 5.0
    second = ShelfEntry.from_curve_info(
        "a", source.curves["a"], origin_label="页1·图1", origin_plot=source
    )
    assert second.key != first.key
    assert widget.add_entry(second) == "updated"

    stored = widget.entry_for_name("a")
    assert stored is first
    assert widget.count() == 1
    assert stored.key == first.key
    assert stored.origin_plot() is source
    assert stored.origin_label == "页1·图1"
    assert stored.color == "#123456"
    assert stored.visible is False
    assert stored.draw_mode == DRAW_MODE_STEP_PRE
    assert stored.symbol_policy == SYMBOL_POLICY_ON
    assert stored.display_name == "新显示名"
    assert stored.factor == 4.0
    assert stored.offset == 5.0


def test_intake_rejected_while_loading(shelf):
    widget, host = shelf
    host.loading = True
    _drop_into_shelf(widget, build_var_mimedata(["c"]))
    assert widget.count() == 0
    assert "数据加载中，暂不可搬运变量" in host.messages


# ---------------------------------------------------------------------------
# stale / 删除 / 清空
# ---------------------------------------------------------------------------


def test_revalidate_marks_stale_and_recovers(shelf):
    widget, _ = shelf
    _drop_into_shelf(widget, build_var_mimedata(["c"]))
    assert widget.revalidate({"a", "b"}) == ["c"]
    entry = widget.entry_for_name("c")
    assert entry.stale is True
    assert "✕" in widget.item(0).text()
    assert not (widget.item(0).flags() & Qt.ItemFlag.ItemIsDragEnabled)

    assert widget.revalidate({"a", "b", "c"}) == []
    assert entry.stale is False
    assert widget.item(0).flags() & Qt.ItemFlag.ItemIsDragEnabled


def test_clear_confirm_no_keeps_entries(shelf, monkeypatch):
    widget, _ = shelf
    _drop_into_shelf(widget, build_var_mimedata(["b", "c"]))
    calls = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: calls.append(a) or QMessageBox.StandardButton.No),
    )
    assert widget.request_clear() == 0
    assert widget.count() == 2
    assert len(calls) == 1  # N>1 一次汇总确认

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    assert widget.request_clear() == 2
    assert widget.count() == 0


# ---------------------------------------------------------------------------
# 拖出 MIME（startDrag）
# ---------------------------------------------------------------------------


def test_start_drag_emits_shelf_mime_and_clears_registry(shelf, monkeypatch):
    widget, _ = shelf
    import src.ui.widgets.curve_shelf as shelf_mod

    class FakeDrag:
        instances = []

        def __init__(self, parent):
            self.mime = None
            self.pixmap = None
            FakeDrag.instances.append(self)

        def setMimeData(self, mime):
            self.mime = mime

        def setPixmap(self, pixmap):
            self.pixmap = pixmap

        def setHotSpot(self, point):
            pass

        def exec(self, actions):
            FakeDrag.exec_actions = actions
            return Qt.DropAction.MoveAction

        def deleteLater(self):
            # P1-6（第 5 轮）：真实代码 exec 后释放 QDrag，替身实现该接口
            pass

    monkeypatch.setattr(shelf_mod, "QDrag", FakeDrag)
    _drop_into_shelf(widget, build_var_mimedata(["b", "c"]))
    for row in range(widget.count()):
        widget.item(row).setSelected(True)

    widget.startDrag(Qt.DropAction.MoveAction)

    assert len(FakeDrag.instances) == 1
    drag = FakeDrag.instances[0]
    assert drag.mime.hasFormat(SHELF_MIME_FORMAT)
    assert drag.mime.text() == "b;;c"
    keys = parse_shelf_mime_payload(drag.mime)["keys"]
    assert keys == [widget.item(r).data(shelf_mod.SHELF_KEY_ROLE) for r in range(2)]
    # exec 返回后注册表必须清空
    assert get_active_shelf_drag_source(drag.mime) is None


def test_start_drag_stale_entry_not_draggable(shelf, monkeypatch):
    widget, _ = shelf
    import src.ui.widgets.curve_shelf as shelf_mod

    created = []
    monkeypatch.setattr(
        shelf_mod,
        "QDrag",
        lambda parent: created.append(parent) or _NoopDrag(),
    )
    _drop_into_shelf(widget, build_var_mimedata(["c"]))
    widget.revalidate({"zzz"})
    widget.item(0).setSelected(True)
    widget.startDrag(Qt.DropAction.MoveAction)
    assert created == []


class _NoopDrag:
    def setMimeData(self, mime):
        pass

    def setPixmap(self, pixmap):
        pass

    def setHotSpot(self, point):
        pass

    def exec(self, actions):
        return Qt.DropAction.IgnoreAction

    def deleteLater(self):
        # P1-6（第 5 轮）：真实代码 exec 后释放 QDrag，替身实现该接口
        pass


# ---------------------------------------------------------------------------
# 几何记忆（str 防御 + 对话框 roundtrip）
# ---------------------------------------------------------------------------


def test_geometry_setting_type_defence(app_settings):
    settings = AppSettings()
    packed = QByteArray(b"geometry-bytes")
    settings.set_curve_shelf_geometry(packed)
    assert bytes(settings.get_curve_shelf_geometry()) == b"geometry-bytes"

    # ini 被写成普通字符串时必须视为无存档（防 restoreGeometry TypeError）
    settings.set_value(ConfigKey.CURVE_SHELF_GEOMETRY, "not-a-bytearray")
    assert settings.get_curve_shelf_geometry() is None


def test_shelf_dialog_geometry_roundtrip(qapp, app_settings):
    host = ShelfHost()
    host.show()
    dialog = CurveShelfDialog(main_window=host, parent=host)
    dialog.resize(320, 240)
    dialog.move(120, 90)
    dialog._save_geometry()
    dialog.deleteLater()

    restored = CurveShelfDialog(main_window=host, parent=host)
    assert restored.size() == dialog.size() or restored.size().width() > 0
    # 落屏幕守卫放行（offscreen 有虚拟屏幕）时尺寸应与存档一致
    assert restored.width() == 320 and restored.height() == 240
    restored.deleteLater()
    host.deleteLater()


def test_dialog_count_title_and_clear_button(shelf):
    widget, _ = shelf
    host = ShelfHost()
    dialog = CurveShelfDialog(main_window=host, parent=host)
    dialog.shelf.add_entry(ShelfEntry.from_name("a"))
    dialog._on_entries_changed()
    assert "暂存区 (1)" in dialog.title_label.text()
    assert dialog.clear_btn.isEnabled() is True
    dialog.deleteLater()
    host.deleteLater()
