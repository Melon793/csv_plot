"""变量区细边栏折叠（C1，D1–D4）组件测试。

覆盖统一入口 ``LayoutManager.set_var_pane_collapsed`` 的六条入口路径
（« 按钮 / rail » 与整条热区 / Ctrl+B / 手柄双击 / 折叠态拖手柄 / Ctrl+F
自动展开）、28px 落地护栏、宽度记忆、折叠×隐藏绘图区矩阵守卫、rail 卫生
（不冒幽灵控件）与播报规格。

用真实 MainWindow（offscreen）：splitter 钳制/最小宽/stacked 切页都是
Qt 布局引擎行为，替身测不到。夹具模式沿用 tests/e2e/conftest 的
``_build_main_window``（重置 sys.argv、缩小默认矩阵），settings 隔离到
tmp_path，折叠态/宽度持久化不泄漏给其他用例。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.core.config import VAR_PANE_RAIL_WIDTH
from tests.fixtures.waits import flush_deferred_deletes

CTRL = Qt.KeyboardModifier.ControlModifier
LEFT = Qt.MouseButton.LeftButton

COLLAPSED_MSG = "变量区已折叠（Ctrl+B 展开）"
COLLAPSED_MSG_FIRST = COLLAPSED_MSG + "；导入文件可用 Ctrl+O"
EXPANDED_MSG = "变量区已展开"


@pytest.fixture()
def isolated_settings(qapp, app_settings, monkeypatch, tmp_path):
    """指向 tmp_path 的隔离 AppSettings 单例（折叠态/宽度不泄漏给其他用例）。"""
    monkeypatch.setenv("CSV_PLOT_CONFIG_DIR", str(tmp_path / "config"))
    from src.core.settings import AppSettings

    AppSettings._reset_for_tests()
    yield AppSettings()
    AppSettings._reset_for_tests()


def _build_window(monkeypatch, qtbot):
    import src.ui.main_window as main_window_mod

    # 与 e2e _small_plot_matrix 同口径：1×1 矩阵省构造耗时
    for name in (
        "PLOT_ROW_MAX_DEFAULT",
        "PLOT_COL_MAX_DEFAULT",
        "PLOT_ROW_CURRENT_DEFAULT",
        "PLOT_COL_CURRENT_DEFAULT",
    ):
        monkeypatch.setattr(main_window_mod, name, 1)
    monkeypatch.setattr("sys.argv", ["csv_plot_rail_test"])
    mw = main_window_mod.MainWindow()
    # 刻意不 qtbot.addWidget：MainWindow 带 WA_DeleteOnClose，收尾 flush 后
    # C++ 对象已销毁，qtbot 的自动 close 会撞 RuntimeError 并污染下一例。
    # 生命周期由 _close_window 自管（close + flush）。
    mw.resize(1200, 800)
    mw.show()
    QApplication.processEvents()
    return mw


def _close_window(mw, qapp):
    try:
        mw.close()
    except RuntimeError:
        pass  # 用例体内可能已自行关闭
    QApplication.processEvents()
    flush_deferred_deletes()


@pytest.fixture()
def rail_env(isolated_settings, monkeypatch, qtbot, qapp):
    """真实 MainWindow + 隔离 AppSettings，返回 (mw, settings)。"""
    mw = _build_window(monkeypatch, qtbot)
    yield mw, isolated_settings
    _close_window(mw, qapp)


def _collapse(mw):
    mw.layout_manager.set_var_pane_collapsed(True, reason="test")
    QApplication.processEvents()


def _expand(mw):
    mw.layout_manager.set_var_pane_collapsed(False, reason="test")
    QApplication.processEvents()


def _handle(mw):
    return mw.main_splitter.handle(1)


def _drag_handle_to(mw, target: int):
    """合成真实手柄拖动到目标左栏宽度。

    实测本 Qt 版本 ``setSizes`` **不发射** splitterMoved（只有交互式拖动
    发射），所以「用户拖过 splitter」必须走 press/move/release 全链路。
    """
    handle = _handle(mw)
    cur = mw.main_splitter.sizes()[0]
    press_pos = QPoint(handle.width() // 2, handle.height() // 2)
    move_pos = press_pos + QPoint(target - cur, 0)
    QTest.mousePress(handle, LEFT, Qt.KeyboardModifier.NoModifier, press_pos)
    QTest.mouseMove(handle, move_pos)
    QApplication.processEvents()
    QTest.mouseRelease(handle, LEFT, Qt.KeyboardModifier.NoModifier, move_pos)
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# 六入口同态同播报
# ---------------------------------------------------------------------------


class TestEntryPaths:
    def test_title_button_collapses_to_rail(self, rail_env, qtbot):
        mw, _ = rail_env
        qtbot.mouseClick(mw.var_collapse_btn, LEFT)
        QApplication.processEvents()

        assert mw._var_pane_collapsed is True
        assert mw.main_splitter.sizes()[0] == VAR_PANE_RAIL_WIDTH
        assert mw._var_pane_stack.currentIndex() == 1
        assert mw.var_pane_rail.isVisible()
        assert mw._message_text == COLLAPSED_MSG_FIRST

    def test_rail_button_expands_back_to_saved_width(self, rail_env, qtbot):
        mw, _ = rail_env
        width_before = mw.main_splitter.sizes()[0]
        _collapse(mw)

        qtbot.mouseClick(mw.var_pane_rail.expand_btn, LEFT)
        QApplication.processEvents()

        assert mw._var_pane_collapsed is False
        assert mw.main_splitter.sizes()[0] == width_before
        assert mw._message_text == EXPANDED_MSG

    def test_rail_body_is_click_hotzone(self, rail_env, qtbot):
        """整条竖条可点：点在竖排文字下方的空白区也要能展开。"""
        mw, _ = rail_env
        _collapse(mw)
        rail = mw.var_pane_rail
        # 底部 stretch 区域（避开 » 按钮与文字标签）
        pos = QPoint(rail.width() // 2, rail.height() - 8)
        QTest.mouseClick(rail, LEFT, Qt.KeyboardModifier.NoModifier, pos)
        QApplication.processEvents()

        assert mw._var_pane_collapsed is False
        assert mw._message_text == EXPANDED_MSG

    def test_ctrl_b_toggles_both_ways(self, rail_env, qtbot):
        mw, _ = rail_env
        qtbot.keyClick(mw, Qt.Key.Key_B, CTRL)
        QApplication.processEvents()
        assert mw._var_pane_collapsed is True
        assert mw._message_text == COLLAPSED_MSG_FIRST

        qtbot.keyClick(mw, Qt.Key.Key_B, CTRL)
        QApplication.processEvents()
        assert mw._var_pane_collapsed is False
        assert mw._message_text == EXPANDED_MSG

    def test_handle_double_click_toggles(self, rail_env):
        mw, _ = rail_env
        QTest.mouseDClick(_handle(mw), LEFT)
        QApplication.processEvents()
        assert mw._var_pane_collapsed is True
        assert mw._message_text == COLLAPSED_MSG_FIRST

        QTest.mouseDClick(_handle(mw), LEFT)
        QApplication.processEvents()
        assert mw._var_pane_collapsed is False
        assert mw._message_text == EXPANDED_MSG

    def test_ctrl_f_auto_expands_and_focuses_search(self, rail_env, qtbot):
        mw, _ = rail_env
        _collapse(mw)
        mw.filter_input.clearFocus()

        qtbot.keyClick(mw, Qt.Key.Key_F, CTRL)
        QApplication.processEvents()

        assert mw._var_pane_collapsed is False
        assert mw.filter_input.hasFocus()
        assert mw._message_text == EXPANDED_MSG

    def test_ctrl_f_when_expanded_only_focuses(self, rail_env, qtbot):
        mw, _ = rail_env
        mw.filter_input.clearFocus()
        qtbot.keyClick(mw, Qt.Key.Key_F, CTRL)
        QApplication.processEvents()

        assert mw._var_pane_collapsed is False
        assert mw.filter_input.hasFocus()

    def test_all_collapse_entries_share_announcement(self, rail_env, qtbot):
        """« / Ctrl+B / 手柄双击三条折叠入口播报逐字一致。"""
        mw, _ = rail_env
        # 首次提示已消耗的场景：三条入口都报常规文案
        mw._var_pane_collapse_announced = True
        messages = []

        qtbot.mouseClick(mw.var_collapse_btn, LEFT)
        QApplication.processEvents()
        messages.append(mw._message_text)
        _expand(mw)

        qtbot.keyClick(mw, Qt.Key.Key_B, CTRL)
        QApplication.processEvents()
        messages.append(mw._message_text)
        _expand(mw)

        QTest.mouseDClick(_handle(mw), LEFT)
        QApplication.processEvents()
        messages.append(mw._message_text)

        assert messages == [COLLAPSED_MSG] * 3


# ---------------------------------------------------------------------------
# 几何护栏：28px 落地 / 宽度记忆 / 不动窗口几何 / 不污染宽度记忆
# ---------------------------------------------------------------------------


class TestGeometryGuards:
    def test_28px_lands_and_min_width_restored(self, rail_env):
        mw, _ = rail_env
        _collapse(mw)
        assert mw.main_splitter.sizes()[0] == VAR_PANE_RAIL_WIDTH
        assert mw.left_widget.minimumWidth() == VAR_PANE_RAIL_WIDTH

        _expand(mw)
        assert mw.left_widget.minimumWidth() == 0
        assert mw.main_splitter.sizes()[0] > VAR_PANE_RAIL_WIDTH

    def test_window_geometry_untouched(self, rail_env):
        mw, _ = rail_env
        before = mw.geometry()
        _collapse(mw)
        assert mw.geometry() == before
        _expand(mw)
        assert mw.geometry() == before

    def test_programmatic_toggle_does_not_pollute_width_memory(self, rail_env):
        """统一入口的 setSizes 必须带 QSignalBlocker：不写 user_adjusted/default_width。"""
        mw, _ = rail_env
        default_width = mw.var_table_default_width
        assert mw.var_table_user_adjusted is False

        _collapse(mw)
        _expand(mw)

        assert mw.var_table_user_adjusted is False
        assert mw.var_table_default_width == default_width

    def test_user_dragged_width_wins_over_initial(self, rail_env):
        """用户拖过 splitter 的宽度优先：折叠再展开回到拖动后的宽度。"""
        mw, _ = rail_env
        _drag_handle_to(mw, 350)
        assert mw.var_table_user_adjusted is True
        assert mw.var_table_default_width == 350

        _collapse(mw)
        _expand(mw)
        assert mw.main_splitter.sizes()[0] == 350

    def test_collapsed_handle_drag_expands_with_dragged_width(self, rail_env):
        """折叠态拖手柄即展开，且拖出来的宽度成为新的展开宽度。

        目标宽必须 ≥ 展开态 minimumSizeHint（实测本夹具 304）：低于它会被
        splitter 钳回最小宽，那是 Qt 布局引擎的固有约束而非记忆失效。
        """
        mw, _ = rail_env
        _collapse(mw)
        target = max(400, mw.left_widget.minimumSizeHint().width() + 50)

        _drag_handle_to(mw, target)

        assert mw._var_pane_collapsed is False
        assert mw.main_splitter.sizes()[0] == target
        assert mw._var_pane_saved_width == target
        assert mw._message_text == EXPANDED_MSG
        # 评审 r1 P1-1 回归：单 move 拖动（首 move 即末 move）也要走正常
        # 分支的记忆逻辑 —— user_adjusted/default_width 更新，且下次
        # resize 的定宽维持不得把新宽度拉回旧默认宽
        assert mw.var_table_user_adjusted is True
        assert mw.var_table_default_width == target
        mw.layout_manager._apply_fixed_splitter_width()
        QApplication.processEvents()
        assert mw.main_splitter.sizes()[0] == target

    def test_collapsed_manual_signal_without_mouse_stays_collapsed(self, rail_env):
        """mouseButtons 守卫的真覆盖：无按键的 splitterMoved 不得自动展开。

        本 Qt 版本 setSizes 不发射 splitterMoved，直接 emit 才能打到守卫。
        """
        mw, _ = rail_env
        _collapse(mw)
        mw.main_splitter.splitterMoved.emit(150, 1)
        QApplication.processEvents()

        assert mw._var_pane_collapsed is True
        assert mw.var_table_user_adjusted is False
        assert mw.var_table_default_width != VAR_PANE_RAIL_WIDTH

    def test_collapsed_signal_with_left_button_expands(self, rail_env, monkeypatch):
        """守卫的另一侧：左键按住的 splitterMoved 即展开；合成信号没改几何，
        sizes 仍 28，宽度记忆不得被 rail 宽污染。"""
        mw, _ = rail_env
        _collapse(mw)
        monkeypatch.setattr(
            QApplication, "mouseButtons",
            staticmethod(lambda: Qt.MouseButton.LeftButton),
        )
        mw.main_splitter.splitterMoved.emit(150, 1)
        QApplication.processEvents()

        assert mw._var_pane_collapsed is False
        assert mw.var_table_default_width != VAR_PANE_RAIL_WIDTH
        assert mw._var_pane_saved_width != VAR_PANE_RAIL_WIDTH

    def test_programmatic_setsizes_while_collapsed_no_auto_expand(self, rail_env):
        """程序化改宽（setSizes 实测不发 splitterMoved）不得触发自动展开/污染记忆。"""
        mw, _ = rail_env
        _collapse(mw)
        default_width = mw.var_table_default_width

        mw.main_splitter.setSizes([400, 600])
        QApplication.processEvents()

        assert mw._var_pane_collapsed is True
        assert mw.var_table_default_width == default_width
        assert mw.var_table_user_adjusted is False

    def test_fixed_width_adjuster_skipped_while_collapsed(self, rail_env):
        """_apply_fixed_splitter_width 折叠态守卫：不得把 28 拉回默认宽。"""
        mw, _ = rail_env
        _collapse(mw)
        mw._splitter_ready = True
        mw.var_table_user_adjusted = False

        mw.layout_manager._apply_fixed_splitter_width()
        QApplication.processEvents()

        assert mw.main_splitter.sizes()[0] == VAR_PANE_RAIL_WIDTH


# ---------------------------------------------------------------------------
# 折叠 × 隐藏绘图区矩阵守卫（生产不可达、测试路径可达）
# ---------------------------------------------------------------------------


class TestMatrixGuard:
    def test_collapse_rejected_while_plot_area_hidden(self, rail_env, qtbot):
        mw, _ = rail_env
        mw.toggle_plot_btn.setChecked(True)  # 隐藏绘图区（按钮已撤入口，toggled 仍在）
        QApplication.processEvents()
        assert mw._plot_area_visible is False

        qtbot.mouseClick(mw.var_collapse_btn, LEFT)
        QApplication.processEvents()

        assert mw._var_pane_collapsed is False
        assert "绘图区已隐藏" in mw._message_text
        assert mw._message_level == "warn"
        assert mw.main_splitter.sizes()[0] > VAR_PANE_RAIL_WIDTH

    def test_hide_plot_while_collapsed_expands_var_pane_first(self, rail_env):
        mw, _ = rail_env
        _collapse(mw)

        mw.toggle_plot_btn.setChecked(True)
        QApplication.processEvents()

        # 任何时刻至少一区可见可恢复：变量区先展开，绘图区才隐藏
        assert mw._var_pane_collapsed is False
        assert mw._plot_area_visible is False
        assert mw.main_splitter.sizes()[0] > VAR_PANE_RAIL_WIDTH


# ---------------------------------------------------------------------------
# rail 卫生与播报规格
# ---------------------------------------------------------------------------


class TestRailHygiene:
    def test_splitter_child_count_unchanged(self, rail_env):
        mw, _ = rail_env
        assert mw.main_splitter.count() == 2
        _collapse(mw)
        assert mw.main_splitter.count() == 2

    def test_rail_hidden_when_expanded(self, rail_env):
        """stacked 非当前页必须隐藏：rail 不得以未布局子控件身份冒出来。"""
        mw, _ = rail_env
        assert mw.var_pane_rail.isHidden()
        assert mw._var_pane_stack.currentIndex() == 0

    def test_content_hidden_when_collapsed(self, rail_env):
        mw, _ = rail_env
        _collapse(mw)
        assert mw._var_pane_stack.widget(0).isHidden()
        assert mw.var_pane_rail.isVisible()
        # 撤入口的隐藏绘图区按钮也不得冒头
        assert mw.toggle_plot_btn.isHidden()

    def test_first_collapse_hint_once_per_session(self, rail_env):
        mw, _ = rail_env
        _collapse(mw)
        assert mw._message_text == COLLAPSED_MSG_FIRST
        _expand(mw)
        _collapse(mw)
        assert mw._message_text == COLLAPSED_MSG

    def test_collapse_tooltip_mentions_ctrl_o(self, rail_env):
        mw, _ = rail_env
        assert "Ctrl+O" in mw.var_collapse_btn.toolTip()
        assert "Ctrl+B" in mw.var_collapse_btn.toolTip()
        assert "Ctrl+B" in mw.var_pane_rail.toolTip()

    def test_collapse_btn_does_not_inflate_window_min_width(self, rail_env):
        """« 按钮不得顶大整窗最小宽（shelf 胶囊锚点出屏回归的根因护栏）。

        隐藏按钮前后最小宽之差只允许是布局 spacing；若按钮贡献了自身宽度
        （mac QPushButton minimumSizeHint ~80px / 上限 28px），差值 ≥ 28。
        """
        mw, _ = rail_env
        with_btn = mw.minimumSizeHint().width()
        mw.var_collapse_btn.hide()
        without_btn = mw.minimumSizeHint().width()
        mw.var_collapse_btn.show()
        QApplication.processEvents()

        assert with_btn - without_btn < VAR_PANE_RAIL_WIDTH
        assert mw.var_collapse_btn.width() == VAR_PANE_RAIL_WIDTH

    def test_help_btn_narrow_like_collapse(self, rail_env):
        """? 与 « 同口径窄化（作者反馈 Win 上 ? 偏宽）：28px 且不顶最小宽。"""
        mw, _ = rail_env
        assert mw.help_btn_small.width() == VAR_PANE_RAIL_WIDTH
        with_btn = mw.minimumSizeHint().width()
        mw.help_btn_small.hide()
        without_btn = mw.minimumSizeHint().width()
        mw.help_btn_small.show()
        QApplication.processEvents()
        assert with_btn - without_btn < VAR_PANE_RAIL_WIDTH

    def test_title_row_order_clone_help_collapse(self, rail_env):
        """D-a P1：« 在标题行最右（? 之后），不夹在动作组中间。"""
        mw, _ = rail_env
        content = mw._var_pane_stack.widget(0)
        title = content.layout().itemAt(0).layout()
        widgets = [
            title.itemAt(i).widget()
            for i in range(title.count())
            if title.itemAt(i).widget() is not None
        ]
        assert widgets[-3:] == [mw.clone_btn, mw.help_btn_small, mw.var_collapse_btn]

    def test_collapse_btn_same_glyph_family_as_rail_chip(self, rail_env):
        """F3：« 与 rail » 同族矢量 ACCENT glyph＋同高（旧文本 glyph 灰小不一致）。"""
        from src.ui import theme
        from src.ui.widgets.chrome_button import ChromeButton

        mw, _ = rail_env
        btn = mw.var_collapse_btn
        assert isinstance(btn, ChromeButton)
        assert btn._glyph_color().name().upper() == theme.ACCENT.upper()
        assert btn.height() == mw.var_pane_rail.expand_btn.height()
        assert btn.testAttribute(Qt.WidgetAttribute.WA_MacNormalSize)  # F4


# ---------------------------------------------------------------------------
# D-b A：静置 affordance 包（rail 条化 / » accent 小按钮 / 手柄 grip＋hover）
# ---------------------------------------------------------------------------


class TestAffordance:
    def test_rail_strip_tokens(self, rail_env):
        """rail 静置浅底＋右弱边框，全复用 theme token（不新增颜色）。"""
        from src.ui import theme

        mw, _ = rail_env
        ss = mw.var_pane_rail.styleSheet()
        assert theme.TINT_SOFT in ss
        assert theme.BORDER_SOFT in ss
        assert ":hover" in ss

    def test_expand_chip_is_accent_button(self, rail_env):
        """C10a（D-e）新语义：原生壳＋ACCENT 矢量 glyph，取代 QSS 实色壳。

        旧断言（QSS 含 ACCENT/border-radius）随 chrome 归平台更新，
        计划 §5.1 披露；品牌蓝改在 glyph 内容色上断言。
        """
        from src.ui import theme
        from src.ui.widgets.chrome_button import ChromeButton

        mw, _ = rail_env
        chip = mw.var_pane_rail.expand_btn
        assert isinstance(chip, ChromeButton)
        assert not chip.styleSheet()
        assert chip._glyph_color().name().upper() == theme.ACCENT.upper()
        # F2：芯片跟标题«同高（平台自然高），宽仍 24；不再置 Mini 属性
        # （mac 自然高 32＝Large 档默认高→圆角；置 Mini 反而落方壳）
        assert chip.width() == 24
        assert chip.height() == mw.var_collapse_btn.height()
        assert not chip.testAttribute(Qt.WidgetAttribute.WA_MacMiniSize)

    def test_handle_is_grip_handle(self, rail_env):
        from src.ui.widgets.var_pane_rail import RailSplitter, _GripHandle

        mw, _ = rail_env
        assert isinstance(mw.main_splitter, RailSplitter)
        assert isinstance(_handle(mw), _GripHandle)

    def test_handle_hover_no_layout_change(self, rail_env):
        """1E：手柄 hover 回退 main 原样——不变色不改宽，绘图区零跳动。

        反馈分工：拖拽面靠光标↔＋tooltip＋静置 grip；单击面（rail）才给
        视觉 hover。改宽会重排 splitter 扣 pane 宽 → 绘图区 resize 跳动。
        """
        mw, _ = rail_env
        plot_w = mw.plot_widget.width()
        dot_before = _handle(mw).grab().toImage().pixelColor(2, _handle(mw).height() // 2)
        QApplication.sendEvent(_handle(mw), QEvent(QEvent.Type.Enter))
        QApplication.processEvents()
        assert mw.main_splitter.handleWidth() == 5
        assert mw.plot_widget.width() == plot_w
        dot_after = _handle(mw).grab().toImage().pixelColor(2, _handle(mw).height() // 2)
        assert dot_after == dot_before
        QApplication.sendEvent(_handle(mw), QEvent(QEvent.Type.Leave))
        QApplication.processEvents()
        assert mw.main_splitter.handleWidth() == 5
        assert mw.plot_widget.width() == plot_w

    def test_rail_blue_same_source_as_tab_selection(self, rail_env):
        """作者诉求：rail 蓝与 tab 选中/按钮激活同源同值（跨平台一致）。"""
        from src.ui import theme

        assert theme.ACCENT == theme.BRAND_BLUE
        assert theme.CHIP_SEL_BD == theme.ACCENT
        assert theme.TINT_SOFT == theme.CHIP_CUR_BG
        assert theme.BORDER_SOFT == theme.CHIP_CUR_BD

    def test_handle_tooltip_verbs_follow_state(self, rail_env):
        """D-c：手柄 tooltip 双击语义随折叠态翻转。"""
        mw, _ = rail_env
        assert "双击折叠变量区" in _handle(mw).toolTip()
        _collapse(mw)
        assert "双击展开变量区" in _handle(mw).toolTip()
        _expand(mw)
        assert "双击折叠变量区" in _handle(mw).toolTip()

    def test_rail_tooltip_click_verb(self, rail_env):
        mw, _ = rail_env
        assert "单击展开变量区" in mw.var_pane_rail.toolTip()
        assert "单击展开变量区" in mw.var_pane_rail.expand_btn.toolTip()

    def test_rail_background_survives_parent_render(self, rail_env):
        """WA_StyledBackground 根因回归：不置时 plain QWidget 子类的样式表
        背景自 grab 有、经父级渲染路径被跳过（实测像素差）。"""
        from src.ui import theme

        mw, _ = rail_env
        _collapse(mw)
        for w in (mw.var_pane_rail, mw.var_pane_rail.expand_btn, _handle(mw)):
            QApplication.sendEvent(w, QEvent(QEvent.Type.Leave))
        QApplication.processEvents()
        img = mw.left_widget.grab().toImage()
        assert img.pixelColor(14, 100).name() == theme.TINT_SOFT.lower()
        assert img.pixelColor(27, 100).name() == theme.BORDER_SOFT.lower()


# ---------------------------------------------------------------------------
# D12：折叠态/宽度持久化与启动恢复（C1b）
# ---------------------------------------------------------------------------


def _raw_ini(settings):
    """另开一个 QSettings 实例直读 ini 文件（模拟跨进程读回 str 的场景）。"""
    from PySide6.QtCore import QSettings

    return QSettings(
        str(settings.config_dir / "app.ini"), QSettings.Format.IniFormat
    )


def _seed_raw_ini(tmp_path, monkeypatch, pairs):
    """在 AppSettings 单例建立前直写 ini（启动恢复用例的脏值/存档布景）。"""
    from PySide6.QtCore import QSettings

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CSV_PLOT_CONFIG_DIR", str(cfg))
    from src.core.settings import AppSettings

    AppSettings._reset_for_tests()
    raw = QSettings(str(cfg / "app.ini"), QSettings.Format.IniFormat)
    for key, value in pairs:
        raw.setValue(key, value)
    raw.sync()


class TestPersistence:
    def test_collapse_persists_state_and_width(self, rail_env):
        mw, settings = rail_env
        _drag_handle_to(mw, 350)
        _collapse(mw)

        assert settings.get_var_pane_collapsed() is True
        assert settings.get_var_pane_width() == 350
        # ini 落盘形态直接看文件文本：QSettings 同进程读回会把 "true" 自动
        # 转 bool，跨进程/手改 ini 才见 str —— getter 两种都必须吃得下
        ini_text = (settings.config_dir / "app.ini").read_text(encoding="utf-8")
        assert "collapsed=true" in ini_text
        assert "width=350" in ini_text

    def test_expand_persists_false(self, rail_env):
        mw, settings = rail_env
        _collapse(mw)
        _expand(mw)
        assert settings.get_var_pane_collapsed() is False

    def test_drag_width_persisted_after_debounce(self, rail_env, qtbot):
        mw, settings = rail_env
        _drag_handle_to(mw, 350)
        assert settings.get_var_pane_width() is None  # 防抖窗口内还没落盘
        qtbot.wait(450)
        assert settings.get_var_pane_width() == 350

    def test_close_flushes_pending_width(self, rail_env, qapp):
        mw, settings = rail_env
        _drag_handle_to(mw, 350)
        mw.close()  # 不等防抖：正常退出必须冲掉未落盘的宽度
        QApplication.processEvents()
        assert settings.get_var_pane_width() == 350

    def test_collapsed_single_move_drag_width_persisted(self, rail_env, qtbot):
        """评审 r1 P1-1：折叠态单 move 拖出的宽度也要防抖落盘。"""
        mw, settings = rail_env
        _collapse(mw)
        target = max(400, mw.left_widget.minimumSizeHint().width() + 50)
        _drag_handle_to(mw, target)
        qtbot.wait(450)
        assert settings.get_var_pane_width() == target

    def test_missing_keys_default(self, isolated_settings):
        assert isolated_settings.get_var_pane_collapsed() is False
        assert isolated_settings.get_var_pane_width() is None

    def test_dirty_collapsed_values_fallback(self, isolated_settings):
        """type=bool 会把 "banana" 转成 True（实测）——脏值必须回退展开。"""
        raw = _raw_ini(isolated_settings)
        for dirty in ("banana", "1", "yes", "on", ""):
            raw.setValue("var_pane/collapsed", dirty)
            raw.sync()
            isolated_settings.sync()
            assert isolated_settings.get_var_pane_collapsed() is False, dirty
        # 合法字面量（大小写不敏感）照常认
        raw.setValue("var_pane/collapsed", "true")
        raw.sync()
        isolated_settings.sync()
        assert isolated_settings.get_var_pane_collapsed() is True
        raw.setValue("var_pane/collapsed", "false")
        raw.sync()
        isolated_settings.sync()
        assert isolated_settings.get_var_pane_collapsed() is False

    def test_dirty_width_values_fallback(self, isolated_settings):
        raw = _raw_ini(isolated_settings)
        for dirty in ("abc", "-5", "28", "0", "99999", "true"):
            raw.setValue("var_pane/width", dirty)
            raw.sync()
            isolated_settings.sync()
            assert isolated_settings.get_var_pane_width() is None, dirty
        raw.setValue("var_pane/width", "350")
        raw.sync()
        isolated_settings.sync()
        assert isolated_settings.get_var_pane_width() == 350


class TestStartupRestore:
    def test_restore_collapsed_and_width(self, qapp, app_settings, monkeypatch, qtbot, tmp_path):
        _seed_raw_ini(
            tmp_path, monkeypatch,
            [("var_pane/collapsed", "true"), ("var_pane/width", "350")],
        )
        mw = _build_window(monkeypatch, qtbot)
        try:
            assert mw._var_pane_collapsed is True
            assert mw.main_splitter.sizes()[0] == VAR_PANE_RAIL_WIDTH
            assert mw.var_pane_rail.isVisible()

            _expand(mw)
            assert mw.main_splitter.sizes()[0] == 350
        finally:
            _close_window(mw, qapp)

    def test_restore_width_only(self, qapp, app_settings, monkeypatch, qtbot, tmp_path):
        _seed_raw_ini(tmp_path, monkeypatch, [("var_pane/width", "350")])
        mw = _build_window(monkeypatch, qtbot)
        try:
            assert mw._var_pane_collapsed is False
            assert mw.var_table_default_width == 350
            assert mw.main_splitter.sizes()[0] == 350
        finally:
            _close_window(mw, qapp)

    def test_dirty_ini_falls_back_to_expanded(self, qapp, app_settings, monkeypatch, qtbot, tmp_path):
        _seed_raw_ini(
            tmp_path, monkeypatch,
            [("var_pane/collapsed", "banana"), ("var_pane/width", "abc")],
        )
        mw = _build_window(monkeypatch, qtbot)
        try:
            assert mw._var_pane_collapsed is False
            assert mw.var_table_default_width == 280  # 默认宽，未被脏值污染
        finally:
            _close_window(mw, qapp)

    def test_restore_collapsed_announces_without_consuming_first_hint(
        self, qapp, app_settings, monkeypatch, qtbot, tmp_path
    ):
        """启动恢复走统一入口同规格播报，但不消耗首次 Ctrl+O 提示额度
        （评审 r1 P2-3：构造期播报可能被 CLI 加载消息即刻覆盖）。"""
        _seed_raw_ini(tmp_path, monkeypatch, [("var_pane/collapsed", "true")])
        mw = _build_window(monkeypatch, qtbot)
        try:
            assert mw._message_text == COLLAPSED_MSG
            assert mw._var_pane_collapse_announced is False
            _expand(mw)
            _collapse(mw)
            assert mw._message_text == COLLAPSED_MSG_FIRST
        finally:
            _close_window(mw, qapp)
