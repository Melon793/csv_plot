"""标记区间 x 域完整性回归（数据换源/新建页场景）。

覆盖三处病征：
1. 新建页激活时 build_matrix 不得用空图默认视图污染全局 saved_mark_range
   （新页拖入曲线后统计 count=0/1）；
2. 同名换文件曲线原位重建时，钉在旧数据坐标的 region 必须域校验重置
   （否则统计 count=0 / x1=x2 钳边界）；
3. 换不同变量文件（B2：曲线降级为占位、不清空 plot）后，标记区间作为布局
   资产保留并域校验，saved_mark_range 落在新数据域内，不得是脱离新城的死值。
"""

from __future__ import annotations

from PySide6.QtCore import Qt

from tests.fixtures.data_factory import make_simple_rows, write_csv


def _load_csv(qtbot, mw, dialog_stubs, path):
    # 上一轮加载的收尾链（0ms + 50ms 双延迟定时器）必须先落地清锁，
    # 否则 load_btn_click 的 _is_loading_new_data 入口守卫会静默吃掉点击
    qtbot.waitUntil(
        lambda: not getattr(mw, "_is_loading_new_data", False), timeout=5000
    )
    dialog_stubs["open_file"] = (str(path), "CSV/TXT Files (*.csv)")
    qtbot.mouseClick(mw.load_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: mw.loaded_path == str(path), timeout=5000)


def _stat_rows(mw):
    """统计窗口当前行（未打开时返回空列表）。"""
    window = getattr(mw, "mark_stats_window", None)
    if not window or not window.isVisible():
        return []
    model = window.table.model()
    src = model.sourceModel() if hasattr(model, "sourceModel") else model
    return list(src.rows())


def test_new_page_keeps_global_mark_range(loaded_window, qtbot, qapp):
    """新建页不得污染全局区间；新页拖入曲线后统计应有非零样本数。"""
    mw = loaded_window
    pw1 = mw.pages[0].plot_widgets[0].plot_widget
    assert pw1.plot_variable("speed")

    mw.layout_manager.toggle_mark_region(True)
    qapp.processEvents()
    opened_range = tuple(mw.saved_mark_range)
    assert opened_range[0] < opened_range[1]

    page2 = mw.new_page()
    qtbot.waitUntil(lambda: len(page2.plot_widgets) > 0, timeout=3000)
    qapp.processEvents()
    # 回归点：新页空图默认 viewRange [0,1] 不得改写全局区间
    assert tuple(mw.saved_mark_range) == opened_range

    pw2 = page2.plot_widgets[0].plot_widget
    assert pw2.plot_variable("rpm")
    qapp.processEvents()
    qtbot.wait(200)

    rows = [r for r in _stat_rows(mw) if r.page == page2.title]
    assert rows, "新页拖入曲线后统计表应出现该页行"
    # 回归点：污染区间 (1/3, 2/3) 罩不住 1~50 的 x 网格，count 恒 0/1
    assert rows[0].entry.count > 1


def test_reload_shorter_file_resets_stale_region(loaded_window, qtbot, qapp,
                                                 dialog_stubs, tmp_path):
    """同名换文件（x 域变短）时钉在旧域的 region 必须重置，统计不得 count=0。"""
    mw = loaded_window
    pw = mw.pages[0].plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")

    mw.layout_manager.toggle_mark_region(True)
    qapp.processEvents()
    stale = tuple(mw.saved_mark_range)
    # 前提：loaded_window 为 50 行数据，开启区间必须脱离 10 行新城才有回归意义
    assert stale[0] > 10.0

    csv_short = write_csv(
        tmp_path / "short.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(10),
    )
    _load_csv(qtbot, mw, dialog_stubs, csv_short)
    qapp.processEvents()
    qtbot.wait(200)

    # 变量匹配 → 曲线原位重建 → 按钮保持选中，但区间必须重置进新城 [1, 10]
    assert mw.mark_region_btn.isChecked()
    new_range = tuple(mw.saved_mark_range)
    assert new_range != stale
    assert 1.0 <= new_range[0] < new_range[1] <= 10.0
    mark = mw.pages[0].plot_widgets[0].plot_widget.mark_region
    assert mark is not None
    assert tuple(mark.getRegion()) == new_range

    rows = [r for r in _stat_rows(mw) if r.entry.var_name == "speed"]
    assert rows, "换文件后统计表应保留 speed 行"
    assert rows[0].entry.count > 0


def test_load_unrelated_file_demotes_to_placeholder_keeps_region_in_domain(
    loaded_window, qtbot, qapp, dialog_stubs, tmp_path
):
    """换不同变量文件（B2）：曲线降级为占位而非清空，标记区间作为布局资产保留。

    旧契约（B2 前）是"变量全不匹配 → 全清 reset → saved_mark_range 归 None"。
    B2 下 speed 降级为占位（invalid_curves），reset 只在"无曲线且无占位"时进入
    （设计 §5.2/§5.5），故标记区间不被强制清除，而是由 revalidate_mark_region_domain
    域校验后保留。回归点从"必须归 None"改为"必须落在新数据域内、不是脱离新城的死值"。
    """
    mw = loaded_window
    pw = mw.pages[0].plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")

    mw.layout_manager.toggle_mark_region(True)
    qapp.processEvents()
    assert mw.saved_mark_range is not None

    csv_b = write_csv(
        tmp_path / "unrelated.csv",
        header=["time", "voltage", "current"],
        units=["s", "V", "A"],
        rows=[[f"{i * 0.1:.1f}", f"{12.0 + i * 0.3:.2f}", i * 2] for i in range(50)],
    )
    _load_csv(qtbot, mw, dialog_stubs, csv_b)
    qapp.processEvents()
    qtbot.wait(200)

    # B2：speed 降级为占位，不进 curves（布局资产保留，等原数据回来复活）
    assert list(pw.curves) == []
    assert "speed" in pw.invalid_curves
    assert not (set(pw.curves) & set(pw.invalid_curves))

    # 标记区间作为布局资产保留（未被全清 reset 强制关闭）
    assert mw.mark_region_btn.isChecked()
    assert pw.mark_region is not None
    # 回归点：区间必须落在新数据域（50 行 → index [1, 50]）内，
    # 不是钉在旧坐标、脱离新城的死值（域校验保证统计不会 count=0/钳边界）
    new_range = tuple(mw.saved_mark_range)
    assert 1.0 <= new_range[0] < new_range[1] <= 50.0
    assert tuple(pw.mark_region.getRegion()) == new_range
    # 占位不参与标记统计（§9）：无可绘制曲线 → 统计表空
    assert _stat_rows(mw) == []


def test_clear_plots_then_new_page_region_stays_in_data_domain(
    loaded_window, qtbot, qapp
):
    """清除绘图后新建页：区间必须落在全局数据域，且不得把 (1/3, 2/3) 回写全局。

    回归点：``clear_all_plots`` 把全局 ``saved_mark_range`` 置 None 但保留标记
    按钮勾选，此时新建页的 ``restore_mark_regions`` 走 fallback；若拿空页 master
    的默认 viewRange [0,1] 算中间 1/3，会建出 (0.33, 0.67) 的错位区间，并在
    下一轮 restore 时把它回写成全局 saved，污染其它页（统计 count=0 / x1=x2 钳边）。
    """
    mw = loaded_window
    pw = mw.pages[0].plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")

    mw.layout_manager.toggle_mark_region(True)
    qapp.processEvents()
    assert mw.saved_mark_range is not None

    mw.cursor_sync_manager.clear_all_plots()
    qapp.processEvents()
    # 前提：全局区间已清空但标记按钮仍勾选——这正是 fallback 的触发条件
    assert mw.saved_mark_range is None
    assert mw.mark_region_btn.isChecked()

    # 画回曲线使全局数据域可用（loaded_window 为 50 行 → index [1, 50]）
    assert pw.plot_variable("speed")
    qapp.processEvents()
    qtbot.wait(200)

    page2 = mw.new_page()
    qtbot.waitUntil(lambda: len(page2.plot_widgets) > 0, timeout=3000)
    qapp.processEvents()
    qtbot.wait(200)

    mark2 = page2.plot_widgets[0].plot_widget.mark_region
    assert mark2 is not None, "标记按钮勾选时新页应建出区间"
    region = tuple(mark2.getRegion())
    assert region[0] > 1.0, f"新页区间落在空页默认域 (0.33, 0.67)：{region}"
    assert 1.0 <= region[0] < region[1] <= 50.0

    saved = tuple(mw.saved_mark_range)
    assert saved[0] > 1.0, f"全局 saved_mark_range 被空页默认域污染：{saved}"

    # 新页区间罩得住数据 → 拖入曲线后统计应有非零样本数（错位区间的病征）
    pw2 = page2.plot_widgets[0].plot_widget
    assert pw2.plot_variable("rpm")
    qapp.processEvents()
    qtbot.wait(200)
    rows = [r for r in _stat_rows(mw) if r.page == page2.title]
    assert rows, "新页拖入曲线后统计表应出现该页行"
    assert rows[0].entry.count > 1
