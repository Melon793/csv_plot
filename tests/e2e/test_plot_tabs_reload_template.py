"""多绘图页 P3：重载全页解锁（R15）与模板 v2 往返（真实 MainWindow）。

设计文档 §5.5 / §6.5 / §9.2：
- reload 的拆卸/重建/解锁必须遍历所有页；隐藏页只要 ``_is_updating_data``
  或 ``updatesEnabled`` 漏复位，切过去就是白屏/冻结（显示页看不出问题）。
- 模板 v2 导出两页后必须有 ``pages`` 键，顶层 plot 字段仍保留第 1 页
  （旧版软件回退视图）；再导入后页数 / 每页曲线 / 行高 / 当前页一致。
"""

from __future__ import annotations

import json

from PySide6.QtWidgets import QInputDialog

from src.core.plot_config import PlotSessionConfig


def _add_page_with_curve(mw, qtbot, var_name: str):
    """新建一页并画一条曲线；返回该页（仍为当前页）。"""
    page = mw.new_page()
    assert page is not None
    qtbot.waitUntil(lambda: len(page.plot_widgets) > 0, timeout=3000)
    assert page.plot_widgets[0].plot_widget.plot_variable(var_name)
    return page


def test_reload_unlocks_hidden_page_and_keeps_curves(loaded_window, qtbot):
    """R15：reload 后隐藏页也要解锁，曲线与场景保持可用。"""
    mw = loaded_window
    first = mw.current_page
    first_pw = first.plot_widgets[0].plot_widget
    assert first_pw.plot_variable("speed")

    second = _add_page_with_curve(mw, qtbot, "rpm")
    second_pw = second.plot_widgets[0].plot_widget

    # 切回第一页，让第二页在 reload 时处于隐藏态（缺陷只在隐藏页暴露）
    mw.activate_page(first)
    qtbot.waitUntil(lambda: mw.current_page is first, timeout=3000)
    assert second.widget.isHidden()
    assert second._scene_dirty is False

    # 初次加载的 50ms 收尾若还没落地，reload_data 会被入口守卫静默吃掉
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    version_before = mw._data_version
    mw.file_loader_manager.reload_data()
    qtbot.waitUntil(lambda: mw._data_version > version_before, timeout=5000)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    # 等 _post_reload_ui_refresh 的 50ms 延迟 + 尾部 cursor 回调落地
    qtbot.wait(150)

    # 两页都必须解锁：旧实现只遍历当前页，隐藏页会永久停在 updating/禁更新
    for page in mw.pages:
        for container in page.plot_widgets:
            widget = container.plot_widget
            assert widget._is_updating_data is False, f"{page.title} 未解锁"
            assert widget.updatesEnabled() is True, f"{page.title} 未恢复更新"

    # 隐藏页的数据层也重建了：曲线存在且仍挂在场景上（白屏的前置条件）
    assert "rpm" in second_pw.curves
    assert second_pw.curves["rpm"].curve is not None
    assert second_pw.curves["rpm"].curve.scene() is not None
    assert "speed" in first_pw.curves

    # reload 期间隐藏页的 cursor/region 场景写入被推迟：这里应已置脏
    assert second._scene_dirty is True

    # 切到隐藏页：补建场景后不得再冻结
    mw.activate_page(second)
    qtbot.waitUntil(lambda: not second._scene_dirty, timeout=3000)
    assert second_pw._is_updating_data is False
    assert second_pw.updatesEnabled() is True
    assert second_pw.curves["rpm"].curve.isVisible() is True


def test_template_v2_export_apply_roundtrip(loaded_window, qtbot, monkeypatch):
    """模板 v2：两页导出有 pages 键，重新导入还原页/曲线/行高/当前页。"""
    mw = loaded_window
    first = mw.current_page
    assert first.plot_widgets[0].plot_widget.plot_variable("speed")
    first.set_row_height(0, 40)

    second = _add_page_with_curve(mw, qtbot, "rpm")
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        staticmethod(lambda *args, **kwargs: ("对比页", True)),
    )
    mw._on_tab_rename_requested(mw.plot_tabs.indexOf(second.widget))
    assert second.title == "对比页"
    assert mw.current_page is second

    manager = mw.plot_config_manager
    config = manager.export_current_config(mw)
    assert len(config.pages) == 2
    assert config.active_page == 1
    assert [page_cfg.title for page_cfg in config.pages] == ["图 1", "对比页"]
    assert config.pages[0].plots[0].curves == ["speed"]
    assert config.pages[1].plots[0].curves == ["rpm"]
    assert config.pages[0].row_height_factors[0] == 40
    # 顶层 v1 字段是第 1 页的回退视图
    assert config.plots == config.pages[0].plots
    assert config.layout_rows == config.pages[0].layout_rows

    # 走一次 JSON 往返：存储层会把 int 行号键写成字符串
    payload = json.loads(json.dumps(config.to_dict()))
    assert len(payload["pages"]) == 2
    restored = PlotSessionConfig.from_dict(payload)
    assert len(restored.pages) == 2
    assert restored.active_page == 1
    assert restored.pages[0].row_height_factors == {0: 40}

    # 打乱现状：删掉第二页、清空第一页曲线、行高复原
    mw.close_page(second, confirm=False)
    assert len(mw.pages) == 1
    first.plot_widgets[0].plot_widget.clear_plot_item()
    first.set_row_height(0, 100)
    assert not first.plot_widgets[0].plot_widget.curves

    assert manager.apply_config(mw, restored) is True
    qtbot.waitUntil(lambda: len(mw.pages) == 2, timeout=3000)

    assert [page.title for page in mw.pages] == ["图 1", "对比页"]
    assert mw.plot_tabs.tabText(0) == "图 1"
    assert mw.plot_tabs.tabText(1) == "对比页"
    assert mw.current_page is mw.pages[1], "active_page=1 应回到第二页"
    assert mw.pages[0].get_row_height(0) == 40
    assert set(mw.pages[0].plot_widgets[0].plot_widget.curves) == {"speed"}
    # 新建页（scene=True）的曲线应直接建好
    assert set(mw.pages[1].plot_widgets[0].plot_widget.curves) == {"rpm"}

    # 隐藏页（第 1 页）只写数据层：切过去后场景补齐、曲线可见
    mw.activate_page(mw.pages[0])
    qtbot.waitUntil(lambda: not mw.pages[0]._scene_dirty, timeout=3000)
    first_pw = mw.pages[0].plot_widgets[0].plot_widget
    assert first_pw.curves["speed"].curve.isVisible() is True
