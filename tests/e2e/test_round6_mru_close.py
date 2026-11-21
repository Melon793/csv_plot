"""第 6 轮批次 C（P2 精选）：MRU 关当前页保留历史 / 退出收页目录 popup。

C1：close_page 置 _current_page=None 再激活邻居时 _switch_to_page(old=None)
不得覆盖仍存活的 _prev_page（否则 Ctrl+Tab 静默失效）。
C3：closeEvent 收页目录 popup（独立 Qt.Popup 顶层，grab 悬置同类抽屉问题）。
"""
from __future__ import annotations


def test_mru_history_survives_closing_current_page(loaded_window, qtbot):
    mw = loaded_window
    a = mw.current_page
    b = mw.new_page()
    assert b is not None
    qtbot.waitUntil(lambda: len(b.plot_widgets) > 0, timeout=3000)
    c = mw.new_page()
    assert c is not None
    qtbot.waitUntil(lambda: len(c.plot_widgets) > 0, timeout=3000)

    mw.activate_page(a)
    qtbot.waitUntil(lambda: mw.current_page is a, timeout=3000)
    mw.activate_page(b)
    qtbot.waitUntil(lambda: mw.current_page is b, timeout=3000)
    mw.activate_page(c)
    qtbot.waitUntil(lambda: mw.current_page is c, timeout=3000)
    # 评审 N2：再翻 A→C，使 prev=A ≠ 关 C 后的邻居 B，翻切链路才有判别力
    mw.activate_page(a)
    qtbot.waitUntil(lambda: mw.current_page is a, timeout=3000)
    mw.activate_page(c)
    qtbot.waitUntil(lambda: mw.current_page is c, timeout=3000)
    assert mw._prev_page is a

    # 关当前页 C → 激活邻居 B，old=None 不得清掉 MRU 历史 A
    mw.close_page(c, confirm=False)
    qtbot.waitUntil(lambda: c not in mw.pages, timeout=3000)
    assert mw._prev_page is a, "关当前页后 MRU 历史必须保留"
    assert mw.current_page is b
    mw._toggle_mru_page()
    qtbot.waitUntil(lambda: mw.current_page is a, timeout=3000)
    assert mw.current_page is a


def test_close_event_hides_directory_popup(loaded_window, qtbot):
    from PySide6.QtGui import QCloseEvent

    mw = loaded_window
    popup = getattr(mw.plot_tabs, "_dir_popup", None)
    assert popup is not None
    popup.show()
    qtbot.waitUntil(lambda: popup.isVisible(), timeout=3000)
    # 直接投递 closeEvent：走真实收尾逻辑但不销毁 C++ 对象
    # （qtbot 夹具 teardown 还会 close 一次，真 close 会双重销毁）
    mw.closeEvent(QCloseEvent())
    assert not popup.isVisible(), "退出时页目录 popup 必须收起（grab 悬置）"