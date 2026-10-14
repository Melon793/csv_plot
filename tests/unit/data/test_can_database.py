"""``src/data/can_database.py`` 的单元测试。

真值来源有两类：

1. **合成 DBC**（``data_factory.write_dbc``）—— 位段/因子/枚举/mux 全部由我们
   写死，断言可以精确到个位。
2. **第三方真实夹具**（``bus_data/``）—— 覆盖 canmatrix 的真实行为，包括
   「解析失败」「0 报文」这类本项目自己造不出来的形态。夹具缺失即 skip。

覆盖方案 §7 的 P1 验收项：DBC 解码 / 与 canmatrix 逐位一致的属性、cluster
flatten、first-match-wins、坏文件容错、canmatrix 不支持特性的用例化。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.can_database import (  # noqa: E402
    SUPPORTED_DB_EXTS,
    CanDatabaseSet,
    CanFrameSpec,
    CanSignalSpec,
    import_type_for,
)
from tests.fixtures import bus_data_access as bd  # noqa: E402
from tests.fixtures.data_factory import (  # noqa: E402
    CAN_FRAMES,
    CAN_UNMATCHED_ID,
    write_arxml,
    write_dbc,
)


@pytest.fixture(scope="module")
def dbc_path(tmp_path_factory) -> Path:
    """全量合成 DBC（4 报文，覆盖 Intel / Motorola / signed / float / mux / 枚举）。"""
    return write_dbc(tmp_path_factory.mktemp("candb") / "syn.dbc")


@pytest.fixture(scope="module")
def db(dbc_path) -> CanDatabaseSet:
    return CanDatabaseSet([str(dbc_path)])


# ---------------------------------------------------------------------------
# 扩展名与 import_type 探测
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.dbc", "dbc"),
        ("A.DBC", "dbc"),
        ("x/arxml/y.arxml", "arxml"),
        ("z.ARXML", "arxml"),
        ("matrix.ldf", None),       # LIN：v1 明确不做
        ("fibex.arxml.bak", None),
        ("noext", None),
        ("", None),
        ("weird.dbc.txt", None),
    ],
)
def test_import_type_for(name, expected):
    assert import_type_for(name) == expected


def test_supported_db_exts():
    assert SUPPORTED_DB_EXTS == (".dbc", ".arxml")


# ---------------------------------------------------------------------------
# 单文件加载：帧与信号属性
# ---------------------------------------------------------------------------


def test_load_reports_one_row_per_input(dbc_path):
    reports = CanDatabaseSet().load([str(dbc_path)]).reports
    assert len(reports) == 1
    assert reports[0].ok is True
    assert reports[0].error == ""
    assert reports[0].frame_count == len(CAN_FRAMES)


def test_frame_identity(db):
    for spec in CAN_FRAMES:
        # 只有跨总线样本才显式带 channel；缺省归 0 总线
        frame = db.find_frame(spec.get("channel", 0), spec["can_id"])
        assert frame is not None, spec["name"]
        assert frame.name == spec["name"]
        assert frame.can_id == spec["can_id"]
        assert frame.dlc == spec["size"]
        assert frame.cycle_time_s == pytest.approx(spec["cycle_ms"] / 1000.0)
        assert frame.comment == spec.get("comment", "")
        assert frame.db_path.endswith("syn.dbc")


def test_intel_signal_attributes(db):
    sig = db.find_frame(0, 0x1A0).signal_by_name("EngineSpeed")
    assert sig is not None
    assert sig.start_bit == 0
    assert sig.size == 16
    assert sig.little_endian is True
    assert sig.signed is False
    assert sig.is_float is False
    assert sig.factor == pytest.approx(0.125)
    assert sig.offset == pytest.approx(0.0)
    assert sig.unit == "rpm"
    # original_name 契约：db_path / frame_name / can_id 反查信息齐备
    assert sig.frame_name == "EngineData"
    assert sig.can_id == 0x1A0
    assert sig.extended is False


def test_signed_signal_keeps_sign_and_offset(db):
    sig = db.find_frame(0, 0x1A0).signal_by_name("CoolantTemp")
    assert sig.signed is True
    assert sig.little_endian is True
    assert sig.start_bit == 16
    assert sig.size == 8
    assert sig.offset == pytest.approx(-40.0)


def test_motorola_signal_attributes(db):
    frame = db.find_frame(0, 0x2C0)
    assert frame is not None
    fl = frame.signal_by_name("WheelSpeedFL")
    fr = frame.signal_by_name("WheelSpeedFR")
    assert fl.little_endian is False and fr.little_endian is False
    # start_bit 是 canmatrix 的**内部**位号（解码器用的就是它），Motorola 下
    # 与 DBC 文本位号不同：见 test_dbc_start_bit_is_the_text_number_not_the_internal_one
    assert fl.start_bit == 0 and fl.size == 12
    assert fr.start_bit == 12 and fr.size == 12
    assert fr.signed is True
    assert fl.receivers == ("Cluster", "DiagTool")
    # 未写真实接收者时 DBC 里是 `[Vector__XXX]` 占位节点，canmatrix 会直接
    # 丢掉它。本项目不把占位节点当接收者，否则会污染变量信息对话框。
    assert fr.receivers == ()


def test_float_signal_flag_and_bit_size(db):
    sig = db.find_frame(0, 0x1A0).signal_by_name("TorqueLimit")
    assert sig.is_float is True
    assert sig.size == 32
    # float 信号的 factor/offset 在 DBC 里仍可能存在，但不该被当作量化步长看待
    assert sig.unit == "Nm"


def test_enum_signal_builds_int_keyed_map(db):
    sig = db.find_frame(0, 0x1A0).signal_by_name("StarterMode")
    assert sig.is_enum is True
    assert sig.values == {0: "off", 1: "run", 2: "error"}
    # 契约：enum_map 必须是 dict[int, str]，绘图侧靠它做码值→文本
    assert all(isinstance(k, int) and isinstance(v, str) for k, v in sig.values.items())


def test_multiplexed_frame_marks_selector(db):
    frame = db.find_frame(0, 0x3B0)
    assert frame.is_multiplexed is True
    assert frame.mux_signal == "MuxSel"
    sel = frame.signal_by_name("MuxSel")
    assert sel.is_mux_selector is True
    # 选择器自己不归属任何 mux 组
    assert sel.mux_value is None
    assert frame.signal_by_name("BranchA").mux_value == 0
    assert frame.signal_by_name("BranchB").mux_value == 1
    # mux 组号与 ``mux`` 原始标记都要留存：解码用 mux_value，回落
    # canmatrix 逐帧解码时靠 mux 区分选择器 / 组号
    assert frame.signal_by_name("BranchA").mux == 0
    assert sel.mux == "Multiplexor"


def test_signals_outside_a_mux_group_stay_addressable(db):
    """同一起始位的两个 mux 分支必须都在 spec 里（由解码层用 NaN 屏蔽）。

    否则「先解选择器再按值掩码」（§2）无从实施：建模阶段就丢了一半信号。
    """
    frame = db.find_frame(0, 0x3B0)
    a = frame.signal_by_name("BranchA")
    b = frame.signal_by_name("BranchB")
    assert (a.start_bit, a.size) == (b.start_bit, b.size) == (8, 16)
    assert a.mux_value != b.mux_value


def test_dbc_start_bit_is_the_text_number_not_the_internal_one(db):
    """canmatrix 内部对 Motorola 换算过位号，UI 要展示的是 DBC 文本里的数。"""
    frame = db.find_frame(0, 0x2C0)
    fl = frame.signal_by_name("WheelSpeedFL")
    assert fl.dbc_start_bit == 7
    assert fl.start_bit == 7 - (7 % 8) + 7 - (7 % 8)
    assert fl.start_bit != fl.dbc_start_bit
    # Intel 两者一致
    intel = db.find_frame(0, 0x1A0).signal_by_name("CoolantTemp")
    assert intel.dbc_start_bit == intel.start_bit == 16


def _frame_with_comment(text: str) -> dict:
    """把帧注释换成指定文本，并**去掉全部信号注释**。

    合成帧里本来就有中文注释，它们在 cp1252 里不可表——要测「cp1252 存盘的
    欧洲文 DBC」必须先把中文抖掉，否则报的是夹具的错而不是编码逻辑的错。
    """
    fr = dict(CAN_FRAMES[0])
    fr["comment"] = text
    fr["signals"] = tuple(
        {k: v for k, v in s.items() if k != "comment"} for s in CAN_FRAMES[0]["signals"]
    )
    return fr


def test_utf8_dbc_comments_survive_the_round_trip(tmp_path):
    """UTF-8 DBC 回归：canmatrix 默认按 iso-8859-1 读，而 iso-8859-1 **永不
    报错**，于是中文注释静默变成 mojibake（实测读回来是 ``åå¨æºæ°æ®``）。

    这不是洁癖：中文车厂的 DBC 里报文 / 信号注释几乎都是中文，错了会一路
    传到变量列表与悬停提示。
    """
    path = write_dbc(tmp_path / "cn.dbc", [CAN_FRAMES[0]])   # utf-8
    frame = CanDatabaseSet([str(path)]).find_frame(0, 0x1A0)
    assert frame.comment == "发动机数据"
    assert frame.signal_by_name("EngineSpeed").comment == "发动机转速"


def test_cp1252_dbc_is_not_mistaken_for_utf8(tmp_path):
    """欧洲语言注释的 DBC 通常以 cp1252 存盘：字节序列不是合法 UTF-8，必须
    猜到 cp1252 而不是兜回 iso-8859-1（后者在 0x80–0x9F 给出控制字符）。
    """
    text = "Motor Ürlib ±§€"   # 全部 cp1252 可表、且编码后不是合法 UTF-8
    path = write_dbc(tmp_path / "eu.dbc", [_frame_with_comment(text)],
                     encoding="cp1252")
    assert text.encode("utf-8") != text.encode("cp1252")
    frame = CanDatabaseSet([str(path)]).find_frame(0, 0x1A0)
    assert frame.comment == text


def test_latin1_fallback_never_raises(tmp_path):
    """utf-8 / cp1252 都解不了的字节（0x81/0x8D/… 在 cp1252 里未定义）也要
    能读完：兜底编码的任务是「不阻断导入」，而不是「注释好看」。"""
    p = tmp_path / "raw.dbc"
    text = write_dbc(p, [_frame_with_comment("发动机数据")]).read_bytes()
    junk = b"Eng\x81\x8d\x8f\x90\x9d"
    with pytest.raises(UnicodeDecodeError):
        junk.decode("cp1252")          # 证明这份字节确实需要兜底
    p.write_bytes(text.replace("发动机数据".encode("utf-8"), junk))

    db = CanDatabaseSet().load([str(p)])
    assert db.reports[0].ok is True
    assert db.frame_count == 1          # 报文结构仍完整，只有注释不可读


def test_signal_defaults_are_safe():
    sig = CanSignalSpec(
        name="Bare", start_bit=0, size=8, little_endian=True, signed=False,
        is_float=False, factor=1.0, offset=0.0, unit="", minimum=None,
        maximum=None, values=None, comment="", mux=None, mux_value=None,
        receivers=(), cycle_time_s=None, db_path="", frame_name="",
        can_id=0, extended=False,
    )
    assert sig.dbc_start_bit == 0
    assert sig.is_enum is False
    assert sig.is_mux_selector is False


def test_frame_vectorizable_rule():
    plain = CanFrameSpec(name="F", can_id=1, extended=False, dlc=8)
    assert plain.is_vectorizable is True
    container = CanFrameSpec(
        name="F", can_id=1, extended=False, dlc=8, is_pdu_container=True
    )
    complex_mux = CanFrameSpec(
        name="F", can_id=1, extended=False, dlc=8, is_complex_multiplexed=True
    )
    # container / 复杂 mux 的字节布局不是「整帧一张平铺位图」，向量化不成立
    assert container.is_vectorizable is False
    assert complex_mux.is_vectorizable is False


# ---------------------------------------------------------------------------
# 多库合并：顺序即优先级（first-match-wins）
# ---------------------------------------------------------------------------


def _make_engine_only(tmp_path: Path) -> Path:
    """只含 EngineData 的库，但把 factor 改成 1.0，以便区分「哪一份库生效」。"""
    frame = dict(CAN_FRAMES[0])
    signals = []
    for s in frame["signals"]:
        s = dict(s)
        if s["name"] == "EngineSpeed":
            s["factor"] = 1.0
        signals.append(s)
    return write_dbc(tmp_path / "engine_a.dbc", [{**frame, "signals": signals}])


def _make_engine_conflicting(tmp_path: Path) -> Path:
    frame = dict(CAN_FRAMES[0])
    signals = []
    for s in frame["signals"]:
        s = dict(s)
        if s["name"] == "EngineSpeed":
            s["factor"] = 100.0
            s["unit"] = "conflicted"
        signals.append(s)
    return write_dbc(tmp_path / "engine_b.dbc", [{**frame, "signals": signals}])


def test_first_match_wins_follows_list_order(tmp_path):
    a = _make_engine_only(tmp_path)
    b = _make_engine_conflicting(tmp_path)

    first = CanDatabaseSet([str(a), str(b)]).find_frame(0, 0x1A0)
    assert first.signal_by_name("EngineSpeed").factor == pytest.approx(1.0)

    second = CanDatabaseSet([str(b), str(a)]).find_frame(0, 0x1A0)
    assert second.signal_by_name("EngineSpeed").factor == pytest.approx(100.0)
    assert second.signal_by_name("EngineSpeed").unit == "conflicted"


def test_merge_keeps_frames_unique_by_id_but_counts_signals_from_both(tmp_path):
    a = _make_engine_only(tmp_path)
    b = write_dbc(tmp_path / "chassis.dbc", [CAN_FRAMES[2]])
    db = CanDatabaseSet([str(a), str(b)])
    assert db.find_frame(0, 0x1A0) is not None
    assert db.find_frame(0, 0x2C0) is not None
    assert db.find_frame(0, CAN_UNMATCHED_ID) is None


def test_has_frame_ignores_channel_lookup():
    db = CanDatabaseSet()
    assert db.has_frame(0x1A0) is False


# ---------------------------------------------------------------------------
# 容错：坏文件只 warning，不中断
# ---------------------------------------------------------------------------


def test_missing_file_is_reported_not_raised(tmp_path):
    db = CanDatabaseSet().load([str(tmp_path / "nope.dbc")])
    report = db.reports[0]
    assert report.ok is False
    assert report.frame_count == 0
    assert report.error
    assert db.load_errors == [f"{tmp_path / 'nope.dbc'}: {report.error}"]


def test_garbage_file_does_not_break_the_good_one(tmp_path):
    bad = tmp_path / "broken.dbc"
    bad.write_bytes(b"\x00\x01\x02 not a dbc at all \xff\xfe")
    good = write_dbc(tmp_path / "good.dbc", [CAN_FRAMES[0]])

    db = CanDatabaseSet().load([str(bad), str(good)])
    # canmatrix 的 DBC 导入是逐行正则匹配，读不懂的行直接略过而不是报错，
    # 所以垃圾文件的形态是「解析成功但 0 报文」（§5.2 状态列的 ⚠），
    # 而不是 ✖。真正会走 ✖ 的是 canmatrix 内部抛异常（见 ARXML 用例）。
    assert db.reports[0].ok is True
    assert db.reports[0].frame_count == 0
    assert db.reports[0].warning == "无 CAN 报文"
    assert db.reports[1].ok is True
    assert db.find_frame(0, 0x1A0) is not None
    # warning 不算 error：load_errors 只收解析失败，否则提示会谎报错误数
    assert db.load_errors == []


def test_empty_file_is_rejected_safely(tmp_path):
    p = tmp_path / "empty.dbc"
    p.write_bytes(b"")
    db = CanDatabaseSet().load([str(p)])
    assert db.reports[0].frame_count == 0
    assert db.load_errors == []
    assert db.frame_count == 0


def test_unsupported_extension_is_a_hard_failure(tmp_path):
    """后缀不在白名单 → 明确报「不支持」，不能静默当成 0 报文。"""
    ldf = tmp_path / "matrix.ldf"
    ldf.write_text("LIN_description_file;\n", encoding="utf-8")
    db = CanDatabaseSet().load([str(ldf)])
    assert db.reports[0].ok is False
    assert "不支持的数据库类型" in db.reports[0].error
    assert len(db.load_errors) == 1


def test_load_twice_resets_state(dbc_path, tmp_path):
    db = CanDatabaseSet()
    db.load([str(dbc_path)])
    assert db.frame_count == len(CAN_FRAMES)
    other = write_dbc(tmp_path / "one.dbc", [CAN_FRAMES[2]])
    db.load([str(other)])
    assert db.frame_count == 1
    assert db.find_frame(0, 0x1A0) is None


def test_source_files_and_counts(dbc_path):
    db = CanDatabaseSet([str(dbc_path)])
    assert db.source_files == [str(dbc_path)]
    assert db.loaded_files == [str(dbc_path)]
    assert db.signal_count == sum(
        len(f["signals"]) for f in CAN_FRAMES
    )
    assert len(db) == db.frame_count
    assert [f.name for f in db.frames] == [f["name"] for f in CAN_FRAMES]


# ---------------------------------------------------------------------------
# ARXML：cluster flatten 与 canmatrix 的真实覆盖面
# ---------------------------------------------------------------------------


def test_arxml_cluster_is_flattened(tmp_path):
    """canmatrix 的 ARXML 导入返回 ``{name: CanMatrix}``，必须展平。"""
    arxml = write_arxml(tmp_path / "syn.arxml", [CAN_FRAMES[0]])
    db = CanDatabaseSet([str(arxml)])
    frame = db.find_frame(0, 0x1A0)
    assert frame is not None
    # canmatrix 的 ARXML 导出会给帧名加前缀，故只断言信号集合而非帧名
    assert frame.signal_by_name("EngineSpeed") is not None
    assert frame.signal_by_name("StarterMode") is not None


def test_arxml_empty_cluster_dict_is_tolerated(tmp_path):
    """canmatrix 对某些 ARXML 返回 ``{}``（实测 ARXMLCompuMethod1.arxml）。"""
    arxml = tmp_path / "empty.arxml"
    arxml.write_text(
        '<?xml version="1.0"?>\n'
        '<AUTOSAR xmlns="http://autosar.org/schema/r4.0">\n'
        "  <AR-PACKAGES/>\n"
        "</AUTOSAR>\n",
        encoding="utf-8",
    )
    db = CanDatabaseSet().load([str(arxml)])
    assert db.reports[0].ok is True
    assert db.reports[0].frame_count == 0
    assert db.reports[0].warning == "无 CAN 报文"
    assert db.load_errors == []


def test_canmatrix_cannot_read_complex_multiplexed_arxml():
    """R4 的定型结论（见 bus_data/README.md）：canmatrix 读 cantools 的
    ``system-4.2.arxml`` 会在 ``get_frame_from_multiplexed_ipdu`` 抛
    ``AttributeError``。本项目把它降级成「单文件解析失败」，绝不冒到 UI。"""
    try:
        path = bd.path_for("ct_system-4.2.arxml")
    except bd.BusDataMissing:
        pytest.skip("真实夹具未下载")
    except bd.BusDataCorrupted as e:
        pytest.fail(str(e))

    db = CanDatabaseSet().load([str(path)])
    report = db.reports[0]
    # 无论 canmatrix 哪天修好这个问题，都必须是「不抛异常」这一条不变量
    assert report.ok is False
    assert "AttributeError" in report.error or report.frame_count == 0


@pytest.mark.parametrize(
    "fixture_name,expect_frames",
    [
        ("ct_multiplex.dbc", 1),
        ("ct_floating_point.dbc", 2),
        ("ct_choices.dbc", 1),
        ("ct_big_numbers.dbc", 2),
        ("ct_bus_comment.dbc", 1),
        ("ct_cp1252.dbc", 0),      # 只有注释、无 BO_：cantools 同样读出 0 条
        ("blfviewer_two-channel.dbc", 6),
        ("ct_system-float-values.arxml", 1),
        ("ct_ecu-extract-4.2.arxml", 0),   # 返回空 dict
        ("ct_system-3.2.3.arxml", 0),      # 找不到 CAN cluster
    ],
)
def test_third_party_database_matrix(fixture_name, expect_frames):
    """真实数据库兼容性矩阵：逐份过 canmatrix，帧数即回归基线。"""
    try:
        path = bd.path_for(fixture_name)
    except bd.BusDataMissing:
        pytest.skip("真实夹具未下载")
    except bd.BusDataCorrupted as e:
        pytest.fail(str(e))

    db = CanDatabaseSet().load([str(path)])
    assert db.reports[0].ok is True, db.reports[0].error
    assert db.reports[0].frame_count == expect_frames


def test_blfviewer_pair_matches_canmatrix_exactly():
    """交叉验证（数据库层）：cantools 与 canmatrix 对同一份真实 DBC 的
    帧列表 / 位段 / factor / offset / 字节序必须逐项一致。"""
    cantools = pytest.importorskip("cantools")
    try:
        path = bd.path_for("blfviewer_two-channel.dbc")
    except bd.BusDataMissing:
        pytest.skip("真实夹具未下载")
    except bd.BusDataCorrupted as e:
        pytest.fail(str(e))

    ref = cantools.database.load_file(str(path))
    db = CanDatabaseSet().load([str(path)])

    ours = {f.can_id: f for f in db.frames}
    assert set(ours) == {m.frame_id for m in ref.messages}

    for message in ref.messages:
        frame = ours[message.frame_id]
        assert frame.dlc == message.length
        assert frame.name == message.name
        for sig in message.signals:
            got = frame.signal_by_name(sig.name)
            assert got is not None, f"{frame.name}.{sig.name}"
            assert got.size == sig.length
            # cantools 的 byte_order 取值是 little_endian / big_endian
            assert got.little_endian == (sig.byte_order == "little_endian")
            assert got.signed == sig.is_signed
            assert got.factor == pytest.approx(float(sig.scale))
            assert got.offset == pytest.approx(float(sig.offset))
            assert got.dbc_start_bit == sig.start


# ---------------------------------------------------------------------------
# canmatrix 噪声降噪：compu method 缺失逐条 ERROR → 每文件一条汇总 WARNING
# ---------------------------------------------------------------------------


def _emit_compu_method_error(name: str) -> None:
    import logging

    logging.getLogger("canmatrix.formats.arxml").error(
        "No valid compu method found for isignal/systemsignal "
        "%s/%s_54edbe9fa0e84f47 - check ARXML file!!",
        name, name,
    )


def test_compu_method_error_spam_is_captured_and_summarized(
    monkeypatch, tmp_path, caplog
):
    """逐信号 ERROR 在解析窗口内被吞掉，收敛为每文件一条 WARNING + tooltip 注记。

    真实诱因是 ARXML 演示库 5 个 isignal 未挂 compu method，而一次 BLF 导入
    会把数据库解析 2~4 遍 → 同一批 ERROR 刷屏。用假 _load_matrix_list 在捕获
    窗口内复现日志行为，不依赖手造坏 ARXML。
    """
    import logging

    def fake_load_matrix_list(self, path):
        _emit_compu_method_error("Signal1")
        _emit_compu_method_error("Signal1")  # 同名去重
        _emit_compu_method_error("Signal9")
        return []

    monkeypatch.setattr(CanDatabaseSet, "_load_matrix_list", fake_load_matrix_list)
    with caplog.at_level(logging.DEBUG):
        db = CanDatabaseSet().load([str(tmp_path / "bad.arxml")])

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], "逐条 compu method ERROR 应在窗口内被吞掉"
    summaries = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "缺少有效换算方法" in r.getMessage()
    ]
    assert len(summaries) == 1
    assert "Signal1, Signal9" in summaries[0].getMessage()
    note = db.reports[0].warning
    assert "2 个信号缺少换算方法" in note and "Signal1" not in note


def test_other_canmatrix_records_still_reach_root(
    monkeypatch, tmp_path, caplog
):
    """降噪只针对 compu method 模式，canmatrix 的其余记录必须原样放行。"""
    import logging

    def fake_load_matrix_list(self, path):
        logging.getLogger("canmatrix.formats.arxml").warning("真实告警别吞我")
        return []

    monkeypatch.setattr(CanDatabaseSet, "_load_matrix_list", fake_load_matrix_list)
    with caplog.at_level(logging.DEBUG):
        CanDatabaseSet().load([str(tmp_path / "bad.arxml")])

    assert any(
        r.name.startswith("canmatrix") and "真实告警别吞我" in r.getMessage()
        for r in caplog.records
    )


def test_capture_window_restores_logger_state():
    """捕获窗口退出后 propagate 与 handler 列表必须恢复原状。"""
    import logging

    from src.data.can_database import _capture_canmatrix_noise

    cm_logger = logging.getLogger("canmatrix")
    handlers_before = list(cm_logger.handlers)
    propagate_before = cm_logger.propagate
    with _capture_canmatrix_noise() as collector:
        assert cm_logger.propagate is False
        _emit_compu_method_error("Signal14")
    assert collector.missing_signals == ["Signal14"]
    assert cm_logger.propagate == propagate_before
    assert list(cm_logger.handlers) == handlers_before
