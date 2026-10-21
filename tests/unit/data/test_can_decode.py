"""``src/data/can_decode.py`` 的单元测试（方案 §7 P1 的验收主体）。

真值来源是 **canmatrix 自己的 ``Frame.decode()``**：向量化路径与逐帧路径必须
逐位一致，否则画出来的曲线与 CANoe / Vector 工具链对不上。为此

* 每类位段（Intel 无符号 / Intel 有符号 / Motorola 无符号 / Motorola 有符号 /
  IEEE float / 枚举 / 多路复用）各跑 ``CATEGORY_MIN`` 条**随机**报文，并把实际
  比对次数断言出来 —— 「各 ≥200 随机例」是验收条款，不该靠人肉数样本；
* 兜底路径（``size > 64``）拿不到 canmatrix 的 DBC 定义（实测 canmatrix 直接
  拒绝 72 bit 的 ``SG_`` 行），因此改用**手算的确定值**（2^72-1、-1、2^64）。

随机种子固定，失败可复现。
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from src.data.can_database import CanDatabaseSet, CanFrameSpec, CanSignalSpec  # noqa: E402
from src.data.can_decode import (  # noqa: E402
    _rows_view,
    decode_physical,
    decode_rows,
    decode_single,
    extract_raw,
    mux_validity,
    row_validity,
)
from tests.fixtures.data_factory import write_dbc  # noqa: E402

# 每类位段要求的随机对照样本数（方案 §7 P1：各 ≥200 例）
CATEGORY_MIN = 200
ROWS = 240
SEED = 0xC0FFEE


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> CanDatabaseSet:
    path = write_dbc(tmp_path_factory.mktemp("candec") / "syn.dbc")
    return CanDatabaseSet([str(path)])


def _payloads(n: int, dlc: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, dlc), dtype=np.uint8)


def _canmatrix_truth(frame: CanFrameSpec, row: bytes) -> dict:
    """canmatrix 的逐帧解码结果：name -> (raw_value, phys_value)。"""
    decoded = frame.cm_frame.decode(bytearray(row))
    return {k: (v.raw_value, v.phys_value) for k, v in decoded.items()}


def _same(a: float, b: float) -> bool:
    """NaN 与 NaN 视为一致：float 信号会解出 signaling NaN（未初始化数据区）。"""
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, rtol=1e-12, atol=1e-12))


# ---------------------------------------------------------------------------
# 与 canmatrix 的逐位随机对照
# ---------------------------------------------------------------------------

# can_id -> (帧名, 该帧覆盖的位段类别)
_FRAMES = {
    0x1A0: ("EngineData", ["intel_unsigned", "intel_signed", "enum", "float32"]),
    0x2C0: ("ChassisData", ["motorola_unsigned", "motorola_signed"]),
    0x3B0: ("DiagMux", ["mux"]),
}


def test_vectorized_decode_matches_canmatrix_on_random_frames(db):
    """核心 property test：每帧每行都过 canmatrix，逐信号比对 raw 与 phys。

    覆盖类别与样本数在末尾断言，避免「只跑到某一类就结束」的假绿。
    """
    covered: dict[str, int] = {
        "intel_unsigned": 0, "intel_signed": 0, "enum": 0, "float32": 0,
        "motorola_unsigned": 0, "motorola_signed": 0, "mux": 0,
    }

    for can_id, (_name, categories) in _FRAMES.items():
        frame = db.find_frame(0, can_id)
        assert frame is not None, hex(can_id)
        rows = _payloads(ROWS, frame.dlc, SEED + can_id)
        with np.errstate(invalid="ignore"):
            decoded_all = {
                sig.name: decode_physical(rows[:, :], sig, frame=frame)
                for sig in frame.signals
            }
            raw_all = {sig.name: extract_raw(rows, sig) for sig in frame.signals}

        for i in range(ROWS):
            row = rows[i].tobytes()
            truth = _canmatrix_truth(frame, row)
            assert truth, f"{frame.name} 第 {i} 行 canmatrix 解不出任何信号"
            for sig_name, (raw_ref, phys_ref) in truth.items():
                sig = frame.signal_by_name(sig_name)
                assert sig is not None, sig_name
                got_raw = raw_all[sig_name][i]
                got_phys = decoded_all[sig_name][i]
                assert _same(float(got_raw), float(raw_ref)), (
                    f"{frame.name}.{sig_name} row={row.hex()} "
                    f"raw {got_raw!r} != canmatrix {raw_ref!r}"
                )
                assert _same(float(got_phys), float(phys_ref)), (
                    f"{frame.name}.{sig_name} row={row.hex()} "
                    f"phys {got_phys!r} != canmatrix {phys_ref!r}"
                )
            for cat in categories:
                covered[cat] += 1

    for category, count in covered.items():
        assert count >= CATEGORY_MIN, f"{category} 类只比对了 {count} 例"


def test_decode_rows_fallback_agrees_with_vectorized(db):
    """两条路径必须给同一个数：回落路径不是「近似」，是等价。"""
    for can_id in _FRAMES:
        frame = db.find_frame(0, can_id)
        rows = _payloads(60, frame.dlc, SEED ^ can_id)
        for sig in frame.signals:
            with np.errstate(invalid="ignore"):
                fast = decode_physical(rows, sig, frame=frame)
                slow = decode_rows(rows, sig, frame=frame)
            for a, b in zip(fast, slow):
                assert _same(float(a), float(b)), f"{frame.name}.{sig.name}: {a} != {b}"


# ---------------------------------------------------------------------------
# 有符号补码 / 上下界
# ---------------------------------------------------------------------------

def test_signed_int8_covers_every_code_exhaustively(db):
    """CoolantTemp（8 bit signed, offset -40）：256 个码值穷举，无抽样。

    补码换算是「一个符号位错就整条曲线偏 512」的那类 bug，只测随机样例不够。
    """
    sig = db.find_frame(0, 0x1A0).signal_by_name("CoolantTemp")
    assert (sig.start_bit, sig.size) == (16, 8)
    # 256 行，每行第 2 字节就是 CoolantTemp；其余字节置 0（不影响本信号的位段）
    rows = np.zeros((256, 8), np.uint8)
    rows[:, 2] = np.arange(256, dtype=np.uint8)

    raw = extract_raw(rows, sig)
    assert list(raw[:128]) == list(range(0, 128))
    assert list(raw[128:132]) == [-128, -127, -126, -125]
    assert raw[255] == -1

    phys = decode_physical(rows, sig)
    assert phys[0] == pytest.approx(-40.0)
    assert phys[127] == pytest.approx(87.0)
    assert phys[128] == pytest.approx(-168.0)


def test_signed_16bit_boundary(db):
    sig = db.find_frame(0, 0x1A0).signal_by_name("EngineSpeed")
    assert sig.signed is False
    rows = np.zeros((1, 8), np.uint8)
    rows[0, 0:2] = [0xFF, 0x7F]        # 0x7FFF 小端
    assert decode_physical(rows, sig)[0] == pytest.approx(0x7FFF * 0.125)


def test_motorola_two_complement_uses_declared_width_not_byte_width(db):
    """WheelSpeedFR：12 bit 有符号，符号位在段内第 12 位而不是第 16 位。

    按字节宽度判符号是最常见的实现错误，会静默让所有负值变巨大正值。
    """
    sig = db.find_frame(0, 0x2C0).signal_by_name("WheelSpeedFR")
    assert (sig.size, sig.signed) == (12, True)
    row = bytes([0x00, 0xF8])          # 内部位号 12 → 取到 0xFF8 的低 12 位 = 0xFF8
    decoded = db.find_frame(0, 0x2C0)
    truth = _canmatrix_truth(decoded, row + b"\x00" * 6)["WheelSpeedFR"]
    rows = np.frombuffer(row + b"\x00" * 6, dtype=np.uint8).reshape(1, 8)
    got = decode_physical(rows, sig)[0]
    assert _same(float(got), float(truth[1])), f"{got} != canmatrix {truth[1]}"
    assert got < 0, "0xFF8 的 12 bit 补码应为负"


# ---------------------------------------------------------------------------
# IEEE float
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "size,little_endian,fmt",
    [
        (32, True, "<f"), (32, False, ">f"),
        (64, True, "<d"), (64, False, ">d"),
    ],
)
def test_float_bits_are_viewed_not_scaled(size, little_endian, fmt):
    """float 信号的 raw 就是浮点数本身，factor 不得再乘一次。

    字节序两者不同不能合并：Intel 下报文字节是浮点的 **小端** 存放，Motorola
    下是大端；而解码统一把已取出的位段当大端位模式重解（``>u4 → >f4``），
    所以预期值得按各自的字节序打包。
    """
    import struct

    payload = np.zeros((1, 16), np.uint8)
    packed = np.frombuffer(struct.pack(fmt, -12.75), dtype=np.uint8)
    payload[0, : len(packed)] = packed
    sig = _spec(size=size, little_endian=little_endian, is_float=True,
                factor=2.0, offset=100.0)
    row = payload[0, : size // 8].tobytes()
    assert decode_single(row, sig) == pytest.approx(-12.75)
    assert decode_rows(payload[:, : size // 8], sig)[0] == pytest.approx(-12.75)
    assert extract_raw(payload[:, : size // 8], sig)[0] == pytest.approx(-12.75)
    # factor/offset 对 float 信号不再生效（否则会变成双重换算）
    assert decode_single(row, sig) != pytest.approx(-12.75 * 2.0 + 100.0)


def test_float32_nan_from_uninitialised_region_is_not_an_error(db):
    """TorqueLimit 解出 NaN 是**数据状态**，不是解码失败。

    报文的未初始化区常是 0xff8b0af1 这类 signaling NaN 位模式；这里既要得到
    NaN，也不能让 numpy 的 invalid 告警冒出来（会污染测试输出与真实日志）。
    """
    sig = db.find_frame(0, 0x1A0).signal_by_name("TorqueLimit")
    assert sig.is_float is True and sig.start_bit == 32
    # 字节 4..7 小端读出 0xFF8B0AF1 → 全 1 指数 + 非零尾数 = NaN
    rows = np.frombuffer(bytes.fromhex("00000000f10a8bff"), np.uint8).reshape(1, 8)
    with np.errstate(invalid="raise"):
        value = decode_physical(rows, sig)[0]
    assert np.isnan(value)


# ---------------------------------------------------------------------------
# 多路复用
# ---------------------------------------------------------------------------

def test_mux_validity_masks_the_other_branch_to_nan(db):
    """非本 mux 值的采样点必须填 NaN（§2）：否则残留值会画成假台阶。"""
    frame = db.find_frame(0, 0x3B0)
    sel = frame.signal_by_name("MuxSel")
    a = frame.signal_by_name("BranchA")
    b = frame.signal_by_name("BranchB")
    assert (a.mux_value, b.mux_value) == (0, 1)

    rows = np.zeros((4, 8), np.uint8)
    rows[:, 0] = [0, 1, 0, 1]
    rows[:, 1:3] = np.array([[0x11, 0x22]] * 4)

    with np.errstate(invalid="ignore"):
        va = decode_physical(rows, a, mux_validity(rows, frame, a))
        vb = decode_physical(rows, b, mux_validity(rows, frame, b))

    assert not np.isnan(va[0]) and not np.isnan(va[2])
    assert np.isnan(va[1]) and np.isnan(va[3])
    assert np.isnan(vb[0]) and np.isnan(vb[2])
    assert not np.isnan(vb[1]) and not np.isnan(vb[3])

    # 选择器自身永远有效
    assert mux_validity(rows, frame, sel) is None
    assert np.all(np.isfinite(decode_physical(rows, sel)))


def test_mux_validity_is_none_for_plain_frames_and_public_signals(db):
    frame = db.find_frame(0, 0x1A0)
    rows = _payloads(8, frame.dlc, 7)
    assert mux_validity(rows, frame, frame.signal_by_name("EngineSpeed")) is None

    muxed = db.find_frame(0, 0x3B0)
    public = CanSignalSpec(
        name="Public", start_bit=24, size=8, little_endian=True, signed=False,
        is_float=False, factor=1.0, offset=0.0, unit="", minimum=None,
        maximum=None, values=None, comment="", mux=None, mux_value=None,
        receivers=(), cycle_time_s=None, db_path="", frame_name="DiagMux",
        can_id=0x3B0, extended=False,
    )
    assert mux_validity(_payloads(8, 8, 9), muxed, public) is None


def test_mux_group_falls_back_to_mux_value_when_the_name_is_missing(db):
    """L2 回归：spec 只填了 ``mux_value`` 时仍要按组掩码。

    canmatrix 1.2 把组名放在 ``mux``；若某个解析器（或手搓 spec）只填组号，
    只看 ``mux`` 会把非本组采样点当公共信号全放行，残留值就被画成台阶。
    """
    frame = db.find_frame(0, 0x3B0)
    legacy = replace(frame.signal_by_name("BranchB"), mux=None)
    assert legacy.mux_value == 1
    assert not legacy.is_mux_selector

    rows = np.zeros((4, 8), np.uint8)
    rows[:, 0] = [0, 1, 0, 1]
    np.testing.assert_array_equal(
        mux_validity(rows, frame, legacy), [False, True, False, True]
    )


def test_a_validity_mask_of_the_wrong_length_is_ignored_but_logged(db, caplog):
    """L7 回归：长度不一致不能静默当「全部有效」。

    行为仍是忽略错位掩码（按位应用会张冠李戴），但必须留下可追的日志证据，
    否则 mux / 短行掩码失效在线上完全看不出来。
    """
    frame = db.find_frame(0, 0x1A0)
    sig = frame.signal_by_name("EngineSpeed")
    rows = np.zeros((3, 8), np.uint8)
    with caplog.at_level("WARNING", logger="src.data.can_decode"):
        values = decode_physical(rows, sig, np.array([True, False]))
    assert np.isfinite(values).all()
    assert any("掩码长度" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 向量化不适用 → 回落
# ---------------------------------------------------------------------------

def _spec(**kw) -> CanSignalSpec:
    base = dict(
        name="S", start_bit=0, size=8, little_endian=True, signed=False,
        is_float=False, factor=1.0, offset=0.0, unit="", minimum=None,
        maximum=None, values=None, comment="", mux=None, mux_value=None,
        receivers=(), cycle_time_s=None, db_path="", frame_name="F",
        can_id=1, extended=False,
    )
    base.update(kw)
    return CanSignalSpec(**base)


@pytest.mark.parametrize("size", [0, 65, 72, 256])
def test_extract_raw_rejects_widths_outside_uint64(size):
    sig = _spec(size=size)
    rows = np.zeros((3, 64), np.uint8)
    with pytest.raises(ValueError, match="位宽|uint64"):
        extract_raw(rows, sig)


def test_wide_signal_falls_back_to_bigint_with_hand_computed_values():
    """``size > 64`` 的期望值**手算**：canmatrix 拒绝 72 bit 的 DBC 行（实测
    ``error with line no: 10``），拿不到它的逐帧真值，只能用确定的边界数。
    """
    sig = _spec(size=72)
    all_ff = bytes([0xFF]) * 9
    assert decode_single(all_ff, sig) == float(2 ** 72 - 1)

    signed = _spec(size=72, signed=True)
    assert decode_single(all_ff, signed) == -1.0
    assert decode_single(bytes(8) + b"\x01", signed) == 2 ** 64
    assert decode_single(bytes(8) + b"\x80", signed) == -(2 ** 71)

    rows = np.frombuffer(all_ff, np.uint8).reshape(1, 9)
    assert decode_rows(rows, sig)[0] == float(2 ** 72 - 1)
    # 同一行走向量化路径必须显式不适用，而不是悄悄截断成 64 bit
    with pytest.raises(ValueError):
        extract_raw(rows, sig)


def test_rows_view_avoids_a_copy_when_the_window_is_the_whole_row():
    """M1：``use == stride`` 时窗口就是整行，瞬时副本必须省掉。

    ``ascontiguousarray`` 对该情形本就返回原数组；但这里用「写穿原数组」把
    「共享内存」钉成契约，避免以后改成切片/astype 时悄悄多一份 N×stride。
    """
    payload = np.arange(24, dtype=np.uint8).reshape(3, 8)
    whole = _rows_view(payload, 8)
    payload[1, 2] = 0xAB
    assert whole[8 + 2] == 0xAB

    # 切列（use < stride）必须仍是逐行连续的新窗口
    cut = _rows_view(payload, 4)
    assert cut.tobytes() == payload[:, :4].tobytes()


class _FakeCanMatrixFrame:
    """canmatrix ``Frame`` 的最小替身：记录每帧喂进来的字节，返回预设解码表。

    刻意不用真 DBC 造容器帧 —— canmatrix 的 PDU container 需要 COM_TX 属性 +
    子报文嵌套，合成成本高，而这里要钉的是**分流**（谁被调用、传了什么），
    不是 canmatrix 自身的解码正确性（那由上面各节随机对照用例钉）。
    """

    def __init__(self, table: dict[bytes, dict] | None = None, fail: bool = False):
        self.calls: list[bytes] = []
        self._table = table or {}
        self._fail = fail

    def decode(self, data):
        self.calls.append(bytes(data))
        if self._fail:
            raise ValueError("canmatrix 拒绝该帧")
        return self._table.get(bytes(data), {})


def _frame(**kw) -> CanFrameSpec:
    base = dict(name="Container", can_id=0x100, extended=False, dlc=8,
                is_pdu_container=True)
    base.update(kw)
    return CanFrameSpec(**base)


def test_container_frame_goes_to_canmatrix_even_for_narrow_signals():
    """容器帧的信号几乎都 ≤64 bit：按旧的 ``size > 64`` 门禁分流的话，这类帧会整帧
    走向量化位提取，照着父帧的连续位序去切子报文 —— 不报错，只是画出一条**错曲线**。
    """
    sig = _spec(name="Sub", size=16, start_bit=0, frame_name="Container", can_id=0x100)
    rows = np.array([[0x01, 0x02, 0, 0, 0, 0, 0, 0],
                     [0x03, 0x04, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    truth = {rows[0].tobytes(): {"Sub": 7.0}, rows[1].tobytes(): {"Sub": 8.0}}
    frame = _frame(cm_frame=_FakeCanMatrixFrame(truth))

    got = decode_rows(rows, sig, frame)
    assert list(got) == [7.0, 8.0]
    assert frame.cm_frame.calls == [rows[0].tobytes(), rows[1].tobytes()]
    # 反向对照：不传 frame 时走位提取，得到的正是那个「看起来很像样」的错值
    assert decode_rows(rows, sig)[0] == 513.0


def test_complex_muxed_frame_is_not_decoded_by_the_vectorized_path():
    """分流只看帧布局，不看信号位宽：复杂多路复用与容器帧同一条回落路径。"""
    sig = _spec(name="Sub", size=8, start_bit=0, mux=1, mux_value=2)
    frame = _frame(is_pdu_container=False, is_complex_multiplexed=True,
                   is_multiplexed=True, cm_frame=_FakeCanMatrixFrame(
                       {bytes([0xFF] + [0] * 7): {"Sub": 42.0}}))
    rows = np.array([[0xFF] + [0] * 7], dtype=np.uint8)
    assert decode_rows(rows, sig, frame)[0] == 42.0
    assert len(frame.cm_frame.calls) == 1


def test_container_signal_absent_from_the_decode_table_stays_nan():
    """canmatrix 解不出的信号（未选中的复用组、别的子报文）必须是 NaN。

    掩码不另外叠加：缺键本身就是 canmatrix 给出的有效性语义，再叠一层向量化
    mux 掩码会把「本帧确实没有这个信号」和「残留值」混成同一个数。
    """
    sig = _spec(name="Absent", size=8, start_bit=0)
    rows = np.zeros((2, 8), dtype=np.uint8)
    frame = _frame(cm_frame=_FakeCanMatrixFrame({rows[0].tobytes(): {"Other": 1.0}}))

    got = decode_rows(rows, sig, frame)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert len(frame.cm_frame.calls) == 2


def test_row_fallback_warns_with_the_sample_count_for_large_series(
    monkeypatch, caplog
):
    """M1：逐帧回落是纯 Python 循环，卡顿必须能在日志里对上规模。"""
    monkeypatch.setattr("src.data.can_decode._ROW_FALLBACK_WARN_ROWS", 1)
    frame = _frame(cm_frame=_FakeCanMatrixFrame({}))
    sig = _spec(name="Sub", size=8)
    rows = np.zeros((2, 8), np.uint8)

    with caplog.at_level("WARNING", logger="src.data.can_decode"):
        decode_rows(rows, sig, frame)

    messages = [r.getMessage() for r in caplog.records]
    assert any("2 个采样点" in m and "Container" in m for m in messages), messages


def test_container_fallback_without_cm_frame_is_nan_and_warned(monkeypatch):
    """``cm_frame`` 缺失时不静默退回位提取：宁可断线，不给错值，并且要留告警。"""
    warnings: list[tuple] = []

    class _Recorder:
        def warning(self, msg, *args, **kw):
            warnings.append((msg, args))

        def __getattr__(self, _name):
            return lambda *a, **kw: None

    monkeypatch.setattr("src.data.can_decode.logger", _Recorder())

    sig = _spec(name="Sub", size=16, start_bit=0)
    frame = _frame(cm_frame=None)
    got = decode_rows(np.zeros((3, 8), dtype=np.uint8), sig, frame)
    assert np.all(np.isnan(got))
    assert warnings, "无 canmatrix 兜底时必须告警，否则用户只看到一条空曲线"
    assert any("不适用向量化" in m % a if a else "不适用向量化" in m
               for m, a in warnings)


def test_container_decode_error_fills_nan_once_warned(monkeypatch):
    """逐帧解码抛异常：本帧及后续都置空，但**只告警一次**（万帧日志会淹没文件）。"""
    warnings: list[str] = []

    class _Recorder:
        def warning(self, msg, *args, **kw):
            warnings.append(str(msg) % args if args else str(msg))

        def __getattr__(self, _name):
            return lambda *a, **kw: None

    monkeypatch.setattr("src.data.can_decode.logger", _Recorder())

    sig = _spec(name="Sub", size=8, start_bit=0)
    frame = _frame(cm_frame=_FakeCanMatrixFrame(fail=True))
    got = decode_rows(np.zeros((50, 8), dtype=np.uint8), sig, frame)
    assert np.all(np.isnan(got))
    assert len(frame.cm_frame.calls) == 50, "失败后仍逐帧尝试（保持行对齐）"
    assert len(warnings) == 1


def test_motorola_window_crossing_the_uint64_boundary():
    """Motorola 且 ``start_bit + size > 64``：低 64 位装不下，缺口来自第 9 字节。

    这条分支无法从合成 DBC 得到（帧只有 8 字节），故直接构造 spec，并用一份
    **独立写出的**大整数参考式对照 —— 参考式只依赖模块头部声明的移位规则。
    """
    sig = _spec(size=64, little_endian=False, start_bit=7)
    rows = np.arange(9, dtype=np.uint8).reshape(1, 9) + 1

    total = 9 * 8
    ref = (int.from_bytes(rows[0].tobytes(), "big") >> (total - (7 + 64))) & ((1 << 64) - 1)
    got = extract_raw(rows, sig)[0]
    # 无符号 raw 按契约走 float64（见 extract_raw 文档），> 2^53 必丢精度，
    # 所以比的是「同一量级」而不是逐位相等；逐位由下面的 int 路径保证。
    assert _same(float(got), float(ref)), f"{int(got)} != {int(ref)}"
    assert int(got) != 0
    assert _same(float(decode_rows(rows, sig)[0]), float(ref))
    # 缺口必须真的来自第 9 字节：去掉它就与 64 位窗口不同，能区分「静默截断」
    assert int(extract_raw(rows[:, :8], sig)[0]) != int(ref)


def test_short_payload_yields_nan_instead_of_truncating_silently():
    """``valid_bytes`` 小于 dlc 的历史样本（pcan_issue_1905 那一类）：位段越界
    必须给 NaN，不能按缺 0 补全后给出一个看似合理的错值。
    """
    sig = _spec(size=32, start_bit=16)
    row = bytes([0x01, 0x02])            # 只有 2 字节，位段要到 48
    assert np.isnan(decode_single(row, sig))
    rows = np.frombuffer(row, np.uint8).reshape(1, 2)
    assert np.isnan(decode_rows(rows, sig)[0])


def test_row_validity_uses_the_internal_bit_span(db):
    """行级有效性按 canmatrix 内部位号判断，Intel 与 Motorola 同一公式。"""
    engine = db.find_frame(0, 0x1A0)
    speed = engine.signal_by_name("EngineSpeed")      # start 0, size 16
    torque = engine.signal_by_name("TorqueLimit")     # start 32, size 32
    lengths = np.array([0, 1, 2, 5, 6, 8], dtype=np.uint8)
    np.testing.assert_array_equal(
        row_validity(lengths, speed), [False, False, True, True, True, True]
    )
    np.testing.assert_array_equal(
        row_validity(lengths, torque), [False, False, False, False, False, True]
    )

    chassis = db.find_frame(0, 0x2C0)
    fl = chassis.signal_by_name("WheelSpeedFL")       # internal start 0, size 12
    fr = chassis.signal_by_name("WheelSpeedFR")       # internal start 12, size 12
    lengths = np.array([0, 1, 2, 3], dtype=np.uint8)
    np.testing.assert_array_equal(
        row_validity(lengths, fl), [False, False, True, True]
    )
    np.testing.assert_array_equal(
        row_validity(lengths, fr), [False, False, False, True]
    )
    assert row_validity(None, speed) is None


def test_decode_paths_apply_row_validity_to_padded_rows(db):
    """同一桶里混有长短行：补 0 只用于对齐存储，不能变成物理值。"""
    frame = db.find_frame(0, 0x1A0)
    sig = frame.signal_by_name("TorqueLimit")         # float32 @ byte 4..7
    payload = np.zeros((2, 8), dtype=np.uint8)
    payload[0, 4:8] = [0x00, 0x00, 0x80, 0x3F]        # 1.0f
    payload[1, 4:8] = [0x00, 0x00, 0x80, 0x3F]
    lengths = np.array([4, 8], dtype=np.uint8)
    mask = row_validity(lengths, sig)

    physical = decode_physical(payload, sig, mask, frame)
    assert np.isnan(physical[0])
    assert physical[1] == pytest.approx(1.0)

    rows = decode_rows(payload, sig, frame, mask, lengths)
    assert np.isnan(rows[0])
    assert rows[1] == pytest.approx(1.0)


def test_mux_validity_rejects_a_selector_from_padding(tmp_path):
    """选择器本身在缺失字节里时，补出来的 0 不能被当成 mux 组 0。"""
    from src.data.can_database import CanDatabaseSet
    from tests.fixtures.data_factory import write_dbc as _write_dbc

    # 反向布局：目标信号在 byte0，选择器在 byte1。1 字节行只含目标信号，
    # 选择器是补出来的 0；若不做选择器行级校验，它会被误判为组 0 有效。
    path = _write_dbc(
        tmp_path / "tail_mux.dbc",
        [{
            "name": "TailMux", "can_id": 0x400, "size": 8,
            "signals": (
                {"name": "Tail", "start": 0, "size": 8, "order": "intel",
                 "factor": 1.0, "offset": 0.0, "minimum": 0.0,
                 "maximum": 255.0, "mux": 0},
                {"name": "Sel", "start": 8, "size": 8, "order": "intel",
                 "factor": 1.0, "offset": 0.0, "minimum": 0.0,
                 "maximum": 255.0, "mux": "M"},
            ),
        }],
    )
    frame = CanDatabaseSet([str(path)]).find_frame(0, 0x400)
    tail = frame.signal_by_name("Tail")
    rows = np.zeros((3, 8), dtype=np.uint8)
    rows[:, 1] = [0, 0, 5]                            # 第 3 行 Sel=5，不属于组 0
    lengths = np.array([1, 2, 2], dtype=np.uint8)
    # 第 1 行选择器缺失（补 0）⇒ 无效；第 2 行选择器完整且为组 0 ⇒ 有效；
    # 第 3 行选择器为 5 ⇒ 不属于组 0。目标信号自身的位段由 row_validity 合并。
    np.testing.assert_array_equal(
        mux_validity(rows, frame, tail, lengths), [False, True, False]
    )
    # 去掉行级信息时退回旧语义（只看字节，选择器补 0 会命中组 0）：
    np.testing.assert_array_equal(
        mux_validity(rows, frame, tail), [True, True, False]
    )


def test_mux_validity_only_checks_the_selector_span(db):
    """mux 掩码只管选择器；目标位段缺失由调用方与 row_validity 合并。"""
    frame = db.find_frame(0, 0x3B0)
    branch = frame.signal_by_name("BranchA")           # start 8, size 16
    rows = np.zeros((4, 8), dtype=np.uint8)            # Sel=0，BranchA 组
    lengths = np.array([0, 1, 2, 3], dtype=np.uint8)
    selector_only = mux_validity(rows, frame, branch, lengths)
    np.testing.assert_array_equal(selector_only, [False, True, True, True])
    combined = selector_only & row_validity(lengths, branch)
    np.testing.assert_array_equal(combined, [False, False, False, True])


def test_payload_shape_and_dtype_are_validated():
    sig = _spec()
    with pytest.raises(ValueError, match="uint8"):
        extract_raw(np.zeros((3, 8), np.int16), sig)
    with pytest.raises(ValueError, match="uint8"):
        extract_raw(np.zeros(8, np.uint8), sig)


# ---------------------------------------------------------------------------
# 物理换算
# ---------------------------------------------------------------------------

def test_factor_and_offset_are_applied_after_the_raw_extraction(db):
    """顺序错了不会报错，只会整体偏移：用 factor≠1 的信号显式锁定。"""
    sig = db.find_frame(0, 0x1A0).signal_by_name("EngineSpeed")
    rows = np.zeros((1, 8), np.uint8)
    rows[0, 0:2] = [0x10, 0x27]                      # raw = 10000
    assert extract_raw(rows, sig)[0] == pytest.approx(10000)
    assert decode_physical(rows, sig)[0] == pytest.approx(1250.0)
    truth = _canmatrix_truth(db.find_frame(0, 0x1A0), rows[0].tobytes())["EngineSpeed"]
    assert Decimal(str(1250.0)) == Decimal(truth[1])


def test_enum_signal_decodes_raw_codes_not_text(db):
    """枚举信号走 raw 码值（与 MDF 一致），文本留给上层用 enum_map 映射。"""
    sig = db.find_frame(0, 0x1A0).signal_by_name("StarterMode")
    assert sig.is_enum is True
    rows = np.zeros((3, 8), np.uint8)
    rows[:, 3] = [0, 1, 2]
    values = decode_physical(rows, sig)
    assert list(values) == [0.0, 1.0, 2.0]
    assert sig.values == {0: "off", 1: "run", 2: "error"}
