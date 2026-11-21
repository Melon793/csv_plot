"""编码判定链测试：单位符号投票（A‴）纯函数 + 端到端矩阵。

背景：cp1252 导出的 °C(B0 43) 会被 gb18030 吞成「癈」；charset_normalizer
对单位文件常误判 cp850/iso8859_16/cp949/big5。投票规则与跳过条件见
src/data/loader.py 的 _build_encoding_chain。
"""

from __future__ import annotations

import csv
import io

import pytest

from src.data.loader import (
    ENCODING_FALLBACKS,
    FastDataLoader,
    _build_encoding_chain,
    _skip_symbol_vote,
)
from tests.fixtures.data_factory import make_simple_rows, write_csv


def _csv_bytes(units: list[str], *, sep: str = ",", crlf: bool = False,
               desc: str | None = None) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL, delimiter=sep,
                   lineterminator="\r\n" if crlf else "\n")
    if desc is not None:
        buf.write(desc + ("\r\n" if crlf else "\n"))
    w.writerow([f"VAR_{i:04d}" for i in range(len(units))])
    w.writerow(units)
    w.writerow([1.5] * len(units))
    return buf.getvalue().encode("utf-8")


def _load_units(path, *, has_unit: bool | None = None) -> tuple[list[str], str]:
    """走生产路径：auto_detect 选码 → 显式传码读表头/单位。

    has_unit 可显式覆盖：单位行关键词检测是另一层逻辑（中文/西里尔单位词
    不在关键词表内属既有行为），编码测试聚焦解码层。
    """
    fmt = FastDataLoader._auto_detect_format(str(path))
    names, units, used, _ = FastDataLoader._load_header_units(
        str(path), desc_rows=fmt.header_row, sep=fmt.sep or ",",
        has_unit=fmt.has_unit if has_unit is None else has_unit,
        encoding=fmt.encoding,
    )
    return [units[n] for n in names], used


# ---------- 纯函数：投票与跳过条件 ----------

class TestBuildEncodingChain:
    def test_det_none_defaults_utf8_first(self):
        chain = _build_encoding_chain(b"a,b\n1,2\n", None)
        assert chain[0] == "utf-8"
        assert chain[-1] == "latin-1"
        assert len(chain) == len(set(chain))

    def test_pure_ascii_keeps_fallback_order(self):
        chain = _build_encoding_chain(b"a,b\n1,2\n", "ascii")
        assert chain == ["ascii", *ENCODING_FALLBACKS]

    @pytest.mark.parametrize("det", ["cp850", "cp949", "big5", "iso8859_16"])
    def test_vote_promotes_cp1252_over_misdetection(self, det):
        sample = '"V1","V2"\n"°C","µm"\n1,2\n'.encode("cp1252")
        chain = _build_encoding_chain(sample, det)
        assert chain[0] == "cp1252"
        assert det in chain  # 不删候选

    def test_sparse_single_degree_still_votes(self):
        sample = ('"' + '","'.join([""] * 100 + ["°C"]) + '"\n').encode("cp1252")
        chain = _build_encoding_chain(sample, "big5")
        assert chain[0] == "cp1252"

    def test_skip_when_det_maps_symbols_identically(self):
        sample = '"°C","°C"\n'.encode("cp1252")
        assert _skip_symbol_vote(sample, "cp1250") is True
        chain = _build_encoding_chain(sample, "cp1250")
        assert chain[0] == "cp1250"

    def test_skip_cp1251_with_cyrillic_content(self):
        sample = '"°C","об/мин","Скорость"\n'.encode("cp1251")
        assert _skip_symbol_vote(sample, "cp1251") is True
        chain = _build_encoding_chain(sample, "cp1251")
        assert chain[0] == "cp1251"

    def test_iso8859_5_not_trusted_for_cyrillic(self):
        # iso8859_5 的 B0 是西里尔 А 而非 °，°C 占多数时投票须纠正而非信任
        sample = _csv_bytes(["°C"] * 20 + ["об/мин", "Скорость"]) \
            .decode("utf-8").encode("cp1251")
        chain = _build_encoding_chain(sample, "iso8859_5")
        assert chain[0] == "cp1252"

    def test_chinese_gb18030_no_vote(self):
        sample = '"摄氏度","牛·米","转每分","千帕"\n'.encode("gb18030") * 20
        chain = _build_encoding_chain(sample, "gb18030")
        assert chain[0] == "gb18030"

    @pytest.mark.parametrize("det", ["gb2312", "gbk"])
    def test_gb_alias_to_superset(self, det):
        sample = '"摄氏度","牛·米"\n'.encode("gb18030") * 20
        chain = _build_encoding_chain(sample, det)
        assert chain[0] == "gb18030"

    def test_utf8_det_keeps_utf8_first(self):
        sample = '"°C","µm"\n'.encode("utf-8")
        chain = _build_encoding_chain(sample, "utf_8")
        assert chain[0] == "utf-8"

    @pytest.mark.parametrize("bom", [
        b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff", b"\xff\xfe", b"\xfe\xff",
    ])
    def test_bom_trusts_det_without_vote(self, bom):
        chain = _build_encoding_chain(bom + "a\x00b\x00".encode("latin-1"), "utf_16_le")
        assert chain[0] == "utf_16_le"

    @pytest.mark.parametrize("det", ["utf_16_le", "utf_16_be", "utf_32_le"])
    def test_utf16_32_without_bom_trusted(self, det):
        sample = '"°C","µm"\n1,2\n'.encode(det)
        chain = _build_encoding_chain(sample, det)
        assert chain[0] == det

    def test_chain_tail_always_latin1_and_unique(self):
        for det in ["ascii", "cp850", "gb18030", "cp1251", "utf_8", "big5"]:
            sample = '"°C"\n'.encode("cp1252")
            chain = _build_encoding_chain(sample, det)
            assert chain[-1] == "latin-1"
            assert len(chain) == len(set(chain))


# ---------- 端到端矩阵（生产路径） ----------

class TestEncodingE2E:
    def test_cp1252_long_header_degree_units(self, tmp_path):
        units = ["°C", "°CA", "°KW", "°F", "µm", "µg", "m³", "‰", "Nm", "%"] * 40
        p = tmp_path / "long_header.csv"
        p.write_bytes(_csv_bytes(units).decode("utf-8").encode("cp1252"))
        got, used = _load_units(p)
        assert got == units
        assert used in ("cp1252", "cp1250")

    def test_cp1252_small_file_micro_units(self, tmp_path):
        # 小文件非 ASCII 落在采样窗内，探测器易误判 iso8859_16（µ→”）
        units = ["µS/cm", "µg", "µm", "µm/m", "µs"]
        p = tmp_path / "micro.csv"
        p.write_bytes('"V1","V2","V3","V4","V5"\n"µS/cm","µg","µm","µm/m","µs"\n1,2,3,4,5\n'
                      .encode("cp1252"))
        got, _ = _load_units(p)
        assert got == units

    @pytest.mark.parametrize("crlf", [False, True])
    def test_cp1252_all_degree_columns(self, tmp_path, crlf):
        # 400 列全 °C：探测器实测判 cp949（跨家族误判），投票须纠正
        units = ["°C"] * 400
        p = tmp_path / "all_c.csv"
        p.write_bytes(_csv_bytes(units, crlf=crlf).decode("utf-8").encode("cp1252"))
        got, _ = _load_units(p)
        assert got == units

    def test_cp1252_sparse_single_degree(self, tmp_path):
        units = [""] * 961 + ["°C"]
        p = tmp_path / "sparse.csv"
        p.write_bytes(_csv_bytes(units).decode("utf-8").encode("cp1252"))
        got, _ = _load_units(p)
        assert got == units

    def test_chinese_gb18030_long_header_with_desc(self, tmp_path):
        units = ["摄氏度", "牛·米", "转每分", "千帕", "°C", "%"] * 50
        text = ("试验描述：增压压力与排温测量\n"
                + ",".join(f"VAR_{i:04d}" for i in range(300)) + "\n"
                + ",".join(units) + "\n" + ",".join(["1.5"] * 300) + "\n")
        p = tmp_path / "cn_gb.csv"
        p.write_bytes(text.encode("gb18030"))
        got, used = _load_units(p)
        assert got == units
        assert used == "gb18030"

    def test_chinese_utf8(self, tmp_path):
        units = ["摄氏度", "牛·米", "转每分", "千帕", "°C", "%"] * 50
        p = tmp_path / "cn_utf8.csv"
        write_csv(p, header=[f"VAR_{i:04d}" for i in range(300)],
                  units=units, rows=[[1.5] * 300],
                  desc_rows=["试验描述：增压压力与排温测量"])
        got, used = _load_units(p)
        assert got == units
        assert used == "utf-8"

    def test_real_big5_not_swallowed(self, tmp_path):
        units = ["時間", "速度", "溫度", "壓力", "轉速", "流量"] * 50
        text = ("測試描述：增壓壓力與排溫測量\n"
                + ",".join(f"VAR_{i:04d}" for i in range(300)) + "\n"
                + ",".join(units) + "\n" + ",".join(["1.5"] * 300) + "\n")
        p = tmp_path / "big5.csv"
        p.write_bytes(text.encode("big5"))
        got, used = _load_units(p, has_unit=True)
        assert got == units
        assert used == "big5"

    def test_real_cp1251_cyrillic(self, tmp_path):
        units = ["Время", "Скорость", "Температура", "Давление", "Обороты", "Расход"] * 50
        text = ("Описание: измерение давления\n"
                + ",".join(f"VAR_{i:04d}" for i in range(300)) + "\n"
                + ",".join(units) + "\n" + ",".join(["1.5"] * 300) + "\n")
        p = tmp_path / "ru.csv"
        p.write_bytes(text.encode("cp1251"))
        got, used = _load_units(p, has_unit=True)
        assert got == units
        assert used == "cp1251"

    def test_real_cp1250_czech(self, tmp_path):
        units = ["Čas", "Rychlost", "Teplota", "Tlak", "Otáčky", "Průtok"] * 50
        text = ("Popis: měření tlaku a teploty\n"
                + ",".join(f"VAR_{i:04d}" for i in range(300)) + "\n"
                + ",".join(units) + "\n" + ",".join(["1.5"] * 300) + "\n")
        p = tmp_path / "cz.csv"
        p.write_bytes(text.encode("cp1250"))
        got, used = _load_units(p)
        assert got == units
        assert used == "cp1250"

    @pytest.mark.parametrize("enc", ["utf-16", "utf-16-le", "utf-16-be", "utf-32-le"])
    def test_utf16_32_variants(self, tmp_path, enc):
        units = ["°C", "µm", "m³", "Nm", "bar", "%"] * 50
        text = (",".join(f"VAR_{i:04d}" for i in range(300)) + "\n"
                + ",".join(units) + "\n" + ",".join(["1.5"] * 300) + "\n")
        p = tmp_path / "u16.csv"
        p.write_bytes(text.encode(enc))
        got, _ = _load_units(p)
        assert got == units

    def test_semicolon_sep_cp1252(self, tmp_path):
        units = ["°C", "µm", "m³", "Nm"] * 100
        text = (";" .join(f"VAR_{i:04d}" for i in range(400)) + "\n"
                + ";".join(units) + "\n" + ";".join(["1.5"] * 400) + "\n")
        p = tmp_path / "semi.csv"
        p.write_bytes(text.encode("cp1252"))
        got, _ = _load_units(p)
        assert got == units

    def test_desc_rows_cp1252(self, tmp_path):
        units = ["°C", "µm", "Nm", "%"] * 100
        text = ("File,export.csv\nAuthor,tool\n"
                + ",".join(f"VAR_{i:04d}" for i in range(400)) + "\n"
                + ",".join(units) + "\n" + ",".join(["1.5"] * 400) + "\n")
        p = tmp_path / "desc.csv"
        p.write_bytes(text.encode("cp1252"))
        got, _ = _load_units(p)
        assert got == units

    def test_has_unit_false_path(self, tmp_path):
        p = tmp_path / "no_unit.csv"
        write_csv(p, header=["time", "speed", "rpm", "flag"], units=None,
                  rows=make_simple_rows()[:5])
        fmt = FastDataLoader._auto_detect_format(str(p))
        names, units, used, has_unit = FastDataLoader._load_header_units(
            str(p), desc_rows=fmt.header_row, sep=fmt.sep,
            has_unit=False, encoding=fmt.encoding,
        )
        assert has_unit is False
        assert all(units[n] == "-" for n in names)

    def test_pure_ascii_regression(self, tmp_path):
        p = tmp_path / "ascii.csv"
        write_csv(p, header=["time", "speed", "rpm", "flag"],
                  units=["s", "km/h", "rpm", "-"], rows=make_simple_rows())
        got, _ = _load_units(p)
        assert got == ["s", "km/h", "rpm", "-"]

    def test_undefined_cp1252_byte_does_not_crash(self, tmp_path):
        # 0x81 是 cp1252 未定义字节：strict 失败后链仍须兜底完成加载
        p = tmp_path / "raw81.csv"
        p.write_bytes(b'"V1","V2"\n"\x81C","\xb0C"\n1,2\n')
        got, used = _load_units(p)
        assert len(got) == 2
        assert used in ENCODING_FALLBACKS or used == "cp949"