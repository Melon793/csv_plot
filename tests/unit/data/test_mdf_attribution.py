"""``src/data/mdf_attribution`` 的纯函数单测。

全部用**手工构造的基础类型 dict**，不读任何 MDF 文件 —— 这些规则的价值就在
于可脱离真实数据回归。字符串样本一律取自实测的真实文件（见
``tmp/mdf_function_meta_extraction_report.md``）。
"""

import pytest

from src.data import mdf_attribution as A


def make_info(
    *,
    version="4.10",
    name="BMS_CellVolt082",
    comment=None,
    description=None,
    display_names=None,
    acq_name=None,
    cg_comment=None,
    source=None,
    hd_comment=None,
):
    """构造一份与 ``get_channel_info()`` 同形的最小 dict。"""
    channel = {"name": name, "comment": comment, "description": description}
    if display_names is not None:
        channel["display_names"] = display_names
    return {
        "version": version,
        "channel": channel,
        "channel_group": {"acq_name": acq_name, "comment": cg_comment},
        "source": source or {},
        "header": {"comment": hd_comment},
    }


HD_TEXT = (
    "<HDcomment><TX>Database: DEMO_DB01\r\n"
    "Experiment: Demo_2\r\n"
    "Workspace: Workspace\r\n"
    "Devices: XCP:1,CAN-Monitoring:1,CalcDev\r\n"
    "Program Description: DEMOSW_AR_D1p0, ENG01_AA_All_01_DBC\r\n"
    "WP: DEMO_0001_Freeze_1\r\n"
    "RP: DEMO_0001_Freeze_0\r\n"
    "Date: 05/09/2026\r\n"
    "Time: 01:36:14 PM\r\n"
    "Pre-trigger Time: 0[s]\r\n"
    "§@</TX><common_properties>"
    '<e name="author">Administrator</e></common_properties></HDcomment>'
)


def labels(rows):
    return [label for label, _ in rows]


# ---------------------------------------------------------------------------
# repair_text：latin-1 误读 GBK 的回转（D5）
# ---------------------------------------------------------------------------


class TestRepairText:
    def test_gbk_misread_is_restored(self):
        assert A.repair_text("82ºÅµ¥ÌåµçÑ¹") == "82号单体电压"
        assert A.repair_text("±¨ÎÄ¼ÆÊýÆ÷") == "报文计数器"

    @pytest.mark.parametrize(
        "text",
        [
            "BMS_CellVolt082",  # 纯 ASCII：回转恒等
            "82号单体电压",  # 已正确解码的中文：含 >U+00FF 字符，直接跳过
            "Théorie du moteur",  # 合法 latin-1 法文：解码后不含 CJK → 原样
            "",
        ],
    )
    def test_identity_for_safe_inputs(self, text):
        assert A.repair_text(text) == text

    def test_none_returns_empty(self):
        assert A.repair_text(None) == ""

    def test_disabled_by_flag(self, monkeypatch):
        monkeypatch.setattr(A, "MDF_GBK_TEXT_REPAIR", False)
        assert A.repair_text("82ºÅµ¥ÌåµçÑ¹") == "82ºÅµ¥ÌåµçÑ¹"


# ---------------------------------------------------------------------------
# extract_tx / channel_aux_text：v3 与 v4 的注释落点互为兜底
# ---------------------------------------------------------------------------


class TestExtractTx:
    def test_single_line_xml_from_real_mdf4(self):
        raw = (
            '<CNcomment xmlns="http://www.asam.net/mdf/v4"><TX>82ºÅµ¥ÌåµçÑ¹</TX>'
            "<names><display>BMS_CellVolt082</display></names>"
            "<raster>0.00100000000000000</raster></CNcomment>"
        )
        assert A.extract_tx(raw) == "82ºÅµ¥ÌåµçÑ¹"

    def test_multiline_xml_from_synthetic_mdf4(self):
        raw = "<CNcomment>\n<TX>\n多行注释\n</TX>\n</CNcomment>"
        assert A.extract_tx(raw) == "多行注释"

    def test_plain_text_is_returned_as_is(self):
        assert A.extract_tx("Lambda actual value sensor 1") == (
            "Lambda actual value sensor 1"
        )

    def test_double_escaped_tx_is_unwrapped_repeatedly(self):
        """asammdf 写 MDF3 的 header 注释时会套两层（实测）。"""
        raw = (
            "<HDcomment><TX>&lt;HDcomment&gt;&lt;TX&gt;Database: SYN_DB"
            "&lt;/TX&gt;&lt;common_properties /&gt;&lt;/HDcomment&gt;"
            "\nSun Sep 13 2026: updated\n</TX><common_properties/></HDcomment>"
        )
        assert A.extract_tx(raw) == "Database: SYN_DB"

    def test_trailing_canape_noise_line_is_dropped(self):
        assert A.extract_tx("<TX>Database: DB\n§@\n</TX>") == "Database: DB"

    def test_xml_entities_are_unescaped(self):
        assert A.extract_tx("<TX>a &lt; b &amp; c</TX>") == "a < b & c"

    @pytest.mark.parametrize("bad", [None, "", "   ", 123, {"x": 1}])
    def test_non_string_is_empty(self, bad):
        assert A.extract_tx(bad) == ""

    def test_channel_aux_text_falls_back_to_description(self):
        """实测合成 v3 只写 CN.description、CN.comment 为空。"""
        info = make_info(version="3.30", description="RBArithmeticElement")
        assert A.channel_aux_text(info["channel"]) == "RBArithmeticElement"

    def test_channel_aux_text_prefers_comment(self):
        info = make_info(comment="来自 comment", description="来自 description")
        assert A.channel_aux_text(info["channel"]) == "来自 comment"

    def test_channel_aux_text_repairs_gbk(self):
        info = make_info(comment="<TX>82ºÅµ¥ÌåµçÑ¹</TX>")
        assert A.channel_aux_text(info["channel"]) == "82号单体电压"

    @pytest.mark.parametrize("bad", [None, {}, {"comment": None}])
    def test_channel_aux_text_never_raises(self, bad):
        assert A.channel_aux_text(bad) == ""


# ---------------------------------------------------------------------------
# 通道名 / HD 注释解析
# ---------------------------------------------------------------------------


class TestSplitNameDevice:
    def test_colon_suffix(self):
        assert A.split_name_device("b_Idle_Flag\\XCP:1") == ("b_Idle_Flag", "XCP:1")

    def test_hash_suffix(self):
        base, dev = A.split_name_device("x\\XCP:1#RAMCal")
        assert base == "x"
        assert dev == "XCP:1#RAMCal"

    def test_array_brackets_kept_in_base(self):
        assert A.split_name_device("EpmCaS_phiAdapRefPosI1_[6]\\XCP:1") == (
            "EpmCaS_phiAdapRefPosI1_[6]",
            "XCP:1",
        )

    def test_no_device_when_no_colon(self):
        assert A.split_name_device("BMS_CellVolt082") == ("BMS_CellVolt082", "")

    def test_device_without_colon_needs_whitelist(self):
        # 无法确认为设备时，**整个名字当作基名**返回（不能把信息弄丢）
        assert A.split_name_device("TorqueCalc\\CalcDev") == ("TorqueCalc\\CalcDev", "")
        assert A.split_name_device("TorqueCalc\\CalcDev", frozenset({"CalcDev"})) == (
            "TorqueCalc",
            "CalcDev",
        )

    def test_name_starting_with_backslash_is_not_split(self):
        assert A.split_name_device("\\XCP:1") == ("\\XCP:1", "")

    @pytest.mark.parametrize("bad", [None, ""])
    def test_empty_name(self, bad):
        assert A.split_name_device(bad) == ("", "")


class TestHdComment:
    def test_parse_hd_comment_from_double_escaped_mdf3_header(self):
        """实测合成 v3 文件读回的 HD 注释被二次转义，解析必须同样兼容。"""
        raw = (
            "<HDcomment><TX>&lt;HDcomment&gt;&lt;TX&gt;Database: SYN_DB\n"
            "Devices: XCP:1,CAN-Monitoring:1&lt;/TX&gt;&lt;common_properties /&gt;"
            "&lt;/HDcomment&gt;\nSun Sep 13 2026: updated\n</TX></HDcomment>"
        )
        assert A.parse_hd_comment(raw)["Database"] == "SYN_DB"
        assert A.parse_hd_devices(raw) == frozenset({"XCP:1", "CAN-Monitoring:1"})

    def test_parse_hd_comment(self):
        parsed = A.parse_hd_comment(HD_TEXT)
        assert parsed["Database"] == "DEMO_DB01"
        assert parsed["Experiment"] == "Demo_2"
        assert parsed["Program Description"] == "DEMOSW_AR_D1p0, ENG01_AA_All_01_DBC"
        # 'Time: 01:36:14 PM' 只在首个冒号切分
        assert parsed["Time"] == "01:36:14 PM"
        # 非 Key:Value 的行（common_properties）不会被误当成键
        assert "author" not in parsed

    def test_parse_hd_devices(self):
        assert A.parse_hd_devices(HD_TEXT) == frozenset(
            {"XCP:1", "CAN-Monitoring:1", "CalcDev"}
        )

    @pytest.mark.parametrize("bad", [None, "", "no colon here"])
    def test_hd_parsing_is_tolerant(self, bad):
        assert A.parse_hd_comment(bad) == {}
        assert A.parse_hd_devices(bad) == frozenset()

    def test_hd_attribution_keys_cover_canape_fields(self):
        parsed = A.parse_hd_comment(HD_TEXT)
        for raw_key, _label in A.HD_ATTRIBUTION_KEYS:
            assert raw_key in parsed, f"试验级归属键 {raw_key} 应能在实测样本中命中"


# ---------------------------------------------------------------------------
# 函数归属的三级回退与误判防护
# ---------------------------------------------------------------------------


class TestFunctionInference:
    def test_rule1_from_name_slash(self):
        info = make_info(
            version="3.30", name="EpmCaS_phiSegOfs_CA/isx\\XCP:1", comment="#Index"
        )
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_FUNCTION] == "EpmCaS_phiSegOfs_CA"
        assert rows[A.LABEL_FUNCTION_BASIS] == A.BASIS_NAME
        assert A.LABEL_HINT not in rows  # 只有注释推断才需要提示
        assert A.LABEL_DB_SOURCE not in rows  # '#Index' 不该被当成任何出处

    def test_rule2_from_single_shot_group_comment(self):
        info = make_info(
            version="3.30", name="KF\\XCP:1", cg_comment="KF\\x\\XCP:1\\SingleShotGroup"
        )
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_FUNCTION] == "KF\\x"
        assert rows[A.LABEL_FUNCTION_BASIS] == A.BASIS_CG_COMMENT

    def test_rule3_from_identifier_aux(self):
        info = make_info(
            version="3.30",
            name="EmsEopPumpMotorSpeedReq\\CAN-Monitoring:1",
            comment="EMSEOP",
            cg_comment="Ems27A",
        )
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_FUNCTION] == "EMSEOP"
        assert rows[A.LABEL_FUNCTION_BASIS] == A.BASIS_AUX_TEXT
        assert rows[A.LABEL_HINT] == A.HINT_INFERRED

    def test_group_prefix_aux_is_signal_not_function(self):
        """实测 'AI50' 组里的 'AI50_3' 是报文内信号名，属信息而非噪声。"""
        info = make_info(
            version="3.30",
            name="CKA_XQR_50_CY03\\CAN-Monitoring:1",
            comment="AI50_3",
            cg_comment="AI50",
        )
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_SIGNAL_IN_GROUP] == "AI50_3"
        assert A.LABEL_FUNCTION not in rows

    def test_aux_equal_group_carries_no_new_info(self):
        info = make_info(
            version="3.30", name="CKA_XQR_50\\CAN-Monitoring:1",
            comment="AI50", cg_comment="AI50",
        )
        rows = dict(A.build_attribution_rows(info))
        assert A.LABEL_FUNCTION not in rows
        assert A.LABEL_SIGNAL_IN_GROUP not in rows

    @pytest.mark.parametrize(
        "aux",
        [
            "Lambda actual value sensor 1 bank 1",  # 含空格的自然语言描述
            "t.b.d.",  # 实测存在的缩写占位（点号不在标识符字符集内）
            "T.B.D.",
            "#Index",  # 首字符非标识符
            "adapted reference-positions of inlet-camshaft",
        ],
    )
    def test_non_identifier_aux_never_becomes_function(self, aux):
        info = make_info(version="3.30", name="UEGO_rLamS1B1\\XCP:1", comment=aux)
        rows = dict(A.build_attribution_rows(info))
        assert A.LABEL_FUNCTION not in rows, f"{aux!r} 被误判为函数"
        assert A.LABEL_SIGNAL_IN_GROUP not in rows

    def test_aux_equal_channel_name_is_not_function(self):
        info = make_info(version="3.30", name="Press\\XCP:1", comment="Press")
        assert A.LABEL_FUNCTION not in dict(A.build_attribution_rows(info))

    def test_raster_group_name_is_never_treated_as_function(self):
        """D6：'100ms time synchronous' 这类光栅名只当测量组展示。"""
        info = make_info(
            version="3.30", name="WaterTemp\\XCP:1", cg_comment="100ms time synchronous"
        )
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_GROUP] == "100ms time synchronous"
        assert A.LABEL_FUNCTION not in rows

    def test_rule1_wins_over_rule3(self):
        info = make_info(
            version="3.30", name="EpmCaS_phiSegOfs_CA/isx\\XCP:1", comment="RBArithmeticElement"
        )
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_FUNCTION] == "EpmCaS_phiSegOfs_CA"
        assert rows[A.LABEL_FUNCTION_BASIS] == A.BASIS_NAME


# ---------------------------------------------------------------------------
# 来源数据库
# ---------------------------------------------------------------------------


class TestDbSource:
    def test_extract_db_source(self):
        assert A.extract_db_source("created from: CANDB: BMC_Spannung") == (
            "CANDB: BMC_Spannung",
            "",
        )

    def test_extract_db_source_keeps_remaining_text(self):
        db, rest = A.extract_db_source("过压保护 created from: CANDB: Uq_Bat")
        assert db == "CANDB: Uq_Bat"
        assert rest == "过压保护"

    @pytest.mark.parametrize("aux", ["", None, "普通注释"])
    def test_extract_db_source_miss(self, aux):
        assert A.extract_db_source(aux) == ("", (aux or "").strip())

    def test_db_row_present_and_rest_not_mistaken_for_function(self):
        info = make_info(name="BMC_Spannung", comment="created from: CANDB: BMC_Spannung")
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_DB_SOURCE] == "CANDB: BMC_Spannung"
        assert A.LABEL_FUNCTION not in rows


# ---------------------------------------------------------------------------
# SI 源块与显示名
# ---------------------------------------------------------------------------


class TestSourceRows:
    MDF4_SOURCE = {
        "name": "BMCe",
        "path": "CAN-Monitoring:1",
        "bus_type": 2,
        "source_type": 1,
        "comment": "",
        "address": 2704712,
    }

    def test_device_ecu_and_labels(self):
        info = make_info(source=self.MDF4_SOURCE)
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_DEVICE] == "CAN-Monitoring:1"
        assert rows[A.LABEL_ECU] == "BMCe"
        assert rows[A.LABEL_BUS_TYPE] == "CAN"
        assert rows[A.LABEL_SOURCE_TYPE] == "ECU"

    @pytest.mark.parametrize(
        "bus_type,expected",
        [(2, "CAN"), (3, "LIN"), (5, "FLEXRAY"), (7, "ETHERNET")],
    )
    def test_bus_type_labels(self, bus_type, expected):
        info = make_info(source={**self.MDF4_SOURCE, "bus_type": bus_type})
        assert dict(A.build_attribution_rows(info))[A.LABEL_BUS_TYPE] == expected

    @pytest.mark.parametrize("bus_type", [0, 1, 99, None])
    def test_unmapped_bus_type_is_hidden(self, bus_type):
        info = make_info(source={**self.MDF4_SOURCE, "bus_type": bus_type})
        assert A.LABEL_BUS_TYPE not in dict(A.build_attribution_rows(info))

    @pytest.mark.parametrize("source_type", [0, None, 42])
    def test_unmapped_source_type_is_hidden(self, source_type):
        info = make_info(source={**self.MDF4_SOURCE, "source_type": source_type})
        assert A.LABEL_SOURCE_TYPE not in dict(A.build_attribution_rows(info))

    def test_noise_source_name_filtered(self):
        """D7：asammdf 给合成 v3 文件注入的源块名不该出现在界面上。"""
        info = make_info(
            version="3.30",
            name="Press\\XCP:1",
            source={**self.MDF4_SOURCE, "name": "Channel inserted by Python Script"},
        )
        assert A.LABEL_ECU not in dict(A.build_attribution_rows(info))

    def test_mdf3_without_source_block_has_no_ecu_row(self):
        info = make_info(version="3.30", name="WaterTemp\\XCP:1", source={})
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_DEVICE] == "XCP:1"
        assert A.LABEL_ECU not in rows
        assert A.LABEL_BUS_TYPE not in rows

    def test_group_prefers_acq_name_over_comment(self):
        info = make_info(acq_name="BMS_CellVoltInfo", cg_comment="same")
        assert dict(A.build_attribution_rows(info))[A.LABEL_GROUP] == "BMS_CellVoltInfo"

    def test_group_falls_back_to_comment(self):
        info = make_info(version="3.30", name="P\\XCP:1", acq_name=None, cg_comment="AI50")
        assert dict(A.build_attribution_rows(info))[A.LABEL_GROUP] == "AI50"

    def test_long_name_skips_empty_key(self):
        """实测 v3 恒为 {} 或 {'': 'display_name'}，不能显示空标签。"""
        assert A.LABEL_LONG_NAME not in dict(
            A.build_attribution_rows(make_info(display_names={"": "display_name"}))
        )
        assert A.LABEL_LONG_NAME not in dict(
            A.build_attribution_rows(make_info(display_names={}))
        )
        info = make_info(display_names={"CAN-Monitoring:1.BMS_x": "source_path"})
        assert dict(A.build_attribution_rows(info))[A.LABEL_LONG_NAME] == (
            "CAN-Monitoring:1.BMS_x"
        )

    def test_non_dict_display_names_is_tolerated(self):
        info = make_info(display_names=None)
        info["channel"]["display_names"] = "不是字典"
        assert A.LABEL_LONG_NAME not in dict(A.build_attribution_rows(info))


# ---------------------------------------------------------------------------
# 聚合器的整体契约
# ---------------------------------------------------------------------------


class TestBuildAttributionRows:
    FULL_INFO = None  # 由 _full() 构造，避免在类属性里共享可变 dict

    @staticmethod
    def full():
        return make_info(
            name="BMS_CellVolt082",
            comment="<CNcomment><TX>82号单体电压</TX></CNcomment>",
            display_names={"CAN-Monitoring:1.BMS_CellVolt082": "source_path"},
            acq_name="BMS_CellVoltInfo",
            source={
                "name": "BMCe",
                "path": "CAN-Monitoring:1",
                "bus_type": 2,
                "source_type": 1,
            },
            hd_comment=HD_TEXT,
        )

    def test_row_order_is_stable(self):
        assert labels(A.build_attribution_rows(self.full())) == [
            A.LABEL_DEVICE,
            A.LABEL_ECU,
            A.LABEL_BUS_TYPE,
            A.LABEL_SOURCE_TYPE,
            A.LABEL_GROUP,
            A.LABEL_LONG_NAME,
        ]

    def test_empty_input_yields_no_rows(self):
        assert A.build_attribution_rows(make_info(name="Press")) == []

    @pytest.mark.parametrize("bad", [None, {}, 0, "x", {"channel": None}])
    def test_garbage_input_yields_no_rows(self, bad):
        assert A.build_attribution_rows(bad) == []

    def test_disabled_by_flag(self, monkeypatch):
        monkeypatch.setattr(A, "MDF_ATTRIBUTION_ENABLED", False)
        assert A.build_attribution_rows(self.full()) == []

    def test_long_value_is_not_truncated(self):
        """行值不得截断：真实 mf4 有 141/4704 个通道注释超 120 字符。

        截断的原始理由“不截断会撑宽值列”已被实测推翻：值列是
        ``QHeaderView.Stretch``，内容由 Qt 绘制层 elide，tooltip / 复制按钮 /
        Markdown 导出都拿得到全值 —— 在数据层截断只会永久丢信息。
        列宽不受影响的守卫在组件层（test_long_value_does_not_widen_layout）。
        """
        info = make_info(name="x", acq_name="G" * 300)
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_GROUP] == "G" * 300

    def test_value_is_whitespace_stripped(self):
        """不截断不等于不清洗：行值首尾空白会影响列对齐。"""
        info = make_info(
            name="Press",
            acq_name="  NormalGroup  ",
            source={"name": " BMCe ", "path": " XCP:1 "},
        )
        rows = dict(A.build_attribution_rows(info))
        assert rows[A.LABEL_GROUP] == "NormalGroup"
        assert rows[A.LABEL_DEVICE] == "XCP:1"
        assert rows[A.LABEL_ECU] == "BMCe"

    def test_no_empty_value_rows(self):
        info = self.full()
        info["source"]["name"] = ""
        rows = A.build_attribution_rows(info)
        assert all(value.strip() for _, value in rows)
        assert A.LABEL_ECU not in dict(rows)


class TestClassifyAuxText:
    @pytest.mark.parametrize(
        "aux,expected",
        [
            ("RBArithmeticElement", "function"),
            ("AI50_3", "signal"),
            ("AI50", ""),  # 与组名相同 → 零信息
            ("Some description here", "description"),
            ("", ""),
            ("Press", ""),  # 与通道基名相同
        ],
    )
    def test_classification(self, aux, expected):
        assert A.classify_aux_text(
            aux, base_name="Press", group="AI50"
        ) == expected
