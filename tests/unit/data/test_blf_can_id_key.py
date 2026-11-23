"""BLF get_channel_info 增补原始 int can_id 键（身份匹配 C2 / MJ-4）。"""

from src.data.blf_loader import BLFLazyLoader
from tests.fixtures.data_factory import CAN_FRAMES, write_blf_pair


def test_can_dict_has_raw_int_can_id(tmp_path):
    blf, dbc = write_blf_pair(tmp_path, n=20, buses=(0,))
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        # EngineSpeed 跨报文重名 → 限定名；单总线无 _G 后缀
        info = loader.get_channel_info("EngineData.EngineSpeed")
        can = info["can"]
        assert can["can_id"] == CAN_FRAMES[0]["can_id"]
        assert isinstance(can["can_id"], int)
        assert not isinstance(can["can_id"], bool)
        # 装饰串保持原样（additive，不解析 can_id_hex）
        assert can["can_id_hex"] == "0x1A0"
        assert can["bus_channel_index"] == 0
        assert can["is_extended"] is False
    finally:
        loader.close()