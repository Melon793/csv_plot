# `bus_data/` — 第三方总线夹具（BLF / DBC / ARXML）

BLF 只存原始报文，必须搭配 DBC 或 ARXML 数据库才能解码出信号，因此本目录是
**CAN 数据源用例的唯一真值来源**。这里不放任何本仓库自产的文件——合成样本一律
由 [`../data_factory.py`](../data_factory.py) 在 tmp 目录现做现用。

- 来源、URL、sha256、字节数、许可的完整机器可读清单：[`manifest.json`](manifest.json)
- 最近一次下载/校验的结果快照：[`fetch_report.json`](fetch_report.json)
- 各来源仓库的 LICENSE 全文：[`licenses/`](licenses/)
- 重新取回：`uv run python scripts/fetch_blf_fixtures.py`（首次锁定哈希加 `--lock`）
- 只校验不猜测：`uv run python scripts/fetch_blf_fixtures.py --check`

**本目录不放任何运行时代码或打包资源**（见仓库根 `CLAUDE.md`）。测试读不到文件
时 `pytest.skip`，不红。

## 入库前的许可核对

逐个取回 LICENSE 全文（`licenses/`）后确认可再分发：

| 仓库 | 许可 | 结论 |
| --- | --- | --- |
| `hardbyte/python-can` | LGPL-3.0-or-later | 这些 `.blf` 是**测试数据**而非代码，不构成 LGPL 作品的衍生/链接；且随包附 LICENSE 全文 |
| `cantools/cantools` | MIT | 可再分发，保留版权声明即可 |
| `mrafatpanah/blf-viewer` | MIT | 同上 |
| `zariiii9003/vblf` | MIT | 同上（作者即 asammdf 维护者） |
| `ebroecker/canmatrix` | BSD-3-Clause | 只归档 LICENSE：本项目从 `.venv` 里已装好的那份取样本，不下载、不复制其文件 |

四个来源均为公开的开源仓库测试目录，合计 **184,882 字节**。

## 文件清单

`sha256` 只列前 16 位，完整值以 `manifest.json` 为准；测试断言读的是后者。

### BLF 二进制层（`hardbyte/python-can` · LGPL-3.0-or-later）

覆盖 BLF 里**每一种** CAN 对象类型，且由第三方工具产出，非本项目自产自验。

| 文件 | 字节 | sha256 | 用途 |
| --- | --- | --- | --- |
| `pcan_test_CanMessage.blf` | 420 | `c04d0cb0ed2a3466` | 经典 CAN 报文基线样本 |
| `pcan_test_CanMessage2.blf` | 436 | `ef135387f6f5b4bc` | 经典 CAN 报文（第二种产出工具） |
| `pcan_test_CanFdMessage.blf` | 564 | `5ee6a21f3eba6f82` | CAN FD 报文（码率切换） |
| `pcan_test_CanFdMessage64.blf` | 612 | `d6c4551541fd9ce7` | CAN FD 64 B payload，决定桶 `stride` 上限 |
| `pcan_test_CanErrorFrameExt.blf` | 452 | `d45c5f0debd642b2` | 扩展错误帧：扫描期必须跳过、不产生采样点 |
| `pcan_issue_1905.blf` | 524 | `192b89141ff35aeb` | 历史回归：`valid_bytes` 大于实际数据长度 |

### 端到端最小闭环（`mrafatpanah/blf-viewer` · MIT）

| 文件 | 字节 | sha256 | 用途 |
| --- | --- | --- | --- |
| `blfviewer_two-channel.blf` | 10,324 | `77748eadd4ae53ef` | 双通道 BLF：`group = (bus 通道, CAN ID)` 分桶的直接证据 |
| `blfviewer_two-channel.dbc` | 2,277 | `f40a70c65bfd2a1c` | 与上一行**配对**的真实 DBC |

这一对是本目录里唯一能做「导入 → 选库 → 出信号 → 绘图」全链路验收的样本：
多总线通道 + 真实数据库，且都是第三方产出。

### 数据库兼容性矩阵（`cantools/cantools` · MIT）

同时充当 `canmatrix` 解析结果的**对照真值**（cantools 独立实现）。
下表的报文 / 信号数是 `CanDatabaseSet` 的**实测**结果而不是预期值 ——
`canmatrix` 对 ARXML 的覆盖面是**逐文件**的，不能按「AUTOSAR 版本」笼统推断。

| 文件 | 字节 | sha256 | 实测 | 用途 |
| --- | --- | --- | --- | --- |
| `ct_multiplex.dbc` | 2,355 | `2b34c7ade5bb1e0f` | 1 帧 / 12 信号 | 多路复用：先解 mux 选择器再按值掩码，非本 mux 值填 NaN |
| `ct_floating_point.dbc` | 891 | `b50c9bbf22bcdd89` | 2 帧 / 3 信号 | IEEE float32/64 的 `view` 解码与大小端 |
| `ct_choices.dbc` | 766 | `30a760dcbd3ddf6b` | 1 帧 / 1 信号 | `VAL_` 枚举表 → `is_enum` + `enum_map` |
| `ct_big_numbers.dbc` | 3,303 | `9335158afda4cff0` | 2 帧 / 2 信号 | > 32 bit 大整数位段的向量化边界 |
| `ct_bus_comment.dbc` | 2,728 | `680c43dcae5644be` | 1 帧 / 12 信号 | 总线级注释（非报文注释），属性读取兜底 |
| `ct_cp1252.dbc` | 694 | `cdd06bd74b6e8e8d` | **0 帧** | cp1252 注释编码；cantools 同样读出 0 条 → 交叉一致的**空库**，恰好喂状态列的「⚠ 无 CAN 报文」 |
| `ct_system-float-values.arxml` | 30,885 | `95af032ad40b5d2d` | 1 帧 / 2 信号 | 本目录里**唯一能出帧**的 ARXML：ARXML 侧浮点 + 物理值上下限 |
| `ct_system-3.2.3.arxml` | 37,616 | `ca9f5d128d17420c` | cluster `Network` / **0 帧** | canmatrix 在其中找不到 CAN cluster → 「⚠ 无 CAN 报文」；直接证明「按 AUTOSAR 版本推断支持度」不成立 |
| `ct_ecu-extract-4.2.arxml` | 18,249 | `415aca333d9f724f` | **空 dict** `{}` | flatten 路径必须容忍空 cluster 而不是 `AttributeError` |
| `ct_system-4.2.arxml` | 71,059 | `c6bf5671279bc7ec` | **抛异常** | `canmatrix/formats/arxml.py` 的 `get_frame_from_multiplexed_ipdu` 拿到 `None` → `AttributeError: 'NoneType' object has no attribute 'text'`。复杂多路复用 IPDU 目前读不动；本文件因此是「✖ 解析失败」状态列与「单文件失败只 warning 不中断」的**真实**（非人造）用例 |

### 逐对象头/边界样本（`zariiii9003/vblf` · MIT）

`.lobj` 是**单个 BLF 对象**的原样字节，用于验证对象头解析与边界，不进 `BLFReader`。
后四个对象类型 v1 明确不解析，入库只为证明「读到了、按设计跳过」。

| 文件 | 字节 | sha256 | 用途 |
| --- | --- | --- | --- |
| `vblf_CAN_MESSAGE.lobj` | 48 | `8acce3461c34f579` | 单对象：经典 CAN |
| `vblf_CAN_MESSAGE2.lobj` | 56 | `295cee512d055b85` | 单对象：经典 CAN（第二种） |
| `vblf_CAN_FD_MESSAGE.lobj` | 120 | `a6379b07174b3d51` | 单对象：CAN FD |
| `vblf_CAN_FD_MESSAGE_64.lobj` | 88 | `9c83420c21904df5` | 单对象：CAN FD 64 B |
| `vblf_CAN_ERROR_EXT.lobj` | 64 | `e5f62672f053cba7` | 单对象：扩展错误帧 |
| `vblf_LOG_CONTAINER.lobj` | 120 | `d21cc522073fe9f6` | 单对象：日志容器（v1 不解析） |
| `vblf_ENV_DOUBLE.lobj` | 59 | `e22287c12b3cacb7` | 单对象：ENV 变量（v1 不解析） |
| `vblf_SYS_VARIABLE__DOUBLE.lobj` | 75 | `61d3540c6fa60cd1` | 单对象：SYS 变量（v1 不解析） |
| `vblf_APP_TEXT.lobj` | 97 | `b0cf8089d215d7d3` | 单对象：APP 文本（v1 不解析） |

## canmatrix 的 ARXML 覆盖面结论（实测，已固化为单测）

用本目录的 4 份 ARXML，加上 `.venv` 里 canmatrix 自带的
`ARXMLContainerTest.arxml` / `ARXMLSecuredPDUTest.arxml` /
`ARXMLCompuMethod1.arxml` / `ARXML_min_max.arxml` / `MyECU.ecuc.arxml`
逐个过 `CanDatabaseSet`（不碰 canmatrix 内部），结论是**逐文件**的：

| 被测项 | 实测结论 |
| --- | --- |
| ISignal / ISignalGroup / PDU / Frame（`ARXML_min_max` 1/6、`MyECU.ecuc` 2/4、`system-float-values` 1/2） | **支持**，帧 / 信号 / factor / offset / 字节序完整 |
| PDU container（`ARXMLContainerTest.arxml`） | **支持**：5 帧 / 10 信号 |
| 复杂多路复用 IPDU（`ct_system-4.2.arxml`） | **抛 `AttributeError`**（`get_frame_from_multiplexed_ipdu` 里拿到 `None`）→ 本项目按「单文件解析失败」处理：记 warning、状态列 ✖、不中断其余库 |
| SecOC / 加密帧（`ARXMLSecuredPDUTest.arxml`） | **不支持**：返回 0 帧 → 计入未匹配反馈，不报错 |
| COMPU METHOD 变体（`ARXMLCompuMethod1.arxml`） | 返回**空 dict** `{}`，必须容忍而不是 `AttributeError` |
| AUTOSAR 3.2 系统描述（`ct_system-3.2.3.arxml`） | 能加载但 **0 报文**（找不到 CAN cluster）；不能按版本号推断可用性 |
| ECU extract（`ct_ecu-extract-4.2.arxml`） | 返回**空 dict**，同上 |
| ARXML **导出** | 可写（须传 `{cluster: CanMatrix}`，传 `CanMatrix` 会 `TypeError: object is not subscriptable`），但 **cantools 读不回**：`BASE-TYPE-ENCODING in base type "sint16" does not exist.` |

最后一行直接决定了：**ARXML 夹具只能验链路，不能与 cantools 交叉验证真值**。
数据库层的交叉验证因此只走 DBC；ARXML 侧的回归靠「能解析出预期数量的帧/信号」
这类结构性断言。

对用户的实际影响：现场 ARXML 若含复杂多路复用 / SecOC，会走「未匹配报文 ID」
反馈路径（banner 状态③ + 详情对话框）而不是崩掉 —— 这也是上面这些「读不出帧」
的文件被保留、而不是被换成人造好文件的原因。

## v1 已知未覆盖

现场 ARXML 里的 SecOC / 复杂多路复用 IPDU 混合布局、J1939 PGN 自动解码、UDS
诊断帧、ENV / SYS 变量对象、报文级通道（ID 计数 / 总线负载）。前两项是
**v1.1 的现场数据回归项**（需要一份脱敏真实文件），其余见方案「v1 明确不做」。
