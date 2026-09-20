"""文件级信息 —— 状态栏左段「文件信息抽屉」的数据源。

产出扁平的 ``[(键, 值)]``，值已经是可直接显示的字符串，UI 层不感知数据源
类型。全程不读数据列、不读盘（唯一例外是 ``file_size`` 缺失时的一次
``os.path.getsize``），因此可在 UI 线程同步调用。

口径约束（与状态栏信息准则一致）：

- 只放"系统在文件里看到了什么"。有效性/常量/无效这类逐列判定不放：抽屉
  要的是矮，而这三行既不能直接指导绘图（要看具体变量仍得开变量信息窗口），
  又会随格式给出无法对比的数字。
- 同族的小字段并进「格式」一行，而不是各占一行：编码、分隔符、工作表这些
  都是"怎么把这个文件读成表的"，一行读完，抽屉高度按行数算。
- 缺字段就**不给这一行**，不用 "-" 占位：抽屉是点开才看到的信息，空占位
  只会稀释可信度。
- 数字含义按格式说清：MDF 的行数是"最大通道组的声明记录数"，与 CSV 的实际
  行数不同口径，混在同一个标签下会误导。
"""

from __future__ import annotations

import os

from src.core.logger import get_logger
from src.data.var_info import ROW_KEY_FILE_PATH, format_size
from src.utils.paths import display_path

logger = get_logger(__name__)

#: 「文件名」「所在文件夹」两行的键名。UI 层按键决定行尾挂哪个动作按钮，
#: 与 ROW_KEY_FILE_PATH 同一机制，字面不得再各处手写
KEY_FILE_NAME = "文件名"
KEY_FOLDER = "所在文件夹"

#: LOADER_TYPE → 可读格式名。未登记的值回落到按扩展名判断
_FORMAT_NAME = {"csv": "CSV 文本", "excel": "Excel 工作簿", "mdf": "MDF 二进制"}


def build_file_rows(loader, elapsed_s: float | None = None) -> list:
    """汇总文件级信息。

    ``elapsed_s`` 是本次加载耗时（只有加载链路 ``_post_load_actions`` 知道），
    None 时整行不出现 —— 宁可少一行，也不写"未知"这种读不出信息的值。
    """
    if loader is None:
        return []

    rows = _file_rows(loader)
    rows.append(_format_row(loader))
    rows += _size_rows(loader, elapsed_s)
    return rows


# ---------------------------------------------------------------------------
# 「文件」几行
# ---------------------------------------------------------------------------


def _split_path(path: str) -> tuple[str, str]:
    """拆成 (文件夹, 文件名)。

    刻意不用 ``os.path.split``：Windows 上拖进主窗口的 UNC 路径会带着反斜杠
    出现在 macOS/Linux 上（``\\\\host\\share\\a.csv``），此时 ``os.path`` 只认
    ``/``，整串会被当成文件名。两种分隔符都切，展示才不错位。
    """
    s = str(path or "")
    cut = max(s.rfind("/"), s.rfind("\\"))
    if cut < 0:
        return "", s
    return s[:cut], s[cut + 1:]


def _file_rows(loader) -> list:
    raw = display_path(str(getattr(loader, "path", "") or ""))
    folder, name = _split_path(raw)
    rows = [
        (KEY_FILE_NAME, name or "-"),
        (ROW_KEY_FILE_PATH, raw or "-"),
        (KEY_FOLDER, folder or "-"),
    ]

    size = getattr(loader, "file_size", None)
    if size is None and raw and os.path.exists(raw):
        try:
            size = os.path.getsize(raw)
        except OSError:
            logger.debug("取 %s 的文件大小失败", raw, exc_info=True)
            size = None
    if size is not None:
        rows.append(("文件大小", format_size(size)))
    return rows


# ---------------------------------------------------------------------------
# 「格式」一行
# ---------------------------------------------------------------------------


def _format_text(sep) -> str:
    """分隔符的可读写法：制表符与控制字符直接写死会看成空白。"""
    if sep is None or sep == "":
        return "-"
    s = str(sep)
    if s == "\t":
        return "Tab"
    if s == " ":
        return "空格"
    if not s.isprintable():
        return repr(s)
    return f"'{s}'"


def _format_notes(loader) -> list:
    """并进「格式」行的小字段：只说"读法"，且默认值不占字。"""
    notes = []

    sheet = str(getattr(loader, "sheet_name", "") or "")
    if sheet:
        notes.append(f"工作表：{sheet}")

    encoding = str(getattr(loader, "encoding_used", "") or "")
    if encoding:
        notes.append(f"编码：{encoding}")

    sep = getattr(loader, "sep", None)
    if sep:
        notes.append(f"分隔符：{_format_text(sep)}")

    # 单位行默认是"有"、说明行默认是 0，只有偏离默认才值得占字数
    if getattr(loader, "has_unit", None) is False:
        notes.append("无单位行")
    desc_rows = getattr(loader, "desc_rows", None)
    if desc_rows:
        notes.append(f"说明行：{int(desc_rows)}")

    groups = _positive_int(loader, "group_count")
    if groups:
        notes.append(f"通道组：{groups}")
    return notes


def _format_row(loader) -> tuple:
    kind = getattr(loader, "LOADER_TYPE", "") or ""
    label = _FORMAT_NAME.get(kind) or _format_by_suffix(loader)
    notes = _format_notes(loader)
    if notes:
        label = f"{label}（{'，'.join(notes)}）"
    return ("格式", label)


def _format_by_suffix(loader) -> str:
    suffix = os.path.splitext(str(getattr(loader, "path", "") or ""))[1].lstrip(".")
    return f"{suffix.upper()} 文件" if suffix else "未知格式"


# ---------------------------------------------------------------------------
# 「数据规模」几行
# ---------------------------------------------------------------------------


def _size_rows(loader, elapsed_s: float | None) -> list:
    rows = []
    count = _count_attr(loader, "row_count")
    if count is not None:
        if getattr(loader, "LOADER_TYPE", "") == "mdf":
            # MDF 的行数取自最大通道组的 cycles_nr，不是全文件总点数，也不是
            # CSV 那种"实际读到的行数"，标签必须把这个口径写出来
            rows.append(("记录数", f"{count} 行（最大通道组的声明记录数）"))
        else:
            rows.append(("数据行数", f"{count} 行"))

    if elapsed_s is not None:
        rows.append(("加载耗时", f"{elapsed_s:.2f} s"))
    return rows


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _count_attr(obj, name: str) -> int | None:
    """取整数属性，读失败返回 None。

    单个字段读不出来不该让整扇抽屉打不开，因此一律吞异常并记 debug 日志。
    0 是有效值（空文件的"0 行"是要说给用户听的结论），照实返回。
    """
    try:
        value = getattr(obj, name)
    except Exception:
        logger.debug("读取文件信息字段 %s 失败", name, exc_info=True)
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _positive_int(obj, name: str) -> int | None:
    """取正整数属性；缺失、非数值或 <=0 时返回 None（0 的计数不展示）。"""
    value = _count_attr(obj, name)
    return value if value and value > 0 else None


__all__ = [
    "KEY_FILE_NAME",
    "KEY_FOLDER",
    "ROW_KEY_FILE_PATH",
    "build_file_rows",
]
