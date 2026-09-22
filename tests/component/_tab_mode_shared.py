"""变量数值表 tab 模式测试的共享替身（拆分产物，勿直接收集）。

两个拆分文件都要**调用**的 loader 替身（FakeCsvLoader）留在这里；
夹具（mdf_loader / tab_dialog / shown_tab_dialog）在
tests/component/conftest.py。拆分只搬代码，未改任何断言。
"""




class FakeCsvLoader:
    """最小 CSV loader 替身：update_data 非 MDF 分支只用 df/units。"""

    LOADER_TYPE = "csv"
    path = "fake.csv"
    var_names = ()

    def __init__(self, df):
        self.df = df
        self.units = {c: "" for c in df.columns}
