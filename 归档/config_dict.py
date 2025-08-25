import json
from pathlib import Path

def save_dict(data: dict, path: str, *, indent: int = 2, ensure_ascii=False):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)   # 自动创建目录
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

def load_dict(path: str , *, default=None) -> dict:

    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {} if default is None else default
    
def create_data_loader_config(path):
    config_dict={}
    sample_dict={'skiprows':int(0),
                'hasunit':bool(1),
                'sep':','}
    
    # csv type
    csv=sample_dict.copy()
    config_dict['csv']=csv
    
    # txt type
    txt=csv.copy()
    config_dict['txt']=txt

    t00=sample_dict.copy()
    t00['skiprows']=int(2)
    t00['hasunit']=bool(1)
    t00['sep']='\t'

    mfile=t00.copy()
    config_dict['mfile']=mfile

    t01=t00.copy()
    config_dict['t01']=t01

    save_dict(config_dict,path=path)
    


if __name__ == "__main__":
    create_data_loader_config('config_dict.json')
