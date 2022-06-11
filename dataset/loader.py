"""
todo 支持模糊匹配数据集
"""
from dataset.ee_dataset import *

dataset_registry = {
    "NYT": None,
    "duie": None,
    "WebNLG": None,
    'FewFC': load_FewFC_ee,
    'Duee': load_Duee_ee_formated
}

