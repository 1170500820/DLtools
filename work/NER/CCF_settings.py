"""
https://www.datafountain.cn/competitions/529/datasets
按这个来的
"""

default_plm = 'bert-base-chinese'

ner_tags = [
    'O',  # 代表不属于标注的范围
    'B-BANK',  # 代表银行实体的开始
    'I-BANK',  # 代表银行实体的内部
    'B-PRODUCT',  # 代表产品实体的开始
    'I-PRODUCT',  # 代表产品实体的内部
    'B-COMMENTS_N',  # 代表用户评论（名词）
    'I-COMMENTS_N',  # 代表用户评论（名词）实体的内部
    'B-COMMENTS_ADJ',  # 代表用户评论（形容词）
    'I-COMMENTS_ADJ',  # 代表用户评论（形容词）实体的内部
]
ner_tags_idx = {s: i for i, s in enumerate(ner_tags)}

sentiment_label_cnt = 3


plm_lr = 2e-5
others_lr = 1e-4


# for naive bert ner hybrid
lmd = 0.7

default_bsz = 8
default_shuffle = True
default_train_val_split = 0.9

# bert info
bert_CLS_id = 101
bert_SEP_id = 102
