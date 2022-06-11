"""
SemEval 新闻相似度任务
基础设定
"""
# 模型配置
pretrain_path = 'xlm-roberta-base'


# 数据性质
legal_lang_pairs = ['en-en','de-de','de-en','ar-ar','es-es','fr-fr','pl-pl','tr-tr']
train_ratio = 0.9
max_seq_length = 512

# 路径配置
crawl_file_dir = 'data/NLP/news_sim/final_transcode_el'
newspair_file = 'data/NLP/news_sim/data/semeval-2022_task8_train-data_batch.csv'

# 输入数据参数
bsz = 8
title_len = 30
text_len = 100   #  100
summary_len = 50


# 训练参数
linear_lr = 2e-4
plm_lr = 1e-5

# 特殊参数
two_tower = True