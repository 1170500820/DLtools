
# 情感分析任务定义
sentiment_label = [0, 1, 2]
sentiment_label_idx = {x: i for (i, x) in enumerate(sentiment_label)}


# default model setting
default_plm = 'bert-base-chinese'
default_plm_lr = 2e-5
default_others_lr = 1e-4
default_bsz = 8
