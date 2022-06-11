from type_def import *

# 两个子任务的输入维度与label数量
input_dim_1 = 76
label_cnt_1 = 10

input_dim_2 = 32 * 32  # 1024
label_cnt_2 = 100


# 训练参数
bsz = 32
lr = 0.0005
train_val_split = 0.9