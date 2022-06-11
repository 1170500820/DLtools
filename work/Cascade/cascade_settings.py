"""
DeepCase模型的配置文件

- walks_per_graph
    每个图中的随机游走路径的条数
- walk_length
    每条随机游走路径的长度
- trasn_type
    随机游走中，计算转移改了的方式，0，1，2分别代表edge, deg, DEG三种
- pseudo_count
    防止出现0值，用一个较小的值代替
"""
from type_def import *


# 随机游走参数
walks_per_graph = 200
walk_length = 20
trans_type = 2
pseudo_count = 0.00001  # 0.001 origin


# DeepCase模型参数
gru_hidden = 32
activation = 'relu'
l1 = 5e-5
l2 = 0
l1l2 = 1.0
dense1_hidden = 32
dense2_hidden = 16


# 训练参数
others_lr = 0.01
embedding_lr = 5e-5
sequence_batch_size = 20
embedding_size = 50
shuffle = True



# node2vec 参数
#    - p:
#        Return hyperparameter in node2vec.
#    - q:
#        Inout hyperparameter in node2vec.


node2vec_p = 1.0
node2vec_q = 1.0


# word2vec 参数
#   - dimensions
#       生成的每个node的嵌入向量的维度
#   - window_size
#       序列中一个node的前后窗口的大小
#   - iter_epoch
#       训练用的epoch数
#   - workers
#       使用的cpu数量
#   - min_count
#   - sg
#       这俩我也不知道

word2vec_dimensions = embedding_size
word2vec_window_size = 10
word2vec_iter_epoch = 5
word2vec_workers = 8
word2vec_min_count = 0
word2vec_sg = 1


# 路径配置
cascade_directory = '../../data/cascade/'
globalgraph_directory = '../../data/cascade/'
output_directory = '../../data/cascade'

cascade_train_file = 'cascade_train.txt'
cascade_val_file = 'cascade_val.txt'
cascade_test_file = 'cascade_test.txt'
global_graph_file = 'global_graph.txt'
random_walks_train_file = 'random_walks_train.txt'
random_walks_val_file = 'random_walks_val.txt'
random_walks_test_file = 'random_walks_test.txt'
vocab_index_file = 'vocab_index.pk'
word2vec_file = 'word2vec.pk'
