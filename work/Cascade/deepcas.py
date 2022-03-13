from type_def import *

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from itertools import chain
from torch.optim import AdamW
import pickle
import numpy as np
from tqdm import tqdm

from .attention_model import StepWeight, GeoWeight
from analysis.recorder import BaseRecorder
from utils import tools
from utils.batch_tool import simple_collate_fn
from work.Cascade import cascade_settings, cascade_utils
from utils.data import SimpleDataset
from models.model_utils import get_init_params
from evaluate.evaluator import BaseEvaluator, \
    l1_l2_regularizer, \
    mean_relative_square_loss, \
    mean_squared_logarithmic_error, \
    median_relative_square_loss, \
    median_squared_logarithmic_error, mean_squared_error, median_squared_error


class DeepCas(nn.Module):
    """
    完全按照model.py复制
    然后再看看论文，看看有什么地方不对吧
    init param:
        - n_sequences: 一个cascade所包含的sequence个数
        - sequence_batch_size: K-attention所使用的bsz
        - embedding_size:
        - embedding_vec_path: 加载embedding的路径
        - n_input: ?
        - seq_l: 每个sequence的node个数
        - gru_hidden_size:
        - dense1_hidden: 第一层dense的hidden size
        - dense2_hidden: 第二层dense的hidden size
        - activation: 激活层的类型, tanh or relu
        - embedding_lr
        - others_lr
    """
    def __init__(self,
                 bsz: int = cascade_settings.sequence_batch_size,
                 walk_length: int = cascade_settings.walk_length,
                 embedding_size: int = cascade_settings.embedding_size,
                 embedding_word2vec_path: str = cascade_settings.word2vec_file,
                 embedding_word2vec_dir: str = cascade_settings.output_directory,
                 gru_hidden_size: int = cascade_settings.gru_hidden,
                 dense1_hidden: int = cascade_settings.dense1_hidden,
                 dense2_hidden: int = cascade_settings.dense2_hidden,
                 activation: str = cascade_settings.activation,
                 embedding_lr: float = cascade_settings.embedding_lr,
                 others_lr: float = cascade_settings.others_lr):
        super(DeepCas, self).__init__()
        self.init_params = get_init_params(locals())
        if embedding_word2vec_dir[-1] != '/':
            embedding_word2vec_dir += '/'
        word2vec_file = embedding_word2vec_dir + embedding_word2vec_path

        # 初始化参数
        #   网络中集成随机游走序列的batch大小
        self.sequence_batch_size = bsz
        #   每一个序列的长度
        self.seq_l = walk_length
        #   word2vec的维度与文件的路径。此处embedding_size是模型的embedding大小，因此word2vec也需要是这个大小
        self.embedding_size = embedding_size
        self.embedding_vec_path = word2vec_file
        #   网络层配置
        self.gru_hidden_size = gru_hidden_size
        self.dense1_hidden = dense1_hidden
        self.dense2_hidden = dense2_hidden
        self.activation = activation
        self.embedding_lr = embedding_lr
        self.others_lr = others_lr

        # 加载word2vec嵌入
        wordvec = np.array(pickle.load(open(self.embedding_vec_path, 'rb')))  # np array (node cnt, embed dim)
        node_cnt = wordvec.shape[0]
        print(f'wordvec\'s shape:{wordvec.shape}')
        wordvec_tensor = torch.tensor(wordvec)

        # 初始化网络结构
        #   Embedding
        self.embedding = nn.Embedding(node_cnt, self.embedding_size)
        self.embedding.weight.data = nn.Parameter(wordvec_tensor)
        #   BiGRU
        # self.BiGru = nn.GRU(input_size=self.embedding_size, hidden_size=self.gru_hidden_size, num_layers=2, bidirectional=True)
        self.LSTM = nn.LSTM(input_size=self.embedding_size, hidden_size=self.gru_hidden_size, num_layers=2, bidirectional=True)
        #       expect input: (seq_l, bsz, hidden)
        #   attention
        self.step_weight = StepWeight(self.seq_l)
        self.geo_weight = GeoWeight(self.sequence_batch_size)
        #   dense
        self.dense1 = nn.Linear(self.gru_hidden_size * 2, self.dense1_hidden)
        self.dense2 = nn.Linear(self.dense1_hidden, self.dense2_hidden)
        self.dense3 = nn.Linear(self.dense2_hidden, 1)
        if self.activation == 'relu':
            self.activation_layer = nn.LeakyReLU()
        else:
            self.activation_layer = nn.Tanh()

    def forward(self,
                x_vector: torch.Tensor,
                cascade_sizes: torch.Tensor):
        """

        :param x_vector: (bsz, n_sequences, seq_l)
        :param cascade_sizes: (bsz), batch中每个sample的cascade的size，
        :return:
        """
        # 首先把几个常量取出来
        bsz, n_seq, seq_l = x_vector.shape
        embed_size = self.embedding_size

        # 获得embedding，然后转化成GRU适合输入的格式
        x_vector = self.embedding(x_vector)  # (bsz, n_sequences, seq_l, embed_size)
        x_vector = x_vector.permute([1, 0, 2, 3])  # (n_sequences, bsz, seq_l, embed_size)
        x_vector = x_vector.reshape([n_seq * bsz, seq_l, embed_size])  # (n_seq * bsz, seq_l, embed_size)
        x_vector = x_vector.permute([1, 0, 2]).float()  # (seq_l, n_seq * bsz, embed_size)

        # GRU part
        # output, _ = self.BiGru(x_vector)
        output, (_, __) = self.LSTM(x_vector)  # (seq_l, n_seq * bsz, 2 * hidden)
        output = output.reshape([seq_l, n_seq, bsz, -1]).permute([2, 1, 3, 0])  # (bsz, n_seq, 2 * hidden, seq_l)

        # weight part
        #   sequential attention
        output = self.step_weight(output)
        #   geometric attention
        output = output.permute([0, 1, 3, 2])
        output = self.geo_weight(output, cascade_sizes)  # (bsz, n_seq, seq_l, 2 * hidden)

        # weighted sum
        output = output.permute([0, 3, 1, 2])  # （bsz, 2 * hidden, n_seq, seq_l）
        output = output.reshape([bsz, -1, n_seq * seq_l])  # (bsz, 2 * hidden, n_seq * seq_l)
        output = torch.sum(output, dim=-1) / (n_seq * seq_l)  # (bsz, 2 * hidden)
        output = self.activation_layer(self.dense1(output))  # (bsz, dense1_hidden)
        output = self.activation_layer(self.dense2(output))  # (bsz, dense2_hidden)
        output = self.activation_layer(self.dense3(output))  # (bsz, 1)
        if self.training:
            return {
                "output": output,  # (bsz, 1)
                "model": self
            }
        else:
            return {
                "output": output,
                "model": self
            }

    def get_optimizers(self):
        self.embedding_params = self.embedding.parameters()
        # self.gru_parameters = self.BiGru.parameters()
        self.step_weight_params = self.step_weight.parameters()
        self.geo_weight_params = self.geo_weight.parameters()
        self.gru_parameters = self.LSTM.parameters()
        self.dense1_params = self.dense1.parameters()
        self.dense2_params = self.dense2.parameters()
        self.dense3_params = self.dense3.parameters()

        self.embed_optimizer = AdamW(self.embedding_params, lr=self.embedding_lr)
        self.others_optimizer = AdamW(chain(self.geo_weight_params, self.step_weight_params, self.gru_parameters, self.dense1_params, self.dense2_params, self.dense3_params), lr=self.others_lr)
        return [self.embed_optimizer, self.others_optimizer]


class DeepCasLoss(nn.Module):
    def forward(self,
                output: torch.Tensor,
                model,
                label: torch.Tensor,
                l1: float = cascade_settings.l1,
                l2: float = cascade_settings.l2,
                l1l2: float = cascade_settings.l1l2):
        """
        并不知道tensorflow版本的regularization方法到底是什么
        :param output: (bsz, 1)
        :param model:
        :param label:
        :param l1:
        :param l2:
        :param l1l2:
        :return:
        """
        reduce_mean = torch.mean((output - label) ** 2)
        regular = l1_l2_regularizer(model, lambda_l1=l1, lambda_l2=l2)
        regularized_loss = reduce_mean + l1l2 * regular
        return regularized_loss


class DeepCasEvaluator(BaseEvaluator):
    def __init__(self):
        self.cnt = 0
        self.errors = []
        self.origin_outputs = []
        self.origin_labels = []
        self.model = None

    def eval_single(self,
                    output: torch.Tensor,
                    label: torch.Tensor,
                    model):
        self.cnt += 1
        self.errors.append(float(torch.mean((output - label) ** 2)))
        # self.origin_outputs.append(2 ** float(output) - 1)
        # self.origin_labels.append(2 ** float(label) - 1)
        self.origin_outputs.append(float(output))
        self.origin_labels.append(float(label))
        self.model = model

    def eval_step(self) -> str:
        error = sum(self.errors) / self.cnt
        self.cnt = 0
        self.errors = []

        # origin
        origin_out = np.array(self.origin_outputs)
        origin_tgt = np.array(self.origin_labels)
        self.origin_outputs = []
        self.origin_labels = []

        origin_MRSE = mean_relative_square_loss(origin_tgt, origin_out)
        origin_mRSE = median_relative_square_loss(origin_tgt, origin_out)
        origin_MSLE = mean_squared_error(origin_tgt, origin_out)
        origin_mSLE = median_squared_error(origin_tgt, origin_out)

        step_weight = self.model.step_weight.step_weight.tolist()
        geo_weight = float(self.model.geo_weight.geo_weight)

        return {
            "MSLE": origin_MSLE,
            "mSLE": origin_mSLE,
            "MRSE": origin_MRSE,
            "mRSE": origin_mRSE,
            "step_weight": step_weight,
            "geo_weight": geo_weight
        }


def dataset_factory(cascade_directory: str = cascade_settings.cascade_directory,
                    globalgraph_directory: str = cascade_settings.globalgraph_directory,
                    output_directory: str = cascade_settings.output_directory,
                    bsz: int = cascade_settings.sequence_batch_size):
    """
    需要读取的文件：
    - vocab_index
    - random walk
    - cascade
    :param cascade_directory:
    :param globalgraph_directory:
    :param output_directrory:
    :return:
    """
    if output_directory[-1] != '/':
        output_directory += '/'
    if cascade_directory[-1] != '/':
        cascade_directory += '/'
    if globalgraph_directory[-1] != '/':
        globalgraph_directory += '/'
    # 首先读取vocab_index。vocab_index用于将cascade与random_walk中的id转换为word2vec用到的id
    print('loading vocab index')
    vocab_index = pickle.load(open(output_directory + cascade_settings.vocab_index_file, 'rb'))

    # 接下来分别处理train、val、test的数据
    print('loading random walk and cascade file')
    random_walk_train = cascade_utils.load_random_walks_txt(output_directory + cascade_settings.random_walks_train_file)
    random_walk_test = cascade_utils.load_random_walks_txt(output_directory + cascade_settings.random_walks_test_file)
    random_walk_val = cascade_utils.load_random_walks_txt(output_directory + cascade_settings.random_walks_val_file)
    cascade_train = cascade_utils.read_cascade(cascade_directory + cascade_settings.cascade_train_file)
    cascade_test = cascade_utils.read_cascade(cascade_directory + cascade_settings.cascade_test_file)
    cascade_val = cascade_utils.read_cascade(cascade_directory + cascade_settings.cascade_val_file)

    print('generating train_dataloader')
    train_dataloader = train_dataset_function(cascade_train, random_walk_train, vocab_index, bsz=bsz, shuffle=cascade_settings.shuffle)
    print('generating val_dataloader')
    val_dataloader = val_dataset_function(cascade_val, random_walk_val, vocab_index)

    return train_dataloader, val_dataloader


def train_dataset_function(train_cascades: List[dict], train_random_walks: Dict[str, Any], vocab_index, bsz: int = cascade_settings.sequence_batch_size, shuffle=True):
    """
    生成训练数据
    :param train_cascades:
    :param train_random_walks:
    :param vocab_index:
    :return:
    """
    # 先把random walk的数据都转为int类型
    for key, value in train_random_walks.items():
        for idx, elem in enumerate(value):
            value[idx] = list(map(int, elem))

    # 然后从cascade中解析出label，size等信息
    train_cascade = tools.transpose_list_of_dict(train_cascades)
    # id, edge_cnt, edges, label, time(s)

    x = []  # 每一个cascade所对应的所有random_walk序列
    y = []  # 每一个cascade的增长数值，也就是要预测的值
    sizes = train_cascade['edge_cnt']  # 每一个cascade的节点数目
    times = train_cascade['time']  # 每一个cascade中的时间
    for elem_id, elem_label in tqdm(list(zip(train_cascade['id'], train_cascade['label']))):
        walk = train_random_walks[str(elem_id)]  # List[list]
        padded_walk = []
        for elem_walk in walk:
            if len(elem_walk) < cascade_settings.walk_length:
                for i in range(cascade_settings.walk_length - len(elem_walk)):
                    elem_walk.append(-1)
            padded_walk.append(vocab_index.new(elem_walk))
        x.append(padded_walk)
        y.append(np.log2(elem_label + 1.0))

    data_dict = {
        "x": x,
        "y": y,
        "sizes": sizes,
        "times": times
    }
    data_dicts = tools.transpose_dict_of_list(data_dict)
    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        rst = simple_collate_fn(lst)
        # x - list of [bsz, n_seq, seq_l]
        # y - [bsz] of int
        # sz - [bsz] of int
        x_vector = torch.tensor(rst['x'], dtype=torch.int)
        cascade_sizes = torch.tensor(rst['sizes'], dtype=torch.int)
        label = torch.tensor(rst['y'])
        return {
            "x_vector": x_vector,  # (bsz, n_seq, seq_l)
            "cascade_sizes": cascade_sizes  # (bsz)
               }, {
            "label": label  # (bsz)
        }

    train_dataloader = DataLoader(train_dataset, bsz, shuffle=shuffle, collate_fn=collate_fn)
    return train_dataloader


def val_dataset_function(val_cascades: List[dict], val_random_walks: Dict[str, Any], vocab_index):
    """
    生成评价数据
    train和val的生成方法是一样的，因为都是直接以预测结果作为label
    :param val_cascades:
    :param val_random_walks:
    :param vocab_index:
    :return:
    """
    return train_dataset_function(val_cascades, val_random_walks, vocab_index, bsz=1, shuffle=False)


model_registry = {
    "model": DeepCas,
    "loss": DeepCasLoss,
    "evaluator": DeepCasEvaluator,
    "train_val_data": dataset_factory,
    'recorder': BaseRecorder
}