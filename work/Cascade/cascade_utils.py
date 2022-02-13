import pickle

from type_def import *

from tqdm import tqdm
import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from utils import tools, dir_tools
from utils.graph import node2vec
from utils.graph.index_edit import IndexDict
from utils.graph.tools import cascade2edges, get_cascade_idx
import cascade_settings


def parse_cascade_line(cascade_line: str) -> Dict[str, Any]:
    """
    解析一个cascade文件中的一行，

    一行数据里面的不同信息按\t隔开
        - parts[0]: 不确定，但是应该是一个id？
        - parts[1]: 是一些用户
        - parts[2]: 似乎是年份？
        - parts[3]: 不像是用户id，可能是时间？
        - parts[4]: cascade中的边的信息

    :param cascade_line:
    :return:
    """
    parts = cascade_line.split('\t')
    edge_strs = parts[4].split(" ")
    node_to_edges = dict()
    edge_cnt = int(parts[3])
    edges = []
    if edge_cnt != 0:
        for elem_edge in edge_strs:
            edge_parts = elem_edge.split(":")
            src_node = int(edge_parts[0])
            tgt_node = int(edge_parts[1])
            edges.append((src_node, tgt_node))
        # node2edges: src node -> (tgt node, tgt degree)
    cascade = {
        "id": int(parts[0]),  # str(int)
        "parts_1": parts[1],
        "parts_2": parts[2],
        "edge_cnt": edge_cnt,  # int
        "edges": edges,  # List[Tuple[int, int]]
        'label': int(parts[5]),  # int
        "time": list(map(int, parts[6].split(' ')))  # List[int]
    }
    return cascade


def read_cascade(cascade_path: str) -> List[dict]:
    """
    读取一个cascade文件

    一个cascade文件的格式如下：
    每一行都是一个cascade，具体的行格式见parse_cascade_line函数的注释部分

    :param cascade_path:
    :return:
    """
    cascade_lines = open(cascade_path, 'r', encoding='utf-8').read().strip().split('\n')
    cascade_infos = list(parse_cascade_line(x) for x in cascade_lines)
    return cascade_infos


def read_global_graph(global_graph_path: str):
    """
    读取一个Global Graph文件

    一个Global Graph文件的格式如下：
    todo
    :param global_graph_path:
    :return:edge2weight, node2degree
    """
    lines = open(global_graph_path, 'r', encoding='utf-8').read().strip().split('\n')
    edge2weight, node2degree = {}, {}
    print('gathering global graph info...')
    for elem_line in lines:
        parts = elem_line.split('\t\t')  # StrList, len==2
        source_node = int(parts[0])  # source node of current path?
        if parts[1] != 'null':  # 当source node存在后继节点
            node_freq_strs = parts[1].split('\t')
            # StrList, str be like target_node:weight
            for elem_node_freq in node_freq_strs:
                node_freq = elem_node_freq.split(':')
                weight = int(node_freq[1])
                target_node = int(node_freq[0])
                if cascade_settings.trans_type == 0:
                    edge2weight[(source_node, target_node)] = weight
            degree = len(node_freq_strs)
        else:
            degree = 0
        node2degree[source_node] = degree
    return edge2weight, node2degree


def generate_random_walk(
        edge2weight: dict,
        node2degree: dict,
        cascades: List[dict],
        walks_per_graph: int = cascade_settings.walks_per_graph,
        trans_type: int = cascade_settings.trans_type,
        walk_length: int = cascade_settings.walk_length,
        pseudo_count: int = cascade_settings.pseudo_count,
        p: float = cascade_settings.node2vec_p,
        q: float = cascade_settings.node2vec_q):
    """
    在一个global graph中，为每个cascade计算随机游走路径


    :param edge2weight:
    :param node2degree:
    :param cascades:
    :param walks_per_graph: 每一个cascade生成的随机游走序列的个数
    :param trans_type:
    :param walk_length:
    :param pseudo_count:
    :param p:
    :param q:
    :return:
        List[str] 每个str都是用\t隔开的随机游走序列
    """
    random_walk_lines = []
    random_walk_dict = {}
    # key = cascade_id
    # value = list of random walk list
    print('random walking...')
    for elem_cascade in tqdm(cascades):
        cascade_id = elem_cascade['id']
        edge_cnt = elem_cascade['edge_cnt']
        edges = elem_cascade['edges']
        node_to_edges = dict()
        if edge_cnt != 0:
            for (elem_src, elem_tgt) in edges:
                try:
                    if not elem_src in node_to_edges:
                        neighbors = list()
                        node_to_edges[elem_src] = neighbors
                    else:
                        neighbors = node_to_edges[elem_src]
                    neighbors.append((elem_tgt, node2degree.get(elem_tgt, 0)))
                except:
                    pass
        nx_G = nx.DiGraph()
        for source, nbr_weights in node_to_edges.items():
            # 这是老版本的dict.items()吗
            for nbr_weight in nbr_weights:
                target = nbr_weight[0]

                if trans_type == 0:  # trans_type不是string吗，怎么又012了
                    edge_weight = pseudo_count + edge2weight.get((source, target), 0)
                    weight = edge_weight
                elif trans_type == 1:
                    target_nbrs = node_to_edges.get(target, None)
                    local_degree = 0 if target_nbrs is None else len(target_nbrs)
                    local_degree += pseudo_count
                    weight = local_degree
                else:
                    global_degree = nbr_weight[1] + pseudo_count
                    weight = global_degree
                # 应该分别对于edge，deg，DEG
                nx_G.add_edge(source, target, weight=weight)
            # 这里就是为每一条边定义了weight，用来计算转移概率的

        # List of the starting nodes.
        roots = list()
        # List of the starting nodes excluding nodes without outgoing neighbors.
        roots_noleaf = list()
        # exclude?

        str_list = list()
        str_list.append(cascade_id)
        random_walk_dict[cascade_id] = []

        probs = list()
        probs_noleaf = list()
        weight_sum_noleaf = 0.0
        weight_sum = 0.0

        # Obtain sampling probabilities of roots.
        for node, weight in nx_G.out_degree(weight="weight"):
            org_weight = weight
            if weight == 0:
                weight += pseudo_count
            weight_sum += weight
            if org_weight > 0:
                weight_sum_noleaf += weight

        for node, weight in nx_G.out_degree(weight="weight"):
            org_weight = weight
            if weight == 0:
                weight += pseudo_count
            roots.append(node)
            prob = weight / weight_sum
            probs.append(prob)
            if org_weight > 0:
                roots_noleaf.append(node)
                prob = weight / weight_sum_noleaf
                probs_noleaf.append(prob)

        sample_total = walks_per_graph
        first_time = True
        G = node2vec.Graph(nx_G, True, p, q)
        G.preprocess_transition_probs()

        while True:
            if first_time:
                first_time = False
                node_list = roots
                prob_list = probs
            else:
                node_list = roots_noleaf
                prob_list = probs_noleaf
            n_sample = min(len(node_list), sample_total)
            if n_sample <= 0:
                break
            sample_total -= n_sample

            sampled_nodes = np.random.choice(node_list, n_sample, replace=False, p=prob_list)
            walks = G.simulate_walks(len(sampled_nodes), walk_length, sampled_nodes)
            random_walk_dict[cascade_id].extend(walks)
            for walk in walks:
                str_list.append(' '.join(str(k) for k in walk))
        result = '\t'.join(str_list)
        random_walk_lines.append(result)
    return random_walk_dict


def output_random_walks_txt(random_walks: dict, filename: str):
    """
    将random_walk_dict转换为旧的字符串格式，然后输出
    这样做是为了兼容旧的代码
    :param random_walks:
    :param filename:
    :return:
    """
    lines = []
    for key, value in random_walks.items():
        walk = [key]
        for elem_walk in value:
            walk.append(' '.join(list(str(x) for x in elem_walk)))
        lines.append('\t'.join(walk))
    f = open(filename, 'w', encoding='utf-8')
    for elem in lines:
        f.write(elem + '\n')
    f.close()


def load_random_walks_txt(filename: str):
    """
    读取旧的字符串格式的random_walk文件，然后转换为dict格式
    :param filename:
    :return:
    """
    lines = open(filename, 'r', encoding='utf-8').read().strip().split('\n')
    random_walk_dict = {}
    for elem_line in lines:
        parts = elem_line.split('\t')
        cascade_id = parts[0]
        walks = list(x.split(' ') for x in parts[1:])
        random_walk_dict[cascade_id] = walks
    return random_walk_dict


def generate_vocab_index(original_ids):
    """
    为每个id重新生成一段连续的整数id列表
    :param original_ids:
    :return:
    """
    if -1 not in original_ids:
        if isinstance(original_ids, set):
            original_ids.add(-1)
        else:
            original_ids.append(-1)
    print('generating vocab index...')
    vocab_index = IndexDict(original_ids)
    return vocab_index


def convert_random_walk_to_original_ids(random_walk_dicts: dict):
    """
    根据随机游走序列获取所有包含的节点的id，转化成int类型，生成一个集合
    :param random_walk_dict:
    :return:
    """
    vocab_set = set()
    for elem_walks in random_walk_dicts.values():
        for elem_walk in elem_walks:
            for elem_node in elem_walk:
                vocab_set.add(int(elem_node))
    return vocab_set


def convert_random_walk_lines_to_original_ids(random_walk_lines: List[str]):
    """
    根据随机游走序列获取所有包含的节点的id，生成一个集合
    :param random_walk_lines:
    :return:
    """

    vocab_set = set()
    print('converting random walk to original ids...')
    for elem_line in random_walk_lines:
        walks = elem_line.split('\t')
        for elem_walk in walks[1:]:
            for elem_node in elem_walk.split():
                vocab_set.add(int(elem_node))
    return vocab_set


def convert_cascade_to_original_ids(cascades: List[dict]):
    """
    与convert_random_walk_to_original_ids类似
    读取所有cascade中包含的节点的id，生成一个id的集合
    :param cascades:
    :return:
    """
    vocab_set = set()
    print('converting cascade to original ids')
    for elem_cascade in cascades:
        edges = elem_cascade['edges']
        for elem_edge in edges:
            node1, node2 = elem_edge
            vocab_set.add(int(node1))
            vocab_set.add(int(node2))
    return vocab_set


def generate_word2vec(
        random_walk_dicts: dict,
        embedding_size=cascade_settings.word2vec_dimensions,
        window_size=cascade_settings.word2vec_window_size,
        epochs=cascade_settings.word2vec_iter_epoch,
        workers=cascade_settings.word2vec_workers,
        min_count=cascade_settings.word2vec_min_count,
        sg=cascade_settings.word2vec_sg):
    """
    输入所有的非测试集的随机游走序列，为每一个id训练word2vec向量
    :param random_walk_dicts: 非测试集的随机游走序列
    :param embedding_size:
    :param window_size:
    :param epochs:
    :param workers:
    :param min_count:
    :param sg:
    :return:
    """
    walks = []
    for elem_walks in random_walk_dicts.values():
        walks.extend(elem_walks)
    model = Word2Vec(
        walks,
        vector_size=embedding_size,
        window=window_size,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs)
    return model


def resemble_word2vec(word2vec_model, vocab_index):
    """
    为word2vec词向量的每个node，用vocab_index分配新的index
    :param word2vec_model:
    :param vocab_index:
    :return:
    """
    np.random.seed(13)  # 在空白处填充随机值
    wv = word2vec_model.wv
    num_nodes, num_dims = wv.vectors.shape

    # 随机生成一个已有词汇表大小的词嵌入矩阵（）
    node_vec = np.random.normal(size=(vocab_index.length(), num_dims))
    node_vec = node_vec.tolist()
    for i in tqdm(list(range(num_nodes))):
        vec = wv[i]
        vec_idx = int(wv.index_to_key[i])
        if vec_idx not in vocab_index.vocab_set:
            continue
        node_vec[vocab_index.new(vec_idx)] = vec
    return node_vec


def generate_deepcas_data(
        cascade_directory: str = cascade_settings.cascade_directory,
        globalgraph_directory: str = cascade_settings.globalgraph_directory,
        output_directory: str = cascade_settings.output_directory,
        walks_per_graph: int = cascade_settings.walks_per_graph,
        walk_length: int = cascade_settings.walk_length):
    """
    生成一次DeepCas以及以deepcas为基础的模型训练所需的所有数据
    walks_per_graph与walk_length为核心参数，由于经常用到，所以也加入到参数中
    :param cascade_directory: cascade文件的存放路径。
        具体的cascade文件名在settings中配置了
    :param globalgraph_directory: GlobalGraph文件的存放路径
    :param output_directory: 生成的random_walk与vocab_index的保存路径
    :param walks_per_graph:
    :param walk_length:
    :return:
    """
    # 处理一下路径后缀
    if cascade_directory[-1] != '/':
        cascade_directory += '/'
    if globalgraph_directory[-1] != '/':
        cascade_directory += '/'
    if output_directory[-1] != '/':
        output_directory += '/'

    # 首先分别读取train、val、test的cascade文件，以及global_graph的信息
    train_cascades = read_cascade(cascade_directory + cascade_settings.cascade_train_file)
    val_cascades = read_cascade(cascade_directory + cascade_settings.cascade_val_file)
    test_cascades = read_cascade(cascade_directory + cascade_settings.cascade_test_file)
    edge2weight, node2degree = read_global_graph(globalgraph_directory + cascade_settings.global_graph_file)

    # 分别生成train、val、test的random_walk
    train_random_walks = generate_random_walk(edge2weight, node2degree, train_cascades, walks_per_graph=walks_per_graph, walk_length=walk_length)
    val_random_walks = generate_random_walk(edge2weight, node2degree, val_cascades, walks_per_graph=walks_per_graph, walk_length=walk_length)
    test_random_walks = generate_random_walk(edge2weight, node2degree, test_cascades, walks_per_graph=walks_per_graph, walk_length=walk_length)

    # 分别使用train、val、test生成original_ids，然后合并。这里不用random_walk生成，因为random_walk可能不包含某些node
    train_original_ids = convert_cascade_to_original_ids(train_cascades)
    val_original_ids = convert_cascade_to_original_ids(val_cascades)
    test_original_ids = convert_cascade_to_original_ids(test_cascades)
    original_ids = set()
    original_ids = original_ids.union(train_original_ids)
    original_ids = original_ids.union(val_original_ids)
    original_ids = original_ids.union(test_original_ids)

    # 用original_ids生成vocab_index
    vocab_index = generate_vocab_index(original_ids)

    # 使用训练集的random_walk训练word2vec嵌入矩阵
    original_word2vec = generate_word2vec(train_random_walks)
    word2vec = resemble_word2vec(original_word2vec, vocab_index)

    # 保存
    pickle.dump(vocab_index, open(output_directory + cascade_settings.vocab_index_file, 'wb'))
    pickle.dump(word2vec, open(output_directory + cascade_settings.word2vec_file, 'wb'))
    output_random_walks_txt(train_random_walks, output_directory + cascade_settings.random_walks_train_file)
    output_random_walks_txt(val_random_walks, output_directory + cascade_settings.random_walks_val_file)
    output_random_walks_txt(test_random_walks, output_directory + cascade_settings.random_walks_test_file)


if __name__ == '__main__':
    generate_deepcas_data()