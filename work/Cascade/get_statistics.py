import argparse
import sys
import os
import pandas as pd
import numpy as np
from cascade_utils import load_random_walks_txt, read_cascade, read_global_graph, cascade_edges_to_graph
from loguru import logger
import networkx as nx
import matplotlib.pyplot as plt

pd.set_option('display.float_format',lambda x : '%.4f' % x)
tasks = ['random_walk', 'cascade', 'draw_cascade']


def get_random_walk_statisitcs(walk_txt_path: str):
    walk_dict = load_random_walks_txt(walk_txt_path)
    lengths = []
    for cascade_id, random_walks in walk_dict.items():
        temp = []
        for x in random_walks:
            idx = 0
            while idx < len(x) and x[idx] != '-1':
                idx += 1
            temp.append(idx)
        lengths.extend(temp)
    lengths_np = np.array(lengths)  # (cascade_cnt * walk_cnt)
    lengths_pd = pd.DataFrame(lengths_np)
    return lengths_pd.describe()


def get_cascade_statistics(cascade_path: str):
    cascade = read_cascade(cascade_path)
    cascade_cnt = len(cascade)
    cascade_lengths = [x['edge_cnt'] for x in cascade]
    cascade_lengths_pd = pd.DataFrame(cascade_lengths)
    return cascade_lengths_pd.describe()


def show_cascade(
        global_graph_path: str,
        cascade_path: str,
        cascade_id: int,
        trans_type: int,
        pseudo_count: float,
        cascade_index: int = None):
    """
    将cascade_path对应文件中id为cascade_id的cascade绘制出来。
    :param cascade_path:
    :param cascade_id:
    :return:
    """
    edge2weight, node2degree = read_global_graph(global_graph_path)
    cascades = read_cascade(cascade_path)
    if cascade_index != None:
        cascade = cascades[cascade_index]
    else:
        cascade = cascades[cascade_id]

    nx_G = cascade_edges_to_graph(cascade['edges'], cascade['edge_cnt'], node2degree, edge2weight, trans_type, pseudo_count)
    ax = plt.subplot()
    nx.draw(nx_G, with_labels=True)
    plt.show()


def readCommand(argv):
    """
    :param argv:
    :return:
    """
    usageStr = """

    """

    parser = argparse.ArgumentParser(usageStr)

    parser.add_argument('--task', '-t', dest='task', type=str, choices=tasks, default='random_walk')
    parser.add_argument('--cascade_file', '-cf', dest='cascade_file', type=str, default=None)
    parser.add_argument('--global_graph_file', '-gf', dest='global_graph_file', type=str, default=None)
    parser.add_argument('--random_walk_file', '-wf', dest='random_walk_file', type=str, default=None)

    parser.add_argument('--cascade_id', dest='cascade_id', type=int, default=1)
    parser.add_argument('--cascade_index', dest='cascade_index', type=int, default=0)
    parser.add_argument('--trans_type', dest='trans_type', type=int, default=2)
    parser.add_argument('--pseudo_count', dest='pseudo_count', type=float, default=1e-5)

    options = parser.parse_args(argv)
    opt_args = []
    for elem in options.__dir__():
        if elem[0] != '_':
            opt_args.append(elem)
    return_dict = {}
    for ags in opt_args:
        return_dict[ags] = getattr(options, ags)
    return return_dict


def runCommand(param_dict):
    if param_dict['task'] == 'random_walk':
        result = get_random_walk_statisitcs(param_dict['random_walk_file'])
        print(result)
    elif param_dict['task'] == 'cascade':
        result = get_cascade_statistics(param_dict['cascade_file'])
        print(result)
    elif param_dict['task'] == 'draw_cascade':
        show_cascade(
            param_dict['global_graph_file'],
            param_dict['cascade_file'],
            param_dict['cascade_id'],
            param_dict['trans_type'],
            param_dict['pseudo_count'],
            param_dict['cascade_index']
        )
    else:
        logger.error(f'task {param_dict["task"]} 不存在!')
        return


if __name__ == '__main__':
    args = readCommand(sys.argv[1:])
    runCommand(args)
