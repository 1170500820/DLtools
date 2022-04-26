import argparse
import sys
from cascade_utils import read_cascade, read_global_graph, generate_random_walk, output_random_walks_txt
from loguru import logger


# random walk parameters and range
default_walks_per_graph = 200
default_walk_length = 20
default_trans_type = 2
trans_type_choices = [0, 1, 2]
default_pseudo_count = 1e-5
default_p = 1
default_q = 1


def gen_random_walk(
        cascade_file: str,
        globalgraph_file: str,
        output_file: str,
        walks_per_graph: int = default_walks_per_graph,
        walk_length: int = default_walk_length,
        trans_type: int = default_trans_type,
        pseudo_count: float = default_pseudo_count,
        p: float = default_p,
        q: float = default_q):
    logger.info(f'loading cascade')
    cascade = read_cascade(cascade_file)

    logger.info('loading global graph')
    edge2weight, node2degree = read_global_graph(globalgraph_file)

    logger.info('random walking')
    random_walks = generate_random_walk(edge2weight, node2degree, cascade, walks_per_graph, trans_type, walk_length, pseudo_count, p, q)
    output_random_walks_txt(random_walks, output_file)

    logger.info(f'random walk file has been dumped to {output_file}')

def readCommand(argv):
    """
    :param argv:
    :return:
    """
    usageStr = """
    输入global_graph与Cascade文件的路径，根据该global_graph为该cascade生成random_walk
    """

    parser = argparse.ArgumentParser(usageStr)

    # input file
    parser.add_argument('--global_graph_file', '-g', dest='global_graph_file', type=str)
    parser.add_argument('--cascade_file', '-c', dest='cascade_file', type=str)

    # random walk parameters
    parser.add_argument('--walks_per_graph', '-wc', dest='walks_per_graph', type=int, default=default_walks_per_graph)
    parser.add_argument('--walk_length', '-wl', dest='walk_length', type=int, default=default_walk_length)
    parser.add_argument('--trans_type', '-t', dest='trans_type', type=int, choices=trans_type_choices, default=default_trans_type)
    parser.add_argument('--pseudo_count', '-pc', dest='pseudo_count', type=float, default=default_pseudo_count)
    parser.add_argument('-p', dest='p', type=float, default=default_p)
    parser.add_argument('-q', dest='q', type=float, default=default_q)

    # output
    parser.add_argument('--output_file', '-o', dest='output_file', type=str)


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
    gen_random_walk(
        param_dict['cascade_file'],
        param_dict['global_graph_file'],
        param_dict['output_file'],
        param_dict['walks_per_graph'],
        param_dict['walk_length'],
        param_dict['trans_type'],
        param_dict['pseudo_count'],
        param_dict['p'],
        param_dict['q'])


if __name__ == '__main__':
    args = readCommand(sys.argv[1:])
    runCommand(args)
