from type_def import *
from .index_edit import *

def cascade2edges(cascade_string: str) -> List[Tuple[int, int]]:
    """
    输入一个cascade的字符串，输出其所包含的所有边
    :param cascade_string:
    :return:
    """
    cascade_info = cascade_string.split('\t')
    edge_string = cascade_info[4].split()
    edges: List[Tuple[int, int]] = []
    for elem_edge in edge_string:
        nodes = elem_edge.split(':')
        edges.append((int(nodes[0]), int(nodes[1])))

    return edges


def get_cascade_idx(cascade_string: str) -> int:
    """
    输入一个cascade的字符串，输出其序号
    :param cascade_string:
    :return:
    """
    return int(cascade_string.split('\t')[0])


