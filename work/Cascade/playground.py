import sys
import os
import pandas as pd
import numpy as np
from loguru import logger
import networkx as nx
import matplotlib.pyplot as plt

from cascade_utils import load_random_walks_txt, read_cascade, read_global_graph, cascade_edges_to_graph
from get_statistics import show_cascade


cascade_path = '../../data/cascade/APS/7/cascade_val.txt'
global_graph_path = '../../data/cascade/APS/global_graph.txt'
trans_type = 2
pseudo_count = 1e-5
cascade_idx = 0

cascades = read_cascade(cascade_path)
edge2weight, node2degree = read_global_graph(global_graph_path)

nx_G = cascade_edges_to_graph(cascades[cascade_idx]['edges'], cascades[cascade_idx]['edge_cnt'], node2degree, edge2weight, trans_type, pseudo_count)
