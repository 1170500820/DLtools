

def parse_graph(graph_string):
    """
    用于生成随机游走的部分
    :param graph_string:
    :return:
    """
    parts = graph_string.split("\t")
    # parts[0]: 不确定，但是应该是一个id？
    # parts[1]: 是一些用户
    # parts[2]: 似乎是年份？
    # parts[3]: 不像是用户id，可能是时间？

    edge_strs = parts[4].split(" ")
    # src id: tgt id:1

    node_to_edges = dict()
    for edge_str in edge_strs:
        edge_parts = edge_str.split(":")
        source = int(edge_parts[0])
        target = int(edge_parts[1])

        if not source in node_to_edges:
            neighbors = list()
            node_to_edges[source] = neighbors
        else:
            neighbors = node_to_edges[source]
        neighbors.append((target, get_global_degree(target)))
    # node_to_edges: src -> (tgt, tgt degree)
    # 这个dict有啥意思？


    nx_G = nx.DiGraph()
    for source, nbr_weights in node_to_edges.items():
        # 这是老版本的dict.items()吗
        for nbr_weight in nbr_weights:
            target = nbr_weight[0]

            if opts.trans_type == 0:  # trans_type不是string吗，怎么又012了
                edge_weight = get_edge_weight(source, target) + pseudo_count
                weight = edge_weight
            elif opts.trans_type == 1:
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
    str_list.append(parts[0])

    probs = list()
    probs_noleaf = list()
    weight_sum_noleaf = 0.0
    weight_sum = 0.0

    # Obtain sampling probabilities of roots.
    for node, weight in nx_G.out_degree(weight="weight"):
        org_weight = weight
        if weight == 0: weight += pseudo_count
        weight_sum += weight
        if org_weight > 0:
            weight_sum_noleaf += weight

    for node, weight in nx_G.out_degree(weight="weight"):
        org_weight = weight
        if weight == 0: weight += pseudo_count
        roots.append(node)
        prob = weight / weight_sum
        probs.append(prob)
        if org_weight > 0:
            roots_noleaf.append(node)
            prob = weight / weight_sum_noleaf
            probs_noleaf.append(prob)

    sample_total = opts.walks_per_graph
    first_time = True
    G = node2vec.Graph(nx_G, True, opts.p, opts.q)
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
        if n_sample <= 0: break
        sample_total -= n_sample

        sampled_nodes = np.random.choice(node_list, n_sample, replace=False, p=prob_list)
        walks = G.simulate_walks(len(sampled_nodes), opts.walk_length, sampled_nodes)
        for walk in walks:
            str_list.append(' '.join(str(k) for k in walk))
    return '\t'.join(str_list)

