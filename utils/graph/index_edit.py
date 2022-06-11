
class IndexDict:
    def __init__(self, original_ids):
        self.vocab = original_ids
        self.vocab_set = set(self.vocab)
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        #为每个node重新编号,设为node_id
        # original_to_new格式：{node:node_id}
        # new_to_original 格式：[node1,node2…]
        for i in original_ids:
            new = self.original_to_new.get(i,cnt)
            if new == cnt:
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)

    #返回一个节点，或一个节点列表，或一个节点列表的列表的 node_id
    def new(self, original):
        if type(original) is int:
            return self.original_to_new[original]
        else:
            if type(original[0]) is int:
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]

    # 返回一个node_id，或一个node_id列表，或一个node_id列表的列表的node
    def original(self, new):
        if type(new) is int:
            return self.new_to_original[new]
        else:
            if type(new[0]) is int:
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]
    #节点个数
    def length(self):
        return len(self.new_to_original)
