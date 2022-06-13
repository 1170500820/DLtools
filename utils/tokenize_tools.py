"""
与tokenize相关的一些转换函数
*该文件中的所有函数，如没有提及或提供设置项，tokenized中均包含首尾的CLS与SEP。*
"""
from type_def import *

from transformers import BertTokenizerFast, AutoTokenizer

OffsetMapping = List[Tuple[int, int]]
"""
OffsetMapping，由BertTokenizerFast得到的mapping，
- 包含CLS与SEP
- OM[i]代表tokens中的第i个token在原句中所对应的起始位置，即tokens[i] -> sentence[OM[i][0]: OM[i][1]]
- 除CLS与SEP之外，其他token所对应的起始位置不相同（即对应char序列中的子序列长度不为0）
- CLS与SEP的对应charSpan都为(0, 0)
- 不考虑结尾的SEP，OM[i][1] <= OM[i + 1][0]，不一定相等
"""


class bert_tokenizer:
    def __init__(self, max_len=256, plm_path='bert-base-chinese'):
        self.tokenizer = BertTokenizerFast.from_pretrained(plm_path)
        self.max_len = max_len

    def __call__(self, input_lst: List[Union[str,List[str]]]):
        """
        每个tokenize结果包括
        - token
        - input_ids
        - token_type_ids
        - attention_mask
        - offset_mapping
        :param input_lst:
        :return:
        """
        results = []
        for elem_input in input_lst:
            tokenized = self.tokenizer(
                elem_input,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_offsets_mapping=True)
            token_seq = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
            tokenized['token'] = token_seq
            results.append(dict(tokenized))
        return results


class xlmr_tokenizer:
    def __init__(self, max_len = 256, plm_path: str = 'xlm-roberta-base'):
        self.max_length = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)

    def __call__(self, input_lst: List[str]):
        """
        要注意这里的input_lst与bert_tokenizer中的不同。bert_tokenzer会考虑上下句的问题，而xlmr以及其他与xlmrtokenize方案相似的
        模型是没有NSP的，所以没有上下句。
        为了方便处理，xlmr_tokenizer简单的对每一个句子进行tokenize，把上下句的组合方案留给外部

        每个tokenize结果包括
        - token
        - input_ids
        - attention_mask
        - offset_mapping
        :param input_lst:
        :return:
        """
        results = []
        for elem_input in input_lst:
            # 一种简单粗暴的方法，对于较长文本，取其中间部分
            elem_tokens = elem_input.split(' ')
            if len(elem_tokens) > self.max_length:
                truncate_length = (len(elem_tokens) - self.max_length) // 2
                elem_tokens = elem_tokens[2 * truncate_length:]
                elem_input = ' '.join(elem_tokens)
            tokenized = self.tokenizer(
                elem_input,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True
            )
            token_seq = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
            tokenized['token'] = token_seq
            results.append(dict(tokenized))
        return results





# def tokenIndex_to_charIndex(token_index: int, offset_mapping: OffsetMapping, right=False) -> int:
#     """
#     :param token_index: token在包含CLS与SEP的tokens中的index
#     :param offset_mapping:
#     :param right: 是否返回token的最右侧char的index。默认是返回最左侧
#     :return:
#     """
#     if right:
#         return offset_mapping[token_index][1] - 1
#     else:
#         return offset_mapping[token_index][0]
#
#
# def charIndex_to_tokenIndex(char_index: int, offset_mapping: OffsetMapping, left=False) -> int:
#     """
#     todo 这段代码有bug，调用下面的charSpan_to_tokenSpan，结果是不对的。
#     计算sentence中某个character的index在token sequence中的对应index
#     如果char_index不在任何token span当中，一般来说该char位置是空格/制表符。应该是不影响结果的。
#     :param char_index:
#     :param offset_mapping:
#     :param left: 若为True，则当char_index处于两个token span之间，不属于其中某一个时，则选择左侧token的index。默认情况是选择右侧的。
#     :return:
#     """
#     last_end = 0
#     for i, token_span in enumerate(offset_mapping):
#         if token_span[0] == token_span[1]:  # 遇到CLS或SEP，直接跳过
#             continue
#         if char_index < token_span[0]:
#             if left:
#                 return i - 1
#             else:
#                 return i
#         elif token_span[0] <= char_index <= token_span[1]:
#             return i
#         else:  # char_index > token_span[1]，继续向后面找
#             last_end = token_span[1]
#     # raise Exception(f'[charIndex_to_tokenIndex]char_index:[{char_index}]不在offset_mapping:[{offset_mapping}]的范围内！')
#     return len(offset_mapping) - 2
#
#
# def tokenSpan_to_charSpan(token_span: Span, offset_mapping: OffsetMapping) -> Span:
#     """
#     将token序列中的一个span转换为char序列中的对应位置
#     左边界为第一个token对应的第一个char，右边界为最后一个token对应的最后一个char
#     :param token_span:
#     :param offset_mapping:
#     :return:
#     """
#     char_span_start = tokenIndex_to_charIndex(token_span[0], offset_mapping)
#     char_span_end = tokenIndex_to_charIndex(token_span[1], offset_mapping, right=True)
#     return (char_span_start, char_span_end)
#
#
# def charSpan_to_tokenSpan(word_span: Span, offset_mapping: OffsetMapping) -> Span:
#     """
#     将char序列中的一个span转换为token序列的对应span
#     :param word_span:
#     :param offset_mapping:
#     :return:
#     """
#     token_span_start = charIndex_to_tokenIndex(word_span[0], offset_mapping)
#     token_span_end = charIndex_to_tokenIndex(word_span[1], offset_mapping, left=True)
#     return (token_span_start, token_span_end)
#
#
# def tokenSpan_to_word(sentence: str, token_span: Span, offset_mapping: OffsetMapping) -> str:
#     """
#     获取token中一个span所对应的字面量
#     :param sentence:
#     :param token_span:
#     :param offset_mapping:
#     :return:
#     """
#     char_span = tokenSpan_to_charSpan(token_span, offset_mapping)
#     word = sentence[char_span[0]: char_span[1] + 1]
#     if len(word) == 0:
#         raise Exception(f'[tokenSpan_to_word]生成了空的word！token_span:[{token_span}], offset_mapping:[{offset_mapping}]')
#     return word


def regex_tokenize(sentnece: str):
    """
    基于正则的tokenize方法，
    :param sentnece:
    :return:
    """


def offset_mapping_to_matches(offset_mapping: List[Tuple[int, int]]) -> Tuple[dict, dict]:
    """
    将offset_mapping转换为token与char的匹配dict，方便直接转换。

    token序列对char序列是一对多。char序列中的一个character不能再拆分，一定属于某个token的一部分。而token序列中的一个token可能对应多个character
    :param offset_mapping:
    :return:
    """
    token2origin, origin2token = {}, {}


def tokenSpan_to_charSpan(token_span: Span, offset_mapping: OffsetMapping) -> Span:
    """
    将token序列中的一个span转换为char序列中的一个对应的span
    :param token_span: tokens[token_span[0]: token_span[1]]代表该token_span所表示的词。
        也就是说token_span[1]并不属于其中，而是一个外边界。
        token_span[0] <= token_span[1]
    :param offset_mapping: 与该token_span所对应的offset mapping
    :return:
    """
    # 初始时候，start与end都是0，在不断遍历token_span所包含的offset mapping项中更新
    start, end = 0, 0
    if token_span[0] < token_span[1]:
        start = offset_mapping[token_span[0]][0]
    for elem_mapping in offset_mapping[token_span[0]: token_span[1]]:
        map_start, map_end = elem_mapping
        if map_start == map_end:  # 此时该mapping将一个token映射到空区间，也就是说该token在原句中没有对应任何char
            continue
        # 代码运行到下面，说明map_start < map_end，sentence[map_start: map_end]为所对应的区间
        start = min(start, map_start)
        end = max(end, map_end)
    # start_part, end_part = offset_mapping[token_span[0]], offset_mapping[token_span[1]]
    # start = start_part[0]
    # if end_part[0] == end_part[1]:
    #     end = end_part[0]
    # else:
    #     end = end_part[0]
    return (start, end)


def charSpan_to_tokenSpan(char_span: Span, offset_mapping: OffsetMapping) -> Span:
    """
    将char序列上的一个span转换为token序列中的对应span

    首先按照offset_mapping构造出origin2token，可能有一些char中的idx不存在token中的对应
    接下来为每个char_span寻找：
        如果有对应值，则直接成立
        如果没有，先尝试靠近span的另一侧。比如start找不到，那么就将start一直加一，直到加到等于end
        然后再反方向找
    :param char_span:
    :param offset_mapping:
    :return:
    """
    origin2token = {}
    for idx, elem in enumerate(offset_mapping):
        if elem[0] == elem[1]:
            continue
        for num in range(elem[0], elem[1]):
            origin2token[num] = idx
    # 按上述步骤构造的origin2token，
    char_start, char_end = char_span
    if char_start in origin2token:
        start = origin2token[char_start]
    else:
        temp = char_start
        while temp < char_end:
            temp += 1
            if temp in origin2token:
                start = origin2token[temp]
                break
        temp = char_start
        while temp > 0:
            temp -= 1
            if temp in origin2token:
                start = origin2token[temp]
                break
        start = 0
    if char_end in origin2token:
        end = origin2token[char_end]
    else:
        temp = char_end
        while temp > char_start:
            temp -= 1
            if temp in origin2token:
                end = origin2token[temp]
                break
        temp = char_end
        while temp < len(offset_mapping) - 1:
            temp += 1
            if temp in origin2token:
                end = origin2token[temp]
                break
        end = len(offset_mapping) - 1
    return (start, end)


def tokenSpan_to_word(sentence: str, token_span: Span, offset_mapping: OffsetMapping) -> str:
    span = tokenSpan_to_charSpan(token_span, offset_mapping)
    return sentence[span[0]: span[1]]


if __name__ == '__main__':
    d = {'content': '近日， “移动电影院V2.0”产品于北京正式 发布，恰逢移动电影院App首发一周年，引起了业内的高度关注。',
 'event_type': '产品行为-发布',
 'trigger_span': [22, 25],
 'trigger_word': ' 发布',
 'other_mentions': [{'word': '近日', 'span': [0, 2], 'role': '时间'},
  {'word': ' “移动电影院V2.0”产品', 'span': [3, 17], 'role': '发布产品'}],
 'input_ids': [101,
  6818,
  3189,
  8024,
  100,
  4919,
  1220,
  4510,
  2512,
  7368,
  100,
  119,
  121,
  100,
  772,
  1501,
  754,
  1266,
  776,
  3633,
  2466,
  1355,
  2357,
  8024,
  2623,
  6864,
  4919,
  1220,
  4510,
  2512,
  7368,
  100,
  7674,
  1355,
  671,
  1453,
  2399,
  8024,
  2471,
  6629,
  749,
  689,
  1079,
  4638,
  7770,
  2428,
  1068,
  3800,
  511,
  102],
 'token_type_ids': [0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0],
 'offset_mapping': [(0, 0),
  (0, 1),
  (1, 2),
  (2, 3),
  (4, 5),
  (5, 6),
  (6, 7),
  (7, 8),
  (8, 9),
  (9, 10),
  (10, 12),
  (12, 13),
  (13, 14),
  (14, 15),
  (15, 16),
  (16, 17),
  (17, 18),
  (18, 19),
  (19, 20),
  (20, 21),
  (21, 22),
  (23, 24),
  (24, 25),
  (25, 26),
  (26, 27),
  (27, 28),
  (28, 29),
  (29, 30),
  (30, 31),
  (31, 32),
  (32, 33),
  (33, 36),
  (36, 37),
  (37, 38),
  (38, 39),
  (39, 40),
  (40, 41),
  (41, 42),
  (42, 43),
  (43, 44),
  (44, 45),
  (45, 46),
  (46, 47),
  (47, 48),
  (48, 49),
  (49, 50),
  (50, 51),
  (51, 52),
  (52, 53),
  (0, 0)],
 'token': ['[CLS]',
  '近',
  '日',
  '，',
  '[UNK]',
  '移',
  '动',
  '电',
  '影',
  '院',
  '[UNK]',
  '.',
  '0',
  '[UNK]',
  '产',
  '品',
  '于',
  '北',
  '京',
  '正',
  '式',
  '发',
  '布',
  '，',
  '恰',
  '逢',
  '移',
  '动',
  '电',
  '影',
  '院',
  '[UNK]',
  '首',
  '发',
  '一',
  '周',
  '年',
  '，',
  '引',
  '起',
  '了',
  '业',
  '内',
  '的',
  '高',
  '度',
  '关',
  '注',
  '。',
  '[SEP]'],
 'trigger_token_span': (0, 22),
 'argument_token_spans': [(48, (1, 2)), (27, (0, 15))]}
    d2 = {'content': '消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了',
 'id': 'cba11b5059495e635b4f95e7484b2684',
 'events': [{'type': '组织关系-裁员',
   'mentions': [{'word': '裁员', 'span': [15, 17], 'role': 'trigger'},
    {'word': '900余人', 'span': [17, 22], 'role': '裁员人数'},
    {'word': '5月份', 'span': [10, 13], 'role': '时间'}]}],
 'input_ids': [101,
  3867,
  1927,
  4638,
  100,
  1912,
  821,
  1045,
  4384,
  100,
  8024,
  126,
  3299,
  819,
  1762,
  1290,
  6161,
  1447,
  8567,
  865,
  782,
  8024,
  7676,
  7661,
  7661,
  1359,
  100,
  5634,
  100,
  749,
  102],
 'token_type_ids': [0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0],
 'attention_mask': [1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1],
 'offset_mapping': [(0, 0),
  (0, 1),
  (1, 2),
  (2, 3),
  (3, 4),
  (4, 5),
  (5, 6),
  (6, 7),
  (7, 8),
  (8, 9),
  (9, 10),
  (10, 11),
  (11, 12),
  (12, 13),
  (13, 14),
  (14, 15),
  (15, 16),
  (16, 17),
  (17, 20),
  (20, 21),
  (21, 22),
  (22, 23),
  (23, 24),
  (24, 25),
  (25, 26),
  (26, 27),
  (27, 28),
  (28, 29),
  (29, 30),
  (30, 31),
  (0, 0)],
 'token': ['[CLS]',
  '消',
  '失',
  '的',
  '[UNK]',
  '外',
  '企',
  '光',
  '环',
  '[UNK]',
  '，',
  '5',
  '月',
  '份',
  '在',
  '华',
  '裁',
  '员',
  '900',
  '余',
  '人',
  '，',
  '香',
  '饽',
  '饽',
  '变',
  '[UNK]',
  '臭',
  '[UNK]',
  '了',
  '[SEP]']}
    offset_mapping = d2['offset_mapping']
    trigger_span = (17, 22)
    trigger_span = (trigger_span[0], trigger_span[1] - 1)
    trigger_token_span = charSpan_to_tokenSpan(trigger_span, offset_mapping)