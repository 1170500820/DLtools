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
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
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
    :param token_span:
    :param offset_mapping:
    :return:
    """
    start_part, end_part = offset_mapping[token_span[0]], offset_mapping[token_span[1]]
    start = start_part[0]
    if end_part[0] == end_part[1]:
        end = end_part[0]
    else:
        end = end_part[0]
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
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    s = '消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了'
    charSpan = (15, 17)
    result = tokenizer(s, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(result['input_ids'])
    tokenSpan = charSpan_to_tokenSpan(charSpan, result['offset_mapping'])
    word = tokenSpan_to_word(s, tokenSpan, result['offset_mapping'])
    new_charSpan = tokenSpan_to_charSpan(tokenSpan, result['offset_mapping'])
    print(f'tokenSpan: {tokenSpan}, word: {word}, charSpan: {charSpan}')
