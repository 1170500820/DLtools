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





def tokenIndex_to_charIndex(token_index: int, offset_mapping: OffsetMapping, right=False) -> int:
    """
    :param token_index: token在包含CLS与SEP的tokens中的index
    :param offset_mapping:
    :param right: 是否返回token的最右侧char的index。默认是返回最左侧
    :return:
    """
    if right:
        return offset_mapping[token_index][1] - 1
    else:
        return offset_mapping[token_index][0]


def charIndex_to_tokenIndex(char_index: int, offset_mapping: OffsetMapping, left=False) -> int:
    """
    计算sentence中某个character的index在token sequence中的对应index
    如果char_index不在任何token span当中，一般来说该char位置是空格/制表符。应该是不影响结果的。
    :param char_index:
    :param offset_mapping:
    :param left: 若为True，则当char_index处于两个token span之间，不属于其中某一个时，则选择左侧token的index。默认情况是选择右侧的。
    :return:
    """
    last_end = 0
    for i, token_span in enumerate(offset_mapping):
        if token_span[0] == token_span[1]:  # 遇到CLS或SEP，直接跳过
            continue
        if char_index < token_span[0]:
            if left:
                return i - 1
            else:
                return i
        elif token_span[0] <= char_index <=  token_span[1]:
            return i
        else:  # char_index > token_span[1]，继续向后面找
            last_end = token_span[1]
    # raise Exception(f'[charIndex_to_tokenIndex]char_index:[{char_index}]不在offset_mapping:[{offset_mapping}]的范围内！')
    return len(offset_mapping) - 2


def tokenSpan_to_charSpan(token_span: Span, offset_mapping: OffsetMapping) -> Span:
    """
    将token序列中的一个span转换为char序列中的对应位置
    左边界为第一个token对应的第一个char，右边界为最后一个token对应的最后一个char
    :param token_span:
    :param offset_mapping:
    :return:
    """
    char_span_start = tokenIndex_to_charIndex(token_span[0], offset_mapping)
    char_span_end = tokenIndex_to_charIndex(token_span[1], offset_mapping, right=True)
    return (char_span_start, char_span_end)


def charSpan_to_tokenSpan(word_span: Span, offset_mapping: OffsetMapping) -> Span:
    """
    将char序列中的一个span转换为token序列的对应span
    :param word_span:
    :param offset_mapping:
    :return:
    """
    token_span_start = charIndex_to_tokenIndex(word_span[0], offset_mapping)
    token_span_end = charIndex_to_tokenIndex(word_span[1], offset_mapping, left=True)
    return (token_span_start, token_span_end)


def tokenSpan_to_word(sentence: str, token_span: Span, offset_mapping: OffsetMapping) -> str:
    """
    获取token中一个span所对应的字面量
    :param sentence:
    :param token_span:
    :param offset_mapping:
    :return:
    """
    char_span = tokenSpan_to_charSpan(token_span, offset_mapping)
    word = sentence[char_span[0]: char_span[1] + 1]
    if len(word) == 0:
        raise Exception(f'[tokenSpan_to_word]生成了空的word！token_span:[{token_span}], offset_mapping:[{offset_mapping}]')
    return word


def regex_tokenize(sentnece: str):
    """
    基于正则的tokenize方法，
    :param sentnece:
    :return:
    """