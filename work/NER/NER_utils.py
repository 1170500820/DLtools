"""
实现一些NER的工具


"""
from type_def import *
from transformers import BertTokenizer
import work.NER.CCF_settings as CCF_settings
import torch

# todo 将BIOE的string，转化为spanlist

"""
NER类的标注格式有BIOS，也有BIO，每一种都需要写一套转化函数
= 合法性检查
    - check_BIO_string 检查一个BIO序列的合法性

= 转换
    - BIO_to_spandict  char上的BIO标记转化为spandict
    - spandict_to_BIO  spandict转换到char上的BIO标记
    - charBIO_to_tokenBIO  将char上的BIO标注转换为token上的BIO标注
    - tokenBIO_to_charBIO  将token上的BIO标注转换为char上的BIO标注
    - charBIO_to_tokenBIO_with_tail与tokenBIO_to_charBIO_with_tail 分别处理包含CLS与SEP的token标注。char序列都不会有CLS与SEP
= 涉及tensor的转换
"""


def check_BIO_string(BIO_string: List[str]):
    """
    检查一个BIO的strlist是否合法

    - 长度为0是合法的
    - I-type前面要么是B-type，要么是I-type，否则非法
    :param BIO_string:
    :return:
    """
    if len(BIO_string) == 0:
        return
    last = 'O'
    for idx, elem in enumerate(BIO_string):
        if elem[0] == 'I':
            if last == 'B' + elem[1:] or last == elem:
                continue
            else:
                raise Exception(f'[check_BIO_string]非法的I位置:{idx}！')
    return


def BIO_to_spandict(BIO_string: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """
    将BISO格式标注，转化为span的dict。
    不会检查BIO_string合法性。如果BIO_string存在非法标注，抛出异常

    其中BISO格式：
        如果词长为1，则为B-type
        词长>=2，则为 B-type, I-type, ...
        否则为O

    span格式：
        第一个字的坐标为0
        start为该词第一个字的坐标
        end为该词右侧的第一个字的坐标
    :param BIO_string:
    :return:
    """
    check_BIO_string(BIO_string)
    tag_types = list(set(x[2:] for x in set(BIO_string)))
    spandict = {x: [] for x in tag_types}  # Dict[type name, list of span tuple]

    # 根据每一个tag以及其上一个tag，判断该tag是否是标注内
    # 外->内 内作为start ,内->外 外作为end
    # 例外，若最后一个仍在标注内，end=len(BIO_string)
    last_tag = 'O'
    span_list = []
    for idx, tag in enumerate(BIO_string):
        if tag[0] == 'B':
            if last_tag[0] != 'O':
                span_list.append(idx)
                cur_tag_type = last_tag[2:]
                spandict[cur_tag_type].append(tuple(span_list))
                span_list = []
            span_list.append(idx)
            continue
        elif last_tag != 'O' and tag == 'O':
            span_list.append(idx)
            cur_tag_type = last_tag[2:]
            spandict[cur_tag_type].append(tuple(span_list))
            span_list = []
        last_tag = tag
    if len(span_list) == 1:
        span_list.append(len(BIO_string))
        cur_tag_type = last_tag[2:]
        spandict[cur_tag_type].append(tuple(span_list))
        span_list = []

    return spandict


def spandict_to_BIO(spandict: Dict[str, List[Tuple[int, int]]], BIO_string_length: int) -> List[str]:
    """
    根据spandict，生成一个BIO标注的StrList
    :param spandict: 合法：1，start<end 2，end<=len(BIO_string)
    :param BIO_string_length:
    :return:
    """
    BIO_string = ['O'] * BIO_string_length

    for key, value in spandict.items():
        for span in value:
            BIO_string[span[0]] = 'B-' + key
            for idx in range(span[0] + 1, span[1]):
                BIO_string[idx] = 'I-' + key

    return BIO_string


def charBIO_to_tokenBIO(BIO_string: List[str], mappings: List[Tuple[int, int]]) -> List[str]:
    """
    从char到token是一个多对一到映射，因此可能出现B和I，B和O，不同类型到B，I和O被映射到同一个token的情况（当然这些情况可能并不多）
    这也就是有损转换的情况

    处理有损转换的核心原则就是让模型效果更好。说废话。
    所以第二原则时保证产生的label合法。
    尝试定义下面的规则
        1，优先级，B>I>O
        2，如果有多个label，保留最高优先级结果
        3，最高优先级有多个重复的，则保留第一个。
    使用这些规则仍然不能保证生成的label合法，比如[B1, I1, B2] [I2],会产生{B1} {I2},I2独立出现了。
        4，所以对生成的序列再进行一遍检查，如果有Ik前面没有Ik或Bk，则将Ik替换为O

    考虑上面的情况
    1）不同类型的B被映射到同一个token，保留第一个。没办法，只能留一个
    2）B和I在同一个token，保留B。这样就不会生成非法的孤立I了
    3）I和O在同一个token，保留I。这样模型能获得更多信息
    4）不同的I在同一个token，保留I。没办法
    :param BIO_string:
    :param mappings:
    :return: 与token对应的BIO label序列（不包含CLS与SEP）
    """
    # 先移除碍事的CLS与SEP
    if mappings[0] == (0, 0):
        mappings = mappings[1:]
    if mappings[-1] == (0, 0):
        mappings = mappings[:-1]

    # 首先生成对应label
    label_lst = []
    for elem in mappings:
        char_labels = BIO_string[elem[0]: elem[1]]
        if len(char_labels) == 0:  # 如果为空，为了保持token与label一对一，插入O
            label_lst.append('O')
        elif len(char_labels) == 1:  # 只有一个，则直接对应
            label_lst.append(char_labels[0])
        else:
            tag_type_set = set(x[0] for x in char_labels)
            if 'B' in tag_type_set:  # 此时B为最高优先级
                for elem_tag in char_labels:
                    if elem_tag[0] == 'B':
                        label_lst.append(elem_tag)
                        break
            elif 'I' in tag_type_set:  # char_labels中的最高优先级tag为I
                for elem_tag in char_labels:
                    if elem_tag[0] == 'I':
                        label_lst.append(elem_tag)
                        break
            else:  # char_labels里面全都是O啊
                label_lst.append('O')

    # 接下来修补非法的label
    last_label = 'O'
    for idx, elem_label in enumerate(label_lst):
        if elem_label[0] == 'I' and last_label[2:] != elem_label[2:]:
            label_lst[idx] = 'O'
        last_label = label_lst[idx]

    return label_lst


def charBIO_to_tokenBIO_with_tail(BIO_string: List[str], mappings: List[Tuple[int, int]]) -> List[str]:
    """
    生成token下的BIO labels，但是会同时生成CLS与SEP对应的label
    :param BIO_string:
    :param mappings:
    :return:
    """
    BIO_string_token = charBIO_to_tokenBIO(BIO_string, mappings)
    BIO_string_token = ["O"] + BIO_string_token + ['O']
    return BIO_string_token


def tokenBIO_to_charBIO(BIO_string_token: List[str], mappings: List[Tuple[int, int]]):
    """
    从token到char到对应是一对多，所以（大概）不会出现损失到情况

    如果出现了不连续的mapping，若左边是O，则空缺处填O，否则空缺处填与左边对应的I
    :param BIO_string_token: 合法的BIO字符串，且不包含CLS与SEP的标注
    :param mappings:
    :return:
    """
    # 先移除碍事的CLS与SEP
    if mappings[0] == (0, 0):
        mappings = mappings[1:]
    if mappings[-1] == (0, 0):
        mappings = mappings[:-1]

    # 检查一下BIO_string_token能否与mappings一一对应
    if len(mappings) != len(BIO_string_token):
        raise Exception('[tokenBIO_to_charBIO]BIO_string_token与mappings无法构成一一对应！')

    BIO_string = []
    last_token, last_mapping = 'O', (0, 0)
    for (label, mapping) in zip(BIO_string_token, mappings):
        if last_mapping[-1] != mapping[0]:  # 如果mapping不能首尾相连，则先进行填充
            if last_token == 'O':
                fill_token = 'O'
            else:
                fill_token = 'I-' + last_token[2:]
            for idx in range(mapping[0] - last_mapping[-1]):
                BIO_string.append(fill_token)
        if label[0] == 'B':
            BIO_string.append(label)
            for idx in range(mapping[1] - mapping[0] - 1):
                BIO_string.append('I-' + label[2:])
        else:
            for idx in range(mapping[1] - mapping[0]):
                BIO_string.append(label)
        last_token = label
        last_mapping = mapping

    return BIO_string


def tokenBIO_to_charBIO_with_tail(BIO_string_token_tailed: List[str], mappings: List[Tuple[int, int]]):
    """
    将包含CLS与SEP的BIO_string_token转化为不包含CLS与SEP的char对应的BIO标注序列
    :param BIO_string_token_tailed:
    :param mappings:
    :return:
    """
    BIO_string_token = BIO_string_token_tailed[1:-1]
    BIO_string = tokenBIO_to_charBIO(BIO_string_token, mappings)
    return BIO_string


def tensor_to_ner_label(ner_tensor: torch.Tensor, ner_tag_lst=CCF_settings.ner_tags):
    """
    将ner_tensor转换为对应的BIO label list
    :param ner_tensor:
        prob tensor，应当已经使用softmax处理过
        (bsz?, seq_l, ner_cnt) 如果bsz维存在，则返回双层list。否则返回单层
    :param ner_tag_lst:
    :return:
    """
    bsz_not_exist = False
    if len(ner_tensor.shape) == 2:
        bsz_not_exist = True
        ner_tensor = ner_tensor.unsqueeze(dim=0)  # (1, seq_l, ner_cnt)

    v, indices = torch.max(ner_tensor, dim=-1)
    indices = indices.tolist()
    labels = []
    for elem_index in indices:
        new_label_lst = []
        for elem_idx in elem_index:
            new_label_lst.append(ner_tag_lst[elem_idx])
        labels.append(new_label_lst)
    if bsz_not_exist:
        labels = labels[0]
    return labels


if __name__ == '__main__':
    pass