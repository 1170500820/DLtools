"""
专注于将某种存储文件的格式转化为易用的python容器
"""
from type_def import *


# CoNll NER data
#  我不太清楚CoNll的具体格式，这里的代码是按照MultiCoNER评测的数据格式写的
def conllner_to_lst(filepath: str) -> List[Dict[str, Any]]:
    """
    输入文件路径，读取其中的数据，每个sample转化为一个dict
    包含的keys：
        - chars
        - tags
        - id

    conll每个sample的格式类似于这样

    # id a23618fa-10ec-4f5e-bed2-682879bfb054	domain=mix
    华 _ _ O
    盛 _ _ O
    顿 _ _ O
    maynard _ _ B-LOC
    天 _ _ O
    气 _ _ O

    华 O
    盛 O
    顿 O
    maynard B-LOC
    天 O
    气 O

    (两种格式都能够处理)

    不同sample之间用两个\n进行分割
    :param filepath: Conll训练文件的路径
    :return:
    """
    lines = open(filepath).read().strip().split('\n\n')
    dict_lst = []
    for elem_sample in lines:
        elem_sample = elem_sample.strip()  # 有时会遇到三个\n
        sample = {}
        sents = elem_sample.split('\n')
        if sents[0][0] == '#':
            info, taggings = sents[0], sents[1:]

            # id
            info_detail = info.split('\t')
            sample['id'] = info_detail[0].split()[-1]
        else:
            taggings = sents

        # tokens and tags
        sample['chars'], sample['tags'] = [], []
        for elem_tagging in taggings:
            elem_tagging_split = elem_tagging.split()
            cur_token, cur_tagging = elem_tagging_split[0], elem_tagging_split[-1]
            sample['chars'].append(cur_token)
            sample['tags'].append(cur_tagging)

        dict_lst.append(sample)

    return dict_lst


if __name__ == '__main__':
    d = conllner_to_lst('../data/NLP/SemEval/MultiCoNER/training_data/MIX_Code_mixed/mix_train.conll')