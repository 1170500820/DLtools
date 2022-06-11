"""
SemEval 新闻相似度任务
通用工具
"""
import sys
sys.path.append('../..')
sys.path.append('..')
sys.path.append('../../')
import random
from type_def import *
from utils import tools, dir_tools
import json
from tqdm import tqdm
import os
import csv
from work.NewsCorrelation import newsco_settings
import numpy as np
from utils.stanza_tools import stanza_ner_multilingual


"""
新闻相似度比赛 -- 数据的读取与写入相关

这部分的函数主要解决训练数据的读取、整理、写入。
"""


def _parse_crawl_result(crawled_json: dict):
    """
    对原代码中Iterator.parse_one_file_dict的重写
    - 去除了进行tokenize的部分，因为tokenize是比较独立的一个步骤
    - 没有进行分句
    - 没有把keywords转化为句子并tokenize
    - 没有对author进行tokenize
    :param crawled_json: 具体细节见notion
        至少包含：
            核心
            - article_id
            - title
            - text
            其他
            - source_url
            - url
            - keywords
            - authors
            - summary
            - meta_description
            - meta_lang
            - top_image
    :return:
    """
    file_info_dict = crawled_json

    info = {}
    for keys in ['source_url', 'url', 'title', 'text', 'keywords', 'authors', 'summary', 'meta_description', 'meta_lang', 'top_image', 'article_id']:
        info[keys] = file_info_dict[keys] if keys in file_info_dict else None
    return info


def _read_crawl_result(crawled_path: str):
    """
    从文件读取爬取结果，返回读取所得
    :param crawled_path:
    :return:
    """
    load_result = json.load(open(crawled_path, 'r'))
    json_id = os.path.split(crawled_path)[-1].split('.')[0]
    load_result['article_id'] = json_id
    return load_result


def _load_single_crawl_result(crawled_path: str):
    """
    给爬取文件的路径，返回解析后的结果
    :param crawled_path:
    :return:
    """
    loaded = _read_crawl_result(crawled_path)
    parsed = _parse_crawl_result(loaded)
    return parsed


def load_crawl_results(crawled_dir: str):
    """
    给定一个目录，对目录下所有的爬取文件进行读取
    :param crawled_dir:
    :return:
    """
    filenames = dir_tools.dir_deepsearch_with_pattern(crawled_dir, '\d+.json')
    crawled = []
    for p in tqdm(filenames):
        crawled.append(_load_single_crawl_result(p))
    return crawled


def build_id2crawled(crawled_lst: List[Dict[str, Any]]):
    """
    传入所有爬取到的crawled的dict组成的list
    :param crawled_lst: list of crawled dict
        dict必须包含:
            - article_id
    :return:
    """
    id2crawled = {}
    for elem in crawled_lst:
        id2crawled[elem['article_id']] = elem
    return id2crawled


def load_news_pair_csv(newspair_file: str):
    """
    读取新闻配对csv文件，读取其中的id, language, url, scores
    返回一个List of dict
        - index （数据的位置）
        - lang1
        - lang2
        - id1
        - id2
        - link1
        - link2
        下面这些并不一定要求包含
        - Geography
        - Entities
        - Time
        - Narrative
        - Overall
        - Style
        - Tone
    :param newspair_file:
    :return:
    """
    results = []
    with open(newspair_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            pair_id = row['pair_id'].split('_')
            filtered_row = {
                "index": i,
                "lang1": row['url1_lang'],
                "lang2": row['url2_lang'],
                'id1': pair_id[0],
                'id2': pair_id[1],
                'link1': row['link1'],
                'link2': row['link2']
            }
            for key in ['Geography', 'Entities', 'Time', 'Narrative', 'Overall', 'Style', 'Tone']:
                if key in row:
                    filtered_row[key] = row[key]
            results.append(filtered_row)
    return results


def dicts2csv(dicts: List[dict], filename: str):
    """
    将一组dict写入到csv文件当中
    :param dicts: list of dict，key值均相同
    :param filename: 要写入的文件的名字，不含后缀名
    :return:
    """
    if len(dicts) == 0:
        return
    fieldnames = list(dicts[0].keys())
    with open(filename + '.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(dicts)


def build_data_samples(id2crawled: Dict[str, Any], newspair: List[Dict[str, Any]]):
    """
    组装出训练sample
    简单的在newspair表中用crawl1和crawl2保存对应的crawled结果
    :param id2crawled:
    :param newspair:
    :return:
    """
    built_lst = []
    for lineidx, elem in enumerate(newspair):
        lang_pair = elem['lang1'] + '-' + elem['lang2']
        id1, id2 = elem['id1'], elem['id2']
        if id1 not in id2crawled or id2 not in id2crawled:
            continue
        crawl1, crawl2 = id2crawled[id1], id2crawled[id2]
        built_dict = {
            'index': elem['index'],
            'language_pair': lang_pair,
            "id1": id1,
            "id2": id2,
            'crawl1': crawl1,
            "crawl2": crawl2,
            'Geography': elem['Geography'] if 'Geography' in elem else None,
            'Entities': elem['Entities'] if 'Entities' in elem else None,
            'Time': elem['Time'] if 'Time' in elem else None,
            'Narrative': elem['Narrative'] if 'Narrative' in elem else None,
            'Overall': elem['Overall'] if 'Overall' in elem else None,
            'Style': elem['Style'] if 'Style' in elem else None,
            'Tone': elem['Tone'] if 'Tone' in elem else None
        }
        built_lst.append(built_dict)
    return built_lst


def build_news_samples(crawl_dir: str, pair_file: str):
    """
    根据两个路径，读取出samples包含
        -language_pair
        -id1
        -id2
        -crawl1
        -crawl2
        下面的不一定需要包含
        -Geography
        -Entities
        -Time
        -Narrative
        -Overall
        -Style
        -Tone
    :param crawl_dir:
    :param pair_file:
    :return:
    """
    result = load_crawl_results(crawl_dir)
    id2crawled = build_id2crawled(result)
    pairs = load_news_pair_csv(pair_file)
    samples = build_data_samples(id2crawled, pairs)
    return samples


"""
新闻相似度比赛 -- 数据简单预处理

下面的函数主要是对数据进行一些简单的预处理

"""


def filter_by_language_pair(sample_lst: List[Dict], language_pair_lst: List[str] = newsco_settings.legal_lang_pairs):
    """
    按照语言对过滤。
    :param sample_lst:
    :param language_pair_lst:
    :return:
    """
    new_sample_lst = []
    for elem in sample_lst:
        if elem['language_pair'] in language_pair_lst:
            new_sample_lst.append(elem)
    return new_sample_lst


def xlmr_sentence_concat(pieces: List[IntList]):
    """
    xlmr的开始与结束符号是<s> - 0, </s> - 2,
    (<pad> - 1)
    xlmr的tokenize规则是:
    <s> A </s>
    <s> A </s> <s> B </s>
    这里是扩充这个规则为:
    <s> A_1 </s> <s> A_2 <s> </s> ... A_n </s>
    :param pieces:
    :return:
    """
    modified_pieces = []
    # 首先给每个piece的开头和结尾添加占位符
    for elem_piece in pieces:
        if elem_piece[0] != 0:
            elem_piece.insert(0, 0)
        if elem_piece[-1] != 2:
            elem_piece.append(2)
        modified_pieces.extend(elem_piece)
    return modified_pieces


def xlmr_sentence_concat_cza(pieces: List[IntList]):
    """
    使用陈仲安的编码方案
    <s> title1 <s> text1 </s> </s> title2 <s> text2 </s>
    :param pieces: len = 4, [title1, text1, title2, text2]
    :return:
    """
    # 先删除首位的<s>和</s>
    clean_pieces = []
    for elem_piece in pieces:
        if elem_piece[0] == 0:
            elem_piece = elem_piece[1:]
        if elem_piece[-1] == 2:
            elem_piece = elem_piece[:-1]
        clean_pieces.append(elem_piece)

    title1, text1, title2, text2 = clean_pieces
    concatenated = [0] + title1 + [0] + text1 + [2, 2] + title2 + [0] + text2 + [2]
    return concatenated


def xlmr_sentence_concat_ndarray(pieces: List[np.ndarray]):
    """
    处理ndarray的情形
    :param pieces:
    :return:
    """
    intlist_pieces = list(x.astype(np.int).tolist() for x in pieces)
    result = xlmr_sentence_concat(intlist_pieces)
    return np.array(result, dtype=np.int)


def generate_attention_mask(input_ids: IntList, length: int = newsco_settings.max_seq_length):
    """
    为input_ids(list of int)生成一个attention_mask
    就是全1后面补0
    :param input_ids:
    :param length:
    :return:
    """
    mask = [1] * len(input_ids) + [0] * (length - len(input_ids))
    return mask


def pad_input_ids(input_ids: IntList, length: int = newsco_settings.max_seq_length):
    """
    将input_ids补齐到某一长度，空余直接补0
    :param input_ids:
    :param length:
    :return:
    """
    input_ids = input_ids + [0] * (length - len(input_ids))
    return input_ids


def kfold_generator(
        k: int = 10,
        news_pair_file_path: str = 'data/NLP/news_sim/data/semeval-2022_task8_train-data_batch.csv',
        output_dir: str = 'data/NLP/news_sim/kfold/',
        generate_eval: bool = True):
    """
    输入一个
    :param k: 执行k折交叉验证
    :param news_pair_file_path:
    :param output_dir: 输出的路径
    :param generate_eval: 是否先划分一个评价集
    :return:
    """
    if output_dir[-1] != '/':
        output_dir = output_dir + '/'
    # 先读取，并进行随机化
    with open(news_pair_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        pairs = list(reader)
    random.shuffle(pairs)
    pairs_size = len(pairs)
    train_ratio = 0.9
    train_size = int(pairs_size * train_ratio)

    train_pairs, val_pairs = pairs[: train_size], pairs[train_size:]

    # 开始利用分出来的训练集构建folds
    folds = []
    fold_size = train_size // k  # 每一小段的长度
    fold_dict = {}
    for i in range(k):
        if i < k - 1:
            fold_dict[i] = train_pairs[i * fold_size: (i + 1) * fold_size]
        else:
            fold_dict[i] = train_pairs[i * fold_size: ]

    for i in range(k):
        fold_lst = list(range(k))
        fold_lst.remove(i)
        kfold_train = []
        for fold_index in fold_lst:
            kfold_train.extend(fold_dict[fold_index])
        kfold_val = fold_dict[i]

        dicts2csv(kfold_train, output_dir + f'kfold-{i}-train')
        dicts2csv(kfold_val, output_dir + f'kfold-{i}-val')
    dicts2csv(val_pairs, output_dir + f'final_valid')




def get_ner(filepath: str = 'data/NLP/news_sim/final_nlp'):
    try:
        data = json.load(open('ner_result.json', 'r'))
    except:
        json.dump({}, open('ner_result.json', 'w'))
        data = json.load(open('ner_result.json', 'r'))

    result = load_crawl_results(filepath)
    stanzaner = stanza_ner_multilingual()
    for idx, elem in tqdm(enumerate(result)):
        # 已经生成过的就没必要再生成一次了
        if elem['article_id'] in data:
            continue
        try:
            elem_ner = stanzaner(elem['text'])
        except Exception as e:
            print(e)
            json.dump(data, open('ner_result.json', 'w'), ensure_ascii=False)
            return
        data[elem['article_id']] = elem_ner
    json.dump(data, open('ner_result.json', 'w'), ensure_ascii=False)


if __name__ == '__main__':
    # kfold_generator(10,
    #                 news_pair_file_path='../../data/NLP/news_sim/data/semeval-2022_task8_train-data_batch.csv',
    #     output_dir='../../data/NLP/news_sim/kfold/')
    result = load_crawl_results('../../data/NLP/news_sim/final_nlp')
    # id2crawled = build_id2crawled(result)
    # stanzaner = stanza_ner_multilingual()
    # ner_results = []
    # for elem in tqdm(result):
    #     ner_results.append(stanzaner(elem['text']))
    # json.dump(ner_results, open('new_results.json', 'w'), ensure_ascii=False)
    # pairs = load_news_pair_csv('../../data/NLP/news_sim/data/semeval-2022_task8_train-data_batch.csv')
    # samples = build_data_samples(id2crawled, pairs)
