import os

from translate_all import find_files
import json
import random
from tqdm import tqdm

translated_dir = '../../../lab2/translated/'
dataset_dir = '../../data/NLP/translate/'
evaluate_dir = '../../data/NLP/translate_evaluate/'

sample_cnt = 50

unavailable_datasets = {
    ('opus100', 'de-zh'),
    ('opus100', 'nl-zh'),
    ('opus100', 'fr-zh'),
    ('opus100', 'ar-zh'),
    ('opus100', 'ru-zh'),
    ('ASPEC'),
    ('tanzil'),
    ('ParaCrawl'),
    ('un_multi'),
    ('news_commentary'),
    ('un_ga'),
    ('un_pc'),
    ('wmt20_mlqe_task1'),
    ('para_pat'),
    ('WikiMatrix')
}


def organize(data_dir, translated_dir, sample_cnt: int = sample_cnt):
    """
    从路径里搜索train，valid和test，然后找出对应的翻译文件
    随机采样
    :param data_dir:
    :return:
    """
    if data_dir[-1] != '/':
        data_dir += '/'
    if translated_dir[-1] != '/':
        translated_dir += '/'
    origin, translated = [], []
    for sub in ['train.json', 'test.json', 'valid.json']:
        if sub in os.listdir(data_dir):
            origin_file = list(json.loads(x)['zh'] for x in open(data_dir + sub))[:200]
            trans_file = list(json.loads(x)['zh'] for x in open(translated_dir + sub))[:200]
            origin.extend(origin_file)
            translated.extend(trans_file)

    idxes = list(range(len(origin)))
    sampled = random.sample(idxes, sample_cnt)
    chosen_origin = list(origin[x] for x in sampled)
    chosen_trans = list(translated[x] for x in sampled)

    return chosen_origin, chosen_trans


def main():
    dataset_files = find_files(dataset_dir)

    for dataset, subset in tqdm(dataset_files.items()):
        if dataset in unavailable_datasets:
            continue
        if isinstance(subset, list):  # no subset
            origin_path = dataset_dir + dataset + '/processed/'
            trans_path = translated_dir + dataset + '/'
            origin, trans = organize(origin_path, trans_path, sample_cnt)
            f = open(evaluate_dir + f'{dataset}-sample.json', 'w')
            for e_origin, e_trans in zip(origin, trans):
                f.write(json.dumps({
                    'origin': e_origin,
                    'trans': e_trans
                }, ensure_ascii=False) + '\n')
            f.close()
        else:  # contains subset
            for subset_name, subfile in subset.items():
                if (dataset, subset_name) in unavailable_datasets:
                    continue
                origin_path = dataset_dir + f'{dataset}/processed/{subset_name}'
                trans_path = translated_dir + f'{dataset}/{subset_name}/'
                origin, trans = organize(origin_path, trans_path, sample_cnt)
                f = open(evaluate_dir + f'{dataset}-{subset_name}-sample.json', 'w')
                for e_origin, e_trans in zip(origin, trans):
                    f.write(json.dumps({
                        'origin': e_origin,
                        'trans': e_trans
                    }, ensure_ascii=False) + '\n')
                f.close()

if __name__ == '__main__':
    main()
