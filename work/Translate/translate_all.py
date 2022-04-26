import os
from translate_mp import *

dataset_dir = 'data/'
translated_dir = 'translated/'
block_size = 10
max_process = 80


unavailable_datasets = {
    ('opus100', 'de-zh'),
    ('opus100', 'nl-zh'),
    ('opus100', 'fr-zh'),
    ('opus100', 'ar-zh'),
    ('opus100', 'ru-zh'),
    ('ASPEC'),
    ('tanzil')
}


def find_files(dataset_dir=dataset_dir):
    """
    find files that need translation
    :return:
    """
    dataset_info = {}
    listoffile = os.listdir(dataset_dir)
    for elem in listoffile:
        if os.path.isdir(dataset_dir + elem) and 'processed' in os.listdir(dataset_dir + elem):
            subfiles = os.listdir(dataset_dir + elem + '/processed')
            if 'train.json' in subfiles or 'valid.json' in subfiles or 'test.json' in subfiles:
                dataset_info[elem] = []
                for elem_subset in ['train.json', 'valid.json', 'test.json']:
                    if elem_subset in subfiles:
                        dataset_info[elem].append(elem_subset)
            else:
                for elem_sub in subfiles:
                    if not os.path.isdir(dataset_dir + elem + '/processed/' + elem_sub):
                        continue
                    subsubfiles = os.listdir(dataset_dir + elem + '/processed/' + elem_sub)
                    if 'train.json' in subsubfiles or 'valid.json' in subsubfiles or 'test.json' in subsubfiles:
                        if elem not in dataset_info:
                            dataset_info[elem] = {}
                        dataset_info[elem][elem_sub] = []
                        for elem_subset in ['train.json', 'valid.json', 'test.json']:
                            if elem_subset in subsubfiles:
                                dataset_info[elem][elem_sub].append(elem_subset)
    return dataset_info

def trans_files(dataset_info):
    for key, value in dataset_info.items():
        if isinstance(value, list):
            if (key) in unavailable_datasets:
                continue
            print('translating {}'.format(key))
            for elem in value:
                filepath = dataset_dir + key + '/processed/' + elem
                outputpath = translated_dir + key + '/' + elem
                print(filepath, outputpath)
                translate_file(filepath, outputpath, 'en', 'zh', 'en', 'zh_CN', block_size, max_process, 200)
        else:
            for subkey, subvalue in dataset_info[key].items():
                if (key, subkey) in unavailable_datasets:
                    continue
                for elem in subvalue:
                    print('translating {} in {}'.format(subkey, key))
                    filepath = dataset_dir + key + '/processed/' + subkey + '/' + elem
                    outputpath = translated_dir + key + '/' + subkey + '/' + elem
                    translate_file(filepath, outputpath, 'en', 'zh', 'en', 'zh_CN', block_size, max_process, 200)


if __name__ == '__main__':
    result = find_files()
    trans_files(result)