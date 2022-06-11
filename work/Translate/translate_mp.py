#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import sys
import json
import time

import requests
from multiprocessing import Pool

"""
multi process version of translation tool
"""

url = 'http://9.138.36.195:21686'
uin = 111
scene = 16
available_lang = ['zh_CN', 'zh_TW', 'zh_HK', 'en', 'ja', 'ko', 'th', 'vi', 'ru', 'de', 'fr', 'ar', 'es']


def translate(content, fromLang='en', toLang='zh_CN', uin=uin, scene=scene):
    """
    translate one sentence
    :param content: sentence to be translated
    :param fromLang:
    :param toLang:
    :param uin:
    :param scene:
    :return:
    """
    data_json = {
        'fromLang': fromLang,
        'toLang': toLang,
        'q': content,
        'uin': uin,
        'scene': scene
    }
    headers = {}
    response = requests.post(url, json=data_json, headers=headers)
    try:
        result_sentence = response.json()['Result']
    except Exception as e:
        print(e)
        print('[{}] failed'.format(content))
    return response.json()['Result']
    # return 'test'


def translate_block(idx, contents, fromLang='en', toLang='zh_CN', uin=uin, scene=scene):
    """
    translate all sentences in a block
    :param idx: order of the block
    :param contents: list of sentences to be translated
    :param fromLang:
    :param toLang:
    :param uin:
    :param scene:
    :return:
    """
    results = []
    for elem in contents:
        results.append(translate(elem, fromLang, toLang, uin, scene))
    return idx, results


def translate_file(
        input_filename,
        output_filename,
        input_key='content',
        output_key='translated',
        fromLang = 'en',
        toLang = 'zh',
        block_size = 200,
        maximum_process_cnt = 30):
    # load all the sentences
    print('loading sentences from file')
    lines = list(json.loads(x)[input_key] for x in open(input_filename, 'r', encoding='utf-8').read().strip().split('\n'))
    print('loaded {} line of sentence'.format(len(lines)))
    # multi process
    pool_processes = []
    translate_pool = Pool(maximum_process_cnt)
    translate_results_dict, translated = {}, []
    blocked_line = 0
    idx = 0
    start_time = time.time()
    while blocked_line < len(lines):
        pool_processes.append(translate_pool.apply_async(
            translate_block, (
                idx,
                lines[blocked_line: blocked_line + block_size],
                fromLang,
                toLang,
                uin,
                scene)
        ))
        blocked_line += block_size

    translate_pool.close()
    translate_pool.join()
    end_time = time.time()
    print('finished translating, took {:.4f} of {} processes'.format(end_time - start_time, len(pool_processes)))

    for elem in pool_processes:
        trans_idx, translated_content = elem.get()
        translate_results_dict[trans_idx] = translated_content

    for idx in range(len(translate_results_dict)):
        for result in translate_results_dict[idx]:
            translated.append({
                output_key: result
            })
    print('finished organizing, {} blocks in total'.format(len(translate_results_dict)))
    f = open(output_filename, 'w', encoding='utf-8')
    for elem in translated:
        f.write(json.dumps(elem, ensure_ascii=False) + '\n')
    f.close()
    print('done')


def readCommand(argv):
    """
    :param argv:
    :return:
    """
    usageStr = """

    """

    parser = argparse.ArgumentParser(usageStr)

    parser.add_argument('--input', '-i', dest='input_filename', type=str)
    parser.add_argument('--output', '-o', dest='output_filename', type=str)

    parser.add_argument('--toLang', dest='toLang', type=str, default='zh_CN', choices=available_lang)
    parser.add_argument('--fromLang', dest='fromLang', type=str, default='en', choices=available_lang)

    parser.add_argument('--input_key', '-k', dest='input_key', type=str, default='content')
    parser.add_argument('--output_key', '-r', dest='output_key', type=str, default='translated')

    parser.add_argument('--block_size', '-b', dest='block_size', type=int, default=200)
    parser.add_argument('--process_cnt', '-p', dest='process_cnt', type=int, default=30)

    options = parser.parse_args(argv)
    opt_args = []
    for elem in options.__dir__():
        if elem[0] != '_':
            opt_args.append(elem)
    return_dict = {}
    for ags in opt_args:
        return_dict[ags] = getattr(options, ags)
    return return_dict


def runCommand(param_dict):
    translate_file(
        param_dict['input_filename'],
        param_dict['output_filename'],
        param_dict['input_key'],
        param_dict['output_key'],
        param_dict['fromLang'],
        param_dict['toLang'],
        param_dict['block_size'],
        param_dict['process_cnt'])


if __name__ == '__main__':
    args = readCommand(sys.argv[1:])
    runCommand(args)