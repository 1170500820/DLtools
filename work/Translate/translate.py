#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import sys
import json
import requests


url = 'http://9.138.36.195:21686'
uin = 111
scene = 16
available_lang = ['zh_CN', 'zh_TW', 'zh_HK', 'en', 'ja', 'ko', 'th', 'vi', 'ru', 'de', 'fr', 'ar', 'es']


def translate(content, fromLang = 'en', toLang = 'zh'):
    data_json = {
        'fromLang': fromLang,
        'toLang': toLang,
        'q': content,
        'uin': uin,
        'scene': scene
    }
    headers = {}
    response = requests.post(url, json=data_json, headers=headers)
    return response.json()['Result']


def translate_file(input_filename, output_filename, input_key='content', output_key='translated', fromLang = 'en', toLang = 'zh'):
    lines = list(json.loads(x) for x in open(input_filename, 'r', encoding='utf-8').read().strip().split('\n'))
    results = []
    for elem in lines:
        trans = translate(elem[input_key], fromLang=fromLang, toLang=toLang)
        results.append({
            output_key: trans
        })
    f = open(output_filename, 'w', encoding='utf-8')
    for elem in results:
        f.write(json.dumps(elem, ensure_ascii=False) + '\n')
    f.close()


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
        param_dict['toLang'])

if __name__ == '__main__':
    args = readCommand(sys.argv[1:])
    runCommand(args)

