import bminf
from type_def import *
from typing import Optional
from work.BMinf_Experiment.better_fill_blank import fill_blank
import re
from tqdm import tqdm
import json


TOKEN_SPAN = "<span>"

duee_samples = {
    "被证实将再裁员1800人 福特汽车公司在为落后的经营模式买单": [
        '那么，<span>进行了裁员。',
        '在这句话中，哪家公司进行了裁员：<span>',  # 填空
        '这句话<span>(是/否)描述了一个"裁员"事件',  # 选择
        '裁员人数为<span>，裁员方为<span>',
        '福特汽车公司进行了<span>'  # 复述
    ],
    "8月20日消息，据腾讯新闻《一线》报道，知情人士表示，为了控制成本支出，蔚来计划将美国分公司的人员规模除自动驾驶业务相关人员外，减少至200人左右。截至美国时间8月16日，蔚来位于美国硅谷的分公司已裁减100名员工。": [
        '<span>公司进行了裁员',
        '在这句话中，哪家公司进行了裁员：<span>',  # 填空
        '这句话<span>(是/否)描述了一个"裁员"事件',  # 选择
        '简而言之，<span>裁员了<span>人。',  # 复述

    ],
    "近日新闻报道，西门子宣布计划未来几年在油气与电力集团全球裁员2700人，这一裁员数量约占该业务集团全球6.4万名员工的4％。": [
        '<span>公司进行了裁员',
        '在这句话中，哪家公司进行了裁员：<span>',  # 填空
        '这句话<span>(是/否)描述了一个"裁员"事件',  # 选择
        '简而言之，<span>裁员了<span>人。',  # 复述
    ],
    "据生态环境部微信公众号消息，6月13日，生态环境部就2018-2019年秋冬季大气污染综合治理问题约谈河北省保定、廊坊，河南省洛阳、安阳、濮阳，山西省晋中等六市政府。": [
        '<span>约谈了<span>',
        '<span>是约谈方，<span>是被约谈方',  # 填空
        '日期：<span>，约谈方<span>就问题<span>约谈了<span>'  # 复述
    ],
    "中国电信因骚扰电话管控不力被约谈": [
        '<span>被约谈了',
        '<span>约谈了<span>',
        '约谈的原因是<span>'  # 填空
    ],
    "此外，上半年，哈啰单车、青桔单车因存在违规行为被管理部门约谈，两家分别被罚款10万元和5万元。": [
        '<span>被约谈了',
        '<span>约谈了<span>',
        '约谈的原因是<span>'  # 填空
    ]
}

duee_common_templates = [
    '这是一个和<span>有关的新闻。',
    '这句话与<span>有关。',  # 完全无信息
    '这句话是不是说明了一个"裁员"事件？答案为：<span>',
    '这句话是不是说明了一个"约谈"事件？答案为：<span>',  # 判断
    '这句话是不是说明了一个"约谈"事件？答案为：<span>，用"是"与"不是"作答',  # 带提示的判断
    '在"裁员"与"约谈"当中选一个，这句话描述的是<span>事件'  # 选择
]

fewfc_samples = {
    "不止是在美国,苹果在中国和英国等地也发起了对高通的诉讼,并在 5 月份获得了来自三星和英特尔的支持。": [
        '<span>提起了诉讼',
        '<span>被提起了诉讼', # 无提示信息
        '<span>公司被提起了诉讼',  # 提示后缀
        '<span>（公司名）被提起了诉讼',  # 直接提示信息
        '<span>对<span>提起了诉讼',
    ],
    "项目工期40个月,中标合同价为人民币5.05亿元,约占公司2018年经审计营业收入的17.96%。": [
        '中标的价格为<span>',
        '中标的价格为<span>元'
    ],
    "2019年7月25日,据外媒报道,苹果正式宣布收购英特尔智能手机调制解调器部门相关业务,用于研发属于苹果自己的5G芯片。": [
        '<span>收购了<span>',
        '<span>被<span>收购了',
        '<span>公司被<span>公司收购了',
        '<span>（公司名）被<span>收购了',
        '<span>（公司名）被<span>（公司名）收购了',
        '<span>是收购方',
        '<span>是被收购方'
    ]
}

fewfc_common_templates = [
    '这是一个和<span>有关的新闻。',
    '这句话与<span>有关。',  # 完全无信息
    '这句话是不是说明了一个"诉讼"事件？答案为：<span>',
    '这句话是不是说明了一个"中标"事件？答案为：<span>',  # 判断
    '这句话是不是说明了一个"收购"事件？答案为：<span>，用"是"与"不是"作答',  # 带提示的判断
    '在"诉讼"、"中标"与"收购"类型当中选一个，这句话描述的是<span>类型的事件'  # 选择
]


def main():

    # 加载模型
    cpm2 = bminf.models.CPM2()
    repeat = 3
    max_span = 4

    duee_results = {}
    for key, value in tqdm(duee_samples.items()):
        duee_results[key] = {x: [] for x in value}
        for elem in duee_common_templates:
            duee_results[key][elem] = []
        templates = value + duee_common_templates
        for elem_template in templates:
            combined_sentence = key + '\n' + elem_template
            for idx in range(repeat):
                result = fill_blank(cpm2, combined_sentence)
                answer = re.split('<s_\d+>', result.strip())[:max_span]
                duee_results[key][elem_template].append(answer)
            json.dump(duee_results, open('duee_results.json', 'w', encoding='utf-8'), ensure_ascii=False)

    fewfc_results = {}
    for key, value in tqdm(fewfc_samples.items()):
        fewfc_results[key] = {x: [] for x in value}
        for elem in fewfc_common_templates:
            fewfc_results[key][elem] = []
        templates = value + fewfc_common_templates
        for elem_template in templates:
            combined_sentence = key + '\n' + elem_template
            for idx in range(repeat):
                result = fill_blank(cpm2, combined_sentence)
                answer = re.split('<s_\d+>', result.strip())[:max_span]
                fewfc_results[key][elem_template].append(answer)
            json.dump(fewfc_results, open('fewfc_results.json', 'w', encoding='utf-8'), ensure_ascii=False)

    json.dump(duee_results, open('duee_results.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(fewfc_results, open('fewfc_results.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
