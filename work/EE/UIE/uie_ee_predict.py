"""
使用UIE预测
"""
from paddlenlp import Taskflow
import paddle
from work.EE import EE_settings
from work.EE.EE_utils import load_jsonl, dump_jsonl
from work.EE.predict_fewfc import event_extraction_metric, event_extraction_wordlevel_metric
import json
from tqdm import tqdm
from loguru import logger


threshold = 0.3
batch_size = 8
dataset_type = 'FewFC'
use_gpu = True
if use_gpu:
    paddle.device.set_device('gpu:0')

fewfc_schema = {}
for key, value in EE_settings.event_available_roles.items():
    asocs = []
    for elem in value:
        asocs.append(EE_settings.role_types_translate[elem])
    fewfc_schema[key + '触发词'] = asocs
fewfc_sample = {
    "id": "7a80b97b57f5b80a9aba97120d31c91a",
    "content": "天奇股份于近日收到日本日产自动车株式会社雷诺日产采购部通知,公司中标该公司巴西与墨西哥总装项目,二项目合同总价合计为1401.2739万美元,折合人民币8800万元。",
    "events":
        [{
            "type": "中标",
            "mentions":
                [{
                    "word": "天奇股份",
                    "span": [0, 4],
                    "role": "sub"
                }, {
                    "word": "中标",
                    "span": [32, 34],
                    "role": "trigger"
                }, {
                    "word": "1401.2739万美元",
                    "span": [58, 70],
                    "role": "amount"
                }, {
                    "word": "8800万元",
                    "span": [76, 82],
                    "role": "amount"
                }, {
                    "word": "日本日产自动车株式会社",
                    "span": [9, 20],
                    "role": "obj"
                }]
        }]
}
fewfc_types = {
    '质押',
    '股份股权转让',
    '起诉',
    '投资',
    '减持',
    '收购',
    '担保',
    '中标',
    '签署合同',
    '判决'
}
fewfc_fewshot_types = {
    '收购',
    '担保',
    '中标',
    '签署合同',
    '判决'
}

fewfc_test_file_dir = '../../../data/NLP/EventExtraction/FewFC-main/'
fewfc_test_file = 'merged_test.json'
fewfc_raw_result = 'merged_test.uie.raw_result.json'
fewfc_predict_result = 'merged_test.uie.result.json'


duee_schema = {}
for key, value in EE_settings.duee_event_available_roles.items():
    # 因为Duee的事件类型的结构本来就包含'-'，所以这里的任务名也加了一个'-'来保持形式一致
    duee_schema[key + '-触发词'] = value
duee_sample = {
    "text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了",
    "id": "cba11b5059495e635b4f95e7484b2684",
    "event_list":
        [{
            "event_type": "组织关系-裁员",
            "trigger": "裁员",
            "trigger_start_index": 15,
            "arguments":
                [{
                    "argument_start_index": 17,
                    "role": "裁员人数",
                    "argument": "900余人", "alias": []
                }, {
                    "argument_start_index": 10,
                    "role": "时间",
                    "argument": "5月份",
                    "alias": []
                }],
            "class": "组织关系"
        }]
}
duee_types = set(duee_schema.keys())

duee_test_file_dir = '../../../data/NLP/EventExtraction/duee/'
duee_test_file = 'duee_test2.json/duee_test2.json'
duee_raw_result = 'duee_test2.uie.raw_result.json'
duee_predict_result = 'duee_test2.uie.result.json'
duee_submit_result = 'duee_test2.uie.submit.json'


model_type = 'uie-base'
"""
name    layer   hidden  heads
uie-base    12  768 12
uie-medical-base    12  768 12
uie-medium  6   768 12
uie-mini    6   384 12
uie-micro   4   384 12
uie-nano    4   312 12
"""


def predict_fewfc(schema: dict = fewfc_schema, model_type: str = model_type):
    """
    用uie模型预测FewFC，并将模型的直接输出保存。
    :param schema:
    :param model_type:
    :return:
    """
    logger.info('FewFC预测：读取数据中')
    d = list(json.loads(x) for x in open(fewfc_test_file_dir + fewfc_test_file, 'r', encoding='utf-8').read().strip().split('\n'))
    contents = list(x['content'] for x in d)
    ids = list(x['id'] for x in d)

    logger.info('FewFC预测：加载模型中')
    ee = Taskflow('information_extraction', schema=schema, model=model_type)

    logger.info('FewFC预测：模型预测中')
    results = []
    for elem_id, elem_content in tqdm(list(zip(ids, contents))):
        pred = ee(elem_content)
        results.append({
            'id': elem_id,
            'content': elem_content,
            'uie_result': pred
        })

    logger.info('FewFC预测：保存模型输出中')
    f = open(fewfc_test_file_dir + fewfc_raw_result, 'w', encoding='utf-8')
    for elem in results:
        f.write(json.dumps(elem, ensure_ascii=False) + '\n')
    f.close()


def convert_fewfc():
    """
    将fewfc的uie输出转为原格式

    如果触发词的置信度小于threshold，则抛弃整个对应预测
    如果论元的置信度小于threshold，则只抛弃该论元
    :return:
    """
    logger.info('FewFC预测：将模型输出转为原格式中')
    results = list(json.loads(x) for x in open(fewfc_test_file_dir + fewfc_raw_result, 'r', encoding='utf-8').read().strip().split('\n'))
    outputs = []
    for elem in results:
        uie_result = elem['uie_result']
        events = []
        for key_tirgger, value in uie_result[0].items():
            event_type = key_tirgger[:-3]  # 删除后缀的"触发词"三个字
            for elem_event in value:
                trigger_prob = elem_event['probability']
                if trigger_prob < threshold:
                    continue
                trigger_word = elem_event['text']
                trigger_span = (elem_event['start'], elem_event['end'])
                mentions = []
                if 'relations' in elem_event:
                    for key_arg, value_arg in elem_event['relations'].items():
                        arg_type = EE_settings.role_types_back_translate[key_arg]
                        for elem_arg in value_arg:
                            arg_prob = elem_arg['probability']
                            if arg_prob < threshold:
                                continue
                            arg_word = elem_arg['text']
                            arg_span = (elem_arg['start'], elem_arg['end'])
                            mentions.append({
                                'word': arg_word,
                                'span': arg_span,
                                'role': arg_type
                            })
                mentions.append({
                    'word': trigger_word,
                    'span': trigger_span,
                    'role': 'trigger'
                })
                events.append({
                    'type': event_type,
                    'mentions': mentions
                })
        outputs.append({
            'id': elem['id'],
            'content': elem['content'],
            'events': events
        })

    f = open(fewfc_test_file_dir + fewfc_predict_result, 'w', encoding='utf-8')
    for elem in outputs:
        f.write(json.dumps(elem, ensure_ascii=False) + '\n')
    f.close()


def evaluate_fewfc(gt_file_name: str, result_file_name: str, available_types: set, eval_type: str = 'word_level'):
    """
    评价标准：
        首先判断事件类型。事件类型相同的情况下，对触发词和论元进行判断。
        如果gt与pred有多个相同的事件类型，则找到结果分最高的匹配
    :param gt_file_name: 原文件的路径
    :param result_file_name: 预测结果文件的路径
    :param available_types: 需要计算分数的类型
    :param eval_type: ['word_level', 'span_level']，计算哪种分数
    :return:
    """
    def record_filter(record, available_event_types: set):
        available_events = []
        for old_event in record['events']:
            if old_event['type'] in available_event_types:
                available_events.append(old_event)
        return {
            'id': record['id'],
            'content': record['content'],
            'events': available_events
        }

    scores = []
    gts = load_jsonl(gt_file_name)
    results = load_jsonl(result_file_name)
    total, predict, correct = 0, 0, 0
    sample = 0
    for i, (elem_gt, elem_result) in enumerate(zip(gts, results)):
        # p_total, p_predict, p_correct = event_extraction_wordlevel_metric(elem_result, elem_gt)
        gt_types = [x['type'] for x in elem_gt['events']]
        gt = record_filter(elem_gt, available_types)
        result = record_filter(elem_result, available_types)
        if len(gt['events']) == 0 and len(result['events']) == 0:
            continue
        sample += 1
        if eval_type == 'word_level':
            p_total, p_predict, p_correct = event_extraction_wordlevel_metric(result, gt)
        elif eval_type == 'span_level':
            p_total, p_predict, p_correct = event_extraction_metric(result, gt)
        else:
            raise Exception(f'[evaluate_fewfc]不存在的指标类型:{eval_type}')
        total += p_total
        predict += p_predict
        correct += p_correct
        scores.append([p_total, p_predict, p_correct])

    precision = correct / predict if predict != 0 else 0
    recall = correct / total if total != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    print(f'precision: {precision}\nrecall: {recall}\nf1: {f1}\n\ncorrect: {correct}\npredict: {predict}\ntotal: {total}\nsample: {sample}')
    return scores


def fewfc_main():
    # predict_fewfc(fewfc_schema, model_type)
    convert_fewfc()

    eval_types = ['word_level', 'span_level']
    gt_file = fewfc_test_file_dir + fewfc_test_file
    result_file = fewfc_test_file_dir + fewfc_predict_result
    logger.info('开始计算分数')

    logger.info('所有类型')
    evaluate_fewfc(gt_file, result_file, fewfc_types)

    logger.info('少样本类型')
    evaluate_fewfc(gt_file, result_file, fewfc_fewshot_types)

    for elem in fewfc_types:
        logger.info(f'[{elem}]类型单独计算')
        evaluate_fewfc(gt_file, result_file, {elem})


def fewfc_try():
    ee = Taskflow('information_extraction', schema=fewfc_schema, model=model_type)
    result = ee(fewfc_sample['content'])
    return ee, result


def predict_duee(schema: dict = duee_schema, model_type: str = model_type, batch_size: int = batch_size):
    """
    用uie模型预测DuEE，并将模型的直接输出保存。
    :param schema:
    :param model_type:
    :return:
    """
    logger.info('DuEE预测：读取数据中')
    d = list(json.loads(x) for x in open(duee_test_file_dir + duee_test_file, 'r', encoding='utf-8').read().strip().split('\n'))
    contents = list(x['text'] for x in d)
    ids = list(x['id'] for x in d)

    logger.info('DuEE预测：加载模型中')
    ee = Taskflow('information_extraction', schema=schema, model=model_type, batch_size=batch_size)

    logger.info('DuEE预测：模型预测中')
    results = []
    cached_id, cached_content = [], []
    for elem_id, elem_content in tqdm(list(zip(ids, contents))):
        cached_id.append(elem_id)
        cached_content.append(elem_content)
        if len(cached_id) == batch_size:
            preds = ee(cached_content)
            for e_id, e_content, e_pred in zip(cached_id, cached_content, preds):
                results.append({
                    'id': e_id,
                    'content': e_content,
                    'uie_result': e_pred
                })
            cached_id, cached_content = [], []
    if len(cached_id) != 0:
        preds = ee(cached_content)
        for e_id, e_content, e_pred in zip(cached_id, cached_content, preds):
            results.append({
                'id': e_id,
                'content': e_content,
                'uie_result': e_pred
        })

    logger.info('DuEE预测：保存模型输出中')
    f = open(duee_test_file_dir + duee_raw_result, 'w', encoding='utf-8')
    for elem in results:
        f.write(json.dumps(elem, ensure_ascii=False) + '\n')
    f.close()


def convert_duee():
    """
    将fewfc的uie输出转为原格式

    如果触发词的置信度小于threshold，则抛弃整个对应预测
    如果论元的置信度小于threshold，则只抛弃该论元
    :return:
    """
    logger.info('DuEE预测：将模型输出转为FewFC格式中')
    results = list(json.loads(x) for x in open(duee_test_file_dir + duee_raw_result, 'r', encoding='utf-8').read().strip().split('\n'))
    outputs = []
    for elem in results:
        uie_result = elem['uie_result']
        events = []
        for key_tirgger, value in uie_result.items():
            event_type = key_tirgger[:-4]  # 删除后缀的"-触发词"四个字
            for elem_event in value:
                trigger_prob = elem_event['probability']
                if trigger_prob < threshold:
                    continue
                trigger_word = elem_event['text']
                trigger_span = (elem_event['start'], elem_event['end'])
                mentions = []
                if 'relations' in elem_event:
                    for key_arg, value_arg in elem_event['relations'].items():
                        arg_type = EE_settings.role_types_back_translate[key_arg]
                        for elem_arg in value_arg:
                            arg_prob = elem_arg['probability']
                            if arg_prob < threshold:
                                continue
                            arg_word = elem_arg['text']
                            arg_span = (elem_arg['start'], elem_arg['end'])
                            mentions.append({
                                'word': arg_word,
                                'span': arg_span,
                                'role': arg_type
                            })
                mentions.append({
                    'word': trigger_word,
                    'span': trigger_span,
                    'role': 'trigger'
                })
                events.append({
                    'type': event_type,
                    'mentions': mentions
                })
        outputs.append({
            'id': elem['id'],
            'content': elem['content'],
            'events': events
        })

    f = open(duee_test_file_dir + duee_predict_result, 'w', encoding='utf-8')
    for elem in outputs:
        f.write(json.dumps(elem, ensure_ascii=False) + '\n')
    f.close()


def convert_for_submit(origin: dict):
    event_list = []
    content = origin['content'].replace(' ', '_')
    for elem_e in origin['events']:
        event_type = elem_e['type']
        arguments = []
        for elem_m in elem_e['mentions']:
            role, span = elem_m['role'], elem_m['span']
            if role == 'trigger':
                 continue
            if role not in EE_settings.duee_event_available_roles[event_type]:
                continue
            word = content[span[0]: span[1]]
            arguments.append({
                'role': role,
                'argument': word
            })
        event_list.append({
            'event_type': event_type,
            'arguments': arguments
        })
    return event_list


def duee_convert_to_submit():
    result = load_jsonl(duee_test_file_dir + duee_predict_result)
    submit_results = []
    for elem in result:
        r = convert_for_submit(elem)
        cid = elem['id']
        submit_results.append({
            'id': cid,
            'event_list': r
        })
    dump_jsonl(submit_results, duee_test_file_dir + duee_submit_result)


def duee_main():
    predict_duee(duee_schema, model_type)
    convert_duee()
    duee_convert_to_submit()


def main():
    pass


if __name__ == '__main__':
    # fewfc_main()
    duee_main()
