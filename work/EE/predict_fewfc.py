"""
对FewFC事件抽取数据集进行完整的预测、评价
"""
from work.EE.JointEE_rebuild.jointee_mask import UseModel as ArgModel
from work.EE.ED.bert_event_detection import UseModel as EventModel
from work.EE.EE_utils import load_jsonl, dump_jsonl
from work.EE import EE_settings
from tqdm import tqdm
from loguru import logger

dataset_type = 'FewFC'  # 不要更改


# 模型配置部分
plm_path = 'hfl/chinese-roberta-wwm-ext-large'

arg_init_params_path = '../../checkpoint/save.init_params.JointEE.FewFC.mask.RoBERTa.merge4.best.pk'
arg_state_dict_path = '../../checkpoint/save.state_dict.JointEE.FewFC.mask.RoBERTa.merge4.best.pth'
event_state_dict_path = '../../checkpoint/save.state_dict.BertED.FewFC.RoBERTa.merge6.local.best.pth'
event_init_params_path = '../../checkpoint/save.init_params.BertED.FewFC.RoBERTa.merge6.local.best.pk'

use_gpu = True

# 文件读取配置部分
train_file_dir = '../../data/NLP/EventExtraction/FewFC-main/'
train_file_name = 'merged_train2.json'
test_file_dir = '../../data/NLP/EventExtraction/FewFC-main/'
test_file_name = 'rearranged/test_trans.json'
valid_file_dir = '../../data/NLP/EventExtraction/FewFC-main/'
valid_file_name = 'val.json'
result_file_name = 'test_predicted_merge6.json'

examples = []


test_types = {
    '判决', '收购', '中标', '担保', '签署合同'
}

def predict_test():
    """
    读取测试文件，预测结果，然后输出
    :return:
    """
    logger.info('读取测试文件')
    test_data = load_jsonl(test_file_dir + test_file_name)

    # 首先预测事件类型
    logger.info('加载事件类型预测模型')
    event_model = EventModel(
        state_dict_path=event_state_dict_path,
        init_params_path=event_init_params_path,
        use_gpu=use_gpu,
        plm_path=plm_path,
        dataset_type=dataset_type
    )
    event_types = []
    logger.info(f'正在预测事件类型')
    for elem in tqdm(test_data):
        sentence = elem['content']
        result = event_model(sentence)
        event_types.append(result)
    del event_model

    # 预测触发词与论元
    logger.info('加载论元预测模型')
    argument_model = ArgModel(
        state_dict_path=arg_state_dict_path,
        init_params_path=arg_init_params_path,
        use_gpu=use_gpu,
        plm_path=plm_path,
        dataset_type=dataset_type
    )
    arg_results = []
    for (elem_sentence, elem_event_types) in tqdm(list(zip(test_data, event_types))):
        argresult = argument_model(elem_sentence['content'], elem_event_types)
        arg_results.append(argresult)
    del argument_model

    dump_jsonl(arg_results, test_file_dir + result_file_name)


def event_extraction_metric(pred: dict, gt: dict):
    """

    :param pred:
    :param gt: 均为fewfc格式的数据
    :return:
    """
    total, predict, correct = 0, 0, 0

    # 构造event_type -> mentions dict
    pred_dict, gt_dict = {}, {}
    for elem in pred['events']:
        event_type, mentions = elem['type'], elem['mentions']
        words = set(f'{x["role"]}-{x["word"]}' for x in mentions)
        if event_type not in pred_dict:
            pred_dict[event_type] = words
        else:
            pred_dict[event_type] = pred_dict[event_type].union(words)
    for elem in gt['events']:
        event_type, mentions = elem['type'], elem['mentions']
        words = set(f'{x["role"]}-{x["word"]}' for x in mentions)
        if event_type not in gt_dict:
            gt_dict[event_type] = words
        else:
            gt_dict[event_type] = gt_dict[event_type].union(words)

    # 首先找到二者预测的事件类型
    pred_events = list(x['type'] for x in pred['events'])
    gt_events = list(x['type'] for x in gt['events'])
    pred_events_set, gt_events_set = set(pred_events), set(gt_events)
    common_events_set = pred_events_set.intersection(gt_events_set)
    pred_only = pred_events_set - gt_events_set
    gt_only = gt_events_set - pred_events_set

    # 先计算非公共的
    for e_type in pred_only:
        for e_mentions in pred_dict[e_type]:
            predict += 1
    for e_type in gt_only:
        for e_mentions in gt_dict[e_type]:
            total += 1

    # 计算公共的
    for e_type in common_events_set:
        gt_word_set = gt_dict[e_type]
        pred_word_set = pred_dict[e_type]
        total += len(gt_word_set)
        predict += len(pred_word_set)
        correct += len(gt_word_set.intersection(pred_word_set))

    return total, predict, correct


def event_extraction_wordlevel_metric(pred: dict, gt: dict):
    total, predict, correct = 0, 0, 0

    # 构造event_type -> mentions dict
    pred_dict, gt_dict = {}, {}
    for elem in pred['events']:
        event_type, mentions = elem['type'], elem['mentions']
        words = set(f'{x["role"]}-{x["word"]}' for x in mentions)
        if event_type not in pred_dict:
            pred_dict[event_type] = words
        else:
            pred_dict[event_type] = pred_dict[event_type].union(words)
    for elem in gt['events']:
        event_type, mentions = elem['type'], elem['mentions']
        words = set(f'{x["role"]}-{x["word"]}' for x in mentions)
        if event_type not in gt_dict:
            gt_dict[event_type] = words
        else:
            gt_dict[event_type] = gt_dict[event_type].union(words)

    # 首先找到二者预测的事件类型
    pred_events = list(x['type'] for x in pred['events'])
    gt_events = list(x['type'] for x in gt['events'])
    pred_events_set, gt_events_set = set(pred_events), set(gt_events)
    common_events_set = pred_events_set.intersection(gt_events_set)
    pred_only = pred_events_set - gt_events_set
    gt_only = gt_events_set - pred_events_set

    # 先计算非公共的
    for e_type in pred_only:
        for e_mentions in pred_dict[e_type]:
            predict += len(e_mentions)
    for e_type in gt_only:
        for e_mentions in gt_dict[e_type]:
            total += len(e_mentions)

    # 计算公共的
    for e_type in common_events_set:
        gt_word_set = gt_dict[e_type]
        pred_word_set = pred_dict[e_type]

        gt_char_set = ''.join(list(gt_word_set))
        pred_char_set = ''.join(list(pred_word_set))
        common_char = set(gt_char_set).intersection(set(pred_char_set))  # 共有的字
        common_char_cnt = 0
        for e_char in common_char:
            common_char_cnt += min(gt_char_set.count(e_char), pred_char_set.count(e_char))
        # total += len(gt_word_set)
        # predict += len(pred_word_set)
        # correct += len(gt_word_set.intersection(pred_word_set))
        total += len(gt_char_set)
        predict += len(pred_char_set)
        correct += common_char_cnt

    return total, predict, correct


def evaluate_result():
    """
    评价标准：
        首先判断事件类型。事件类型相同的情况下，对触发词和论元进行判断。
        如果gt与pred有多个相同的事件类型，则找到结果分最高的匹配
    :return:
    """
    scores = []
    gts = load_jsonl(test_file_dir + test_file_name)
    results = load_jsonl(test_file_dir + result_file_name)
    total, predict, correct = 0, 0, 0
    sample = 0
    for i, (elem_gt, elem_result) in enumerate(zip(gts, results)):
        # p_total, p_predict, p_correct = event_extraction_wordlevel_metric(elem_result, elem_gt)
        gt_types = [x['type'] for x in elem_gt['events']]
        next_loop = False
        for e_type in gt_types:
            if e_type not in test_types:
                next_loop = True
        # if next_loop:
        #     continue
        sample += 1
        p_total, p_predict, p_correct = event_extraction_wordlevel_metric(elem_result, elem_gt)
        total += p_total
        predict += p_predict
        correct += p_correct
        scores.append([p_total, p_predict, p_correct])

    precision = correct / predict if predict != 0 else 0
    recall = correct / total if total != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    print(f'precision: {precision}\nrecall: {recall}\nf1: {f1}\n\ncorrect: {correct}\npredict: {predict}\ntotal: {total}\nsample: {sample}')
    return scores


def sample_cnt():
    train_data = load_jsonl(train_file_dir + train_file_name)
    test_data = load_jsonl(test_file_dir + test_file_name)
    valid_data = load_jsonl(valid_file_dir + valid_file_name)

    train_total, test_total, valid_total = len(train_data), len(test_data), len(valid_data)

    def cnt(d):
        half, full = 0, 0
        for elem in d:
            event_types = set(x['type'] for x in elem['events'])
            contains, all = False, True
            for elem in event_types:
                if elem in test_types:
                    contains = True
                else:
                    all = False
            if contains and not all:
                half += 1
            if all:
                full += 1
        return half, full

    train_half, train_full = cnt(train_data)
    test_half, test_full = cnt(test_data)
    valid_half, valid_full = cnt(valid_data)

    print(f'train:\n\ttotal: {train_total}\n\thalf: {train_half}\n\tfull: {train_full}')
    print('')
    print(f'test:\n\ttotal: {test_total}\n\thalf: {test_half}\n\tfull: {test_full}')
    print('')
    print(f'valid:\n\ttotal: {valid_total}\n\thalf: {valid_half}\n\tfull: {valid_full}')


if __name__ == '__main__':
    predict_test()
    scores = evaluate_result()
    # sample_cnt()
