"""
实现一系列的evaluate模块
- 接耦合，与数据结构、模型结构、模型输出等因素无关
- 复用，要能够在run_train.py中复用

"""
from numpy import argmax
import copy
import torch
from type_def import *
import numpy as np
import math


class BaseEvaluator:
    """
    eval_single记录每一个测试数据以及对应gt
    eval_step在每次eval结束时调用，对保存的测试数据和gt计算评价指标
    """

    def __init__(self):
        pass

    def eval_single(self, test=None):
        pass

    def eval_step(self) -> Dict[str, Any]:
        pass


"""
我认为直接使用源数据的组织格式即可，
有些数据比如word，id是没有收集在模型当中的，这里仍然保持着他们的key，但是value留空
这样得出的数据结构是训练数据格式的子集，输入、输出、评价都要方便
"""


# todo 这些数据类型的定义最好都移到ee_types.py当中，统一管理
class Mention(TypedDict):
    word: str
    span: SpanL
    role: str


Mentions = List[Mention]


class Event(TypedDict):
    type: str
    mentions: Mentions


Events = List[Event]


class SentenceWithEvent(TypedDict):
    id: str
    content: str
    events: Events


# ==========CCKS eval==========


def eventscore(gt_event: Event, prd_events: Events) -> float:
    gt_type = gt_event['type']
    gt_mentions = gt_event['mentions']
    gt_num = len(gt_mentions)
    score_lst = []
    match_trigger_index = []
    f1_lst = []
    max_score = 0.0
    for i_pred, elem_predi in enumerate(prd_events):
        _score = 0.0
        _recall = 0.0
        pred_type = elem_predi['type']
        pred_mentions = elem_predi['mentions']
        if gt_type == pred_type:
            match_trigger_index.append(i_pred)
            mention_num = mention_word_count(gt_mentions, pred_mentions)
            _score += mention_num / gt_num
            _recall += mention_num / len(pred_mentions) if len(pred_mentions) != 0 else 0
            f1 = 2 * _score * _recall / (_score + _recall) if (_score + _recall) != 0 else 0
            score_lst.append(_score)
            f1_lst.append(f1)
        if len(score_lst) > 0:
            max_score = score_lst[argmax(f1_lst)]
            prd_events.pop(match_trigger_index[argmax(f1_lst)])
        else:
            max_score = 0.0
    return max_score


def mention_word_count(true_mentions: Mentions, pred_mentions: Mentions) -> int:
    '''
    按照span对比真实与预测的mention内容， 包含trigger
    :param true_mentions:
    :param pred_mentions:
    :return: tp_int
    '''
    tp_int = 0
    pred_mentions_copy = copy.deepcopy(pred_mentions)
    for true_element in true_mentions:
        for ind, pred_element in enumerate(pred_mentions_copy):
            if true_element["role"] == pred_element["role"]:
                true_word = true_element["span"]
                pred_word = pred_element["span"]
                if true_word == pred_word:
                    tp_int += 1
                    pred_mentions_copy.pop(ind)
                break
    return tp_int


def ccks_evaluate(predict_result: List[SentenceWithEvent], ground_truth: List[SentenceWithEvent]):
    """
    calculate ccks score
    https://www.biendata.xyz/competition/ccks_2020_3/evaluation/
    ccks3.py的重写版本

    - predict_result与ground_truth应当长度相等且一一对应
    :param predict_result:
    :param ground_truth:
    :return:
    """
    if len(predict_result) != len(ground_truth):
        raise Exception('predict_result与ground_truth应当长度相等!')

    true_event_num = 0
    pred_event_num = 0
    tp = 0.0
    for idx, elem_gt in enumerate(ground_truth):
        elem_pred = predict_result[idx]
        pred_event_num += len(elem_pred['events'])
        true_event_num += len(elem_gt['events'])
        for elem_event in elem_gt['events']:
            score = eventscore(elem_event, elem_pred['events'])
            tp += score
    precision = tp / pred_event_num if pred_event_num != 0 else 0
    recall = tp / true_event_num
    f1score = 2 * precision * recall / (precision + recall) if precision + recall != 0.0 else 0
    return (precision, recall, f1score), (true_event_num, pred_event_num, tp)


# ==========f1 fullword eval==========


def fullword_f1_count_trigger(pred_events: Events, gt_events: Events) -> Tuple[int, int, int]:
    """
    不考虑trigger的对应关系，直接求所有词的f1统计量
    :param pred_events:
    :param gt_events:
    :return:
    """
    pred_triggers: SpanSet = set()
    gt_triggers: SpanSet = set()
    for elem_event in pred_events:
        for elem_mention in elem_event['mentions']:
            if elem_mention['role'] == 'trigger':
                pred_triggers.add(tuple(elem_mention['span']))
    for elem_event in gt_events:
        for elem_mention in elem_event['mentions']:
            if elem_mention['role'] == 'trigger':
                gt_triggers.add(tuple(elem_mention['span']))
    total = len(gt_triggers)
    predict = len(pred_triggers)
    correct = len(gt_triggers.intersection(pred_triggers))
    return total, predict, correct


def fullword_f1_count_argument(pred_events: Events, gt_events: Events) -> Tuple[int, int, int]:
    """
    不考虑trigger的对应关系，直接求所有词的f1统计量
    :param pred_events:
    :param gt_events:
    :return:
    """
    pred_arguments: SpanSet = set()
    gt_arguments: SpanSet = set()
    for elem_event in pred_events:
        for elem_mention in elem_event['mentions']:
            if elem_mention['role'] != 'trigger':
                pred_arguments.add(tuple(elem_mention['span']))
    for elem_event in gt_events:
        for elem_mention in elem_event['mentions']:
            if elem_mention['role'] != 'trigger':
                gt_arguments.add(tuple(elem_mention['span']))
    total = len(gt_arguments)
    predict = len(pred_arguments)
    correct = len(gt_arguments.intersection(pred_arguments))
    return total, predict, correct


def fullword_f1_evaluate(predict_result: List[SentenceWithEvent], ground_truth: List[SentenceWithEvent]):
    if len(predict_result) != len(ground_truth):
        raise Exception('predict_result和ground_truth应当长度相等！')
    trig_total, trig_predict, trig_correct = 0, 0, 0
    arg_total, arg_predict, arg_correct = 0, 0, 0
    for idx, elem_gt in enumerate(ground_truth):
        elem_pred = predict_result[idx]
        elem_trig_total, elem_trig_predict, elem_trig_correct = fullword_f1_count_trigger(elem_pred['events'],
                                                                                          elem_gt['events'])
        elem_arg_total, elem_arg_predict, elem_arg_correct = fullword_f1_count_argument(elem_pred['events'],
                                                                                        elem_gt['events'])
        trig_total += elem_trig_total
        trig_predict += elem_trig_predict
        trig_correct += elem_trig_correct
        arg_total += elem_arg_total
        arg_predict += elem_arg_predict
        arg_correct += elem_arg_correct
    trig_precision = trig_correct / trig_predict if trig_predict != 0 else 0
    trig_recall = trig_correct / trig_total if trig_total != 0 else 0
    trig_f1 = 2 * trig_precision * trig_recall / (trig_precision + trig_recall) if (
                                                                                               trig_precision + trig_recall) != 0 else 0
    arg_precision = arg_correct / arg_predict if arg_predict != 0 else 0
    arg_recall = arg_correct / arg_total if arg_total != 0 else 0
    arg_f1 = 2 * arg_precision * arg_recall / (arg_precision + arg_recall) if (arg_precision + arg_recall) != 0 else 0
    return (trig_precision, trig_recall, trig_f1), \
           (trig_total, trig_predict, trig_correct), \
           (arg_precision, arg_recall, arg_f1), \
           (arg_total, arg_predict, arg_correct)


# ==========f1 strict eval==========
# todo 搞一个类似于ccks的，但是统计的是f1


# ==========f1 strict stable eval==========
# todo 对可用匹配遍历，得到最优结果。简单strict的结果不稳定？匈牙利算法？
# https://en.wikipedia.org/wiki/Hungarian_algorithm
"""
要把这个函数包装到一个Evaluator里面，需要一个转格式到函数
"""


def spans2events(event_types: StrList, triggers: List[SpanList], arguments: List[List[List[SpanList]]],
                 role_types: List[str], content: str = '') -> SentenceWithEvent:
    cur_events = []
    for (elem_event, elem_triggers, elem_arguments) in zip(event_types, triggers, arguments):
        # elem_event: str
        # elem_trigger: SpanList
        # elem_arguments: List[List[SpanList]]
        for (elem_trigger, elem_argument) in zip(elem_triggers, elem_arguments):
            # elem_trigger: Span
            # elem_argument: List[SpanList]
            cur_mentions = [{'word': "", 'span': list(elem_trigger), 'role': 'trigger'}]
            for idx_arg, elem_type_args in enumerate(elem_argument):
                # elem_type_args: SpanList
                cur_role_type = role_types[idx_arg]
                for elem_arg in elem_type_args:
                    # elem_arg: Span
                    cur_mentions.append({"word": "", "span": list(elem_arg), "role": cur_role_type})
            cur_event = {
                'type': elem_event,
                'mentions': cur_mentions
            }
            cur_events.append(cur_event)
    result: SentenceWithEvent = {
        "id": '',
        "content": content,
        "events": cur_events}
    return result


def tokenspans2events(event_types: StrList, triggers: List[SpanList], arguments: List[List[List[SpanList]]],
                      role_types: List[str], content: str = '', token2origin: dict = None) -> SentenceWithEvent:
    """
    将token span转化为 origin span
    然后生成SentenceWithEvents
    :param event_types:
    :param triggers:
    :param arguments:
    :param role_types:
    :param content:
    :param token2origin_lst:
    :return:
    """

    def span_converter(span: Span, token2origin: dict) -> Span:
        """
        origin -> token
            token start = origin2token[origin start] - 1
            token end = origin2token[origin end - 1] - 1
        token -> origin
            origin start = token2origin[token start + 1]
            origin end = token2origin[token end + 1] + 1

        特殊情况下，会遇到预测超出范围的问题
        情况一：
            span[0/1] + 1超出token2origin范围这种情况，只能将二者均设置为0
        情况二：
            token2origin[-]为空list，只能取邻近？
        :param span:
        :param token2origin:
        :return:
        """
        pref = 0  # 0 or -1
        try:
            origin_start = token2origin[span[0] + 1][pref]
            origin_end = token2origin[span[1] + 1][pref] + 1
        except:
            print(f'span:{span}')
            print(f'token2origin:{token2origin}')
            origin_start, origin_end = 0, 0
        return origin_start, origin_end

    # convert triggers span
    for i in range(len(triggers)):
        triggers[i] = list(map(lambda x: span_converter(x, token2origin), triggers[i]))

    # convert arguments span
    for i in range(len(arguments)):
        # List[List[SpanList]]
        for j in range(len(arguments[i])):
            # List[SpanList]
            for k in range(len(arguments[i][j])):
                arguments[i][j][k] = list(map(lambda x: span_converter(x, token2origin), arguments[i][j][k]))

    result = spans2events(event_types, triggers, arguments, role_types, content)
    return result


def events2spans(events: SentenceWithEvent):
    raise NotImplementedError


"""
然后只需要简答的包装一下，就能得到一个Evaluator
如果需要别的格式，也可以另行包装。
甚至可以包装多个

上面的evaluator function没有固定的形式，所有格式转换的工作都在BaseEvaluator子类里面进行即可
"""


class CcksEvaluator(BaseEvaluator):
    def __init__(self):
        super(CcksEvaluator, self).__init__()
        self.predevents: List[SentenceWithEvent] = []
        self.trueevents: List[SentenceWithEvent] = []

    def eval_single(self, trueevent: SentenceWithEvent, predevent: SentenceWithEvent):
        self.trueevents.append(trueevent)
        self.predevents.append(predevent)

    def eval_step(self) -> Dict[str, Any]:
        (precision, recall, f1score), (true_event_num, pred_event_num, tp) = \
            ccks_evaluate(self.predevents, self.trueevents)
        self.predevents = []
        self.trueevents = []
        result = {
            "precision": precision,
            "recall": recall,
            "f1score": f1score,
            "true num": true_event_num,
            "pred num": pred_event_num,
            "tp": tp
        }
        # return f"|| p: {precision:<6.4f} || r: {recall:<6.4f} || f1: {f1score:<6.4f} ||\n" \
        #        f"|| true num: {true_event_num:5} || pred num: {pred_event_num: 5} || tp: {tp: <6.4f} ||"
        return result


class F1Evaluator(BaseEvaluator):
    def __init__(self):
        super(F1Evaluator, self).__init__()
        self.predevents: List[SentenceWithEvent] = []
        self.trueevents: List[SentenceWithEvent] = []

    def eval_single(self, trueevent: SentenceWithEvent, predevent: SentenceWithEvent):
        self.trueevents.append(trueevent)
        self.predevents.append(predevent)

    def eval_step(self) -> Dict[str, Any]:
        (trig_precision, trig_recall, trig_f1), \
        (trig_total, trig_predict, trig_correct), \
        (arg_precision, arg_recall, arg_f1), \
        (arg_total, arg_predict, arg_correct) = \
            fullword_f1_evaluate(self.predevents, self.trueevents)
        self.predevents = []
        self.trueevents = []
        scores = {
            "trigger p": trig_precision,
            "trigger r": trig_recall,
            "trigger f1": trig_f1,
            "trigger total": trig_total,
            "trigger predict": trig_predict,
            "trigger correct": trig_correct,
            "argument p": arg_precision,
            "argument r": arg_recall,
            "argument f1": arg_f1,
            "argument total": arg_total,
            "argument predict": arg_predict,
            "argument correct": arg_correct
        }
        # return f"|| trigger p: {trig_precision:<6.4f} || trigger r: {trig_recall:<6.4f} || trigger f1: {trig_f1:<6.4f} ||\n" \
        #        f"|| trigger total: {trig_total:5} || trigger predict: {trig_predict:5} || trigger correct: {trig_correct: 5} ||\n" \
        #        f"|| argument p: {arg_precision:<6.4f} || argument r: {arg_recall:<6.4f} || argument f1: {arg_f1:<6.4f} ||\n" \
        #        f"|| argument total: {arg_total:5} || argument predict: {arg_predict:5} || argument correct: {arg_correct: 5}"
        return scores


def dict2fstring(scores: Dict[str, Any], row_cnt: int = 3):
    """

    float - ^6.4f
    int - 5
    :param scores:
    :param row_cnt:
    :return:
    """
    units = []
    for key, value in scores.items():
        if isinstance(value, float):
            units.append(f'| {key}: {value:<6.4f} |')
        elif isinstance(value, int):
            units.append(f'| {key}: {value:5} |')

    final_string = f''
    cnt = 0
    for elem_unit in units:
        if cnt == 0:
            final_string += '|'
        elif cnt == row_cnt:
            final_string += '|\n|'
            cnt = 0
        final_string += elem_unit
        cnt += 1
    return final_string


"""
多分类问题相关
"""


def multilabel_match_score(preds: List[List[Hashable]], gts: List[List[Hashable]]) -> float:
    """
    计算多标签分类问题的完全匹配率
    :param preds: List[List[Hashable]]，其中每个元素的长度相同
    :param gts: List[List[Hashable]]，其中每个元素的长度相同
    :return:
    """
    total, match = 0, 0
    for (elem_pred, elem_gt) in zip(preds, gts):
        if set(elem_pred) == set(elem_gt):
            match += 1
        total += 1
    return match / total if total != 0 else 0.


def multilabel_f1_score(preds: List[List[Hashable]], gts: List[List[Hashable]]):
    """

    :param preds:
    :param gts:
    :return:
    """
    total, predict, correct = 0, 0, 0
    for (elem_pred, elem_gt) in zip(preds, gts):
        pred_set, gt_set = set(elem_pred), set(elem_gt)
        inter_set = pred_set.intersection(gt_set)
        total += len(gt_set)
        predict += len(pred_set)
        correct += len(inter_set)
    recall = correct / total if total != 0 else 0
    precision = correct / predict if predict != 0 else 0
    f1_measure = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0
    return precision, recall, f1_measure


class MultiLabelClsEvaluator(BaseEvaluator):
    def __init__(self):
        super(MultiLabelClsEvaluator, self).__init__()
        self.preds = []
        self.gts = []

    def eval_single(self, pred: List[Hashable], gt: List[Hashable]):
        self.preds.append(pred)
        self.gts.append(gt)

    def eval_step(self) -> Dict[str, Any]:
        match_score = multilabel_match_score(self.preds, self.gts)
        precision, recall, f1_measure = multilabel_f1_score(self.preds, self.gts)
        self.preds, self.gts = [], []
        return {
            "match": match_score,
            "precision": precision,
            "recall": recall,
            "f1": f1_measure
        }


# todo 单分类评价指标

def kappa_score(total: int, correct: int, pred_class: IntList, total_class: IntList) -> float:
    """
    计算Kappa score
    P_0 = correct / total
    P_e = sum(pred_class * total_class) / total^2
    Kappa = (P_0 - P_e) / (1 - P_e)
    :param total: 总个数
    :param correct: 预测正确的个数
    :param pred_class: 每一类的预测个数
    :param total_class: 每一类的真实个数
    :return:
    """
    if len(pred_class) <= 1:
        raise Exception('[kappa_score]pred_class中包含的元素数量为零！应当至少为2！')
    if len(pred_class) != len(total_class):
        raise Exception('[kappa_score]pred_class与total_class所包含的元素数量不同！应当相同！')
    p_0 = correct / total
    p_e = sum([p * t for p, t in zip(pred_class, total_class)]) / total ** 2
    kappa = (p_0 - p_e) / (1 - p_e) if 1 - p_e != 0 else 0
    return kappa


def calculate_kappa_score(pred: List[Hashable], gt: List[Hashable], classes: List[Hashable]=None):
    """
    给定预测结果和实际结果，计算kappa分数
    数学部分的计算由kappa_score实现，本函数主要是管理数据结构，统计结果
    :param pred:
    :param gt:
    :param classes:
    :return:
    """
    if classes is None:
        classes = list(set(gt))

    # 分别统计每个类下的预测个数与实际个数
    pred_per_class, total_per_class = {x: 0 for x in classes}, {x: 0 for x in classes}
    total, correct = 0, 0
    for (elem_pred, elem_gt) in zip(pred, gt):
        total += 1
        total_per_class[elem_gt] += 1
        if elem_pred == elem_gt:
            correct += 1
            pred_per_class[elem_gt] += 1

    kappa = kappa_score(total, correct, list(pred_per_class.values()), list(total_per_class.values()))
    return kappa


def precision_score(total: int, correct: int):
    """
    简单的，直接计算单分类的准确率
    :param total:
    :param correct:
    :return:
    """
    return correct / total if total != 0 else 0


def calculate_precision_score(pred: List[Hashable], gt: List[Hashable]):
    """
    给定预测结果和实际结果，计算precision得分
    :param pred:
    :param gt:
    :return:
    """
    # 分别统计每个类下的预测个数与实际个数
    total, correct = 0, 0
    for (elem_pred, elem_gt) in zip(pred, gt):
        total += 1
        if elem_pred == elem_gt:
            correct += 1
    precision = precision_score(total, correct)
    return precision


class KappaEvaluator(BaseEvaluator):
    def __init__(self):
        super(KappaEvaluator, self).__init__()
        self.pred_lst = []
        self.gt_lst = []

    def eval_single(self, pred: Hashable, gt: Hashable):
        self.pred_lst.append(pred)
        self.gt_lst.append(gt)

    def eval_step(self) -> Dict[str, Any]:
        kappa = calculate_kappa_score(self.pred_lst, self.gt_lst)
        self.pred_lst, self.gt_lst = [], []
        return {
            "Kappa": kappa
        }


class PrecisionEvaluator(BaseEvaluator):
    def __init__(self):
        super(PrecisionEvaluator, self).__init__()
        self.pred_lst = []
        self.gt_lst = []

    def eval_single(self, pred: Hashable, gt: Hashable):
        self.pred_lst.append(pred)
        self.gt_lst.append(gt)

    def eval_step(self) -> Dict[str, Any]:
        precision = calculate_precision_score(self.pred_lst, self.gt_lst)
        self.pred_lst, self.gt_lst = [], []
        return {
            "precision": precision
        }


# todo 标注任务评价指标


"""
上面还实现类一个F1Evaluator，但是他是只针对事件抽取任务的，是不好的设计，应当抽象为纯粹F1计算+事件格式转换,是要删掉的
这里的F1_Evaluator就是纯粹F1计算，是对multilabel_f1_score的封装
"""


class F1_Evaluator(BaseEvaluator):
    def __init__(self):
        super(F1_Evaluator, self).__init__()
        self.pred_lst = []
        self.gt_lst = []

    def eval_single(self, pred: List[Hashable], gt: List[Hashable]):
        self.pred_lst.append(pred)
        self.gt_lst.append(gt)

    def eval_step(self) -> Dict[str, Any]:
        p, r, f1 = multilabel_f1_score(self.pred_lst, self.gt_lst)
        self.pred_lst = []
        self.gt_lst = []

        return {
            "precision": p,
            "recall": r,
            "f1-measure": f1
        }


"""
SemEval评价指标
    皮尔逊系数共有三个实现，
"""


def pearson(vector1, vector2):
    """
    皮尔逊系数，xhq同学的实现
    :param vector1:
    :param vector2:
    :return:
    """
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    #分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den


def cal_pccs(x, y):
    """
    warning: data format must be ndarray
    :param x: Variable 1
    :param y: Variable 2
    :return: pccs
    """
    n = len(x)
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc


def cal_pccs_torch(x, y):
    n = len(x)
    sum_xy = torch.sum(torch.sum(x * y))
    sum_x = torch.sum(torch.sum(x))
    sum_y = torch.sum(torch.sum(y))
    sum_x2 = torch.sum(torch.sum(x * x))
    sum_y2 = torch.sum(torch.sum(y * y))
    pcc = (n * sum_xy - sum_x * sum_y) / torch.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pcc


def pearson_score(vector1: Sequence[float], vector2: Sequence[float]):
    """
    计算vector1和vector2之间的皮尔逊系数

    :param vector1: len(vector1) == len(vector2)
    :param vector2:
    :return:
    """
    if len(vector1) != len(vector2):
        raise Exception(f'[pearson_score]vector1的长度与vector2的长度不相等！len(vector1)={len(vector1)}, len(vector2)={len(vector2)}')
    if len(vector1) == 0:
        raise Exception('[pearson_score]vector1与vector2的长度不能为0！')
    np_v1 = np.array(vector1)
    np_v2 = np.array(vector2)
    score = cal_pccs(np_v1, np_v2)
    return score


class Pearson_Evaluator(BaseEvaluator):
    def __init__(self):
        super(Pearson_Evaluator, self).__init__()
        self.pred_lst = []
        self.gt_lst = []

    def eval_single(self, pred: float, gt: float):
        self.pred_lst.append(pred)
        self.gt_lst.append(gt)

    def eval_step(self) -> Dict[str, Any]:
        pearson = pearson_score(self.pred_lst, self.gt_lst)
        self.pred_lst = []
        self.gt_lst = []

        return {
            "pearson": pearson
        }