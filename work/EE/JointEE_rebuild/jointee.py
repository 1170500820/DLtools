from type_def import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

import numpy as np
from itertools import chain
import pickle
import copy

from evaluate.evaluator import BaseEvaluator, EE_F1Evaluator, CcksEvaluator, SentenceWithEvent, spans2events
from work.EE.PLMEE.sentence_representation_layer import SentenceRepresentation, TriggeredSentenceRepresentation
from work.EE.PLMEE.trigger_extraction_model import TriggerExtractionLayer_woSyntactic
from work.EE.PLMEE.argument_extraction_model import ArgumentExtractionModel_woSyntactic
from work.EE.EE_utils import *
from work.EE.JointEE_rebuild import jointee_settings
from utils import tools
from utils.data import SimpleDataset
from analysis.recorder import NaiveRecorder
from models.model_utils import get_init_params
from utils import tokenize_tools


class JointEE(nn.Module):
    def __init__(self,
                 plm_path=jointee_settings.plm_path,
                 n_head=jointee_settings.n_head,
                 d_head=jointee_settings.d_head,
                 hidden_dropout_prob=0.3,
                 plm_lr=EE_settings.plm_lr,
                 others_lr=EE_settings.others_lr,
                 trigger_threshold=jointee_settings.trigger_extraction_threshold,
                 argument_threshold=jointee_settings.argument_extraction_threshold,
                 dataset_type: str = 'FewFC',
                 use_cuda: bool = False):
        super(JointEE, self).__init__()
        self.init_params = get_init_params(locals())  # 默认模型中包含这个东西。也许是个不好的设计
        # store init params

        if dataset_type == 'FewFC':
            self.role_types = EE_settings.role_types
        elif dataset_type == 'Duee':
            self.role_types = EE_settings.duee_role_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')

        self.plm_path = plm_path
        self.n_head = n_head
        self.d_head = d_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.plm_lr = plm_lr
        self.others_lr = others_lr
        self.trigger_threshold = trigger_threshold
        self.argument_threshold = argument_threshold
        self.use_cuda = use_cuda

        # initiate network structures
        #   Sentence Representation
        self.sentence_representation = SentenceRepresentation(self.plm_path, self.use_cuda)
        self.hidden_size = self.sentence_representation.hidden_size
        #   Trigger Extraction
        self.tem = TriggerExtractionLayer_woSyntactic(
            num_heads=self.n_head,
            hidden_size=self.hidden_size,
            d_head=self.d_head,
            dropout_prob=self.hidden_dropout_prob)
        #   Triggered Sentence Representation
        self.trigger_sentence_representation = TriggeredSentenceRepresentation(self.hidden_size, self.use_cuda)
        self.aem = ArgumentExtractionModel_woSyntactic(
            n_head=self.n_head,
            d_head=self.d_head,
            hidden_size=self.hidden_size,
            dropout_prob=self.hidden_dropout_prob,
            dataset_type=dataset_type)

        self.trigger_spanses = []
        self.argument_spanses = []

    def forward(self,
                sentences: List[str],
                event_types: Union[List[str], List[StrList]],
                triggers: SpanList = None,
                offset_mappings: list = None):
        """
        jointEE对于train和eval模式有着不同的行为

        train：
            event_types为List[str]，与sentences中的句子一一对应，同时与triggers也一一对应
            先通过sentences与event_types获得句子的表示[BERT + Pooling + CLN] (bsz, seq_l, hidden)
            然后用TEM预测出trigger，再用triggers中的Span(gt)通过AEM预测出argument。这里的Span和TEM的预测结果完全无关
        eval：
            event_types为List[StrList]，是sentences中每一个对应的句子的所有包含的事件类型
            先通过sentences与event_types获得句子在每一个事件类型下的表示。
            然后用TEM对每一个类型预测出trigger，再用AEM对每一个trigger预测出argument。所有结果会被转化为源数据格式
            triggers参数完全不会被用到
        :param sentences:
        :param event_types:
        :param triggers:
        :return:
        """
        if self.training:
            self.trigger_spanses = []
            self.argument_spanses = []
            H_styp = self.sentence_representation(sentences, event_types)  # (bsz, max_seq_l, hidden)
            trigger_start, trigger_end = self.tem(H_styp)  # (bsz, max_seq_l, 1)
            arg_H_styps, RPEs = self.trigger_sentence_representation(H_styp, triggers)
            argument_start, argument_end = self.aem(arg_H_styps, RPEs)
            # print(f'argument extraction output: start:{argument_start.shape}, end:{argument_end.shape}')
            # both (bsz, max_seq_l, len(role_types))
            return {
                "trigger_start": trigger_start,  # (bsz, seq_l, 1)
                "trigger_end": trigger_end,  # (bsz. seq_l, 1)
                "argument_start": argument_start,  # (bsz, seq_l, role_cnt)
                "argument_end": argument_end,  # (bsz, seq_l, role_cnt)
            }
        else:  # eval mode
            # sentence_types: List[StrList] during evaluating
            if len(sentences) != 1:
                raise Exception('eval模式下一次只预测单个句子!')
            cur_spanses = []  # List[SpanList]
            arg_spanses = []  # List[List[List[SpanList]]]
            for elem_sentence_type in event_types[0]:
                # elem_sentence_type: str
                H_styp = self.sentence_representation(sentences, [elem_sentence_type])  # (bsz, max_seq_l, hidden)
                trigger_start_tensor, trigger_end_tensor = self.tem(H_styp)  # (bsz, max_seq_l, 1)
                trigger_start_tensor, trigger_end_tensor = trigger_start_tensor.squeeze(), trigger_end_tensor.squeeze()
                # (max_seq_l)

                trigger_start_result = (trigger_start_tensor > self.trigger_threshold).int().tolist()
                trigger_end_result = (trigger_end_tensor > self.trigger_threshold).int().tolist()
                cur_spans = tools.argument_span_determination(trigger_start_result, trigger_end_result, trigger_start_tensor, trigger_end_tensor)
                # cur_spans: SpanList, triggers extracted from current sentence

                arg_spans: List[List[SpanList]] = []
                for elem_trigger_span in cur_spans:
                    arg_H_styp, RPE = self.trigger_sentence_representation(H_styp, [elem_trigger_span])
                    argument_start_tensor, argument_end_tensor = self.aem(arg_H_styp, RPE)
                    # (1, max_seq_l, len(role_types))
                    argument_start_tensor = argument_start_tensor.squeeze().T  # (role cnt, max_seq_l)
                    argument_end_tensor = argument_end_tensor.squeeze().T  # (role cnt, max_seq_l)

                    argument_start_result = (argument_start_tensor > self.argument_threshold).int().tolist()
                    argument_end_result = (argument_end_tensor > self.argument_threshold).int().tolist()
                    argument_spans: List[SpanList] = []
                    for idx_role in range(len(self.role_types)):
                        cur_arg_spans: SpanList = tools.argument_span_determination(argument_start_result[idx_role], argument_end_result[idx_role], argument_start_tensor[idx_role].tolist(), argument_end_tensor[idx_role].tolist())
                        argument_spans.append(cur_arg_spans)
                    arg_spans.append(argument_spans)
                cur_spanses.append(cur_spans)
                arg_spanses.append(arg_spans)
            self.trigger_spanses.append(cur_spanses)
            self.argument_spanses.append(arg_spanses)
            result = tokenspans2events(event_types[0], cur_spanses, arg_spanses, self.role_types, sentences[0], offset_mappings[0])
            return {"pred": result}

    def get_optimizers(self):
        repr_plm_params, repr_other_params = self.sentence_representation.PLM.parameters(), self.sentence_representation.CLN.parameters()
        trigger_repr_params = self.trigger_sentence_representation.parameters()
        aem_params = self.aem.parameters()
        tem_params = self.tem.parameters()
        plm_optimizer = AdamW(params=repr_plm_params, lr=self.plm_lr)
        others_optimizer = AdamW(params=chain(aem_params, tem_params, repr_other_params, trigger_repr_params), lr=self.others_lr)
        return [plm_optimizer, others_optimizer]


class JointEE_Loss(nn.Module):
    """
    需要加入的内容：
    1, focal loss
    2, weight
    3, RF·IEF weight
    """
    def __init__(self, lambd=0.2, alpha=0.3, gamma=2):
        """

        :param lambd: loss = lambd * trigger + (1 - lambd) * argument
        :param alpha: focal weight param
        :param gamma: focal weight param
        """
        super(JointEE_Loss, self).__init__()
        self.lambd = lambd
        self.focal = tools.FocalWeight(alpha, gamma)

        # record
        self.last_trigger_loss, self.last_argument_loss = 0., 0.
        self.last_trigger_preds, self.last_argument_preds = (0., 0.), (0., 0.)

    def forward(self,
                trigger_start: torch.Tensor,
                trigger_end: torch.Tensor,
                trigger_label_start: torch.Tensor,
                trigger_label_end: torch.Tensor,
                argument_start: torch.Tensor,
                argument_end: torch.Tensor,
                argument_label_start: torch.Tensor,
                argument_label_end: torch.Tensor):
        """

        :param trigger_start: (bsz, seq_l, 1)
        :param trigger_end: (bsz, seq_l, 1)
        :param trigger_label_start:
        :param trigger_label_end:
        :param argument_start: (bsz, seq_l, role_cnt)
        :param argument_end: (bsz, seq_l, role_cnt)
        :param argument_label_start:
        :param argument_label_end:
        :return: loss
        """
        if trigger_start.shape != trigger_label_start.shape:
            print('error')
            breakpoint()
        # calculate focal weight
        trigger_start_weight_focal = self.focal(trigger_start, trigger_label_start)
        trigger_end_weight_focal = self.focal(trigger_end, trigger_label_end)
        argument_start_weight_focal = self.focal(argument_start, argument_label_start)
        argument_end_weight_focal = self.focal(argument_end, argument_label_end)

        # combine weights
        trigger_start_weight = trigger_start_weight_focal
        trigger_end_weight = trigger_end_weight_focal
        argument_start_weight = argument_start_weight_focal
        argument_end_weight = argument_end_weight_focal

        trigger_start_loss = F.binary_cross_entropy(trigger_start, trigger_label_start, trigger_start_weight)
        trigger_end_loss = F.binary_cross_entropy(trigger_end, trigger_label_end, trigger_end_weight)
        trigger_loss = trigger_start_loss + trigger_end_loss

        argument_start_loss = F.binary_cross_entropy(argument_start, argument_label_start, argument_start_weight)
        argument_end_loss = F.binary_cross_entropy(argument_end, argument_label_end, argument_end_weight)
        argument_loss = argument_start_loss + argument_end_loss

        loss = self.lambd * trigger_loss + (1 - self.lambd) * argument_loss

        # for recorder
        self.last_trigger_loss = float(trigger_loss)
        self.last_argument_loss = float(argument_loss)
        trigger_start_np, trigger_end_np, argument_start_np, argument_end_np = \
            [trigger_start.cpu().detach().numpy(), trigger_end.cpu().detach().numpy(),
             argument_start.cpu().detach().numpy(), argument_end.cpu().detach().numpy()]
        self.last_trigger_preds = (trigger_start_np, trigger_end_np)
        self.last_argument_preds = (argument_start_np, argument_end_np)

        return loss


class JointEE_MaskLoss(nn.Module):
    def forward(self,
                trigger_start: torch.Tensor,
                trigger_end: torch.Tensor,
                trigger_label_start: torch.Tensor,
                trigger_label_end: torch.Tensor,
                argument_start: torch.Tensor,
                argument_end: torch.Tensor,
                argument_label_start: torch.Tensor,
                argument_label_end: torch.Tensor,
                mask: torch.Tensor):
        """

        :param trigger_start:
        :param trigger_end: (bsz, seq_l, 1)
        :param trigger_label_start:
        :param trigger_label_end: (bsz, seq_l)
        :param argument_start:
        :param argument_end: (bsz, seq_l, role_cnt)
        :param argument_label_start:
        :param argument_label_end:  (bsz, seq_l, role_cnt)
        :param mask: (bsz, seq_l)
        :return:
        """

        bsz = mask.shape[0]
        concat_mask = mask[:, 1:-1]  # mask需要裁剪，因为原句没有CLS与SEP
        role_cnt = argument_start.shape[-1]

        trigger_start_losses, trigger_end_losses = [], []
        # trigger loss
        for i_batch in range(bsz):
            start_loss = F.binary_cross_entropy(trigger_start[i_batch], trigger_label_start[i_batch], reduction='none')
            end_loss = F.binary_cross_entropy(trigger_end[i_batch], trigger_label_end[i_batch], reduction='none')
            start_loss = torch.sum(start_loss * concat_mask[i_batch]) / torch.sum(concat_mask[i_batch])
            end_loss = torch.sum(end_loss * concat_mask[i_batch]) / torch.sum(concat_mask[i_batch])
            trigger_start_losses.append(start_loss)
            trigger_end_losses.append(end_loss)
        trigger_loss = sum(trigger_start_losses) + sum(trigger_end_losses)

        # argument loss
        argument_start_losses, argument_end_losses = [], []
        for i_batch in range(bsz):
            start_loss = F.binary_cross_entropy(argument_start[i_batch], argument_label_start[i_batch], reduction='none')
            end_loss = F.binary_cross_entropy(argument_end[i_batch], argument_label_end[i_batch], reduction='none')
            start_loss = torch.sum(start_loss * concat_mask[i_batch]) / (torch.sum(concat_mask[i_batch]) * role_cnt)
            end_loss = torch.sum(end_loss * concat_mask[i_batch]) / (torch.sum(concat_mask[i_batch]) * role_cnt)
            argument_start_losses.append(start_loss)
            argument_end_losses.append(end_loss)
        argument_loss = sum(argument_start_losses) + sum(argument_end_losses)
        # argument loss

        loss = trigger_loss + argument_loss
        return loss


class JointEE_Evaluator(BaseEvaluator):
    """
    f1 and ccks
    """
    def __init__(self):
        super(JointEE_Evaluator, self).__init__()
        self.gt_lst = []
        self.pred_lst = []
        self.ccks_evaluator = CcksEvaluator()
        self.f1_evaluator = EE_F1Evaluator()
        self.info_dict = {
            'main': 'f1score'
        }

    def eval_single(self, pred, gt):
        self.gt_lst.append(copy.deepcopy(gt))
        self.pred_lst.append(copy.deepcopy(pred))
        self.ccks_evaluator.eval_single(copy.deepcopy(gt), copy.deepcopy(pred))
        self.f1_evaluator.eval_single(copy.deepcopy(gt), copy.deepcopy(pred))

    def eval_step(self) -> Dict[str, Any]:
        ccks_result = self.ccks_evaluator.eval_step()
        f1_result = self.f1_evaluator.eval_step()
        ccks_result.update(f1_result)
        self.gt_lst = []
        self.pred_lst = []
        ccks_result['info'] = self.info_dict
        return ccks_result


class JointEE_Recorder(NaiveRecorder):
    """
    同时还会记录一些别的输出

    首先，为了了解loss的计算情况，我需要
    """
    def __init__(self, save_path):
        super(JointEE_Recorder, self).__init__(save_path='work/PLMEE')
        self.train_ckp_freq = 100

        self.record['loss']: List[Tuple[float, float, float]] = []  # total, trigger, argument
        self.loss_freq = 2
        self.record['trigger_preds']: List[Tuple[np.ndarray, np.ndarray]] = []  # trigger start and end
        self.record['argument_preds']: List[Tuple[np.ndarray, np.ndarray]] = []  # argument start and end
        self.record['trigger_spanses'] = []
        self.record['argument_spanses'] = []
        self.pred_freq = 20
        self.epoch, self.batch = 0, 0

    def train_checkin(self, step_info):
        self.epoch, self.batch = step_info

    def record_after_backward(self, loss_output: torch.Tensor, loss_func: nn.Module):
        """

        :param loss_output:
        :param loss_func:
        :return:
        """
        if self.batch % self.loss_freq == 0:
            self.record['loss'].append((float(loss_output), loss_func.last_trigger_loss, loss_func.last_argument_loss))

        if self.batch % self.pred_freq == 0:
            self.record['trigger_preds'].append(loss_func.last_trigger_preds)
            self.record['argument_preds'].append(loss_func.last_argument_preds)

    def record_after_evaluate(self, model: nn.Module, evaluator: BaseEvaluator, eval_result: Dict[str, Any]):
        super(JointEE_Recorder, self).record_after_evaluate(model, evaluator, eval_result)
        self.record['trigger_spanses'].append(model.trigger_spanses)
        self.record['argument_spanses'].append(model.argument_spanses)

    def train_checkpoint(self):
        if self.batch % self.train_ckp_freq == 0:
            pickle.dump(self.record, open(self.save_name, 'wb'))


class UseModel:
    def __init__(self, state_dict_path: str, init_params_path: str, use_gpu: bool = False, plm_path: str = EE_settings.default_plm_path, dataset_type: str = 'Duee'):
        # 首先加载初始化模型所使用的参数
        init_params = pickle.load(open(init_params_path, 'rb'))
        init_params['use_cuda'] = False
        self.model = JointEE(**init_params)
        self.model.eval()
        if not use_gpu:
            self.model.load_state_dict(torch.load(open(state_dict_path, 'rb'), map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(open(state_dict_path, 'rb'), map_location=torch.device('cuda')))

        if dataset_type == 'FewFC':
            self.event_types = EE_settings.event_types_full
            self.role_types = EE_settings.role_types
        elif dataset_type == 'Duee':
            self.event_types = EE_settings.duee_event_types
            self.role_types = EE_settings.duee_role_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')

        self.tokenizer = BertTokenizerFast.from_pretrained(plm_path)

    def __call__(self, sentence: str, event_types: List[str]):
        tokenized = self.tokenizer(sentence, padding=True, truncation=True, return_offsets_mapping=True)
        token_seq = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
        tokenized['token'] = token_seq

        result = self.model(sentences=[sentence], event_types=[event_types], offset_mappings=[tokenized['offset_mapping']])['pred']
        return result

"""
data process part
"""


def train_dataset_factory(data_dicts: List[dict], bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'Duee'):
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
        role_types = EE_settings.role_types
    elif dataset_type == 'Duee':
        event_types = EE_settings.duee_event_types
        role_types = EE_settings.duee_role_types
    else:
        raise Exception(f'{dataset_type}数据集不存在！')
    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """
        expect output:

        {
            sentence,
            event_type,
            trigger_span_gt,
        }, {
            trigger_label_start,
            trigger_label_end,
            argument_label_start,
            argument_label_end,
        }
        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        sentence_lst = data_dict['content']
        input_ids = data_dict['input_ids']
        max_seq_l = max(list(len(x) for x in input_ids)) - 2
        event_type_lst = data_dict['event_type']
        trigger_span_gt_lst = data_dict['trigger_token_span']
        arg_spans_lst = data_dict['argument_token_spans']

        trigger_label_start, trigger_label_end = torch.zeros((bsz, max_seq_l, 1)), torch.zeros((bsz, max_seq_l, 1))
        argument_label_start, argument_label_end = torch.zeros((bsz, max_seq_l, len(role_types))), torch.zeros((bsz, max_seq_l, len(role_types)))

        for i_batch in range(bsz):
            # trigger
            trigger_span = trigger_span_gt_lst[i_batch]
            trigger_label_start[i_batch][trigger_span[0] - 1] = 1
            trigger_label_end[i_batch][trigger_span[1] - 1] = 1
            # argument
            for e_role in arg_spans_lst[i_batch]:
                role_type_idx, role_span = e_role
                argument_label_start[i_batch][role_span[0] - 1][role_type_idx] = 1
                argument_label_end[i_batch][role_span[1] - 1][role_type_idx] = 1

        new_trigger_span_list = []
        for elem in trigger_span_gt_lst:
            new_trigger_span_list.append([elem[0] - 1, elem[1] - 1])

        return {
            'sentences': sentence_lst,
            'event_types': event_type_lst,
            'triggers': new_trigger_span_list
               }, {
            'trigger_label_start': trigger_label_start,
            'trigger_label_end': trigger_label_end,
            'argument_label_start': argument_label_start,
            'argument_label_end': argument_label_end
        }

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)

    return train_dataloader


def valid_dataset_factory(data_dicts: List[dict], dataset_type: str = 'Duee'):
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
        role_types = EE_settings.role_types
    elif dataset_type == 'Duee':
        event_types = EE_settings.duee_event_types
        role_types = EE_settings.duee_role_types
    else:
        raise Exception(f'{dataset_type}数据集不存在！')
    valid_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """
        input:
            - sentences
            - event_types
            - offset_mapping
        eval:
            - gt
        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        sentences = data_dict['content']
        events = data_dict['events'][0]
        event_types = []
        offset_mappings = data_dict['offset_mapping']

        gt = []
        for elem in lst:
            gt.append({
                'id': '',
                'content': elem['content'],
                'events': elem['events']
            })
            event_types.append(list(x['type'] for x in events))
        return {
            'sentences': sentences,
            'event_types': event_types,
            'offset_mappings': offset_mappings
               }, {
            'gt': gt[0]
        }

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return valid_dataloader


def dataset_factory(train_file: str, valid_file: str, bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'Duee'):
    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))
    print(f'dataset_type: {dataset_type}')

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle, dataset_type=dataset_type)
    valid_dataloader = valid_dataset_factory(valid_data_dicts, dataset_type=dataset_type)

    return train_dataloader, valid_dataloader


def tokenspans2events(event_types: StrList, triggers: List[SpanList], arguments: List[List[List[SpanList]]],
                      role_types: List[str], content: str = '', offset_mapping = None) -> SentenceWithEvent:
    """
    todo
    将token span转化为 origin span
    然后生成SentenceWithEvents

    由于模型中AEM和TEM忽略了input_ids开头的<CLS>和结尾的<SEP>，因此预测出的span相对tokenize阶段的offset_mapping，是偏前一位的
    因此需要给start与end均+1
    :param event_types:
    :param triggers:
    :param arguments:
    :param role_types:
    :param content:
    :param offset_mapping:
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
        # triggers[i] = list(map(lambda x: span_converter(x, token2origin), triggers[i]))
        temp = list(map(lambda x: tokenize_tools.tokenSpan_to_charSpan((x[0] + 1, x[1] + 1), offset_mapping), triggers[i]))
        new_temp = list((x[0], x[1] + 1) for x in temp)
        triggers[i] = new_temp

    # convert arguments span
    for i in range(len(arguments)):
        # List[List[SpanList]]
        for j in range(len(arguments[i])):
            # List[SpanList]
            for k in range(len(arguments[i][j])):
                # arguments[i][j][k] = list(map(lambda x: span_converter(x, token2origin), arguments[i][j][k]))
                temp = list(map(lambda x: tokenize_tools.tokenSpan_to_charSpan((x[0] + 1, x[1] + 1), offset_mapping), arguments[i][j][k]))
                new_temp = list((x[0], x[1] + 1) for x in temp)
                arguments[i][j][k] = new_temp


    result = spans2events(event_types, triggers, arguments, role_types, content)
    return result


def generate_trial_data(dataset_type: str):
    if dataset_type == 'Duee':
        train_file = 'temp_data/train.Duee.labeled.pk'
        valid_file = 'temp_data/valid.Duee.tokenized.pk'
    elif dataset_type == 'FewFC':
        # train_file = 'temp_data/train.PLMEE_Trigger.FewFC.labeled.pk'
        # valid_file = 'temp_data/valid.PLMEE_Trigger.FewFC.gt.pk'
        pass
    else:
        return None, None, None, None

    bsz = 4
    shuffle = False

    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle, dataset_type=dataset_type)
    valid_dataloader = valid_dataset_factory(valid_data_dicts, dataset_type=dataset_type)

    limit = 5
    train_data, valid_data = [], []
    for idx, (train_sample, valid_sample) in enumerate(list(zip(train_dataloader, valid_dataloader))):
        train_data.append(train_sample)
        valid_data.append(valid_sample)
    return train_dataloader, train_data, valid_dataloader, valid_data


model_registry = {
    'model': JointEE,
    'loss': JointEE_Loss,
    'evaluator': JointEE_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder,
    'use_model': UseModel
}


if __name__ == '__main__':
    train_dataloader, train_data, valid_dataloader, valid_data = generate_trial_data('Duee')
