"""
只检测论元，不分类。
"""
from work.EE.JointEE_rebuild.jointee import *
from work.EE.PLMEE.argument_extraction_model import ArgumentDetectionModel_woSyntactic
from evaluate.evaluator import fullword_f1_count_trigger


def tokenspans2events_wo_role_type(event_types: StrList, triggers: List[SpanList], arguments: List[List[SpanList]], content: str = '', offset_mapping = None):
    """
    将token span转换为不包含论元类型的SentenceWithEvents
    返回的仍然是正常的SentenceWithEvents，但是role一栏除了trigger之外就为空
    :param event_types:
    :param triggers:
    :param arguments:
    :param content:
    :param offset_mapping:
    :return:
    """

    def spans2events_wo_role_type(event_types: StrList, triggers: List[SpanList], arguments: List[List[SpanList]] , content: str = '') -> SentenceWithEvent:
        cur_events = []
        for (elem_event, elem_triggers, elem_arguments) in zip(event_types, triggers, arguments):
            # elem_event: str
            # elem_trigger: SpanList
            # elem_arguments: List[SpanList]
            for (elem_trigger, elem_argument) in zip(elem_triggers, elem_arguments):
                # elem_trigger: Span
                # elem_argument: SpanList
                cur_mentions = [{'word': "", 'span': list(elem_trigger), 'role': 'trigger'}]

                for elem_arg in elem_argument:
                    # elem_arg: Span
                    cur_mentions.append({"word": "", "span": list(elem_arg), "role": ''})
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
            # arguments[i][j][k] = list(map(lambda x: span_converter(x, token2origin), arguments[i][j][k]))
            temp = list(map(lambda x: tokenize_tools.tokenSpan_to_charSpan((x[0] + 1, x[1] + 1), offset_mapping), arguments[i][j]))
            new_temp = list((x[0], x[1] + 1) for x in temp)
            arguments[i][j] = new_temp


    result = spans2events_wo_role_type(event_types, triggers, arguments, content)
    return result

class JointEE_DetectRole(nn.Module):
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
        super(JointEE_DetectRole, self).__init__()
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
        self.aem = ArgumentDetectionModel_woSyntactic(
            n_head=self.n_head,
            d_head=self.d_head,
            hidden_size=self.hidden_size,
            dropout_prob=self.hidden_dropout_prob)

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
            # both (bsz, max_seq_l, len(role_types))
            return {
                "trigger_start": trigger_start,  # (bsz, seq_l, 1)
                "trigger_end": trigger_end,  # (bsz. seq_l, 1)
                "argument_start": argument_start,  # (bsz, seq_l, 1)
                "argument_end": argument_end,  # (bsz, seq_l, 1)
            }
        else:  # eval mode
            # sentence_types: List[StrList] during evaluating
            if len(sentences) != 1:
                raise Exception('eval模式下一次只预测单个句子!')
            cur_spanses = []  # List[SpanList]
            arg_spanses = []  # List[List[SpanList]]
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

                for elem_trigger_span in cur_spans:
                    arg_H_styp, RPE = self.trigger_sentence_representation(H_styp, [elem_trigger_span])
                    argument_start_tensor, argument_end_tensor = self.aem(arg_H_styp, RPE)
                    # (1, max_seq_l, 1)
                    argument_start_tensor = argument_start_tensor.squeeze().T  # (max_seq_l)
                    argument_end_tensor = argument_end_tensor.squeeze().T  # (max_seq_l)

                    argument_start_result = (argument_start_tensor > self.argument_threshold).int().tolist()
                    argument_end_result = (argument_end_tensor > self.argument_threshold).int().tolist()
                    argument_spans: SpanList = []
                    cur_arg_spans: SpanList = tools.argument_span_determination(argument_start_result, argument_end_result, argument_start_tensor.tolist(), argument_end_tensor.tolist())
                    argument_spans.append(cur_arg_spans)
                cur_spanses.append(cur_spans)
                arg_spanses.append(argument_spans)
            self.trigger_spanses.append(cur_spanses)
            self.argument_spanses.append(arg_spanses)
            result = tokenspans2events_wo_role_type(event_types[0], cur_spanses, arg_spanses, sentences[0], offset_mappings[0])
            return {"pred": result}

    def get_optimizers(self):
        repr_plm_params, repr_other_params = self.sentence_representation.PLM.parameters(), self.sentence_representation.CLN.parameters()
        trigger_repr_params = self.trigger_sentence_representation.parameters()
        aem_params = self.aem.parameters()
        tem_params = self.tem.parameters()
        plm_optimizer = AdamW(params=repr_plm_params, lr=self.plm_lr)
        others_optimizer = AdamW(params=chain(aem_params, tem_params, repr_other_params, trigger_repr_params), lr=self.others_lr)
        return [plm_optimizer, others_optimizer]


class JointEE_DetectRole_Loss(nn.Module):
    def __init__(self, lambd=0.3, alpha=0.3, gamma=2):
        """

        :param lambd: loss = lambd * trigger + (1 - lambd) * argument
        :param alpha: focal weight param
        :param gamma: focal weight param
        """
        super(JointEE_DetectRole_Loss, self).__init__()
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
        :param argument_start: (bsz, seq_l, 1)
        :param argument_end: (bsz, seq_l, 1)
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


class JointEE_DetectRole_Evaluator(BaseEvaluator):
    """
    f1 and ccks
    """
    def __init__(self):
        super(JointEE_DetectRole_Evaluator, self).__init__()
        self.gt_lst = []
        self.pred_lst = []
        self.info_dict = {
            'main': 'argument f1'
        }

    def eval_single(self, pred, gt):
        self.gt_lst.append(copy.deepcopy(gt))
        self.pred_lst.append(copy.deepcopy(pred))

    def eval_step(self) -> Dict[str, Any]:

        # 评测计算过程
        trig_total, trig_predict, trig_correct = 0, 0, 0
        arg_total, arg_predict, arg_correct = 0, 0, 0
        for idx, elem_gt in enumerate(self.gt_lst):
            elem_pred = self.pred_lst[idx]
            elem_trig_total, elem_trig_predict, elem_trig_correct = fullword_f1_count_trigger(elem_pred['events'],
                                                                                              elem_gt['events'])
            trig_total += elem_trig_total
            trig_predict += elem_trig_predict
            trig_correct += elem_trig_correct
            pred_roles = set()
            gt_roles = set()
            for elem_event in elem_pred['events']:
                for elem_mention in elem_event['mentions']:
                    if elem_mention['role'] == '':
                        pred_roles.add(tuple(elem_mention['span']))
            for elem_event in elem_gt['events']:
                for elem_mention in elem_event['mentions']:
                    if elem_mention['role'] == '':
                        gt_roles.add(tuple(elem_mention['span']))
            arg_total += len(gt_roles)
            arg_predict += len(pred_roles)
            arg_correct += len(gt_roles.intersection(pred_roles))
        trig_precision = trig_correct / trig_predict if trig_predict != 0 else 0
        trig_recall = trig_correct / trig_total if trig_total != 0 else 0
        trig_f1 = 2 * trig_precision * trig_recall / (trig_precision + trig_recall) if (
                                                                                               trig_precision + trig_recall) != 0 else 0
        arg_precision = arg_correct / arg_predict if arg_predict != 0 else 0
        arg_recall = arg_correct / arg_total if arg_total != 0 else 0
        arg_f1 = 2 * arg_precision * arg_recall / (arg_precision + arg_recall) if (
                                                                                              arg_precision + arg_recall) != 0 else 0

        f1_result = {
            "trigger precision": trig_precision,
            'trigger recall': trig_recall,
            'trigger f1': trig_f1,
            'argument precision': arg_precision,
            'argument recall': arg_recall,
            'argument f1': arg_f1
        }
        self.gt_lst = []
        self.pred_lst = []
        f1_result['info'] = self.info_dict
        return f1_result


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
        argument_label_start, argument_label_end = torch.zeros((bsz, max_seq_l, 1)), torch.zeros((bsz, max_seq_l, 1))

        for i_batch in range(bsz):
            # trigger
            trigger_span = trigger_span_gt_lst[i_batch]
            trigger_label_start[i_batch][trigger_span[0] - 1][0] = 1
            trigger_label_end[i_batch][trigger_span[1] - 1][0] = 1
            # argument
            for e_role in arg_spans_lst[i_batch]:
                role_type_idx, role_span = e_role
                argument_label_start[i_batch][role_span[0] - 1][0] = 1
                argument_label_end[i_batch][role_span[1] - 1][0] = 1

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
        for elem_event in events:
            mentions = elem_event['mentions']
            for elem_mention in mentions:
                if elem_mention['role'] != 'trigger':
                    elem_mention['role'] = ''
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


class UseModel:
    pass

model_registry = {
    'model': JointEE_DetectRole,
    'loss': JointEE_DetectRole_Loss,
    'evaluator': JointEE_DetectRole_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder,
    'use_model': UseModel
}


if __name__ == '__main__':
    train_dataloader, train_data, valid_dataloader, valid_data = generate_trial_data('Duee')

