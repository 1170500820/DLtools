"""
考虑了mask的JointEE
"""
from work.EE.JointEE_rebuild.jointee import *


class JointEE_Mask(nn.Module):
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
        super(JointEE_Mask, self).__init__()
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

