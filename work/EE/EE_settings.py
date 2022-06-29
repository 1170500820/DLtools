# Data Definitions
role_types = [
    'obj-per',  # 0
    'amount',   # 1
    'title',    # 2
    'sub-org',  # 3
    'number',   # 4
    'way',      # 5
    'collateral',   # 6
    'obj',      # 7
    'target-company',   # 8
    'share-org',    # 9
    'sub-per',  # 10
    'sub',      # 11
    'data',     # 12
    'obj-org',  # 13
    'proportion',   # 14
    'date',     # 15
    'share-per',    # 16
    'institution',  # 17
    'money'     # 18
]
role_types_translate = {
 'obj-per': '目标任务',  # 0
 'amount': '总数',  # 1
 'title': '标题',  # 2
 'sub-org': '发起组织',  # 3
 'number': '数字',  # 4
 'way': '方式',  # 5
 'collateral': '质押品',  # 6
 'obj': '客体',  # 7
 'target-company': '目标公司',  # 8
 'share-org': '股份组织',  # 9
 'sub-per': '发起人',  # 10
 'sub': '主体',  # 11
 'data': '数据',  # 12
 'obj-org': '目标组织',  # 13
 'proportion': '比例',  # 14
 'date': '日期',  # 15
 'share-per': '股份人',  # 16
 'institution': '机构',  # 17
 'money': '金钱',  # 18
}
role_types_back_translate = {
 v: k for (k, v) in role_types_translate.items()
}

role_index = {v: i for i, v in enumerate(role_types)}
event_types_initial = [
    '质押',
    '股份股权转让',
    '起诉',
    '投资',
    '减持'
]
event_types_initial_index = {v: i for i, v in enumerate(event_types_initial)}
event_types_full = [
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
]
event_types_full_index = {v: i for i, v in enumerate(event_types_full)}


event_available_roles = {
    '质押': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'collateral', 'date', 'money', 'number', 'proportion'},
    '股份股权转让': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'collateral', 'date', 'money', 'number', 'proportion',
               'target-company'},
    '起诉': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'date'},
    '投资': {'sub', 'obj', 'money', 'date'},
    '减持': {'sub', 'obj', 'title', 'date', 'share-per', 'share-org'},
    '收购': {'sub-org', 'sub-per', 'obj-org', 'way', 'date', 'money', 'number', 'proportion', 'target-company'},
    '担保': {'sub-org', 'sub-per', 'obj-org', 'way', 'amount', 'date'},
    '中标': {'sub', 'obj', 'amount', 'date'},
    '签署合同': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'amount', 'date'},
    '判决': {'institution', 'sub-org', 'sub-per', 'obj-org', 'obj-per', 'date', 'money'}
}


# duee
duee_role_types = [
    '上市企业',
 '上映影视',
 '上映方',
 '下架产品',
 '下架方',
 '举报发起方',
 '举报对象',
 '交易物',
 '产子者',
 '会见主体',
 '会见对象',
 '停职人员',
 '入狱者',
 '冠军',
 '出售价格',
 '出售方',
 '出生者',
 '出轨对象',
 '出轨方',
 '分手双方',
 '刑期',
 '加息幅度',
 '加息机构',
 '加盟者',
 '原告',
 '原所属组织',
 '参礼人员',
 '发布产品',
 '发布方',
 '受伤人数',
 '召回内容',
 '召回方',
 '地点',
 '坍塌主体',
 '失联者',
 '夺冠赛事',
 '奖项',
 '庆祝方',
 '开庭案件',
 '开庭法院',
 '怀孕者',
 '所加盟组织',
 '所属组织',
 '执法机构',
 '拘捕者',
 '探班主体',
 '探班对象',
 '收购方',
 '时间',
 '晋级方',
 '晋级赛事',
 '死亡人数',
 '死者',
 '死者年龄',
 '求婚对象',
 '求婚者',
 '活动名称',
 '涨价幅度',
 '涨价方',
 '涨价物',
 '涨停股票',
 '游行人数',
 '游行组织',
 '点赞对象',
 '点赞方',
 '生日方',
 '生日方年龄',
 '禁赛时长',
 '禁赛机构',
 '离婚双方',
 '离职者',
 '立案对象',
 '立案机构',
 '约谈发起方',
 '约谈对象',
 '结婚双方',
 '罚款对象',
 '罚款金额',
 '罢工人员',
 '罢工人数',
 '胜者',
 '致谢人',
 '获奖人',
 '融资方',
 '融资轮次',
 '融资金额',
 '被下架方',
 '被告',
 '被感谢人',
 '被拘捕者',
 '被禁赛人员',
 '被解约方',
 '被解雇人员',
 '袭击对象',
 '袭击者',
 '裁员人数',
 '裁员方',
 '解散方',
 '解约方',
 '解雇方',
 '订婚主体',
 '败者',
 '赛事名称',
 '跌停股票',
 '跟投方',
 '退出方',
 '退役者',
 '退赛方',
 '退赛赛事',
 '道歉对象',
 '道歉者',
 '降价幅度',
 '降价方',
 '降价物',
 '降息幅度',
 '降息机构',
 '震中',
 '震源深度',
 '震级',
 '颁奖机构',
 '领投方']
duee_role_index = {v: i for i, v in enumerate(duee_role_types)}

duee_event_types = ['交往-会见',
 '交往-感谢',
 '交往-探班',
 '交往-点赞',
 '交往-道歉',
 '产品行为-上映',
 '产品行为-下架',
 '产品行为-发布',
 '产品行为-召回',
 '产品行为-获奖',
 '人生-产子/女',
 '人生-出轨',
 '人生-分手',
 '人生-失联',
 '人生-婚礼',
 '人生-庆生',
 '人生-怀孕',
 '人生-死亡',
 '人生-求婚',
 '人生-离婚',
 '人生-结婚',
 '人生-订婚',
 '司法行为-举报',
 '司法行为-入狱',
 '司法行为-开庭',
 '司法行为-拘捕',
 '司法行为-立案',
 '司法行为-约谈',
 '司法行为-罚款',
 '司法行为-起诉',
 '灾害/意外-地震',
 '灾害/意外-坍/垮塌',
 '灾害/意外-坠机',
 '灾害/意外-洪灾',
 '灾害/意外-爆炸',
 '灾害/意外-袭击',
 '灾害/意外-起火',
 '灾害/意外-车祸',
 '竞赛行为-夺冠',
 '竞赛行为-晋级',
 '竞赛行为-禁赛',
 '竞赛行为-胜负',
 '竞赛行为-退役',
 '竞赛行为-退赛',
 '组织关系-停职',
 '组织关系-加盟',
 '组织关系-裁员',
 '组织关系-解散',
 '组织关系-解约',
 '组织关系-解雇',
 '组织关系-辞/离职',
 '组织关系-退出',
 '组织行为-开幕',
 '组织行为-游行',
 '组织行为-罢工',
 '组织行为-闭幕',
 '财经/交易-上市',
 '财经/交易-出售/收购',
 '财经/交易-加息',
 '财经/交易-涨价',
 '财经/交易-涨停',
 '财经/交易-融资',
 '财经/交易-跌停',
 '财经/交易-降价',
 '财经/交易-降息']
duee_event_index = {v: i for i, v in enumerate(duee_event_types)}

# duee的主要类型，也就是‘-’前面的部分
duee_main_event_types = ['组织行为', '交往', '组织关系', '人生', '竞赛行为', '产品行为', '财经/交易', '灾害/意外', '司法行为']
duee_main_event_index = {v: i for i, v in enumerate(duee_main_event_types)}

# duee的主要类型与次要类型的对应表
duee_sub_types = {'交往': ['会见', '感谢', '探班', '点赞', '道歉'],
 '产品行为': ['上映', '下架', '发布', '召回', '获奖'],
 '人生': ['产子/女',
  '出轨',
  '分手',
  '失联',
  '婚礼',
  '庆生',
  '怀孕',
  '死亡',
  '求婚',
  '离婚',
  '结婚',
  '订婚'],
 '司法行为': ['举报', '入狱', '开庭', '拘捕', '立案', '约谈', '罚款', '起诉'],
 '灾害/意外': ['地震', '坍/垮塌', '坠机', '洪灾', '爆炸', '袭击', '起火', '车祸'],
 '竞赛行为': ['夺冠', '晋级', '禁赛', '胜负', '退役', '退赛'],
 '组织关系': ['停职', '加盟', '裁员', '解散', '解约', '解雇', '辞/离职', '退出'],
 '组织行为': ['开幕', '游行', '罢工', '闭幕'],
 '财经/交易': ['上市', '出售/收购', '加息', '涨价', '涨停', '融资', '跌停', '降价', '降息']}



duee_event_available_roles = {
 '财经/交易-出售/收购': ['时间', '出售方', '交易物', '出售价格', '收购方'],
 '财经/交易-跌停': ['时间', '跌停股票'],
 '财经/交易-加息': ['时间', '加息幅度', '加息机构'],
 '财经/交易-降价': ['时间', '降价方', '降价物', '降价幅度'],
 '财经/交易-降息': ['时间', '降息幅度', '降息机构'],
 '财经/交易-融资': ['时间', '跟投方', '领投方', '融资轮次', '融资金额', '融资方'],
 '财经/交易-上市': ['时间', '地点', '上市企业', '融资金额'],
 '财经/交易-涨价': ['时间', '涨价幅度', '涨价物', '涨价方'],
 '财经/交易-涨停': ['时间', '涨停股票'],
 '产品行为-发布': ['时间', '发布产品', '发布方'],
 '产品行为-获奖': ['时间', '获奖人', '奖项', '颁奖机构'],
 '产品行为-上映': ['时间', '上映方', '上映影视'],
 '产品行为-下架': ['时间', '下架产品', '被下架方', '下架方'],
 '产品行为-召回': ['时间', '召回内容', '召回方'],
 '交往-道歉': ['时间', '道歉对象', '道歉者'],
 '交往-点赞': ['时间', '点赞方', '点赞对象'],
 '交往-感谢': ['时间', '致谢人', '被感谢人'],
 '交往-会见': ['时间', '地点', '会见主体', '会见对象'],
 '交往-探班': ['时间', '探班主体', '探班对象'],
 '竞赛行为-夺冠': ['时间', '冠军', '夺冠赛事'],
 '竞赛行为-晋级': ['时间', '晋级方', '晋级赛事'],
 '竞赛行为-禁赛': ['时间', '禁赛时长', '被禁赛人员', '禁赛机构'],
 '竞赛行为-胜负': ['时间', '败者', '胜者', '赛事名称'],
 '竞赛行为-退赛': ['时间', '退赛赛事', '退赛方'],
 '竞赛行为-退役': ['时间', '退役者'],
 '人生-产子/女': ['时间', '产子者', '出生者'],
 '人生-出轨': ['时间', '出轨方', '出轨对象'],
 '人生-订婚': ['时间', '订婚主体'],
 '人生-分手': ['时间', '分手双方'],
 '人生-怀孕': ['时间', '怀孕者'],
 '人生-婚礼': ['时间', '地点', '参礼人员', '结婚双方'],
 '人生-结婚': ['时间', '结婚双方'],
 '人生-离婚': ['时间', '离婚双方'],
 '人生-庆生': ['时间', '生日方', '生日方年龄', '庆祝方'],
 '人生-求婚': ['时间', '求婚者', '求婚对象'],
 '人生-失联': ['时间', '地点', '失联者'],
 '人生-死亡': ['时间', '地点', '死者年龄', '死者'],
 '司法行为-罚款': ['时间', '罚款对象', '执法机构', '罚款金额'],
 '司法行为-拘捕': ['时间', '拘捕者', '被拘捕者'],
 '司法行为-举报': ['时间', '举报发起方', '举报对象'],
 '司法行为-开庭': ['时间', '开庭法院', '开庭案件'],
 '司法行为-立案': ['时间', '立案机构', '立案对象'],
 '司法行为-起诉': ['时间', '被告', '原告'],
 '司法行为-入狱': ['时间', '入狱者', '刑期'],
 '司法行为-约谈': ['时间', '约谈对象', '约谈发起方'],
 '灾害/意外-爆炸': ['时间', '地点', '死亡人数', '受伤人数'],
 '灾害/意外-车祸': ['时间', '地点', '死亡人数', '受伤人数'],
 '灾害/意外-地震': ['时间', '死亡人数', '震级', '震源深度', '震中', '受伤人数'],
 '灾害/意外-洪灾': ['时间', '地点', '死亡人数', '受伤人数'],
 '灾害/意外-起火': ['时间', '地点', '死亡人数', '受伤人数'],
 '灾害/意外-坍/垮塌': ['时间', '坍塌主体', '死亡人数', '受伤人数'],
 '灾害/意外-袭击': ['时间', '地点', '袭击对象', '死亡人数', '袭击者', '受伤人数'],
 '灾害/意外-坠机': ['时间', '地点', '死亡人数', '受伤人数'],
 '组织关系-裁员': ['时间', '裁员方', '裁员人数'],
 '组织关系-辞/离职': ['时间', '离职者', '原所属组织'],
 '组织关系-加盟': ['时间', '加盟者', '所加盟组织'],
 '组织关系-解雇': ['时间', '解雇方', '被解雇人员'],
 '组织关系-解散': ['时间', '解散方'],
 '组织关系-解约': ['时间', '被解约方', '解约方'],
 '组织关系-停职': ['时间', '所属组织', '停职人员'],
 '组织关系-退出': ['时间', '退出方', '原所属组织'],
 '组织行为-罢工': ['时间', '所属组织', '罢工人数', '罢工人员'],
 '组织行为-闭幕': ['时间', '地点', '活动名称'],
 '组织行为-开幕': ['时间', '地点', '活动名称'],
 '组织行为-游行': ['时间', '地点', '游行组织', '游行人数']}


default_plm_path = 'bert-base-chinese'


# train
plm_lr = 2e-5
others_lr = 1e-4

default_dropout_prob = 0.3

default_bsz = 8
default_shuffle = True

event_detection_threshold = 0.5


"""数据预处理参数
"""

max_sentence_length = 256