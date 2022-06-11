from jointee_mask import UseModel
from work.EE.EE_utils import load_jsonl, dump_jsonl
from work.EE import EE_settings
from tqdm import tqdm


state_dict_path = '../../../checkpoint/save.state_dict.JointEE.Duee.mask.lambda-05.best.pth'
init_params_path = '../../../checkpoint/save.init_params.JointEE.Duee.mask.lambda-05.best.pk'
plm_path = 'bert-base-chinese'
dataset_type = 'Duee'

test_file_dir = '../../../data/NLP/EventExtraction/duee/duee_test2.json/'
test_file_name = 'test.type.duee.json'


examples = [
    {
        'sentence': "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！",
        'event_types': ["组织关系-裁员"]
    },
    {
        'sentence': '离开德云社，曹云金创办了听云轩，但是生意一直也就那么回事。6月离婚后，更是受到负面消息影响，他的门票最低才49元，而岳云鹏的专场最低是299元。',
        'event_types': ["组织关系-退出", "人生-离婚"]
    }, {
        'sentence': "在昨天晚上7点钟左右，108国道故市段发生重大车祸，一辆蓝色丰田撞上了电动车，电动车车主（一位老年人）直接当场死亡。",
        'event_types': ["灾害/意外-车祸", "人生-死亡"]
    }
]


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


def predict_test():
    test_data = load_jsonl(test_file_dir + test_file_name)
    model = UseModel(
        state_dict_path=state_dict_path,
        init_params_path=init_params_path,
        use_gpu=True,
        plm_path=plm_path,
        dataset_type=dataset_type)
    results = []
    for elem in tqdm(test_data):
        text, cid, etypes = elem['text'], elem['id'], elem['type']
        predicted = model(text, etypes)
        converted = convert_for_submit(predicted)
        results.append({
            'id': cid,
            'event_list': converted
        })
    dump_jsonl(results, test_file_dir + 'duee.mask.json')


def main():
    print('加载模型中...', end='')
    model = UseModel(
        state_dict_path=state_dict_path,
        init_params_path=init_params_path,
        use_gpu=False,
        plm_path=plm_path,
        dataset_type=dataset_type)
    print('done')

    results = []

    for elem in examples:
        results.append(model(elem['sentence'], elem['event_types']))

    print(results)


if __name__ == '__main__':
    predict_test()
