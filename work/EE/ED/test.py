from bert_event_detection import UseModel
from work.EE.EE_utils import load_jsonl, dump_jsonl
from tqdm import tqdm

state_dict_path = '../../../checkpoint/save.state_dict.BertED.Duee.Default.best.pth'
init_params_path = '../../../checkpoint/save.init_params.BertED.Duee.Default.best.pk'
plm_path = 'bert-base-chinese'
dataset_type = 'Duee'

test_file_dir = '../../../data/NLP/EventExtraction/duee/duee_test2.json/'
test_file_name = 'duee_test2.json'

examples = [
    "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！",
    '离开德云社，曹云金创办了听云轩，但是生意一直也就那么回事。6月离婚后，更是受到负面消息影响，他的门票最低才49元，而岳云鹏的专场最低是299元。',
    "在昨天晚上7点钟左右，108国道故市段发生重大车祸，一辆蓝色丰田撞上了电动车，电动车车主（一位老年人）直接当场死亡。"
]


def predict_test():
    test_data = load_jsonl(test_file_dir + test_file_name)
    model = UseModel(
        state_dict_path=state_dict_path,
        init_params_path=init_params_path,
        use_gpu=True,
        plm_path=plm_path,
        dataset_type=dataset_type
    )
    results = []
    for elem in tqdm(test_data):
        content, cid = elem['text'], elem['id']
        e_type = model(content)
        results.append({
            'text': content,
            'id': cid,
            "type": e_type
        })
    dump_jsonl(results, test_file_dir + 'test.type.duee.json')


def main():
    print('加载模型中...', end='')
    model = UseModel(
        state_dict_path=state_dict_path,
        init_params_path=init_params_path,
        use_gpu=True,
        plm_path=plm_path,
        dataset_type=dataset_type
    )
    print('done')

    results = []
    for elem in examples:
        results.append(model(elem))

    print(results)


if __name__ == '__main__':
    predict_test()

