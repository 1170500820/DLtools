from bert_event_detection import UseModel

state_dict_path = '../../../checkpoint/save.state_dict.BertED.Duee.Default.best.pth'
init_params_path = '../../../checkpoint/save.init_patams.BertED.Duee.Default.best.pk'
plm_path = 'bert-base-chinese'
dataset_type = 'Duee'

examples = [
    "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！",
    '离开德云社，曹云金创办了听云轩，但是生意一直也就那么回事。6月离婚后，更是受到负面消息影响，他的门票最低才49元，而岳云鹏的专场最低是299元。',
    "在昨天晚上7点钟左右，108国道故市段发生重大车祸，一辆蓝色丰田撞上了电动车，电动车车主（一位老年人）直接当场死亡。"
]


def main():
    print('加载模型中...', end='')
    model = UseModel(
        state_dict_path=state_dict_path,
        init_params_path=init_params_path,
        use_gpu=False,
        plm_path=plm_path,
        dataset_type=dataset_type
    )
    print('done')

    results = []
    for elem in examples:
        results.append(model(elem))

    print(results)


if __name__ == '__main__':
    main()

