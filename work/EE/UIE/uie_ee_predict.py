"""
使用UIE预测
"""
from paddlenlp import Taskflow
import paddle
from work.EE import EE_settings
import json

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


fewfc_test_file_dir = '../../../data/NLP/EventExtraction/FewFC-main/'
fewfc_test_file = 'merged_test.json'
fewfc_predict_result = 'merged_test.uie_result.json'


duee_schema = {}
for key, value in EE_settings.duee_event_available_roles:
    # 因为Duee的事件类型的结构本来就包含'-'，所以这里的任务名也加了一个'-'来保持形式一致
    duee_schema[key + '-触发词'] = value


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
    d = list(json.loads(x) for x in open(fewfc_test_file_dir + fewfc_test_file, 'r', encoding='utf-8').read().strip().split('\n'))
    contents = list(x['content'] for x in d)
    ee = Taskflow('information_extraction', schema=schema, model=model_type)


def fewfc_try():
    ee = Taskflow('information_extraction', schema=fewfc_schema, model=model_type)
    result = ee(fewfc_sample['content'])
    return ee, result


def main():
    pass


if __name__ == '__main__':
    ee, result = fewfc_try()
