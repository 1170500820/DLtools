"""
一个最简的模拟server端

正经的server做的事是加载两个模型在本地，输入是一个句子str，输出是一个dict：预测结果。然后把这个dict，应该是用json格式，发回去

模拟的话，只需要实现：接收str，根据str返回一个json即可。
后面只要把这个func替换一下就行
"""
import asyncio
import websockets
import functools

from work.EE.PLMEE import UseModel as ED_model
from work.EE.PLMEE import UseModel as EE_joint


def str2json(s: str):
    return {'str': s}


def fill_word(event_dict: dict):
    """
    对于一个SentenceWithEvents类型，通过span补全其word字段
    :param event_dict:
    :return:
    """
    content = event_dict['content']
    events = event_dict['events']
    for elem_event in events:
        for elem_mention in elem_event['mentions']:
            span = elem_mention['span']
            elem_mention['word'] = content[span[0]: span[1]]
    return event_dict


def convert_event(event_dict: dict):
    content = event_dict['content']
    events = event_dict['events']
    new_events = []
    for elem in events:
        # elem: {type: , mentions: ,}
        new_event = {
            "type": elem['type'],
            "trigger": '-',
            "trigger_span": '-'}
        mentions = elem['mentions']
        triggers, arguments = [], []
        for elem_mention in mentions:
            if elem_mention['role'] == 'trigger':
                new_event['trigger'] = elem_mention['word']
                new_event['trigger_span'] = elem_mention['span']
            else:
                arguments.append(elem_mention)
        if len(arguments) == 0:  # 保证arguments内至少有一个元素
            arguments.append({"word": '-', 'role': '-', 'span': '-'})
        new_event['mentions'] = arguments
        new_events.append(new_event)
    return {
        "content": content,
        "events": new_events
    }


class Predictor:
    """
    把这个文件放在DLtools目录下，加载两个模型即可，这部分真的很简单
    """
    def __init__(self):
        self.ed_model = ED_model('work/EE/PLMEE/checkpoint/EventDetection-save-20.pth',
                                 'work/EE/PLMEE/checkpoint/EventDetection-init_param.pk')
        self.ee_model = EE_joint('work/EE/PLMEE/checkpoint/JointEE-save-15.pth',
                                 'work/EE/PLMEE/checkpoint/JointEE-init_param.pk')
        print('-\n-\nloaded\n-\n-')

    def __call__(self, sentence: str) -> dict:
        pred_event_types = self.ed_model(sentence)
        pred_result = self.ee_model(sentence, pred_event_types)
        sample = {
            "id": "aab68a78f7e138ced02a25ff1de2ca76",
            "content": "财华社讯】招商局港口(00144-HK)于9月20日公布,基于现有资料,决定不接纳间接控股股东布罗德福提出收购大连港(02880-HK)全部H股之可能强制性无条件现金要约。",
            "events": [
                {"type": "股份股权转让", "mentions": [
                    {"word": "收购", "span": [53, 55], "role": "trigger"},
                    {"word": "招商局港口", "span": [5, 10], "role": "obj-org"},
                    {"word": "布罗德福", "span": [47, 51], "role": "obj-per"},
                    {"word": "H股", "span": [70, 72], "role": "collateral"},
                    {"word": "9月20日", "span": [21, 26], "role": "date"},
                    {"word": "大连港", "span": [55, 58], "role": "target-company"}]},
                {"type": "投资", "mentions": [
                    {"word": "收购", "span": [53, 55], "role": "trigger"},
                    {"word": "招商局港口", "span": [5, 10], "role": "sub"},
                    {"word": "布罗德福", "span": [47, 51], "role": "sub"},
                    {"word": "9月20日", "span": [21, 26], "role": "date"},
                    {"word": "大连港", "span": [55, 58], "role": "obj"}]}]}
        filled_result = fill_word(pred_result)
        if len(filled_result['events']) == 0:
            filled_result['events'] = [{'type': '-', 'mentions': []}]
        return convert_event(pred_result)
# WS server example



async def hello(websocket, path, lock: asyncio.Lock, func):
    sentence = await websocket.recv()
    print(f"<<< {sentence}")

    await lock.acquire()
    try:
        extracted = f"{func(sentence)}"
    finally:
        lock.release()
    await websocket.send(extracted)
    print(f">>> {extracted}")

async def main():
    lock = asyncio.Lock()
    predictor = Predictor()
    async with websockets.serve(functools.partial(hello, lock=lock, func=predictor), "localhost", 9870):
        await asyncio.Future()  # run forever

asyncio.run(main())
