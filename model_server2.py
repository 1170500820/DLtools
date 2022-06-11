import asyncio
import torch
import websockets
import functools
from bs4 import BeautifulSoup
import numpy as np

# from achieve.model import NeuralNet
# from achieve.data import get_need_text
from transformers import BertModel, BertTokenizer, BertConfig

model_path = ""
rw_model_path = ""
dz_model_path = ""
tokenizer_path = ""
config_path = ""


def sentence_change2_input(tokenizer, sentence):
    pair = sentence.strip().split(',')
    if len(pair) < 2:
        print(pair)
    bf_title = BeautifulSoup(pair[0], 'html.parser')
    bf_content = BeautifulSoup(pair[1:], 'html.parser')
    title = bf_title.get_text().replace("{", "").replace("}", "").replace("$", "")
    title = title if title is not None else "None_title"
    content = bf_content.get_text().replace("{", "").replace("}", "").replace("$", "")
    content = content if content is not None else "None_content"
    # text = get_need_text(title, content, similarity_calculation=True)
    # result = tokenizer(text)
    # return result
    return None

class Predictor:

    def __init__(self):
        # self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        # self.config = BertConfig.from_pretrained(config_path)
        # self.model_rw = NeuralNet(BertModel, None, model_path, self.config)
        # self.model_rw.load_state_dict(torch.load(rw_model_path))
        # self.model_dz = NeuralNet(BertModel, None, model_path, self.config)
        # self.model_dz.load_state_dict(torch.load(dz_model_path))
        print('-- loaded --\n')

    def __call__(self, sentence: str) -> dict:
        # self.model_rw.eval()
        # self.model_dz.eval()
        # with torch.no_grad():
        #     inputs = sentence_change2_input(self.tokenizer, sentence)
        #     logits_rw = self.model_rw(**inputs)["logits"]
        #     logits_dz = self.model_dz(**inputs)["logits"]
        #     logits_rw = logits_rw.detach().cpu().numpy()
        #     logits_dz = logits_dz.detach().cpu().numpy()
        #     whether_rw = np.argmax(logits_rw, axis=1)
        #     whether_dz = np.argmax(logits_dz, axis=1)
        pred_result = {"content": sentence, "是否软文": "是", "是否低质": "否"}
        # pred_result = {"content": sentence, "是否软文": "是" if whether_rw == 1 else "否", "是否低质": "是" if whether_dz == 1 else "否"}
        return pred_result


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
    async with websockets.serve(functools.partial(hello, lock=lock, func=predictor), "0.0.0.0", 9871):
        await asyncio.Future()  # run forever

asyncio.run(main())
