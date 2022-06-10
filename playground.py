import torch
from transformers import BertTokenizerFast

plm_path = 'bert-base-chinese'
tokenizer = BertTokenizerFast.from_pretrained(plm_path)

ts = ['质押', '投资', '股份股权转让']
ss = ['我在人民广场吃炸鸡', '规格严格，功夫到家', '我们的征途是星辰大海']

concat = []
for e_t, e_s in zip(ts, ss):
    concat.append(f'{e_t}[SEP]{e_s}')

tokenized = tokenizer(concat, padding=True, return_offsets_mapping=True, return_tensors='pt')

seq_positions_1 = (tokenized['input_ids'] == 102)
seq_positions_2 = seq_positions_1.nonzero()
seq_positions = seq_positions_2.T[1]


