from itertools import product

default_plm = 'bert-base-chinese'

# data def
msra_ner_tags = [
    'O',
    'B-LOC',
    "I-LOC",
    "B-ORG",
    'I-ORG',
    "B-PER",
    "I-PER"
]
msra_ner_tags_idx = {s: i for i, s in enumerate(msra_ner_tags)}

weibo_ner_tags = ['O'] + list(x[0] + x[1] for x in product(['B', 'I'], [
    'PER.NAM', 'PER.NOM', 'LOC.NAM', 'LOC.NOM', 'GPE.NAM', 'ORG.NAM', 'ORG.NOM'
]))
weibo_nert_tags_idx = {s: i for i, s in enumerate(weibo_ner_tags)}


# model
embedding_dim = 300
num_layers = 2

# train
plm_lr = 2e-5
others_lr = 1e-4

default_bsz = 8
default_shuffle = True
default_train_val_split = 0.9

hidden_size = 256
