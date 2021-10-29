from type_def import *
from process.data_processor_type import *
from analysis.visualize import *
from process.typed_processor_utils import *
from utils.data import SimpleDataset



def get_data(filepath: str):
    p = TSV_Reader() + DoubleListTranspose() + List2Dict(['sent1', 'sent2', 'label'])
    p = p + ReleaseDict() + ListMerger(['sent1', 'sent2'], 'sentence') + ListMapper(len)
    result = p(filepath)
    return result


def get_train(filepath: str):
    p = TSV_Reader() + (ListMapper(lambda x: x[:2]) + BERT_Tokenizer_double() + ListOfDictTranspose() + ReleaseDict()) * ListMapper(lambda x: x[-1]) + Dict2List(['mapped', 'input_ids', 'token_type_ids', 'attention_mask'])
    result = p(filepath)
    return result



result = get_data('../data/NLP/Similarity/bq_corpus/train.tsv')
result2 = get_train('../data/NLP/Similarity/bq_corpus/train.tsv')
dataset = SimpleDataset(list(result2.values())[0])
