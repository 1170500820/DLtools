"""
将斯坦福大学的nlp工具stanza进行包装，提供一些方便的方法
"""
from type_def import *
import stanza

# 有用的实体类型。数字、钱数之类的实体是没必要的。
useful_tags = ['ORG']


def download_lang():
    stanza.download('multilingual')
    stanza.download('zh-hans')
    stanza.download('en')
    stanza.download('de')
    stanza.download('ar')
    stanza.download('fr')
    stanza.download('et')
    stanza.download(lang="es", package=None, processors={"ner": "conll02"})
    stanza.download(lang="es", package=None, processors={"tokenize": "ancora"})


class stanza_ner_multilingual:
    """
    多语言的ner工具。对于一个输入，自动检测语言并进行ner
    """

    # 语言的识别结果与
    lang2ner = {
        'zh-hans': 'zh',
        'en': 'en',
        'de': 'de',
        'ar': 'ar',
        'fr': 'fr',
        'es': 'es',
    }
    lang2processor = {}
    unavailable_langs = ['tr', 'pl', 'nn', 'id', 'et', 'ja', 'hi', 'be']

    def __init__(self):
        self.lang_detector = stanza.Pipeline(lang='multilingual', processors='langid', verbose=False)
        self.lang2processor['zh'] = stanza.Pipeline(lang='zh', processors='tokenize,ner', verbose=False)
        self.lang2processor['en'] = stanza.Pipeline(lang='en', processors='tokenize,ner', verbose=False)

    def __call__(self, sentence: Union[str, Sequence[str]], clean: bool = True):
        if isinstance(sentence, str):
            return self.named_entity_recognize(sentence, clean)
        else:
            results = []
            for elem_sentence in sentence:
                results.append(self.named_entity_recognize(elem_sentence, clean))
            return results

    def named_entity_recognize(self, sentence: str, clean: bool = True):
        """
        对一个句子进行命名实体识别

        先识别语言；然后执行识别；最后抽取识别结果并以dict形式返回
        :param sentence:
        :return:
            按照start_char顺序排列的List[Dict[str, Any]]
            dict contains:
                - text: str
                - type: str
                - start_char: int
                - end_char: int
        """
        if len(sentence) <= 1:  # 空串
            return []
        lang = self.lang_detector(sentence).lang
        if lang in self.unavailable_langs:  # 不支持的语言
            return []
        if lang not in self.lang2processor:
            self.lang2processor[self.lang2ner[lang]] = stanza.Pipeline(lang=self.lang2ner[lang], processors='tokenize,ner', verbose=False)
        result = self.lang2processor[self.lang2ner[lang]](sentence).ents
        if clean:
            new_result = list({
                "text": x.text,
                'type': x.type
            } for x in result)
            return new_result
        return result


if __name__ == '__main__':
    download_lang()
