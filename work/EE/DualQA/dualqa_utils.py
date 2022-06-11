from type_def import *


def concat_token_for_evaluate(tokens: List[str], span: Tuple[int, int]):
    """
    利用预测的span从输入模型的input_ids所对应的token序列中抽取出所需要的词语

    - 删除"##"
    - 对于(0, 0)会直接输出''
    :param tokens:
    :param span:
    :return:
    """
    if span == (0, 0):
        return ''
    result = ''.join(tokens[span[0]: span[1] + 1])
    result = result.replace('##', '')
    return result