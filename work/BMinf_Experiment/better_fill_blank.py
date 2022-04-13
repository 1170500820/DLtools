import bminf
from type_def import *
from typing import Optional


TOKEN_SPAN = "<span>"

input_text0 = "北京环球度假区相关负责人介绍，北京环球影城指定单日门票将采用<span>制度，即推出淡季日、平季日、旺季日和特定日门票。<span>价格为418元，<span>价格为528元，<span>价格为638元，<span>价格为<span>元。北京环球度假区将提供90天滚动价格日历，以方便游客提前规划行程。"
input_text1 = "被证实将再裁员1800人 福特汽车公司在为落后的经营模式买单。那么，<span>进行了裁员"


def fill_blank(cpm2: bminf.models.CPM2,
               input_sentence: str,
               spans_position: Optional[List[int]] = None,
               max_tokens: int = 128,
               top_n: Optional[int] = None,
               top_p: Optional[float] = None,
               temperature: float = 0.9,
               frequency_penalty: float = 0,
               presence_penalty: float = 0,
               ):
    """
    利用cpm2模型进行填空任务，但是尝试解决报错问题
    :param cpm2: 加载的模型本体
    :param input_sentence:  输入的句子，用<span>表示填空位置
    :param spans_position:
    :param max_tokens: 生成token的最大数量
    :param top_n: 从前top_n个候选token中选取
    :param top_p: 我也不懂。Only sampling from tokens that comprising the top p probability in the result.
    :param temperature: 值越高，生成的结果越diverse
    :param frequency_penalty: 避免生成重复token的惩罚项
    :param presence_penalty: 避免生成同一个topic的结果的惩罚项
    :return:
    """
    idx, input_length, spans_position = cpm2._pre_processing(input_sentence, spans_position, 0)
    res = cpm2._gen_iter(
        idx,
        input_length,
        max_tokens,
        cpm2._model.tokenizer.sod_id,
        top_n,
        top_p,
        temperature,
        frequency_penalty,
        presence_penalty,
        filter_tokens=[cpm2._model.tokenizer.unk_id]
    )
    next_span = 0
    blanks = []
    # for token in res:
    #     if token == cpm2._model.tokenizer.get_span(next_span):
    #         blanks.append([])
    #         next_span += 1
    #         if next_span > len(spans_position):
    #             break
    #     elif next_span == 0:
    #         raise RuntimeError("Unexpected model output: %d" % token)
    #     else:
    #         blanks[-1].append(token)
    cpm2.free()
    # return [
    #     {
    #         "position": blank_pos,
    #         "text": cpm2._model.tokenizer.decode(blank_tokens)
    #     }
    #     for blank_pos, blank_tokens in zip(spans_position, blanks)
    # ]
    return cpm2._model.tokenizer.devode(res)


def main():
    print("Loading model")
    cpm2 = bminf.models.CPM2()
    print("Start")
    result = fill_blank(cpm2, input_text1)
    print('Result:\n', result)


if __name__ == '__main__':
    main()