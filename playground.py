from work.NewsCorrelation.raw_BERT import UseModel
from work.NewsCorrelation import newsco_utils as utl
from tqdm import tqdm
from evaluate.evaluator import cal_pccs
import numpy as np

# model = UseModel('checkpoint/', 'checkpoint')

valid_file = utl.load_news_pair_csv('data/NLP/news_sim/semeval-2022_task8_eval_data_202201.csv')
crawls = utl.load_crawl_results('data/NLP/news_sim/eval_final')
id2crawled = utl.build_id2crawled(crawls)
samples = utl.build_data_samples(id2crawled, valid_file)
# gt = list(x['Overall'] for x in samples)
inputs = list({
    'title1': x['crawl1']['title'],
    'text1': x['crawl1']["text"],
    'title2': x['crawl2']['title'],
    'text2': x['crawl2']['text'],
} for x in samples)
ids = list(x['id1'] + '-' + x['id2'] for x in samples)

fold_results = {}
fold_avgs = []
for i in range(10):
    print(f'loading model of fold-{i}', end='...')
    model = UseModel(f'checkpoint/kfold-{i}-save-best.pth', f'checkpoint/kfold-{i}-init_param-best.pk')
    print('done.', end='')

    results = {}
    total, cnt = 0, 0
    for (elem_input, elem_id) in tqdm(zip(inputs, ids)):
        result = model(**elem_input)
        total += result['pred']
        cnt += 1
        results[elem_id] = result['pred']
    fold_avgs.append(total / cnt)  # 计算平均数
    fold_results[i] = results

    output_dicts = []
    for elem in valid_file:
        id1, id2 = elem['id1'], elem['id2']
        pair = id1 + '-' + id2
        if pair in results:
            score = results[pair]
        else:
            score = fold_avgs[-1]

        output_dicts.append({
            "url1lang": elem['lang1'],
            "url2lang": elem['lang2'],
            "pair_id": pair,
            'link1': elem['link1'],
            "link2": elem['link2'],
            "Overall": float(score)
        })
    utl.dicts2csv(output_dicts, f'fold-{i}.csv')


# print('combining scores')
# allscores = np.array(list(fold_results.values()))
# avg_scores = np.average(allscores, axis=0)
#
# output = []
# for i, s in enumerate(avg_scores):
#     output.append({
#         'url1lang': valid_file[i]['lang1'],
#         'url2lang': valid_file[i]['lang2'],
#         'pair_id': valid_file[i]['id1'] + '_' + valid_file[i]['id2'],
#         'link1': valid_file[i]['link1'],
#         'link2': valid_file[i]['link2'],
#         'Overall': float(s)
#     })
# utl.dicts2csv(output, 'submitz.csv')
#
