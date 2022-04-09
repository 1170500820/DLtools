import pickle
from utils import tokenize_tools

d = pickle.load(open('PLMEE_Trigger.averaged_loss-EvalResults.pk', 'rb'))

preds = d['preds'][-1]
gt = d['gts'][-1]


