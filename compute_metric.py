from metric import *
from collections import defaultdict
import torch
import os

def compute_metric(scores, groud_truth, conf):
    topk = conf.topk
    metric_dict = defaultdict(list)
    for k in topk:
        metric_dict['Recall'].append(Recall(k))
        metric_dict['NDCG'].append(NDCG(k))
        metric_dict['MRR'].append(MRR(k))
    metric_name = {'Recall':[], 'NDCG':[], 'MRR':[]}

    with torch.no_grad():
        for i in range(len(topk)):
            for name in metric_name.keys():
                m = metric_dict[name][i]
                m.start()
                m(scores, groud_truth)
                m.stop()
                metric_name[name].append(m.metric)
    return metric_name