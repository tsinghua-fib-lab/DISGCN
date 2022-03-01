#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

_is_hit_cache = {}


def get_is_hit(scores, ground_truth, topk):
    global _is_hit_cache
    cacheid = (id(scores), id(ground_truth))
    if topk in _is_hit_cache and _is_hit_cache[topk]['id'] == cacheid:
        return _is_hit_cache[topk]['is_hit']
    else:
        device = scores.device
        _, col_indice = torch.topk(scores, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            scores.shape[0], device=device, dtype=torch.long).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        _is_hit_cache[topk] = {'id': cacheid, 'is_hit': is_hit}
        return is_hit


class _Metric(object):
    '''
    base class of metrics like HR@k NDCG@k
    '''

    def __init__(self):
        self.start()
        self.epison = 1e-8

    @property
    def metric(self):
        return self._metric

    def __call__(self, scores, ground_truth):
        '''
        - scores: model output
        - ground_truth: one-hot test dataset shape=(users, all_POIs).
        '''
        raise NotImplementedError

    def get_title(self):
        raise NotImplementedError

    def start(self):
        '''
        clear all
        '''
        global _is_hit_cache
        _is_hit_cache = {}
        self._cnt = 0
        self._metric = 0
        self._sum = 0

    def stop(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._metric = self._sum/(self._cnt+self.epison)


class Recall(_Metric):
    '''
    Recall in top-k samples
    '''

    def __init__(self, topk):
        super(Recall, self).__init__()
        self.topk = topk

    def get_title(self):
        return "Recall@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)
        self.each = is_hit/(num_pos+self.epison)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += self.each.sum().item()


class NDCG(_Metric):
    '''
    NDCG in top-k samples
    In this work, NDCG = log(2)/log(1+hit_positions)
    '''

    def DCG(self, hit, device=torch.device('cpu')):
        hit = hit/torch.log2(torch.arange(2, self.topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(self, num_pos):
        hit = torch.zeros(self.topk, dtype=torch.float)
        hit[:num_pos] = 1
        return self.DCG(hit)

    def __init__(self, topk):
        super(NDCG, self).__init__()
        self.topk = topk
        self.IDCGs = torch.empty(1 + self.topk, dtype=torch.float)
        self.IDCGs[0] = 1  # avoid 0/0
        for i in range(1, self.topk + 1):
            self.IDCGs[i] = self.IDCG(i)

    def get_title(self):
        return "NDCG@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        device = scores.device
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        num_pos = ground_truth.sum(dim=1).clamp(0, self.topk).to(torch.long)
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs[num_pos]
        self.each = dcg/idcg.to(device)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += self.each.sum().item()


class MRR(_Metric):
    '''
    Mean reciprocal rank in top-k samples
    '''

    def __init__(self, topk):
        super(MRR, self).__init__()
        self.topk = topk
        self.denominator = torch.arange(1, self.topk+1, dtype=torch.float)

    def get_title(self):
        return "MRR@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        device = scores.device
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        is_hit /= self.denominator.to(device)
        self.each = is_hit.max(dim=1)[0]
        num_pos = ground_truth.sum(dim=1)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += self.each.sum().item()