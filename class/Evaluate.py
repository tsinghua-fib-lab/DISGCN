

import math
import numpy as np
import time
from collections import defaultdict

class Evaluate():
    def __init__(self, conf):
        self.conf = conf

    def getIdcg(self, length):
        idcg = 0.0
        for i in range(length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        return idcg

    def evaluateRankingPerformance(self, evaluate_index_dict, evaluate_real_rating_matrix, \
        evaluate_predict_rating_matrix, topK, num_procs, exp_flag=0, sp_name=None, result_file=None):
        user_list = list(evaluate_index_dict.keys())
        batch_size = len(user_list) / num_procs
#         print('user_list: {}'.format(len(user_list)))
        hr_list, ndcg_list, mrr_list = defaultdict(list), defaultdict(list), defaultdict(list)
        index = 0
        for _ in range(num_procs):
            if index + batch_size < len(user_list):
                batch_user_list = user_list[index:index+batch_size]
                index = index + batch_size
            else:
                batch_user_list = user_list[index:len(user_list)]
            tmp_hr_list, tmp_ndcg_list, tmp_mrr_list = self.getHrNdcgProc(evaluate_index_dict, evaluate_real_rating_matrix, \
                evaluate_predict_rating_matrix, topK, batch_user_list)
            for k in topK:
                hr_list[k].extend(tmp_hr_list[k])
                ndcg_list[k].extend(tmp_ndcg_list[k])
                mrr_list[k].extend(tmp_mrr_list[k])
        for k in topK:
            hr_list[k] = np.mean(hr_list[k])
            ndcg_list[k] = np.mean(ndcg_list[k])
            mrr_list[k] = np.mean(mrr_list[k])
        return hr_list, ndcg_list, mrr_list

    def getHrNdcgProc(self, 
        evaluate_index_dict, 
        evaluate_real_rating_matrix,
        evaluate_predict_rating_matrix, 
        topK, 
        user_list):

        tmp_hr_list, tmp_ndcg_list, tmp_mrr_list = defaultdict(list), defaultdict(list), defaultdict(list)
#         sort_time = 0
        for u in user_list:
            real_item_index_list = evaluate_index_dict[u]
            real_item_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_item_index_list]))
            positive_length = len(real_item_rating_list)
#             print('u: {}, real_item_index_list: {}'.format(u, real_item_index_list))
            
            predict_rating_list = evaluate_predict_rating_matrix[u]
#             print('real_item_rating_list: {}, predict_rating_list: {}'.format(real_item_rating_list, predict_rating_list))
            real_item_rating_list.extend(predict_rating_list)
#             print('u: {}, real_item_rating_list: {}'.format(u, len(real_item_rating_list)))#, len(d_train.positive_data[u])))
#             time.sleep(1)
#             sort_index = np.argsort(real_item_rating_list)
#             tt = time.time()
            
#             sort_time = sort_time + (time.time() - tt)
            sort_index = np.argpartition(real_item_rating_list, -np.max(topK))[-np.max(topK):]
            sort_index = sort_index[np.argsort(np.array(real_item_rating_list)[sort_index])]
            sort_index = sort_index[::-1]
            user_hr_list = defaultdict(list)
            user_ndcg_list = defaultdict(list)
            user_mrr_list = defaultdict(list)
            for k in topK:
                hits_num = 0
                target_length = min(positive_length, k)
                for idx in range(k):
                    ranking = sort_index[idx]
                    if ranking < positive_length:
                        hits_num += 1
                        user_hr_list[k].append(1.0)
                        user_ndcg_list[k].append(math.log(2) / math.log(idx + 2))
                        if hits_num == 1:
                            user_mrr_list[k].append(1.0/(idx+1))
                idcg = self.getIdcg(target_length)
                tmp_hr_list[k].append(np.sum(user_hr_list[k]) / target_length)
                tmp_ndcg_list[k].append(np.sum(user_ndcg_list[k]) / idcg)
                tmp_mrr_list[k].append(np.sum(user_mrr_list[k]))
#         print('sort cost {}s'.format(sort_time))
        return tmp_hr_list, tmp_ndcg_list, tmp_mrr_list
