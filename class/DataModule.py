
from collections import defaultdict
import numpy as np
from time import time
import random
import math
import operator
import pandas as pd
import sys
sys.path.append('/data4/linian/Social_Rec/data/')

class DataModule():
    def __init__(self, conf, filename, links_filename=None):
        self.conf = conf
        self.num_users, self.num_items = conf.num_users, conf.num_items
        self.model_name, self.data_name = conf.model_name, conf.data_name
        self.train_or_not = links_filename is not None
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.links_filename = links_filename
        self.index = 0
        self.index_link = 0
        self.print_once = True
        self.loss_turn = 0

###########################################  Initalize Procedures ############################################
    def prepareModelSupplement(self):
        data_dict = {}
        model_name = self.model_name
        need_S_idx = (model_name in ['gcncsr', 'diffnet', 'graphrec', 'socialbpr', 'samn', 'sorec', 'disengcn', 'disbpr', 'disgcn'])
        need_S_values = (model_name in ['gcncsr', 'diffnet', 'socialbpr', 'disengcn', 'disbpr', 'disgcn'])
        need_C_idx = (model_name in ['gcncsr', 'diffnet', 'graphrec', 'ngcf', 'lightgcn', 'gcn', 'gcmc', 'disengcn', 'disbpr', 'disgcn'])
        need_C_values = (model_name in ['gcncsr', 'diffnet', 'ngcf', 'lightgcn', 'gcn', 'gcmc', 'disengcn', 'disbpr', 'disgcn'])
        if need_S_idx:
            self.generateSocialNeighborsSparseMatrix(need_S_values)
            data_dict['social_edges_user0'] = self.social_edges_user0
            data_dict['social_edges_user1'] = self.social_edges_user1
            if need_S_values:
                data_dict['social_neighbors_values_input'] = self.social_neighbors_values_input
        if need_C_idx:
            self.generateConsumedItemsSparseMatrix(need_C_values)
            data_dict['interaction_user_indices'] = self.interaction_user_indices
            data_dict['interaction_item_indices'] = self.interaction_item_indices
            if need_C_values:
                data_dict['consumed_items_values_input'] = self.consumed_items_values_input
                data_dict['inter_users_values_input'] = self.inter_users_values_input
        if model_name in ['lightgcn', 'gcn', 'ngcf']:
            data_dict['user_self_value'] = self.user_self_value
            data_dict['item_self_value'] = self.item_self_value
        if  model_name in ['gcncsr', 'sorec'] and self.conf.social_loss:
            data_dict['social_edges_user_neg'] = self.negative_link
        if model_name in ['gcmc', 'ngcf'] and self.conf.user_side:
            data_dict['user_side_information'] = self.u_emb

        if model_name in ['disgcn', 'csr']:
            data_dict['u_input'] = self.u_input
            data_dict['f_input'] = self.f_input
            data_dict['i_input'] = self.i_input
            data_dict['seg_in_friends'] = self.seg_in_friends
            data_dict['ufi_att'] = self.ufi_att
            data_dict['int_att'] = self.int_att
            data_dict['social_att_idx'] = self.social_att_idx
            data_dict['social_att'] = self.social_att
            
        return data_dict

    def initializeRankingTrain(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()

    def initializeRankingVT(self):
        self.readData()
        self.arrangePositiveData()

    def initalizeRankingEva(self):
        self.readData()
        self.arrangePositiveData()

    def construct_ground_truth(self):
        ground_truth = np.zeros([len(self.batch_user_list), self.num_items])
        for uid, u in enumerate(self.batch_user_list):
            ground_truth[uid, list(self.user_consumed_items[u])] = 1
        return ground_truth

    def linkedMap(self):
        if self.loss_turn == 0:
            self.data_dict['USER_LIST'] = self.user_list
            self.data_dict['ITEM_LIST'] = self.item_list
            self.data_dict['ITEM_NEG_INPUT'] = self.item_neg_list
            # if self.model_name in ['disengcn', 'disbpr', 'disgcn'] and self.train_or_not:
            #     self.data_dict['cor_idx'] = self.cor_idx
            if not self.terminal_flag and self.model_name in ['disbpr', 'disgcn'] and self.conf.social_loss:
                self.loss_turn = 1

            # if self.model_name in ['Geobpr']:
            #     self.data_dict['Geo0_list'] = self.Geo0_list
            #     self.data_dict['Geo1_list'] = self.Geo1_list
            #     self.data_dict['cor_idx'] = self.cor_idx

        else:
            if self.model_name in ['gcncsr', 'esgcn', 'sbpr', 'sorec'] and self.conf.social_loss:
                self.data_dict['USER_SOCIAL_LOSS_IDX'] = self.user_social_loss_idx
            elif self.model_name in ['disgcn', 'disbpr'] and self.conf.social_loss:
                self.data_dict['ufi_u'] = self.ufi_u
                self.data_dict['ufi_f'] = self.ufi_f
                self.data_dict['ufi_i'] = self.ufi_i
                self.data_dict['ufi_j'] = self.ufi_j
                if self.conf.g:
                    self.data_dict['ufi_g'] = self.ufi_g
            if not self.terminal_flag:
                self.loss_turn = 0
    
    def linkedRankingEvaMap(self):
        self.data_dict['EVA_USER_LIST'] = self.eva_user_list
        self.data_dict['EVA_ITEM_LIST'] = self.eva_item_list

    def readData(self):
        positive_data = np.loadtxt(self.filename).astype(np.int32)[:, :2]
        self.positive_data = positive_data[(positive_data[:,0]*self.num_items+positive_data[:, 1]).argsort()]
        print(f'# of interactions: {len(self.positive_data)}')
        if self.train_or_not:
            positive_link = np.loadtxt(self.links_filename)
            positive_link = np.unique(np.concatenate([positive_link, np.fliplr(positive_link)], 0), axis=0)
            self.positive_link = positive_link[(positive_link[:,0]*self.num_users+positive_link[:,1]).argsort()]
            print(f'# of friendship: {len(self.positive_link)}')
        if self.model_name in ['disbpr', 'disgcn', 'csr'] and self.train_or_not:
            ufi = np.loadtxt(f'data/{self.data_name}/ufi').astype(np.int32)
            ufi = np.unique(np.concatenate([np.concatenate([ufi[:, :2], np.fliplr(ufi[:, :2])], 0), np.expand_dims(np.tile(ufi[:, 2], 2), 1)], 1), axis=0)
            self.ufi = ufi[(ufi[:, 0]*self.num_users+ufi[:, 1]).argsort()]
            if self.model_name in ['disbpr', 'disgcn']:
                self.ufi_batch_size = int(np.ceil(len(self.ufi)/(np.ceil(len(self.positive_data)/self.conf.training_batch_size))))
                print(f'ufi_batch_size: {self.ufi_batch_size}')
        self.total_user_list = np.arange(self.num_users)
        self.total_item_list = np.arange(self.num_items)
        if self.model_name in ['gcmc', 'ngcf'] and self.conf.user_side:
            self.u_emb = np.zeros([self.num_users, self.conf.dimension])
            emb_name_f = 'data/{}/deepwalk'.format(self.data_name)
            emb_name = emb_name_f + '.txt' if self.conf.dimension == 64 else emb_name_f + '_{}.txt'.format(self.conf.dimension)
            with open(emb_name) as f:
                for x in f.readlines()[1:]:
                    tmp = np.fromstring(x, sep=' ')
                    self.u_emb[int(tmp[0])] = tmp[1:]

        # if self.model_name in ['Geobpr']:
        #     with open('/home/linian/KGAT/Data/MDB_yelp/item_coor.txt') as f:
        #         item_coor = pd.read_csv(f, sep=' ')
        #     self.coor = np.array(item_coor[['latitude', 'longitude']])

    def arrangePositiveData(self):
        self.user_consumed_items, self.item_inter_users = defaultdict(set), defaultdict(set)
        for u, i in self.positive_data:
            self.user_consumed_items[u].add(i)
            self.item_inter_users[i].add(u)
        self.user_friends = defaultdict(set)
        for u, fr in self.positive_link:
            self.user_friends[u].add(fr)
        if self.model_name in ['gcncsr', 'sbpr'] and self.conf.social_loss:
            L = len(self.positive_link)
            num_split = 100
            step = L//num_split+1
            self.split_idx = [0]
            for i in range(num_split):
                end = self.split_idx[-1] + step
                self.split_idx.append(np.min([L, end]))

        if self.model_name in ['disgcn', 'csr'] and self.train_or_not:
            from sklearn.preprocessing import LabelEncoder
            ufi_le = LabelEncoder()
            ufi = self.ufi
            label = ufi[:, 0]*self.num_users+ufi[:, 1]
            self.seg_in_friends = ufi_le.fit_transform(label)
            self.u_input, self.f_input, self.i_input = ufi[:, 0], ufi[:, 1], ufi[:, 2]
            _, ufi_att = np.unique(ufi[:, :2], axis=0, return_counts=True)
            self.ufi_att = np.repeat(1.0/ufi_att, ufi_att).reshape([-1, 1])

            _, int_att = np.unique(self.positive_link[:, 0], axis=0, return_counts=True)
            self.int_att = np.repeat(1.0/int_att, int_att).reshape([-1, 1])
            self.social_att_idx = np.unique(ufi[:, :2], axis=0)
            _, social_att = np.unique(self.social_att_idx[:, 0], axis=0, return_counts=True)
            self.social_att = np.repeat(1.0/social_att, social_att).reshape([-1, 1])

    def sample_neg(self, num_neg, num_all, cond_sets):
        tmp = []
        for _ in range(num_neg):
            j = np.random.randint(num_all)
            while True:
                if j not in cond_sets:
                    break
                j = np.random.randint(num_all)
            tmp.append([j])
        return tmp

    def generateTrainNegative(self):
        num_items, num_users = self.num_items, self.num_users
        num_negatives = self.conf.num_negatives
        negative_data = []
        for u, _ in self.positive_data:
            tmp = self.sample_neg(num_negatives, num_items, self.user_consumed_items[u])
            negative_data.append(tmp)
        self.negative_data = np.array(negative_data)
        if self.model_name in ['disbpr', 'disgcn'] and self.conf.social_loss:
            negative_ufi = []
            for u, f, _ in self.ufi:
                tmp = self.sample_neg(num_negatives, num_items, self.user_consumed_items[u].union(self.user_consumed_items[f]))
                negative_ufi.append(tmp)
            self.negative_ufi_i = np.array(negative_ufi)
            if self.conf.g:
                negative_ufi = []
                for u, _, i in self.ufi:
                    tmp = self.sample_neg(num_negatives, num_users, self.user_friends[u].union(self.item_inter_users[i]))
                    negative_ufi.append(tmp)
                self.negative_ufi_f = np.array(negative_ufi)
        if self.model_name in ['gcncsr', 'sorec'] and self.conf.social_loss:
            negative_link = []
            for u, _ in self.positive_link:
                tmp = self.sample_neg(num_negatives, num_users, [self.user_friends[u]])
                negative_link.append(tmp)
            self.negative_link = np.array(negative_link)

        # if self.model_name in ['Geobpr']:
        #     Geo_item0, Geo_item1 = [], []
        #     for i in self.positive_data[:, 1]:
        #         x = np.random.randint(0, num_items, 10)
        #         x_dis = np.sum(np.square(self.coor[x] - self.coor[i]), 1)
        #         Geo_item0.append(np.argmin(x_dis))
        #         Geo_item1.append(np.argmax(x_dis))
        #     self.Geo_item0 = np.reshape(np.array(Geo_item0), [-1, 1])
        #     self.Geo_item1 = np.reshape(np.array(Geo_item1), [-1, 1])
    
    def getTrainRankingBatch(self):
        if self.loss_turn == 0:
            positive_data = self.positive_data
            len_positive_data = len(positive_data)
            batch_size = self.conf.training_batch_size
            index = self.index
            tmp = min((len_positive_data, index+batch_size))
            batch_data = positive_data[index:tmp]
            self.item_neg_list = self.negative_data[index:tmp]
            # if self.model_name in ['disgcn', 'disbpr', 'disengcn'] and self.train_or_not:
            #     cor_user = np.array(random.sample(self.total_user_list, tmp-index))
            #     cor_item = np.array(random.sample(self.total_item_list, tmp-index)) + self.num_users
            #     self.cor_idx = np.concatenate([cor_user, cor_item], 0)
            if tmp >= len_positive_data:
                self.index = 0
                self.terminal_flag = 0
            else:
                self.index = index + batch_size
            
            self.user_list = batch_data[:, 0]
            self.item_list = batch_data[:, 1]

            # if self.model_name in ['Geobpr']:
            #         self.Geo0_list = self.Geo_item0[index:tmp]
            #         self.Geo1_list = self.Geo_item1[index:tmp]

        else:
            if self.model_name in ['gcncsr', 'sorec'] and self.conf.social_loss:
                positive_link = self.positive_link
                len_positive_link = len(positive_link)
                link_batch_size = int(np.ceil(len_positive_link/(np.ceil(len_positive_data/batch_size))))
                index_link = self.index_link
                if index_link + link_batch_size < len_positive_link:
                    batch_link = range(index_link, index_link+link_batch_size)
                    self.index_link = index_link + link_batch_size
                else:
                    batch_link = range(index_link, len_positive_link)
                    self.index_link = 0
                    self.terminal_flag = 0
                self.user_social_loss_idx = batch_link

            elif self.model_name in ['disgcn', 'disbpr'] and self.conf.social_loss:
                ufi = self.ufi
                len_ufi = len(self.ufi)
                index_link = self.index_link
                tmp = min((len_ufi, index_link+self.ufi_batch_size))
                batch_ufi = ufi[index_link:tmp]
                self.ufi_j = self.negative_ufi_i[index_link:tmp]
                if self.conf.g:
                    self.ufi_g = self.negative_ufi_f[index_link:tmp]
                self.ufi_u, self.ufi_f, self.ufi_i = batch_ufi[:, 0], batch_ufi[:, 1], batch_ufi[:, 2]
                if tmp >= len_ufi:
                    self.index_link = 0
                    self.terminal_flag = 0
                else:
                    self.index_link = index_link + self.ufi_batch_size
        
    def getEvaBatch(self):
        index = self.index
        batch_size = self.conf.evaluate_batch_size
        if index + batch_size < self.num_users:
            self.batch_user_list = self.total_user_list[index:index+batch_size]
            self.index = index + batch_size
            len_batch = batch_size
        else:
            self.terminal_flag = 0
            self.batch_user_list = self.total_user_list[index:self.num_users]
            len_batch = self.num_users - index
            self.index = 0
        self.len_batch = len_batch
        if self.model_name in ['gmf', 'mlp', 'neumf', 'hbpr', 'graphrec', 'disengcn']:
            self.eva_user_list = np.repeat(batch_user_list, self.num_items)
            self.eva_item_list = np.tile(range(self.num_items), len_batch)
        else:
            self.eva_user_list = self.batch_user_list
            self.eva_item_list = None


    def generateSocialNeighborsSparseMatrix(self, values=True):
        model_name = self.model_name
        self.social_edges_user0 = np.reshape(self.positive_link[:, 0], [-1, 1])
        self.social_edges_user1 = np.reshape(self.positive_link[:, 1], [-1, 1])
        if values:
            social_neighbors_values_input = []
            if model_name in ['gcncsr', 'disengcn']:
                for u0, u1 in self.positive_link:
                    social_neighbors_values_input.append(1.0/math.sqrt((len(self.user_friends[u0])+1)*(len(self.user_friends[u1])+1)))
                    # social_neighbors_values_input.append(1.0/len(self.user_friends[u0]))
            if model_name in ['diffnet', 'socialbpr', 'disbpr', 'disgcn']:
                for u in self.positive_link[:, 0]:
                    social_neighbors_values_input.append(1.0/len(self.user_friends[u]))
            self.social_neighbors_values_input = np.array(social_neighbors_values_input).astype(np.float32)



    def generateConsumedItemsSparseMatrix(self, values=True):
        model_name = self.model_name
        self.interaction_user_indices = np.expand_dims(self.positive_data[:, 0], 1)
        self.interaction_item_indices = np.expand_dims(self.positive_data[:, 1], 1)
        if values:
            consumed_items_values_input, inter_users_values_input = [], []
            if model_name in ['diffnet', 'graphrec', 'gcncsr', 'disbpr', 'disgcn']:
                for u, i in self.positive_data:
                    consumed_items_values_input.append(1.0/len(self.user_consumed_items[u]))
                    inter_users_values_input.append(1.0/len(self.item_inter_users[i]))
            if model_name in ['ngcf', 'gcn', 'lightgcn', 'gcmc', 'disengcn']:
                self.user_self_value = np.zeros([self.num_users, 1]).astype(np.float32)
                self.item_self_value = np.zeros([self.num_items, 1]).astype(np.float32)
                for u, i in self.positive_data:
                    u, i = int(u), int(i)
                    num_i = len(self.user_consumed_items[u])+1
                    num_u = len(self.item_inter_users[i])+1
                    L = 1.0/math.sqrt(num_i*num_u)
                    consumed_items_values_input.append(L)
                    inter_users_values_input.append(L)
                    self.user_self_value[u, 0] = 1.0/num_i
                    self.item_self_value[i, 0] = 1.0/num_u
            self.consumed_items_values_input = np.array(consumed_items_values_input).astype(np.float32)
            self.inter_users_values_input = np.array(inter_users_values_input).astype(np.float32)