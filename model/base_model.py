import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import os
import numpy as np
class base_model(object):
    def __init__(self, conf, reg, learning_rate):
        self.conf = conf
        self.reg = reg
        self.learning_rate = learning_rate
        self.dim = self.conf.dimension
        self.num_users = self.conf.num_users
        self.num_items = self.conf.num_items
        self.num_negatives = self.conf.num_negatives
        self.batch_size = self.conf.training_batch_size

    def startConstructGraph(self):
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()

    def initializeNodes(self):
        tfv1.disable_eager_execution()
        self.item_input = tfv1.placeholder("int32", [None])
        self.user_input = tfv1.placeholder("int32", [None])
        self.item_neg_input = tfv1.placeholder("int32", [None, self.num_negatives, 1])
        if self.conf.pretrain_flag:
            pre_emb = np.load(os.path.join(os.getcwd(), 'embedding', self.conf.data_name, self.conf.pre_train), encoding='latin1')
            user_embedding, item_embedding = pre_emb['user_embedding'], pre_emb['item_embedding']
            self.user_embedding = tf.Variable(
                user_embedding/np.linalg.norm(user_embedding, axis=1, keepdims=True), name='user_embedding')
            self.item_embedding = tf.Variable(
                item_embedding/np.linalg.norm(item_embedding, axis=1, keepdims=True), name='item_embedding')
            if 'dis' in self.conf.model_name:
                user_social_embedding, item_social_embedding = pre_emb['user_social_embedding'], pre_emb['item_social_embedding']
                self.user_social_embedding = tf.Variable(
                    user_social_embedding/np.linalg.norm(user_social_embedding, axis=1, keepdims=True), name='user_social_embedding')
                self.item_social_embedding = tf.Variable(
                    item_social_embedding/np.linalg.norm(item_social_embedding, axis=1, keepdims=True), name='item_social_embedding')
        else:
            self.user_embedding = tf.Variable(
                tf.random.normal([self.num_users, self.conf.dimension], stddev=0.01), name='user_embedding')
            self.item_embedding = tf.Variable(
                tf.random.normal([self.num_items, self.conf.dimension], stddev=0.01), name='item_embedding')
    
    def constructTrainGraph(self):
        raise NotImplementedError

    def saveVariables(self):
        raise NotImplementedError

    def predict(self, emb_u=None, emb_i=None):
        if emb_u is None:
            emb_u = self.user_embedding
        if emb_i is None:
            emb_i = self.item_embedding
        emb_u_gather = tf.gather(emb_u, self.user_input)
        return tf.matmul(emb_u_gather, tf.transpose(emb_i))

    def BPRloss(self, emb_u=None, emb_i=None, reg=True):
        if emb_u is None:
            emb_u = self.user_embedding
        if emb_i is None:
            emb_i = self.item_embedding
        emb_u_gather = tf.gather(emb_u, self.user_input)
        emb_i_gather = tf.gather(emb_i, self.item_input)
        emb_j_gather = tf.gather_nd(emb_i, self.item_neg_input)
        pos_score = tf.reduce_sum(emb_u_gather*emb_i_gather, -1, keepdims=True)
        neg_score = tf.reduce_sum(tf.expand_dims(emb_u_gather, 1)*emb_j_gather, -1, keepdims=False)
        loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softplus(neg_score-pos_score), -1))
        if reg:
            loss += self.reg*(self.regloss([emb_u_gather, emb_i_gather])+self.regloss([emb_j_gather])/self.num_negatives)
        return loss

    def regloss(self, tensors):
        loss = 0
        for t in tensors:
            loss += tf.nn.l2_loss(t)
        return loss


    def defineMap(self):
        from copy import copy
        map_dict = {}
        tmp = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.item_neg_input: 'ITEM_NEG_INPUT'
        }
        map_dict['train'] = tmp
        map_dict['val'] = copy(tmp)
        map_dict['test'] = copy(tmp)

        map_dict['eva'] = {
            self.user_input: 'EVA_USER_LIST',
            self.item_input: 'EVA_ITEM_LIST'
        }
        map_dict['out'] = {
            'train': self.loss,
            'val': self.loss,
            'test': self.loss,
            'eva': self.prediction#, self.prediction_link]
        }
        self.map_dict = map_dict


