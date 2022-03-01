import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from model.base_model import base_model

class disbpr(base_model):

    def initializeNodes(self):
        super(disbpr, self).initializeNodes()
        self.cor_idx = tfv1.placeholder("int32", [None])
        self.ufi_u = tfv1.placeholder("int32", [None])
        self.ufi_f = tfv1.placeholder("int32", [None])
        self.ufi_g = tfv1.placeholder("int32", [None, self.num_negatives, 1])
        self.ufi_i = tfv1.placeholder("int32", [None])
        self.ufi_j = tfv1.placeholder("int32", [None, self.num_negatives, 1])
        self.user_social_embedding = tf.Variable(
                tf.random.normal([self.num_users, self.conf.dimension], stddev=0.01), name='user_social_embedding')
        self.item_social_embedding = tf.Variable(
                tf.random.normal([self.num_items, self.conf.dimension], stddev=0.01), name='item_social_embedding')

    def Y(self, u, f, i, keepdims=True):
        return tf.reduce_sum(u*i, -1, keepdims=keepdims) + tf.reduce_sum(f*i, -1, keepdims=keepdims) + tf.reduce_sum(u*f, -1, keepdims=keepdims)
    
    def constructTrainGraph(self):
        user_embedding = tf.concat([self.user_embedding, self.user_social_embedding], 1)
        item_embedding = tf.concat([self.item_embedding, self.item_social_embedding], 1)
        self.loss = self.BPRloss(user_embedding, item_embedding)
        self.prediction = self.predict(user_embedding, item_embedding)
        cor_social_embedding = tf.gather(tf.concat([self.user_social_embedding, self.item_social_embedding], 0), self.cor_idx)
        cor_interest_embedding = tf.gather(tf.concat([self.user_embedding, self.item_embedding], 0), self.cor_idx)
        self.cor_loss = self._create_distance_correlation(cor_social_embedding, cor_interest_embedding)
        self.loss += self.beta*self.cor_loss
        emb_u, emb_f, emb_i = tf.gather(self.user_social_embedding, self.ufi_u), \
                                tf.gather(self.user_social_embedding, self.ufi_f), \
                                tf.gather(self.item_social_embedding, self.ufi_i)
        emb_j = tf.gather_nd(self.item_social_embedding, self.ufi_j)
        emb_g = tf.gather_nd(self.user_social_embedding, self.ufi_g)
        ufi_pos = self.Y(emb_u, emb_f, emb_i)
        ufi_neg_i = self.Y(tf.expand_dims(emb_u, 1), tf.expand_dims(emb_f, 1), emb_j, keepdims=False)
        ufi_neg_f = self.Y(tf.expand_dims(emb_u, 1), emb_g, tf.expand_dims(emb_i, 1), keepdims=False)
        self.social_loss = tf.reduce_sum(tf.nn.softplus(tf.reduce_sum(ufi_neg_i-ufi_pos, -1))) + tf.reduce_sum(tf.nn.softplus(tf.reduce_sum(ufi_neg_f-ufi_pos, -1))) +\
                            self.conf.sreg*(self.regloss([emb_u, emb_f, emb_i])+self.regloss([emb_j, emb_g])/self.num_negatives)

        self.opt = tfv1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.social_opt = tfv1.train.AdamOptimizer(self.conf.social_lr).minimize(self.social_loss)
        self.init = tfv1.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['user_social_embedding'] = self.user_social_embedding
        variables_dict['item_social_embedding'] = self.item_social_embedding
        self.saver = tfv1.train.Saver(variables_dict)

    def defineMap(self):
        super(disbpr, self).defineMap()
        tmp_mask = {self.ufi_u:'ufi_u', self.ufi_f:'ufi_f', self.ufi_i:'ufi_i', self.ufi_j:'ufi_j', self.ufi_g:'ufi_g'}
        self.map_dict['train_social'] = tmp_mask
        self.map_dict['train'].update({self.cor_idx:'cor_idx'})