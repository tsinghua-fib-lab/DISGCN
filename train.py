import os, sys, shutil
from collections import deque
from time import time
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import pickle
import torch
from itertools import product
from collections import defaultdict
from visdom import Visdom
from Logging import Logging
from model import *
from compute_metric import compute_metric
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(f'Tensorflow Version: {tf.__version__}')
sys.path.append(os.path.join(os.getcwd(), 'class'))

def test(model, conf, d_test_eva, sess, d_train):
    device0 = torch.device('cuda:{}'.format(conf.gpu_device))
    metrics = defaultdict(list)
    for _ in range(len(conf.topk)):
        metrics['NDCG'].append(0)
        metrics['Recall'].append(0)
        metrics['MRR'].append(0)
    tt0 = time()
    while d_test_eva.terminal_flag:
        d_test_eva.getEvaBatch()
        d_test_eva.linkedRankingEvaMap()
        eva_feed_dict = {}
        if conf.model_name in ['gmf', 'mlp', 'neumf', 'hbpr', 'graphrec']:
            for (key, value) in model.map_dict['eva'].items():
                eva_feed_dict[key] = d_test_eva.data_dict[value]
            predictions = torch.tensor(np.reshape(sess.run(model.map_dict['out']['eva'], feed_dict=eva_feed_dict), [-1, conf.num_items]), dtype=torch.float32, device=device0)
        else:
            eva_feed_dict[model.user_input] = d_test_eva.data_dict['EVA_USER_LIST']
            if conf.model_name in ['ngcf']:
                eva_feed_dict[model.training] = False
            predictions = torch.tensor(sess.run(model.map_dict['out']['eva'], feed_dict=eva_feed_dict), dtype=torch.float32, device=device0)
        for uid, u in enumerate(d_test_eva.batch_user_list):
            predictions[uid, list(d_train.user_consumed_items[u])] = -999
        ground_truth = torch.tensor(d_test_eva.construct_ground_truth(), dtype=torch.float32, device=device0)
        tmp_metrics = compute_metric(predictions, ground_truth, conf)
        for k in range(len(conf.topk)):
            metrics['NDCG'][k] += tmp_metrics['NDCG'][k]*d_test_eva.len_batch
            metrics['Recall'][k] += tmp_metrics['Recall'][k]*d_test_eva.len_batch
            metrics['MRR'][k] += tmp_metrics['MRR'][k]*d_test_eva.len_batch
    for k in range(len(conf.topk)):
        metrics['NDCG'][k] /= conf.num_users
        metrics['Recall'][k] /= conf.num_users
        metrics['MRR'][k] /= conf.num_users
    d_test_eva.index = 0
    d_test_eva.terminal_flag = 1
    print(f'Evaluate cost:{time()-tt0:.1f}s')
    return metrics

def start(conf, data, model_name):
    if conf.data_name in ['Beidian']:
        vis_port = 1496
    elif conf.data_name in ['Beibei']:
        vis_port = 1469

    for reg, lr in product(conf.reg, conf.learning_rate):
        print('reg: {}, lr: {}---------------------------'.format(reg, lr))

        data.initializeRankingHandle()
        d_train, d_val, d_test, d_test_eva = data.train, data.val, data.test, data.test_eva
        print('System start to load data...')
        t0 = time()
        d_train.initializeRankingTrain()
        d_val.initializeRankingVT()
        d_test.initializeRankingVT()
        d_test_eva.initalizeRankingEva()
        print(f'Data has been loaded successfully, cost:{time()-t0:.1f}s')
        data_dict = d_train.prepareModelSupplement()

        model = eval(model_name)
        model = model(conf, reg, lr)
        model.inputSupply(data_dict)
        model.startConstructGraph()

        tf_conf = tfv1.ConfigProto()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf_conf.gpu_options.allow_growth = True
        sess = tfv1.Session(config=tf_conf)
        sess.run(model.init)

        if conf.premodel_flag == 1:
            if conf.model_name == 'neumf' and conf.test == 0:
                hhh = conf.pre_model.split('/')
                checkpoint_gmf = os.path.join('pretrain', conf.data_name, hhh[0])
                model.saver_GMF.restore(sess, checkpoint_gmf)
                print('restore gmf done')
                checkpoint_mlp = os.path.join('pretrain', conf.data_name, hhh[1])
                model.saver_mlp.restore(sess, checkpoint_mlp)
                print('restore mlp done')
            else:
                checkpoint = os.path.join('pretrain', conf.data_name, conf.pre_model)
                model.saver.restore(sess, checkpoint)
                print('restore model done')
        print('Following will output the evaluation of the model:')
        
        ndcg_item = deque([0]*30, 30)
        best_ndcg20 = 0.0
        if not conf.test:
            vis = Visdom(port=vis_port, env=f'{conf.model_name}_reg{reg}-lr{lr}-dim{conf.dimension}-{conf.test_name}')
        if conf.test == 2:
            After_Metric = {'Recall': [], 'NDCG': [], 'MRR': []}
        for epoch in range(1, conf.epochs+1):
            if conf.test != 1:
                tmp_train_loss = []
                t0 = time()
                while d_train.terminal_flag:
                    d_train.getTrainRankingBatch()
                    d_train.linkedMap()
                    train_feed_dict = {}
                    for (key, value) in model.map_dict['train'].items():
                        train_feed_dict[key] = d_train.data_dict[value]
                    if conf.model_name in ['ngcf']:
                        train_feed_dict[model.training] = True
                    if conf.model_name in ['sorec']:
                        train_feed_dict[model.user_social_loss_idx] = d_train.user_social_loss_idx
                    [sub_train_loss, _] = sess.run(\
                        [model.map_dict['out']['train'], model.opt], feed_dict=train_feed_dict)
                    tmp_train_loss.append(sub_train_loss)
                train_loss = np.mean(tmp_train_loss, 0)
                if conf.model_name in ['gcncsr'] and conf.social_loss:
                    tmp_train_social_loss = []
                    d_train.terminal_flag = 1
                    while d_train.terminal_flag:
                        d_train.getTrainRankingBatch()
                        d_train.linkedMap()
                        [sub_train_social_loss, _] = sess.run(\
                            [model.social_loss, model.social_opt], feed_dict={model.user_social_loss_idx: d_train.user_social_loss_idx})
                        tmp_train_social_loss.append(sub_train_social_loss)
                    social_loss = np.mean(tmp_train_social_loss)
                    if conf.att:
                        social_att = []
                        for i in range(len(d_train.split_idx)-1):
                            start, end = d_train.split_idx[i], d_train.split_idx[i+1]
                            att = sess.run(model.social_att, feed_dict={model.u0: d_train.social_edges_user0[start:end], model.u1: d_train.social_edges_user1[start:end]})
                            social_att.extend(list(att))
                        social_att = np.array(social_att)
                        model.update_social_matrix(social_att)
                if conf.model_name in ['disgcn', 'disbpr'] and conf.social_loss:
                    tmp_train_social_loss = []
                    d_train.terminal_flag = 1
                    while d_train.terminal_flag:
                        d_train.getTrainRankingBatch()
                        d_train.linkedMap()
                        train_feed_dict = {}
                        for (key, value) in model.map_dict['train_social'].items():
                            train_feed_dict[key] = d_train.data_dict[value]
                        [sub_train_social_loss, _] = sess.run(\
                            [model.social_loss, model.social_opt], feed_dict=train_feed_dict)
                        tmp_train_social_loss.append(sub_train_social_loss)
                    social_loss = np.mean(tmp_train_social_loss)
                    if conf.model_name in ['disgcn']:
                        ufi_att_list = [sess.run(model.ufi_att_list[k]) for k in range(conf.num_layers)]
                        model.update_ufi_att(ufi_att_list)
                        if conf.att:
                            int_att_list = [sess.run(model.int_att_list[k]) for k in range(conf.num_layers)]
                            social_att_list = [sess.run(model.social_att_list[k]) for k in range(conf.num_layers)]
                            model.update_att(int_att_list, social_att_list)

                d_train.generateTrainNegative()
                d_train.terminal_flag = 1

                t1 = time()
                if conf.model_name in ['disbpr', 'disgcn', 'gcncsr'] and conf.social_loss:
                    print(f'Epoch:{epoch}, compute loss cost:{t1-t0:.1f}s, train loss:{train_loss:.2f}, social loss:{social_loss:.2f}')
                else:
                    print(f'Epoch:{epoch}, compute loss cost:{t1-t0:.1f}s, train loss:{train_loss:.2f}')
                if not conf.test:
                    X = [epoch]
                    if epoch == 1:
                        if conf.model_name in ['disgcn', 'disbpr', 'gcncsr'] and conf.social_loss:
                            vis.line([train_loss], X, win='train loss', opts={'title':'train loss'})
                            vis.line([social_loss], X, win='social loss', opts={'title':'social loss'})
                        else:
                            vis.line([train_loss], X, win='train loss', opts={'title':'train loss'})
                    else:
                        if conf.model_name in ['disgcn', 'disbpr', 'gcncsr'] and conf.social_loss:
                            vis.line([train_loss], X, win='train loss',  update='append', opts={'title':'train loss'})
                            vis.line([social_loss], X, win='social loss',  update='append', opts={'title':'social loss'})
                        else:
                            vis.line([train_loss], X, win='train loss', update='append', opts={'title':'train loss'})

            if epoch%5 == 0 or conf.test:
                metrics = test(model, conf, d_test_eva, sess, d_train)
                for i, k in enumerate(conf.topk):
                    print('Recall@{}: {}, NDCG@{}: {}, MRR@{}: {}'.format(k, metrics['Recall'][i], k, metrics['NDCG'][i], k, metrics['MRR'][i]))
                if conf.test == 1:
                    print('test done')
                    exit()
                if conf.test == 2:
                    for k in After_Metric.keys():
                        After_Metric[k].append(metrics[k])
                    if epoch == conf.epochs:
                        for k in After_Metric.keys():
                            After_Metric[k] = np.mean(After_Metric[k], 0)
                        for i, k in enumerate(conf.topk):
                            print('Recall@{}: {}, NDCG@{}: {}, MRR@{}: {}'.format(k, After_Metric['Recall'][i], k, After_Metric['NDCG'][i], k, After_Metric['MRR'][i]))
                    continue
                which_ndcg = conf.topk.index(20)
                if epoch == 5:
                    vis.line([metrics['NDCG'][which_ndcg]], X, win='NDCG@20', opts={'title':'NDCG@20'})
                    vis.line([metrics['Recall'][which_ndcg]], X, win='Recall@20', opts={'title':'Recall@20'})
                    vis.line([metrics['MRR'][which_ndcg]], X, win='MRR@20', opts={'title':'MRR@20'})
                else:
                    vis.line([metrics['NDCG'][which_ndcg]], X, win='NDCG@20', update='append', opts={'title':'NDCG@20'})
                    vis.line([metrics['Recall'][which_ndcg]], X, win='Recall@20', update='append', opts={'title':'Recall@20'})
                    vis.line([metrics['MRR'][which_ndcg]], X, win='MRR@20', update='append', opts={'title':'MRR@20'})
                if metrics['NDCG'][which_ndcg] > best_ndcg20:
                    best_ndcg20 = metrics['NDCG'][which_ndcg]
                    save_path = f'./pretrain/{conf.data_name}/{conf.model_name}_reg{reg}_lr{lr}_dim{conf.dimension}_{conf.test_name}.ckpt'
                    save_path = model.saver.save(sess, save_path, write_meta_graph=False)
                    print('Model saved in ' + save_path)
                    if conf.model_name in ['gcncsr']:
                        save_path = f'./pretrain/{conf.data_name}/{conf.model_name}_reg{reg}_lr{lr}_dim{conf.dimension}_att_{conf.test_name}.npy'
                        np.save(save_path, np.expand_dims(sess.run(model.social_neighbors_sparse_matrix._values), 1))
                        print('save att values')
                    print('test metric'+'-'*20)
                    metrics_test = test(model, conf, d_test, sess, d_train)
                    for i, k in enumerate(conf.topk):
                        print('Recall@{}: {}, NDCG@{}: {}, MRR@{}: {}'.format(k, metrics_test['Recall'][i], k, metrics_test['NDCG'][i], k, metrics_test['MRR'][i]))
                    print('-'*31)

                ndcg_item.append(metrics['NDCG'][which_ndcg])
                if np.mean(ndcg_item) > metrics['NDCG'][which_ndcg] or epoch == conf.epochs:
                    print('ndcg@20 dose not change, early stopping ...')
                    break