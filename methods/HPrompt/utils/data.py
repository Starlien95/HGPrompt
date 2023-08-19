import sys
import dgl
import numpy as np
import os
import copy
import torch as th
import scipy.sparse as sp
sys.path.append('..')
from utils.data_loader import data_loader,data_loader_lp

def load_data(prefix='DBLP', shotnum=10,tasknum=2,index=None):
    dl = data_loader('../../../data/'+prefix)
    index_dir=os.path.join('../../../data',prefix,index)
    print(index_dir)
    index_exist=os.path.exists(index_dir)
    if index_exist==False:
        print("Please Generate Few shot tasks first,using SHGN")
        sys.exit()
    print("##################")
    print("index for ",shotnum,"shots ",tasknum,"tasks exists: ",index_exist)

    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    #labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    print("Loading Few Shot Tasks")
    save_train = np.load(os.path.join(index_dir, "train_index.npy"), allow_pickle=True)
    save_val = np.load(os.path.join(index_dir, "val_index.npy"), allow_pickle=True)
    save_test = np.load(os.path.join(index_dir, "test_index.npy"), allow_pickle=True)
    save_train_labels = np.load(os.path.join(index_dir, "train_labels.npy"), allow_pickle=True)
    save_val_labels = np.load(os.path.join(index_dir, "val_labels.npy"), allow_pickle=True)

    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = save_train
    train_val_test_idx['val_idx'] = save_val
    train_val_test_idx['test_idx'] = save_test
    multi_task_labels = {}
    multi_task_labels['train'] = save_train_labels
    multi_task_labels['val'] = save_val_labels
    return features,\
           adjM, \
           multi_task_labels,\
           train_val_test_idx,\
            dl


def load_data_lp(prefix='DBLP', shotnum=10,tasknum=2,index=None):
    dl = data_loader_lp('../../../data/'+prefix)
    index_dir=os.path.join('../../../data',prefix,index)
    print(index_dir)
    index_exist=os.path.exists(index_dir)
    if index_exist==False:
        print("Please Generate Few shot tasks first,using SHGN")
        sys.exit()
    print("##################")
    print("index for ",shotnum,"shots ",tasknum,"tasks exists: ",index_exist)

    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    print("Loading Few Shot Tasks")
    save_train = np.load(os.path.join(index_dir, "train_index.npy"), allow_pickle=True)
    save_val = np.load(os.path.join(index_dir, "val_index.npy"), allow_pickle=True)
    save_test = np.load(os.path.join(index_dir, "test_index.npy"), allow_pickle=True)
    save_train_labels = np.load(os.path.join(index_dir, "train_labels.npy"), allow_pickle=True)
    save_val_labels = np.load(os.path.join(index_dir, "val_labels.npy"), allow_pickle=True)

    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = save_train
    train_val_test_idx['val_idx'] = save_val
    train_val_test_idx['test_idx'] = save_test
    multi_task_labels = {}
    multi_task_labels['train'] = save_train_labels
    multi_task_labels['val'] = save_val_labels

    task_source=dl.links_lp_task_s
    task_des=dl.links_lp_task_d

    return features,\
           adjM, \
           multi_task_labels,\
           train_val_test_idx,\
            dl, task_source,task_des


def load_pretrain_data(prefix='DBLP'):
    #from scripts.data_loader import data_loader
    dl = data_loader('../../../data/'+prefix)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())

    return features,\
           adjM, \
            dl

def load_pretrain_data_lp(prefix='DBLP'):
    #from scripts.data_loader import data_loader
    dl = data_loader_lp('../../../data/'+prefix)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    return features,\
           adjM, \
            dl


def load_acm(feat_type=0,shotnum=10,tasknum=2,index=None):
    dl = data_loader('../../../data/ACM')
    index_dir=os.path.join('../../../data/ACM',index)
    index_exist=os.path.exists(index_dir)
    if index_exist==False:
        print("Please Generate Few shot tasks first,using SHGN")
        sys.exit()
    print("##################")
    print("index for ",shotnum,"shots ",tasknum,"tasks exists: ",index_exist)

    link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp', 6: 'pt', 7: 'tp'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # paper feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        features = th.FloatTensor(np.eye(paper_num))

    # paper labels

    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 3

    print("Loading Few Shot Tasks")
    save_train = np.load(os.path.join(index_dir, "train_index.npy"), allow_pickle=True)
    save_val = np.load(os.path.join(index_dir, "val_index.npy"), allow_pickle=True)
    save_test = np.load(os.path.join(index_dir, "test_index.npy"), allow_pickle=True)
    save_train_labels = np.load(os.path.join(index_dir, "train_labels.npy"), allow_pickle=True)
    save_val_labels = np.load(os.path.join(index_dir, "val_labels.npy"), allow_pickle=True)

    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = save_train
    train_val_test_idx['val_idx'] = save_val
    train_val_test_idx['test_idx'] = save_test
    multi_task_labels = {}
    multi_task_labels['train'] = save_train_labels
    multi_task_labels['val'] = save_val_labels

    train_valid_mask = dl.labels_train['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False

    meta_paths = [['pp', 'ps', 'sp'], ['-pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pt', 'tp']]
    return hg, features, labels, num_classes, train_val_test_idx, multi_task_labels, meta_paths
