import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append('..')
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import GCN, GAT, GIN,semantic_GCN,myGAT
import dgl
from dgl.nn.pytorch import GraphConv
import itertools
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from hprompt import hnode_prompt_layer_feature_weighted_sum,node_prompt_layer_feature_weighted_sum,distance2center,center_embedding\
    ,acm_hnode_prompt_layer_feature_weighted_sum,node_prompt_layer_feature_cat,node_prompt_layer_feature_sum,hnode_prompt_layer_feature_sum\
    ,node_prompt_layer_feature_cat_edge,hnode_prompt_layer_feature_cat_edge,prompt_gcn,hprompt_gcn\
    ,acm_eachloss_hnode_prompt_layer_feature_weighted_sum,dblp_hnode_prompt_layer_feature_weighted_sum\
    ,freebase_des_hnode_prompt_layer_feature_weighted_sum,freebase_bidirection_hnode_prompt_layer_feature_weighted_sum\
    ,freebase_source_hnode_prompt_layer_feature_weighted_sum,acm_hnode_semantic_prompt_layer_feature_weighted_sum\
    ,freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum,dblp_hnode_semantic_prompt_layer_feature_weighted_sum

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def subgraph_nodelist(g,subgraphs_dir):
    if os.path.exists(subgraphs_dir) == False:
        os.mkdir(subgraphs_dir)

    if os.path.exists(os.path.join(subgraphs_dir,'0.npy'))==False:
        nodenum = g.number_of_nodes()
        subgraph_list = []
        for i in range(nodenum):
            neighbors = g.successors(i).numpy().tolist()
            two_hop_neighbors = []
            for neighbor in neighbors:
                two_hop_neighbors.extend(g.successors(neighbor).numpy().tolist())
            subgraph_nodes = [i] + neighbors + two_hop_neighbors
            subgraph_nodes = np.array(list(set(subgraph_nodes)))
            subgraph_dir = os.path.join(subgraphs_dir, str(i))
            np.save(subgraph_dir, subgraph_nodes)
            subgraph_list.append(torch.tensor(subgraph_nodes))
    else:
        # Load subgraphs list
        subgraph_list = []
        file_names = [file_name for file_name in os.listdir(subgraphs_dir) if file_name.endswith('.npy')]

        sorted_file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

        for file_name in sorted_file_names:
            file_path = os.path.join(subgraphs_dir, file_name)
            np_array = np.load(file_path)
            subgraph_nodes = torch.tensor(np_array)
            subgraph_list.append(subgraph_nodes)
    return subgraph_list


def run_model_DBLP(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    feats_type = args.feats_type
    index_dir=[str(args.shotnum),'shots',str(args.tasknum),'tasks']
    index = "".join(index_dir)
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset,args.tasknum,args.shotnum,index)
    if args.device==1:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    elif args.device==0:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)


    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])
    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    trans_g=dgl.reverse(g)

    coo_adj=adjM.tocoo()
    values = coo_adj.data
    indices = np.vstack((coo_adj.row, coo_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_adj.shape

    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    eval_result={}
    eval_result['micro-f1']=[]
    eval_result['macro-f1']=[]
    train_time=0
    test_time=0

    for count in range(args.tasknum):
        train_time_one_task = 0
        early_stop=False
        train_labels=labels['train'][count]
        val_labels=labels['val'][count]
        train_labels = torch.LongTensor(train_labels).to(device)
        val_labels = torch.LongTensor(val_labels).to(device)
        train_idx = train_val_test_idx['train_idx'][count]
        val_idx = train_val_test_idx['val_idx'][count]
        test_idx = train_val_test_idx['test_idx']
        test_idx = np.sort(test_idx)

        for _ in range(args.repeat):
            num_classes = dl.labels_train['num_classes']
            if args.model_type == 'gat':
                heads = [args.num_heads] * args.num_layers + [1]
                net = GAT(g, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False)
            elif args.model_type == 'gcn':
                if args.pretrain_semantic:
                    net = semantic_GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                else:
                    net = GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
            elif args.model_type == 'gin':
                net = GIN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.relu, args.dropout)
            elif args.model_type=='SHGN':
                num_classes = dl.labels_train['num_classes']
                heads = [args.num_heads] * args.num_layers + [1]
                net = myGAT(g, args.edge_feats, len(dl.links['count']) * 2 + 1, in_dims, args.hidden_dim, num_classes,
                            args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)

            else:
                raise Exception('{} model is not defined!'.format(args.model_type))
            if args.model_type == 'SHGN':
                net.load_state_dict(torch.load('../checkpoint/pretrain/shgn_checkpoint_{}_{}.pt'.format(
                                                   args.dataset, args.num_layers)))
            else:
                if args.dataset=='Freebase':
                    if args.load_pretrain:
                        if args.hetero_pretrain:
                            if args.hetero_pretrain_subgraph:
                                if args.pretrain_semantic:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                          args.pre_loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                                elif args.pretrain_each_loss:
                                    net.load_state_dict(
                                        torch.load(
                                            '../checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                            format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                   args.pre_loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))

                                else:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                          args.pre_loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                            else:
                                net.load_state_dict(
                                    torch.load('../checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}_{}_{}.pt'.
                                               format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                      args.pre_loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                        else:
                            if args.hetero_pretrain_subgraph:
                                if args.pretrain_semantic:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                          args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                                elif args.pretrain_each_loss:
                                    net.load_state_dict(
                                        torch.load(
                                            '../checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                            format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                   args.pre_loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                                else:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                          args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                            else:
                                net.load_state_dict(
                                    torch.load('../checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}_{}.pt'.
                                               format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                      args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
    ##################### end of if freebase#############
                else:
                    if args.load_pretrain:
                        if args.hetero_pretrain:
                            if args.hetero_pretrain_subgraph:
                                if args.pretrain_semantic:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                          args.pre_loss_weight, args.feats_type)))
                                elif args.pretrain_each_loss:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                          args.pre_loss_weight, args.feats_type)))

                                else:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                          args.pre_loss_weight, args.feats_type)))
                            else:
                                net.load_state_dict(
                                    torch.load('../checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}.pt'.
                                                       format(args.dataset, args.model_type,args.subgraph_hop_num,args.pre_loss_weight,args.feats_type)))
                        else:
                            if args.hetero_pretrain_subgraph:
                                if args.pretrain_semantic:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                          args.feats_type)))
                                elif args.pretrain_each_loss:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                          args.pre_loss_weight, args.feats_type)))
                                else:
                                    net.load_state_dict(
                                        torch.load('../checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}.pt'.
                                                   format(args.dataset, args.model_type, args.subgraph_hop_num, args.feats_type)))
                            else:
                                net.load_state_dict(
                                    torch.load('../checkpoint/pretrain/checkpoint_{}_{}_{}_{}.pt'.
                                                       format(args.dataset, args.model_type,args.subgraph_hop_num,args.feats_type)))
            net.to(device)
            if args.pretrain_semantic:
                prelogits,semantic_weight=net(features_list)
            else:
                if args.model_type=='SHGN':
                    prelogits = net(features_list,e_feat)
                else:
                    prelogits = net(features_list)

            if args.tuning=='linear':
                classify=torch.nn.Linear(args.hidden_dim,num_classes)
            elif args.tuning=='gcn':
                classify=GraphConv(args.hidden_dim, num_classes)
            elif args.tuning in ('weight-sum','weight-sum-center-fixed','bottle-net'):
                if args.model_type=='SHGN':
                    hidden_dim=args.shgn_hidden_dim
                else:
                    hidden_dim=args.hidden_dim
                if args.add_edge_info2prompt:
                    classify = hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                    if args.each_type_subgraph:
                        if args.dataset=='ACM':
                            if args.pretrain_semantic:
                                classify = acm_hnode_prompt_layer_feature_weighted_sum(hidden_dim, semantic_weight)
                            elif args.pretrain_each_loss:
                                classify=acm_eachloss_hnode_prompt_layer_feature_weighted_sum(hidden_dim, semantic_weight)
                            elif args.semantic_prompt==1:
                                classify=acm_hnode_semantic_prompt_layer_feature_weighted_sum(hidden_dim,semantic_prompt_weight=args.semantic_prompt_weight)
                            else:
                                classify = acm_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                        elif args.dataset=='DBLP':
                            if args.semantic_prompt==1:
                                classify=dblp_hnode_semantic_prompt_layer_feature_weighted_sum(hidden_dim,semantic_prompt_weight=args.semantic_prompt_weight)
                            else:
                                classify = dblp_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                        elif args.dataset=='Freebase':
                            if args.semantic_prompt==1:
                                classify=freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum(hidden_dim,semantic_prompt_weight=args.semantic_prompt_weight)
                            else:
                                if args.freebase_type==2:
                                    classify=freebase_bidirection_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                                elif args.freebase_type==1:
                                    classify=freebase_des_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                                else:
                                    classify=freebase_source_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                else:
                    if args.tuning=='bottle-net':
                        print("##############    bottel-net   ###############")
                        classify=node_bottle_net(args.hidden_dim,args.bottle_net_hidden_dim,args.bottle_net_output_dim)
                    else:
                        classify = node_prompt_layer_feature_weighted_sum(hidden_dim)
            elif args.tuning in ('cat'):
                classify=node_prompt_layer_feature_cat(args.cat_prompt_dim)
            elif args.tuning in ('sum'):
                if args.add_edge_info2prompt:
                    classify = hnode_prompt_layer_feature_sum()
                else:
                    classify=node_prompt_layer_feature_sum()
            elif args.tuning in ('cat_edge'):
                if args.add_edge_info2prompt:
                    classify = hnode_prompt_layer_feature_cat_edge(args.cat_prompt_dim,args.cat_hprompt_dim)
                else:
                    classify=node_prompt_layer_feature_cat_edge(args.cat_prompt_dim)
            elif args.tuning in ('prompt_gcn'):
                if args.add_edge_info2prompt:
                    classify = hprompt_gcn(args.hidden_dim)
                else:
                    classify=prompt_gcn(args.hidden_dim)

            else:
                print('tuning model does not exist')
                sys.exit()
            classify.to(device)

            if args.tuning!='sum':
                optimizer = torch.optim.AdamW(classify.parameters(),
                                              lr=args.lr, weight_decay=args.weight_decay)

            # training loop
            classify.train()
            early_stopping_classify = EarlyStopping(patience=args.patience, verbose=True,
                                           save_path='../checkpoint/checkpoint_{}_{}_{}_freeze_classify.pt'.format(args.dataset,
                                                                                             args.model_type,args.tuning,args.shotnum))
            for epoch in range(args.epoch):
                t_start = time.time()
                # training
                net.train()
                classify.train()
                if args.add_edge_info2prompt:
                    if args.tuning == 'gcn':
                        logits = classify(g, prelogits)
                    elif args.tuning in ('weight-sum','weight-sum-center-fixed','bottle-net','sum','cat','cat_edge','prompt_gcn'):
                        if args.dataset=='Freebase':
                            if args.freebase_type==2:
                                logits=classify(g,trans_g,prelogits,e_feat)
                            else:
                                logits = classify(g, prelogits, e_feat)
                        else:
                            logits = classify(g, prelogits, e_feat)
                    elif args.tuning=='linear':
                        logits = classify(prelogits)
                else:
                    if args.tuning in ('gcn','weighted-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                        logits = classify(g, prelogits)
                    else:#linear, weight-sum-center-fixed,cat
                        logits = classify(prelogits)
                if args.tuning in ('gcn','linear'):
                    embedding = logits
                    logp = F.log_softmax(embedding, 1)
                    train_loss = F.nll_loss(logp[train_idx], train_labels)
                else:#args.tuning in ('weight-sum','weight-sum-center-fixed','cat')
                    embedding = logits[train_idx]
                    c_embedding = center_embedding(embedding, train_labels, num_classes)
                    distance = distance2center(embedding, c_embedding)
                    logp = F.log_softmax(distance, dim=1)
                    train_loss = F.nll_loss(logp, train_labels)

                if torch.isnan(train_loss).any():
                    print('embedding',embedding)
                    print('c_embedding',c_embedding)
                    print('distance',distance)
                    print('logp',logp)
                    raise ValueError("Tensor contains NaN values. Program terminated.")
                # autograd
                if args.tuning != 'sum':
                    optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                if args.tuning != 'sum':
                    optimizer.step()
                t_end = time.time()
                train_time_one_task+=t_end-t_start

                if args.tuning in ('weight-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                    _pred = torch.argmax(logp, dim=1, keepdim=False)
                    truth = train_labels.cpu().numpy()
                    output = _pred.cpu().numpy()
                    microf1 = f1_score(truth, output, average='micro')
                    macrof1 = f1_score(truth, output, average='macro')
                else:
                    _pred = torch.argmax(logp[train_idx], dim=1, keepdim=False)
                    truth = train_labels.cpu().numpy()
                    output = _pred.cpu().numpy()
                    microf1 = f1_score(truth, output, average='micro')
                    macrof1 = f1_score(truth, output, average='macro')

                # print training info
                print('Epoch {:05d} | Train_Loss {:.4f} | Microf1 {:.4f} | Macrof1 {:.4f} | Time(s) {:.4f}'.format(
                    epoch, train_loss.item(), microf1,macrof1,t_end - t_start))

                t_start = time.time()
                # validation
                classify.eval()
                with torch.no_grad():
                    # add edge info to prompt, edge info is added same as SHGN
                    if args.add_edge_info2prompt:
                        if args.tuning == 'gcn':
                            logits = classify(g, prelogits)
                        elif args.tuning in ('weight-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                            if args.dataset == 'Freebase':
                                if args.freebase_type == 2:
                                    logits = classify(g, trans_g, prelogits, e_feat)
                                else:
                                    logits = classify(g, prelogits, e_feat)
                            else:
                                logits = classify(g, prelogits, e_feat)
                        else:
                            logits = classify(prelogits)
                    else:
                        if args.tuning in ('gcn', 'weighted-sum', 'weight-sum-center-fixed', 'bottle-net','cat', 'sum','cat_edge','prompt_gcn'):
                            if args.pretrain_each_loss:
                                if args.dataset=='ACM':
                                    logits,logits0,logits2,logits4,logits6 = classify(g, prelogits)
                            else:
                                logits = classify(g, prelogits)
                        else:
                            logits = classify(prelogits)
                    if args.tuning == 'gcn':
                        logits = logits
                        logp = F.log_softmax(logits, 1)
                        val_loss = F.nll_loss(logp[val_idx], val_labels)
                    if args.tuning == 'linear':
                        logits = logits
                        logp = F.log_softmax(logits, 1)
                        val_loss = F.nll_loss(logp[val_idx], val_labels)
                    if args.tuning == 'weight-sum':
                        embedding = logits[val_idx]
                        c_embedding = center_embedding(embedding, val_labels, num_classes)
                        distance = distance2center(embedding, c_embedding)
                        #distance = 1 / F.normalize(distance, dim=1)
                        logp = F.log_softmax(distance, dim=1)
                        val_loss = F.nll_loss(logp, val_labels)
                    if args.tuning in ('weight-sum-center-fixed','cat','bottle-net','sum','cat_edge','prompt_gcn'):
                        #This part hasn't been finished
                        if args.pretrain_each_loss:
                            if args.dataset == 'ACM':
                                embedding = logits[val_idx]
                                distance = distance2center(embedding, c_embedding)
                                # distance = 1 / F.normalize(distance, dim=1)
                                logp = F.log_softmax(distance, dim=1)
                                embedding0 = logits0[val_idx]
                                distance0 = distance2center(embedding0, c_embedding0)
                                # distance = 1 / F.normalize(distance, dim=1)
                                logp0 = F.log_softmax(distance0, dim=1)
                                val_loss = F.nll_loss(logp, val_labels)
                        else:
                            embedding = logits[val_idx]
                            distance = distance2center(embedding, c_embedding)
                            #distance = 1 / F.normalize(distance, dim=1)
                            logp = F.log_softmax(distance, dim=1)
                            val_loss = F.nll_loss(logp, val_labels)

                    t_end = time.time()

                if args.tuning in ('weight-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                    _pred = torch.argmax(logp, dim=1, keepdim=False)
                    truth = val_labels.cpu().numpy()
                    output = _pred.cpu().numpy()
                    microf1 = f1_score(truth, output, average='micro')
                    macrof1 = f1_score(truth, output, average='macro')
                else:
                    _pred = torch.argmax(logp[val_idx], dim=1, keepdim=False)
                    truth = val_labels.cpu().numpy()
                    output = _pred.cpu().numpy()
                    microf1 = f1_score(truth, output, average='micro')
                    macrof1 = f1_score(truth, output, average='macro')

                # print validation info
                print('Epoch {:05d} | Val_Loss {:.4f} | Microf1 {:.4f} | Macrof1 {:.4f} | Time(s) {:.4f}'.format(
                    epoch, val_loss.item(), microf1,macrof1,t_end - t_start))
                # early stopping
                early_stopping_classify(val_loss,classify)
                if early_stopping_classify.early_stop:
                    print('Early stopping!')
                    train_time_one_task = train_time_one_task / epoch
                    early_stop = True
                    break
            if early_stop == False:
                train_time_one_task = train_time_one_task / args.epoch
            train_time += train_time_one_task
            if args.tuning == 'weight-sum-center-fixed':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save('../checkpoint/{}_task_center_embedding'.format(count), save_c_emb)
            elif args.tuning == 'bottle-net':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save('../checkpoint/{}_task_center_embedding'.format(count), save_c_emb)
            elif args.tuning == 'cat':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save('../checkpoint/{}_task_center_embedding_cat'.format(count), save_c_emb)
            elif args.tuning == 'sum':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save('../checkpoint/{}_task_center_embedding_sum'.format(count), save_c_emb)
            elif args.tuning == 'cat_edge':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save('../checkpoint/{}_task_center_embedding_cat_edge'.format(count), save_c_emb)
            elif args.tuning == 'prompt_gcn':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save('../checkpoint/{}_task_center_embedding_prompt_gcn'.format(count), save_c_emb)


            # testing with evaluate_results_nc
            classify.load_state_dict(torch.load('../checkpoint/checkpoint_{}_{}_{}_freeze_classify.pt'.format(args.dataset, args.model_type,args.tuning,args.shotnum)))

            start_test_time=time.time()
            classify.eval()
            test_logits = []
            with torch.no_grad():
                if args.add_edge_info2prompt:
                    if args.tuning=='gcn':
                        logits=classify(g,prelogits)
                    elif args.tuning in ('weight-sum','weight-sum-center-fixed','sum','cat','cat_edge','prompt_gcn'):
                        if args.dataset == 'Freebase':
                            if args.freebase_type == 2:
                                logits = classify(g, trans_g, prelogits, e_feat)
                            else:
                                logits = classify(g, prelogits, e_feat)
                        else:
                            logits = classify(g, prelogits, e_feat)
                    else:
                        logits=classify(prelogits)
                else:
                    if args.tuning in ('gcn','weighted-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                        logits = classify(g, prelogits)
                    else:
                        logits=classify(prelogits)
                if args.tuning == 'gcn':
                    logits = logits
                    test_logits = logits[test_idx]
                    pred = test_logits.cpu().numpy().argmax(axis=1)
                    onehot = np.eye(num_classes, dtype=np.int32)
                    dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{args.run}.txt")
                    pred = onehot[pred]
                    res = dl.evaluate(pred)
                elif args.tuning=='linear':
                    logits = logits
                    test_logits = logits[test_idx]
                    pred = test_logits.cpu().numpy().argmax(axis=1)
                    onehot = np.eye(num_classes, dtype=np.int32)
                    dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{args.run}.txt")
                    pred = onehot[pred]
                    res = dl.evaluate(pred)
                elif args.tuning == 'weight-sum':
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    c_embedding = center_embedding(embedding, test_label, num_classes)
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning == 'weight-sum-center-fixed':
                    load_c_emb = np.load('../checkpoint/{}_task_center_embedding.npy'.format(count), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning == 'bottle-net':
                    load_c_emb = np.load('../checkpoint/{}_task_center_embedding.npy'.format(count), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning == 'cat':
                    load_c_emb = np.load('../checkpoint/{}_task_center_embedding_cat.npy'.format(count), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning == 'sum':
                    load_c_emb = np.load('../checkpoint/{}_task_center_embedding_sum.npy'.format(count), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning == 'cat_edge':
                    load_c_emb = np.load('../checkpoint/{}_task_center_embedding_cat_edge.npy'.format(count), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning == 'prompt_gcn':
                    load_c_emb = np.load('../checkpoint/{}_task_center_embedding_prompt_gcn.npy'.format(count), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')

            end_test_time=time.time()
            test_time+=end_test_time-start_test_time
            eval_result['micro-f1'].append(res['micro-f1'])
            eval_result['macro-f1'].append(res['macro-f1'])
    eval_result['micro-f1']=np.array(eval_result['micro-f1'])
    eval_result['macro-f1']=np.array(eval_result['macro-f1'])
    print("####################################################")
    print('microf mean: ', np.mean(eval_result['micro-f1']), 'acc std: ', np.std(eval_result['micro-f1']))
    print('macroF mean: ', np.mean(eval_result['macro-f1']), 'macroF std: ', np.std(eval_result['macro-f1']))
    print('downstream train time per epoch:',train_time/args.tasknum)
    print('downstream test time per epoch:',test_time/args.tasknum)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' +
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--bottle-net-hidden-dim', type=int, default=2, help='Dimension of the node hidden state. Default is 2.')
    ap.add_argument('--bottle-net-output-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--model-type', type=str, default='gcn', help="gcn or gat")
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--device',type=int,default=1)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-6)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str, default='ACM')
    ap.add_argument('--seed', type=int, default="0")
    ap.add_argument('--tasknum', type=int, default=100)
    ap.add_argument('--shotnum', type=int, default=1)
    ap.add_argument('--load_pretrain', type=int, default=1)
    ap.add_argument('--tuning',type=str, default='weight-sum-center-fixed')
    ap.add_argument('--subgraph_hop_num', type=int, default=1)
    #make sure we have ran pre_train_model with this loss_weight
    ap.add_argument('--pre_loss_weight', type=float, default=1)
    ap.add_argument('--hetero_pretrain', type=int, default=0)
    ap.add_argument('--hetero_pretrain_subgraph', type=int, default=0)
    ap.add_argument('--pretrain_semantic', type=int, default=0)
    ap.add_argument('--add_edge_info2prompt', type=int, default=1)
    ap.add_argument('--each_type_subgraph', type=int, default=1)
    ap.add_argument('--pretrain_each_loss', type=int, default=0)
    ap.add_argument('--cat_prompt_dim', type=int, default=64, help='Dimension of the cat prompt dim. Default is 64.')
    ap.add_argument('--cat_hprompt_dim', type=int, default=64, help='Dimension of the cat prompt dim. Default is 64.')
    ap.add_argument('--tuple_neg_disconnected_num', type=int, default=1, help='Dimension of the cat prompt dim. Default is 64.')
    ap.add_argument('--tuple_neg_unrelated_num', type=int, default=1, help='Dimension of the cat prompt dim. Default is 64.')
    ap.add_argument('--meta_path', type=int, default=0)
    ap.add_argument('--semantic-prompt', type=int, default=1)
    ap.add_argument('--freebase-type', type=int, default=0, help='0:book as source, 1:book as destination, 2: bidirection')
    ap.add_argument('--semantic-prompt-weight', type=float, default=0.1)
    ap.add_argument('--shgn-hidden-dim', type=int, default=3, help='Dimension of the node hidden state. Default is 64.')
    args = ap.parse_args()
    run_model_DBLP(args)
