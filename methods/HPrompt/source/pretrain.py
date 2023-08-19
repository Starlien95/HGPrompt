import sys
sys.path.append('..')
import time
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..')
from utils.pytorchtools import EarlyStopping
from utils.data import load_pretrain_data
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import GCN, GAT, GIN,acm_hGCN,acm_sem_hGCN,acm_hGCN_each_loss,dblp_hGAT,freebase_bi_hGCN\
    ,freebase_source_hGCN,freebase_des_hGCN,myGAT
import dgl
import tqdm
import os

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

#by disconnected neg, we calculate disconnected loss
#for this part, we use the same tuples as GraphPrompt to prove the effectiveness of hprompt_pretrain_sample
def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    print("#############")
    print("start sampling disconnected tuples")
    for i in tqdm.trange(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)

def prompt_pretrain_sample_target_node(dataset,dl,adj,n_unrelated,target_node_tuple_num):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    whole=np.array(range(nodenum))
    print("#############")
    print("start sampling target nodes' disconnected tuples")

    if dataset=="ACM":
        res = np.zeros((np.sum(target_node_tuple_num), 2 + n_unrelated))
        count=0
        for i in tqdm.trange(dl.nodes['shift'][0], dl.nodes['shift'][0] + dl.nodes['count'][0]):
            nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
            zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
            np.random.shuffle(nonzero_index_i_row)
            np.random.shuffle(zero_index_i_row)
            for j in np.arange(target_node_tuple_num[i]):
                res[count][0] = i
                if np.size(nonzero_index_i_row)==0:
                    res[count][1] = i
                else:
                    res[count][1]=nonzero_index_i_row[j]
                res[count][2:2+n_unrelated]=zero_index_i_row[j+j*n_unrelated:j+(j+1)*n_unrelated]
    return res.astype(int)


def hprompt_pretrain_sample(dataset,dl,adj,n_unrelated):
    #target nodes are labeled nodes
    print("#############")
    print("start hetero-sampling tuples")
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    target_node_tuple_num=None
    if dataset=="ACM":
        paper_nodes= np.arange(dl.nodes['shift'][0],dl.nodes['shift'][0]+dl.nodes['count'][0])
        author_nodes= np.arange(dl.nodes['shift'][1],dl.nodes['shift'][1]+dl.nodes['count'][1])
        subject_nodes= np.arange(dl.nodes['shift'][2],dl.nodes['shift'][2]+dl.nodes['count'][2])
        term_nodes= np.arange(dl.nodes['shift'][3],dl.nodes['shift'][3]+dl.nodes['count'][3])
        target_node_tuple_num=np.zeros_like(paper_nodes)
        whole=[]
        whole.append(paper_nodes)
        whole.append(author_nodes)
        whole.append(subject_nodes)
        whole.append(term_nodes)
        res = None
        first_res=True
        count=0
        paper_nodes_dont_have_type={}
        paper_nodes_dont_have_type[0]=[]
        paper_nodes_dont_have_type[1]=[]
        paper_nodes_dont_have_type[2]=[]
        paper_nodes_dont_have_type[3]=[]
        isolated_nodes=[]
        temp=np.zeros((1,2+n_unrelated))
        for w in whole:
            print('\nedge type: ', count)
            for i in tqdm.trange(dl.nodes['shift'][0],dl.nodes['shift'][0]+dl.nodes['count'][0]):
                nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
                nonzero_corresponding_type_index_i_row=np.intersect1d(nonzero_index_i_row,w)
                if not nonzero_corresponding_type_index_i_row.any():
                    paper_nodes_dont_have_type[count].append(i)
                    continue
                target_node_tuple_num[i]+=1
                zero_index_i_row=np.setdiff1d(w,nonzero_index_i_row)
                np.random.shuffle(nonzero_index_i_row)
                np.random.shuffle(zero_index_i_row)
                temp[0][0] = i
                temp[0][1] = nonzero_corresponding_type_index_i_row[0]
                temp[0][2:2 + n_unrelated] = zero_index_i_row[0:n_unrelated]
                if first_res:
                    res=temp
                    first_res=False
                else:
                    res=np.concatenate((res,temp),axis=0)
            count+=1
            # for (target,pos_candidate) in tqdm(edge2type):
        print('isolated nodes:', isolated_nodes)
        print('paper_nodes_dont_have_type 0 num', len(paper_nodes_dont_have_type[0]))
        print('paper_nodes_dont_have_type 1 num',len(paper_nodes_dont_have_type[1]))
        print('paper_nodes_dont_have_type 2 num',len(paper_nodes_dont_have_type[2]))
        print('paper_nodes_dont_have_type 3 num',len(paper_nodes_dont_have_type[3]))

    elif dataset=="DBLP":
        paper_nodes= np.arange(dl.nodes['shift'][1],dl.nodes['shift'][1]+dl.nodes['count'][1])
        target_node_tuple_num=np.zeros_like(paper_nodes)
        whole=[]
        whole.append(paper_nodes)
        res = None
        first_res=True
        ##each relation
        count=0
        author_nodes_dont_have_type={}
        author_nodes_dont_have_type[0]=[]
        isolated_nodes=[]
        temp=np.zeros((1,2+n_unrelated))
        for w in whole:
            ##each paper node
            print('\nedge type: ', count)
            #Is there some author node that is isolated?
            for i in tqdm.trange(dl.nodes['shift'][0],dl.nodes['shift'][0]+dl.nodes['count'][0]):
                #need to select the corresponding type of node
                nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
                ##ACM has no isolated paper_node
                nonzero_corresponding_type_index_i_row=np.intersect1d(nonzero_index_i_row,w)
                #some paper nodes dont have several relation
                if not nonzero_corresponding_type_index_i_row.any():
                    author_nodes_dont_have_type[count].append(i)
                    continue
                zero_index_i_row=np.setdiff1d(w,nonzero_index_i_row)
                np.random.shuffle(nonzero_index_i_row)
                np.random.shuffle(zero_index_i_row)
                temp[0][0] = i
                temp[0][1] = nonzero_corresponding_type_index_i_row[0]
                temp[0][2:2 + n_unrelated] = zero_index_i_row[0:n_unrelated]
                if first_res:
                    res=temp
                    first_res=False
                else:
                    res=np.concatenate((res,temp),axis=0)
            count+=1
        print('isolated nodes:', isolated_nodes)
        print('paper_nodes_dont_have_type 0 num', len(author_nodes_dont_have_type[0]))
    np.random.shuffle(res)
    return res.astype(int),target_node_tuple_num

def mygather(feature, index):
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    # print(tuples)
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))

#tuple[i][0] represent pos node
def compareloss(feature,tuples,temperature,device):
    h_tuples=mygather(feature,tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp=temp.to(device)
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

#tuple[i][0] represent target node
def tcompareloss(feature,tuples,temperature,device):
    h_tuples=mygather(feature,tuples)
    temp=h_tuples.permute(1,0,2)[0]
    temp=temp.unsqueeze(0)
    h_i = temp.permute(1,0,2)
    h_tuples=h_tuples.permute(1,0,2)[1:]
    h_tuples=h_tuples.permute(1,0,2)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

#tuple[i][0] represent target node, add relation to similarity
def hcompareloss(feature,tuples,temperature,device):
    h_tuples=mygather(feature,tuples)
    temp=h_tuples.permute(1,0,2)[0]
    temp=temp.unsqueeze(0)
    h_i = temp.permute(1,0,2)
    h_tuples=h_tuples.permute(1,0,2)[1:]
    h_tuples=h_tuples.permute(1,0,2)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

def two_hop_subgraph_nodelist(g,subgraphs_dir,max_neigbour_num=10):
    if os.path.exists(subgraphs_dir) == False:
        os.mkdir(subgraphs_dir)
    if os.path.exists(os.path.join(subgraphs_dir,'0.npy'))==False:
        nodenum = g.number_of_nodes()
        subgraph_list = []
        for i in range(nodenum):
            neighbors = g.successors(i).numpy().tolist()
            if len(neighbors)>max_neigbour_num:
                neighbors=random.sample(neighbors, max_neigbour_num)
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
    feats_type = args.feats_type
    features_list, adjM, dl = load_pretrain_data(args.dataset)
    if args.device==1:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    elif args.device==0:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device=torch.device('cpu')
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
    #edge feature:0,1,2,3,...
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    # print(dl.links['count'])
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])

    samples_dir=os.path.join('../../../data',args.dataset,"".join([str(args.tuple_neg_disconnected_num),'neg_samples']))
    load_samples_dir=os.path.join('../../../data',args.dataset,"".join([str(args.tuple_neg_disconnected_num),'neg_samples.npy']))
    target_nodes_samples_dir=os.path.join('../../../data',args.dataset,"".join([str(args.target_tuple_neg_disconnected_num),'target_neg_samples']))
    target_nodes_load_samples_dir=os.path.join('../../../data',args.dataset,"".join([str(args.target_tuple_neg_disconnected_num),'target_neg_samples.npy']))
    hsamples_dir=os.path.join('../../../data',args.dataset,"".join([str(args.tuple_neg_unrelated_num),'neg_unrelated_samples']))
    hload_samples_dir=os.path.join('../../../data',args.dataset,"".join([str(args.tuple_neg_unrelated_num),'neg_unrelated_samples.npy']))
    target_nodes_tuple_num_dir=os.path.join('../../../data',args.dataset,"".join([str(args.tuple_neg_unrelated_num),'target_nodes_tuple_num']))
    load_target_nodes_tuple_num_dir=os.path.join('../../../data',args.dataset,"".join([str(args.tuple_neg_unrelated_num),'target_nodes_tuple_num.npy']))

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    trans_g=dgl.reverse(g)

    torch_sparse_adj=g.adj()
    torch_sparse_adj = torch_sparse_adj.to(device)

    if os.path.exists(load_samples_dir)==False:
        if args.dataset=='Freebase':
            samples=prompt_pretrain_sample(adjM+adjM.T, args.tuple_neg_disconnected_num)
        else:
            samples = prompt_pretrain_sample(adjM, args.tuple_neg_disconnected_num)
        np.save(samples_dir, samples)
    else:
        print("load sampleing tuples")
        samples=np.load(load_samples_dir,allow_pickle=True)
    samples=torch.tensor(samples,dtype=int)


    if args.target_pretrain:
        target_samples=target_samples.to(device)
    else:
        samples = samples.to(device)

    if args.hetero_pretrain:
        hsamples=hsamples.to(device)

    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    train_time=0
    early_stop=False
    for _ in range(args.repeat):
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        if args.model_type == 'gat':
            if args.hetero_subgraph:
                if args.dataset == 'DBLP':
                    net = dblp_hGAT(g, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False)
            else:
                net = GAT(g, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False)
        elif args.model_type == 'gcn':
            if args.hetero_subgraph:
                if args.dataset =='ACM':
                    if args.semantic_weight:
                        net=acm_sem_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                    elif args.each_loss:
                        net = acm_hGCN_each_loss(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                    else:
                        net = acm_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                elif args.dataset=='Freebase':
                    if args.freebase_type==0:
                        net=freebase_source_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                    elif args.freebase_type==1:
                        net=freebase_des_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                    else:
                        net=freebase_bi_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
            else:
                net = GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
        elif args.model_type == 'gin':
            net = GIN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.relu, args.dropout)
        elif args.model_type =='SHGN':
            num_classes = dl.labels_train['num_classes']
            heads = [args.num_heads] * args.num_layers + [1]
            net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        else:
            raise Exception('{} model is not defined!'.format(args.model_type))
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        if args.model_type=='SHGN':
            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                           save_path='../checkpoint/pretrain/shgn_checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        else:
            if args.dataset=='Freebase':
                if args.hetero_pretrain:
                    if args.hetero_subgraph:
                        if args.semantic_weight:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num))
                        elif args.each_loss:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num))

                        else:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num))
                    else:
                        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                       save_path='../checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                       format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                              args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num))
                else:
                    if args.hetero_subgraph:
                        if args.semantic_weight:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num))
                        elif args.each_loss:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num))
                        else:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num))
                    else:
                        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                       save_path='../checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}_{}.pt'.
                                                       format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                              args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num))

            else:
                if args.hetero_pretrain:
                    if args.hetero_subgraph:
                        if args.semantic_weight:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type))
                        elif args.each_loss:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type))

                        else:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type))
                    else:
                        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                       save_path='../checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}.pt'.
                                                       format(args.dataset, args.model_type,args.subgraph_hop_num,args.loss_weight,args.feats_type))
                else:
                    if args.hetero_subgraph:
                        if args.semantic_weight:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.feats_type))
                        elif args.each_loss:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type))
                        else:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path='../checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}.pt'.
                                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.feats_type))
                    else:
                        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                       save_path='../checkpoint/pretrain/checkpoint_{}_{}_{}_{}.pt'.
                                                       format(args.dataset, args.model_type,args.subgraph_hop_num,args.feats_type))

        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()
            if args.model_type=='SHGN':
                logits = net(features_list, e_feat)
                if args.subgraph_hop_num != 0:
                    for i in range(args.subgraph_hop_num):
                        logits = torch.sparse.mm(torch_sparse_adj, logits)
            else:
                if args.hetero_subgraph:
                    if args.each_loss:
                        logits,logits0,logits2,logits4,logits6 = net(g, features_list, e_feat)
                    else:
                        if args.dataset=='Freebase':
                            if args.freebase_type==2:
                                logits = net(g, trans_g, features_list, e_feat)
                            else:
                                logits = net(g, features_list, e_feat)
                        else:
                            logits = net(g,features_list,e_feat)
                else:
                    logits = net(features_list)
                #subgraph embedding
                    if args.subgraph_hop_num != 0:
                        for i in range(args.subgraph_hop_num):
                            logits = torch.sparse.mm(torch_sparse_adj, logits)
            if args.hetero_pretrain:
                htrain_loss=hcompareloss(logits,hsamples,args.temperature,device)
                htrain_loss.requires_grad_(True)
                total_loss=htrain_loss
            else:
                if args.each_loss:
                    train_loss = compareloss(logits, samples, args.temperature, device) + compareloss(logits0, samples,
                                                                                                      args.temperature,
                                                                                                      device) \
                                 + compareloss(logits2, samples, args.temperature, device) + compareloss(logits4,
                                                                                                         samples,
                                                                                                         args.temperature,
                                                                                                         device) \
                                 + compareloss(logits6, samples, args.temperature, device)
                else:
                    if args.target_pretrain:
                        train_loss = tcompareloss(logits, target_samples, args.temperature, device)
                    else:
                        train_loss = compareloss(logits, samples, args.temperature, device)
                    train_loss.requires_grad_(True)
                    total_loss=train_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            t_end = time.time()
            train_time+=t_end-t_start

            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, total_loss.item(), t_end-t_start))

            early_stopping(total_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                train_time = train_time / epoch
                early_stop = True
                break
        if early_stop == False:
            train_time = train_time / args.epoch

    print("####################################################")
    print('pretrain train time per epoch:',train_time)

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
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--model-type', type=str, default='gcn', help="gcn or gat")
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--device',type=int,default=1)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-6)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str, default='ACM')
    ap.add_argument('--seed', type=int, default="0")
    ap.add_argument('--tuple_neg_disconnected_num', type=int, default=1)
    ap.add_argument('--tuple_neg_unrelated_num', type=int, default=1)
    ap.add_argument('--target_tuple_neg_disconnected_num', type=int, default=1)
    ap.add_argument('--subgraph_hop_num', type=int, default=1)
    ap.add_argument('--subgraph_neighbor_num_bar', type=int, default=10)
    ap.add_argument('--temperature', type=float, default=1)
    #weight of unrelated loss
    ap.add_argument('--loss_weight', type=float, default=1)
    ap.add_argument('--hetero_pretrain', type=int, default=0)
    ap.add_argument('--target_pretrain', type=int, default=0)
    ap.add_argument('--hetero_subgraph', type=int, default=0)
    ap.add_argument('--semantic_weight', type=int, default=0)
    ap.add_argument('--each_loss', type=int, default=0)
    ap.add_argument('--freebase-type', type=int, default=2)
    ap.add_argument('--edge-feats', type=int, default=64)



    args = ap.parse_args()
    run_model_DBLP(args)
