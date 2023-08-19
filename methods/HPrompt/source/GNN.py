import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv

class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()


    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h

class acm_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(acm_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg
        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6=msg
        return {'m': res+res0+res2+res4+res6}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft2': h,'ft4': h,'ft6': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res



class acm_hGCN_each_loss(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(acm_hGCN_each_loss, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    def message_func(self,edges):
        res=edges.src['ft']
        return {'m': res}
    def message_func0(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def message_func2(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 2
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def message_func4(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 4
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def message_func6(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 6
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft2': h,'ft4': h,'ft6': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft0'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft2'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft4'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft6'))
        res = graph.dstdata['ft']
        res0 = graph.dstdata['ft0']
        res2 = graph.dstdata['ft2']
        res4 = graph.dstdata['ft4']
        res6 = graph.dstdata['ft6']
        return res,res0,res2,res4,res6


class acm_sem_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(acm_sem_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.semantic_weight)
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']*self.semantic_weight[0,0]
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg*self.semantic_weight[0,1]
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg*self.semantic_weight[0,2]
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg*self.semantic_weight[0,3]
        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6=msg*self.semantic_weight[0,4]
        return {'m': res+res0+res2+res4+res6}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft2': h,'ft4': h,'ft6': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class semantic_GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(semantic_GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h,self.semantic_weight

#book as source
class freebase_source_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(freebase_source_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3=msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg
        return {'m': res+res0+res1+res2+res3+res4}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft1': h,'ft2': h,'ft3': h,'ft4': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


#book as destination
class freebase_des_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(freebase_des_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3=msg
        return {'m': res+res0+res1+res2+res3}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft1': h,'ft2': h,'ft3': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

#freebase as source and des
class freebase_bi_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(freebase_bi_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    #book as source and 1hop subgraph
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3=msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg
        return {'m': res+res0+res1+res2+res3+res4}

    #book as destination
    def message_func0(self,edges):
        #type2 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3=msg
        return {'m': res1+res2+res3}
    def forward(self, graph, trans_graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft1': h,'ft2': h,'ft3': h,'ft4': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        trans_graph.edata.update({'e':e_feat})
        trans_graph.srcdata.update({'ft': h,'ft6': h,'ft14': h,'ft30': h})
        trans_graph.update_all(self.message_func0, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']+trans_graph.dstdata['ft']
        return res



class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))


    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


#error version, subgraph readout part has mistakes
# class dblp_hGAT(nn.Module):
#     def __init__(self,
#                  g,
#                  in_dims,
#                  num_hidden,
#                  num_classes,
#                  num_layers,
#                  heads,
#                  activation,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual):
#         super(dblp_hGAT, self).__init__()
#         self.g = g
#         self.num_layers = num_layers
#         self.gat_layers = nn.ModuleList()
#         self.activation = activation
#         self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
#         for fc in self.fc_list:
#             nn.init.xavier_normal_(fc.weight, gain=1.414)
#         self.gat_layers.append(GATConv(
#             num_hidden, num_hidden, heads[0],
#             feat_drop, attn_drop, negative_slope, False, self.activation))
#         # hidden layers
#         for l in range(1, num_layers):
#             # due to multi-head, the in_dim = num_hidden * num_heads
#             self.gat_layers.append(GATConv(
#                 num_hidden * heads[l-1], num_hidden, heads[l],
#                 feat_drop, attn_drop, negative_slope, residual, self.activation))
#         # output projection
#         # self.gat_layers.append(GATConv(
#         #     num_hidden * heads[-2], num_classes, heads[-1],
#         #     feat_drop, attn_drop, negative_slope, residual, None))
#         self.gat_layers.append(GATConv(
#             num_hidden * heads[-2], num_hidden, heads[-1],
#             feat_drop, attn_drop, negative_slope, residual, None))
#     def message_func_onehop(self,edges):
#         _mask = edges.data['e'] == 3
#         mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#         msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
#         #paper->author
#         res0=msg
#         return {'m': res0}
#
#     def message_func_twohop(self, edges):
#         # type0 subgraph
#         res=edges.src['ft']
#         _mask = edges.data['e'] == 0
#         mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#         msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
#         #author->paper->author
#         res0 = msg
#         # type2 subgraph
#         _mask = edges.data['e'] == 4
#         mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#         msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
#         #term->paper->author
#         res1 = msg
#         # type4 subgraph
#         _mask = edges.data['e'] == 5
#         mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#         msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
#         #venue->paper->author
#         res2 = msg
#         return {'m': res+ res0 + res1 + res2}
#     def forward(self, graph, features_list, e_feat):
#         e_feat=e_feat.reshape(-1,1).float()
#         h = []
#         for fc, feature in zip(self.fc_list, features_list):
#             h.append(fc(feature))
#         h = torch.cat(h, 0)
#         for l in range(self.num_layers):
#             h = self.gat_layers[l](self.g, h).flatten(1)
#         # output projection
#         h = self.gat_layers[-1](self.g, h).mean(1)
#         # h=self.sigmoid(h)
#         graph.edata.update({'e':e_feat})
#         graph.srcdata.update({'ft': h,'ft0': h,'ft3': h,'ft4': h,'ft5': h})
#         #1hop: paper->author
#         graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
#         graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
#         res = 2*graph.dstdata['ft_onehop']+graph.dstdata['ft_twohop']
#         return res

class dblp_hGAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(dblp_hGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
    def message_func_onehop(self,edges):
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop'], torch.zeros_like(edges.src['ft']))
        #paper->author
        res0=msg
        return {'m': res0}

    def message_func_twohop(self, edges):
        # type0 subgraph
        res=edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        #author->paper->author
        res0 = msg
        # type2 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        #term->paper->author
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        #venue->paper->author
        res2 = msg
        return {'m': res+ res0 + res1 + res2}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        h = self.gat_layers[-1](self.g, h).mean(1)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft3': h,'ft4': h,'ft5': h})
        graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
        graph.srcdata.update({'ft_twohop': graph.dstdata['ft_twohop']})
        graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
        res = graph.dstdata['ft_onehop']
        return res


class GIN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(GIN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act=torch.nn.ReLU()
        self.activation=activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.ginfunc=torch.nn.Sequential(torch.nn.Linear(num_hidden, num_hidden), self.act, torch.nn.Linear(num_hidden, num_hidden))
        self.GINlayer=dgl.nn.pytorch.conv.GINConv(apply_func=self.ginfunc,aggregator_type='sum')
        self.bn=torch.nn.BatchNorm1d(num_hidden)
        self.layers.append(self.GINlayer)
        self.bn_layers.append(self.bn)
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(self.GINlayer)
            self.bn_layers.append(self.bn)
        # output layer
        # self.output_ginfunc=torch.nn.Sequential(torch.nn.Linear(num_hidden, num_classes), self.act, torch.nn.Linear(num_classes, num_classes))
        # self.output_GINlayer=dgl.nn.pytorch.conv.GINConv(apply_func=self.output_ginfunc,aggregator_type='sum')
        # self.output_bn=torch.nn.BatchNorm1d(num_classes)
        # self.layers.append(self.output_GINlayer)
        # self.bn_layers.append(self.output_bn)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
            h=self.activation(h)
            h=self.bn_layers[i](h)
        return h
