import torch
import torch.nn as nn
from dgl import function as fn
import dgl
import torch.nn.functional as F

class node_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,input_dim):
        super(node_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.dropout = nn.Dropout(p=0.2)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph, graph_embedding):
        # emb=graph_embedding*self.weight
        emb=F.elu(graph_embedding*self.weight)
        # emb = self.dropout(emb)
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']

class freebase_node_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,input_dim, g):
        super(freebase_node_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.dropout = nn.Dropout(p=0.2)
        self.r_graph=dgl.reverse(g)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph, graph_embedding):
        # emb=graph_embedding*self.weight
        emb=F.elu(graph_embedding*self.weight)
        # emb = self.dropout(emb)
        r_graph=self.r_graph
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        r_graph.srcdata.update({'ft': emb})
        r_graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']+r_graph.dstdata['ft']


class node_bottle_net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(node_bottle_net, self).__init__()
        # self.weight0= torch.nn.Parameter(torch.Tensor(input_dim,hidden_dim))
        # self.weight1= torch.nn.Parameter(torch.Tensor(hidden_dim,output_dim))
        self.linear0=torch.nn.Linear(input_dim,hidden_dim)
        self.linear1=torch.nn.Linear(hidden_dim,output_dim)
        # self.reset_parameters()
    # def reset_parameters(self):
    #     torch.nn.init.xavier_uniform_(self.weight0)
    #     torch.nn.init.xavier_uniform_(self.weight1)
    def message_func(edges):
        return {'m': edges.dst['h']}
    def forward(self, graph, graph_embedding):
        # emb=graph_embedding*self.weight
        # emb=F.elu(torch.matmul(torch.matmul(graph_embedding,self.weight0),self.weight1))
        emb=F.elu(self.linear1(self.linear0(graph_embedding)))
        # emb = self.dropout(emb)
        graph.srcdata.update({'ft': emb})
        graph.dstdata.update({'ft':emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']

class hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 negative_slope=0.2
                 ):
        super(hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def message_func(self,edges):
        return {'m': torch.cat((edges.src['ft'],edges.data['e']),dim=1)}
    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        emb=graph_embedding*self.weight
        graph.srcdata.update({'ft': emb})
        graph.edata.update({'e': e_feat})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class acm_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(acm_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
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



    #add pre semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.pre_semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    #add semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        emb=emb0=emb2=emb4=emb6=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        # res=F.sigmoid(res)
        return res

#acm semantic prompt
class acm_hnode_semantic_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 semantic_prompt_weight=0.1
                 ):
        super(acm_hnode_semantic_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.semantic_prompt_weight=semantic_prompt_weight
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
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

    #add pre semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.pre_semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    # add semantic prompt
    # def message_func_semantic(self,edges):
    #     #all type subgraph
    #     semantic=self.semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*(1+self.semantic_prompt_weight*semantic[0,1])
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*(1+self.semantic_prompt_weight*semantic[0,2])
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*(1+self.semantic_prompt_weight*semantic[0,3])
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*(1+self.semantic_prompt_weight*semantic[0,4])
    #     return {'m': res+res0+res2+res4+res6}

    def message_func_semantic(self,edges):
        #all type subgraph
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        res=edges.src['ft']*semantic[0,0]
        #type0 subgraph
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg*semantic[0,1]
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg*semantic[0,2]
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg*semantic[0,3]
        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6=msg*semantic[0,4]
        return {'m': res+res0+res2+res4+res6}


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        emb=emb0=emb2=emb4=emb6=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
        graph.update_all(self.message_func_semantic, fn.sum('m', 'ft_s'))
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']+self.semantic_prompt_weight*graph.dstdata['ft_s']
        # res = graph.dstdata['ft_s']
        # res=F.sigmoid(res)
        return res


class acm_eachloss_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(acm_eachloss_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

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

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        emb=graph_embedding*self.weight
        emb0=graph_embedding*self.weight
        emb2=graph_embedding*self.weight
        emb4=graph_embedding*self.weight
        emb6=graph_embedding*self.weight
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
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


class acm_meta_path_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 negative_slope=0.2
                 ):
        super(acm_meta_path_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']*self.semantic_weight[0,0]
        #type0 subgraph
        _mask = edges.data['e'] == 0
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

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        emb=graph_embedding*self.weight
        emb0=graph_embedding*self.weight
        emb2=graph_embedding*self.weight
        emb4=graph_embedding*self.weight
        emb6=graph_embedding*self.weight

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

# book as source and target
class freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 semantic_prompt_weight=0.1
                 ):
        super(freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,9))
        # self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.pre_semantic_weight=pre_semantic_weight
        # self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.semantic_prompt_weight=semantic_prompt_weight
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

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

    # book as source and 1hop subgraph
    def message_func_semantic(self, edges):
        # all type subgraph
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        res = edges.src['ft']
        res=res*semantic[0,0]
        # type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg*semantic[0,1]
        # type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg*semantic[0,2]
        # type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg*semantic[0,3]
        # type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg*semantic[0,4]
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg*semantic[0,5]
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    # book as destination
    def message_func0_semantic(self, edges):
        # type2 subgraph
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1 = msg*semantic[0,6]
        # type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2 = msg*semantic[0,7]
        # type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3 = msg*semantic[0,8]
        return {'m': res1 + res2 + res3}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # torch.nn.init.xavier_uniform_(self.weight0)
        # torch.nn.init.xavier_uniform_(self.weight2)
        # torch.nn.init.xavier_uniform_(self.weight4)
        # torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, trans_graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb = emb0 = emb1 = emb2 = emb3 = emb4 = emb6 = emb14 = emb30 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft1': emb1, 'ft2': emb2, 'ft3': emb3, 'ft4': emb4})
        # graph.update_all(self.message_func, fn.sum('m', 'ft'))
        graph.update_all(self.message_func_semantic, fn.sum('m', 'ft_semantic'))
        trans_graph.edata.update({'e': e_feat})
        trans_graph.srcdata.update({'ft': emb, 'ft6': emb6, 'ft14': emb14, 'ft30': emb30})
        # trans_graph.update_all(self.message_func0, fn.sum('m', 'ft'))
        trans_graph.update_all(self.message_func0_semantic, fn.sum('m', 'ft_semantic'))
        res = graph.dstdata['ft'] + trans_graph.dstdata['ft']\
              +self.semantic_prompt_weight*(graph.dstdata['ft_semantic'] + trans_graph.dstdata['ft_semantic'])
        # res = graph.dstdata['ft_semantic'] + trans_graph.dstdata['ft_semantic']
        return res



class freebase_bidirection_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(freebase_bidirection_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        # self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.pre_semantic_weight=pre_semantic_weight
        # self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    # book as source and 1hop subgraph
    def message_func(self, edges):
        # all type subgraph
        res = edges.src['ft']
        # type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        # type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        # type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    # book as destination
    def message_func0(self, edges):
        # type2 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        # type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        return {'m': res1 + res2 + res3}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # torch.nn.init.xavier_uniform_(self.weight0)
        # torch.nn.init.xavier_uniform_(self.weight2)
        # torch.nn.init.xavier_uniform_(self.weight4)
        # torch.nn.init.xavier_uniform_(self.weight6)
        # torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, trans_graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb1=emb2=emb3=emb4=emb6=emb14=emb30=F.elu(graph_embedding*self.weight)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft1': emb1,'ft2': emb2,'ft3': emb3,'ft4': emb4})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        trans_graph.edata.update({'e':e_feat})
        trans_graph.srcdata.update({'ft': emb,'ft6': emb6,'ft14': emb14,'ft30': emb30})
        trans_graph.update_all(self.message_func0, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']+trans_graph.dstdata['ft']
        return res


    # book as source
class freebase_source_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(freebase_source_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        # self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.pre_semantic_weight=pre_semantic_weight
        # self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        # all type subgraph
        res = edges.src['ft']
        # type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        # type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        # type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # torch.nn.init.xavier_uniform_(self.weight0)
        # torch.nn.init.xavier_uniform_(self.weight2)
        # torch.nn.init.xavier_uniform_(self.weight4)
        # torch.nn.init.xavier_uniform_(self.weight6)
        # torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb1=emb2=emb3=emb4=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft1': emb1,'ft2': emb2,'ft3': emb3,'ft4': emb4})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        # res=F.sigmoid(res)
        return res

#book as destination
class freebase_des_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(freebase_des_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.pre_semantic_weight=pre_semantic_weight
        # self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

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

    #add pre semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.pre_semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    #add semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # torch.nn.init.xavier_uniform_(self.weight0)
        # torch.nn.init.xavier_uniform_(self.weight2)
        # torch.nn.init.xavier_uniform_(self.weight4)
        # torch.nn.init.xavier_uniform_(self.weight6)
        # torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb1=emb2=emb3=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft1': emb1,'ft2': emb2,'ft3': emb3})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        # res=F.sigmoid(res)
        return res


#error version, mistakes in the subgraph readout part
# class dblp_hnode_prompt_layer_feature_weighted_sum(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  pre_semantic_weight=None,
#                  negative_slope=0.2,
#                  ):
#         super(dblp_hnode_prompt_layer_feature_weighted_sum, self).__init__()
#         self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
#         self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
#         self.weight1= torch.nn.Parameter(torch.Tensor(1,input_dim))
#         self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
#         self.pre_semantic_weight=pre_semantic_weight
#         self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,4))
#         self.leaky_relu = nn.LeakyReLU(negative_slope)
#         self.reset_parameters()
#     def weighted_sum(self, input, weights):
#         weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
#         weighted_inputs = input * weights
#         output = weighted_inputs
#         return output
#
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
#
#     #add pre semantic prompt
#     # def message_func(self,edges):
#     #     #all type subgraph
#     #     semantic=self.pre_semantic_weight
#     #     semantic=F.normalize(semantic, p=2, dim=1)
#     #     res=edges.src['ft']*semantic[0,0]
#     #     #type0 subgraph
#     #     _mask = edges.data['e'] == 0
#     #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#     #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
#     #     res0=msg*semantic[0,1]
#     #     #type2 subgraph
#     #     _mask = edges.data['e'] == 2
#     #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#     #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
#     #     res2=msg*semantic[0,2]
#     #     #type4 subgraph
#     #     _mask = edges.data['e'] == 4
#     #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#     #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
#     #     res4=msg*semantic[0,3]
#     #     #type6 subgraph
#     #     _mask = edges.data['e'] == 6
#     #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#     #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
#     #     res6=msg*semantic[0,4]
#     #     return {'m': res+res0+res2+res4+res6}
#
#     #add semantic prompt
#     # def message_func(self,edges):
#     #     #all type subgraph
#     #     semantic=self.semantic_weight
#     #     semantic=F.normalize(semantic, p=2, dim=1)
#     #     res=edges.src['ft']*semantic[0,0]
#     #     #type0 subgraph
#     #     _mask = edges.data['e'] == 0
#     #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#     #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
#     #     res0=msg*semantic[0,1]
#     #     #type2 subgraph
#     #     _mask = edges.data['e'] == 2
#     #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#     #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
#     #     res2=msg*semantic[0,2]
#     #     #type4 subgraph
#     #     _mask = edges.data['e'] == 4
#     #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#     #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
#     #     res4=msg*semantic[0,3]
#     #     #type6 subgraph
#     #     _mask = edges.data['e'] == 6
#     #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
#     #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
#     #     res6=msg*semantic[0,4]
#     #     return {'m': res+res0+res2+res4+res6}
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight)
#         torch.nn.init.xavier_uniform_(self.weight0)
#         torch.nn.init.xavier_uniform_(self.weight1)
#         torch.nn.init.xavier_uniform_(self.weight2)
#         torch.nn.init.xavier_uniform_(self.semantic_weight)
#
#     def forward(self, graph, graph_embedding,e_feat):
#         e_feat=e_feat.reshape(-1,1).float()
#
#         # emb=graph_embedding*self.weight
#         # emb0=graph_embedding*self.weight0
#         # emb2=graph_embedding*self.weight2
#         # emb4=graph_embedding*self.weight4
#         # emb6=graph_embedding*self.weight6
#         # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
#         emb=emb0=emb3=emb4=emb5=F.elu(graph_embedding*self.weight)
#
#         graph.edata.update({'e':e_feat})
#         graph.srcdata.update({'ft': emb,'ft0': emb0,'ft3': emb3,'ft4': emb4,'ft5': emb5})
#         graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
#         graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
#         res = 2*graph.dstdata['ft_onehop']+graph.dstdata['ft_twohop']
#         # res=F.sigmoid(res)
#         return res

class dblp_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(dblp_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight1= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,4))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

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

    #add pre semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.pre_semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    #add semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb3=emb4=emb5=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft3': emb3,'ft4': emb4,'ft5': emb5})
        graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
        graph.srcdata.update({'ft_twohop': graph.dstdata['ft_twohop']})
        graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
        res = graph.dstdata['ft_onehop']
        # res=F.sigmoid(res)
        return res

class dblp_hnode_semantic_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 semantic_prompt_weight=0.1
                 ):
        super(dblp_hnode_semantic_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight1= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.semantic_prompt_weight=semantic_prompt_weight
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output
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

    def message_func_onehop_semantic(self,edges):
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop_s'], torch.zeros_like(edges.src['ft']))
        #paper->author
        res0=msg*semantic[0,0]
        return {'m': res0}
    def message_func_twohop_semantic(self, edges):
        # type0 subgraph
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        res=edges.src['ft']*semantic[0,1]
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        #author->paper->author
        res0 = msg*semantic[0,2]
        # type2 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        #term->paper->author
        res1 = msg*semantic[0,3]
        # type4 subgraph
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        #venue->paper->author
        res2 = msg*semantic[0,4]
        return {'m': res+ res0 + res1 + res2}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb3=emb4=emb5=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft3': emb3,'ft4': emb4,'ft5': emb5})
        graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
        graph.srcdata.update({'ft_twohop': graph.dstdata['ft_twohop']})
        graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
        graph.update_all(self.message_func_twohop_semantic, fn.sum('m', 'ft_twohop_s'))
        graph.srcdata.update({'ft_twohop_s': graph.dstdata['ft_twohop_s']})
        graph.update_all(self.message_func_onehop_semantic, fn.sum('m', 'ft_onehop_s'))
        res = graph.dstdata['ft_onehop']+self.semantic_prompt_weight*graph.dstdata['ft_onehop_s']
        # res = graph.dstdata['ft_onehop_s']
        # res=F.sigmoid(res)
        return res


class node_prompt_layer_feature_cat(nn.Module):
    def __init__(self,prompt_dim):
        super(node_prompt_layer_feature_cat, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,prompt_dim))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph,graph_embedding):
        graph_embedding=torch.cat([graph_embedding,torch.broadcast_to(self.weight,(graph_embedding.size(0),self.weight.size(1)))],dim=1)
        emb=graph_embedding
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class node_prompt_layer_feature_cat_edge(nn.Module):
    def __init__(self,prompt_dim):
        super(node_prompt_layer_feature_cat_edge, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,prompt_dim))
        self.prompt_dim=prompt_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def message_func(self,edges):
        return {'m': torch.cat((edges.src['ft'],edges.data['p']),dim=1)}
    def forward(self, graph,graph_embedding):
        emb=graph_embedding
        graph.srcdata.update({'ft': emb})
        enum=graph.num_edges()
        graph.edata.update({'p':torch.broadcast_to(self.weight,(enum,self.prompt_dim))})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class hnode_prompt_layer_feature_cat_edge(nn.Module):
    # def __init__(self,prompt_dim):
    def __init__(self,prompt_dim,heterprompt_dim):
        super(hnode_prompt_layer_feature_cat_edge, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,prompt_dim))
        self.hetero_prompt = torch.nn.Parameter(torch.Tensor(1, heterprompt_dim))
        self.hetero_dim=heterprompt_dim
        self.hetero_prompt=torch.nn.Parameter(torch.Tensor(1,heterprompt_dim))
        self.prompt_dim=prompt_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.hetero_prompt)
    def message_func(self,edges):
        return {'m': torch.cat((edges.src['ft']*edges.data['p'],edges.data['e']),dim=1)}
    def forward(self, graph, graph_embedding,e_feat):
        graph.srcdata.update({'ft': graph_embedding})
        enum=graph.num_edges()
        graph.edata.update({'p':torch.broadcast_to(self.weight,(enum,self.prompt_dim))})
        graph.edata.update({'hp':torch.broadcast_to(self.hetero_prompt,(enum,self.hetero_dim))})
        graph.edata.update({'e':e_feat})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class node_prompt_layer_feature_sum(nn.Module):
    def __init__(self):
        super(node_prompt_layer_feature_sum, self).__init__()
    def forward(self, graph,graph_embedding):
        graph.srcdata.update({'ft': graph_embedding})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']

class hnode_prompt_layer_feature_sum(nn.Module):
    def __init__(self,negative_slope=0.2
                 ):
        super(hnode_prompt_layer_feature_sum, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
    def message_func(self,edges):
        return {'m': torch.cat((edges.src['ft'],edges.data['e']),dim=1)}
    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        emb=graph_embedding
        graph.srcdata.update({'ft': emb})
        graph.edata.update({'e': e_feat})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

def distance2center(f,center):
    _f=torch.broadcast_to(f,(center.size(0),f.size(0),f.size(1)))
    _center=torch.broadcast_to(center,(f.size(0),center.size(0),center.size(1)))
    _f=_f.permute(1,0,2)
    _center=_center.reshape(-1,_center.size(2))
    _f=_f.reshape(-1,_f.size(2))
    cos=torch.cosine_similarity(_f,_center,dim=1)
    res=cos
    res=res.reshape(f.size(0),center.size(0))
    return res

#calculate the center embedding of each class
def center_embedding(input,index,label_num=0,debug=False):
    device=input.device
    mean = torch.ones(index.size(0)).to(device)
    _mean = torch.zeros(label_num, device=device).scatter_add_(dim=0, index=index, src=mean)
    index=index.reshape(-1,1)
    index = index.expand(input.size())
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index, src=input)
    _mean=_mean.reshape(-1,1)
    c = c / _mean
    return c

from dgl.nn.pytorch import GraphConv
class hprompt_gcn(nn.Module):
    def __init__(self,
                 input_dim,
                 negative_slope=0.2
                 ):
        super(hprompt_gcn, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, input_dim, weight=False))
        # hidden layers
        for i in range(2 - 1):
            self.layers.append(GraphConv(input_dim, input_dim))
        self.dropout = nn.Dropout(p=0.2)
        # self.gcn=GraphConv(input_dim, input_dim)
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        # res0=msg*self.semantic_weight[0,1]
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        # res2=msg*self.semantic_weight[0,2]
        res2=msg
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        # res4=msg*self.semantic_weight[0,3]
        res4=msg

        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        # res6=msg*self.semantic_weight[0,4]
        res6=msg
        return {'m': res+res0+res2+res4+res6}

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        # graph_embedding=self.gcn(graph,graph_embedding)
        for i, layer in enumerate(self.layers):
            graph_embedding = self.dropout(graph_embedding)
            graph_embedding = layer(graph, graph_embedding)
        emb=graph_embedding
        emb0=graph_embedding
        emb2=graph_embedding
        emb4=graph_embedding
        emb6=graph_embedding
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class prompt_gcn(nn.Module):
    def __init__(self,input_dim):
        super(prompt_gcn, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, input_dim, weight=False))
        # hidden layers
        for i in range(2 - 1):
            self.layers.append(GraphConv(input_dim, input_dim))
        self.dropout = nn.Dropout(p=0.2)
        self.gcn=GraphConv(input_dim, input_dim)
    def forward(self, graph, graph_embedding):
        for i, layer in enumerate(self.layers):
            graph_embedding = self.dropout(graph_embedding)
            graph_embedding = layer(graph, graph_embedding)
        emb=graph_embedding
        # emb=self.gcn(graph,graph_embedding)
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']


