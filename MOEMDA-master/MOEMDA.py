# -*- coding: utf-8 -*-
import dgl.nn.pytorch
import torch as th
from torch import nn
from dgl.nn import pytorch as pt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

# 专家网络
class ExpertNetwork(nn.Module):
    def __init__(self, args):
        super(ExpertNetwork, self).__init__()
        self.args = args
        self.elu = nn.ELU()
        
    def forward(self, graph, features):
        raise NotImplementedError("Expert networks must implement forward method")

# 专家1: Multi-hop GCN (MGCN) Expert 
class MGCNExpert(ExpertNetwork):
    def __init__(self, args, input_dim, output_dim):
        super(MGCNExpert, self).__init__(args)
        self.gcn1 = pt.GraphConv(input_dim, 640, allow_zero_in_degree=True)
        self.gcn2 = pt.GraphConv(640, 320, allow_zero_in_degree=True)
        self.gcn3 = pt.GraphConv(320, output_dim, allow_zero_in_degree=True)
        self.res_l = pt.nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, graph, features):
        res = self.elu(self.res_l(features))
        x = self.elu(self.gcn1(graph, features))
        x = self.dropout(x)
        x = self.elu(self.gcn2(graph, x))
        x = self.dropout(x)
        x = self.gcn3(graph, x)
        return x, res

# 专家2: Graph Isomorphism Network (GIN) Expert 
class GINExpert(ExpertNetwork):
    def __init__(self, args, input_dim, output_dim):
        super(GINExpert, self).__init__(args)
        self.gin1 = pt.GINConv(
            nn.Sequential(
                nn.Linear(input_dim, 640),
                nn.ELU(),
                nn.Linear(640, 640),
                nn.ELU()
            ), 'sum')
        self.gin2 = pt.GINConv(
            nn.Sequential(
                nn.Linear(640, 320),
                nn.ELU(),
                nn.Linear(320, 320),
                nn.ELU()
            ), 'sum')
        self.gin3 = pt.GINConv(
            nn.Sequential(
                nn.Linear(320, output_dim),
                nn.ELU(),
                nn.Linear(output_dim, output_dim),
                nn.ELU()
            ), 'sum')
        self.res_l = pt.nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, graph, features):
        res = self.elu(self.res_l(features))
        x = self.elu(self.gin1(graph, features))
        x = self.dropout(x)
        x = self.elu(self.gin2(graph, x))
        x = self.dropout(x)
        x = self.gin3(graph, x)
        return x, res

# 专家3: GraphSAGE (SAGE) Expert 
class SAGEExpert(ExpertNetwork):
    def __init__(self, args, input_dim, output_dim):
        super(SAGEExpert, self).__init__(args)
        self.sage1 = pt.SAGEConv(input_dim, 640, aggregator_type='mean', activation=nn.ELU())
        self.sage2 = pt.SAGEConv(640, 320, aggregator_type='mean', activation=nn.ELU())
        self.sage3 = pt.SAGEConv(320, output_dim, aggregator_type='mean')
        self.res_l = pt.nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, graph, features):
        res = self.elu(self.res_l(features))
        x = self.dropout(self.sage1(graph, features))
        x = self.dropout(self.sage2(graph, x))
        x = self.elu(self.sage3(graph, x))
        return x, res

# 专家4: Multi-Layer Perceptron (MLP) Expert 
class MLPExpert(ExpertNetwork):
    def __init__(self, args, input_dim, output_dim):
        super(MLPExpert, self).__init__(args)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 640),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(640, 320),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(320, output_dim),
            nn.ELU()
        )
        self.res_l = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, graph, features):
        res = self.elu(self.res_l(features))
        x = self.mlp(features)
        return x, res

# 门控网络 
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
        self.expert_performance = nn.Parameter(th.ones(num_experts))  
        self.temperature = nn.Parameter(th.tensor(1.0))  
        
    def forward(self, x, expert_outputs=None):
        base_weights = self.gate(x)
        
        if expert_outputs is not None:
            quality_scores = []
            for output in expert_outputs:
                variance = th.var(output, dim=1, keepdim=True)
                entropy = -th.sum(output * th.log(output + 1e-10), dim=1, keepdim=True)
                quality = variance * entropy
                quality_scores.append(quality)
            
            quality_scores = th.cat(quality_scores, dim=1)
            
         
            combined_weights = base_weights * self.expert_performance * quality_scores
            weights = F.softmax(combined_weights / self.temperature, dim=1)
        else:
            weights = base_weights
        
        return weights

# MOEMDA模型
class MDA_MOE(nn.Module):
    def __init__(self, args):
        super(MDA_MOE, self).__init__()
        self.args = args
        
        # 特征转换层
        self.lin_m = nn.Linear(args.miRNA_number, args.in_feats, bias=False)
        self.lin_d = nn.Linear(args.disease_number, args.in_feats, bias=False)
        
        # 专家数量
        self.num_experts = 4
        
       
        self.m_experts = nn.ModuleList([
            MGCNExpert(args, args.miRNA_number, args.out_feats),
            GINExpert(args, args.miRNA_number, args.out_feats),
            SAGEExpert(args, args.miRNA_number, args.out_feats),
            GATExpert(args, args.miRNA_number, args.out_feats)
        ])
        
       
        self.d_experts = nn.ModuleList([
            MGCNExpert(args, args.disease_number, args.out_feats),
            GINExpert(args, args.disease_number, args.out_feats),
            SAGEExpert(args, args.disease_number, args.out_feats),
            GATExpert(args, args.disease_number, args.out_feats)
        ])
        
        
        self.md_experts = nn.ModuleList([
            MGCNExpert(args, args.in_feats, args.out_feats),
            GINExpert(args, args.in_feats, args.out_feats),
            SAGEExpert(args, args.in_feats, args.out_feats),
            GATExpert(args, args.in_feats, args.out_feats)
        ])
        
        # 门控网络
        self.m_gate = GatingNetwork(args.miRNA_number, self.num_experts)
        self.d_gate = GatingNetwork(args.disease_number, self.num_experts)
        self.md_gate = GatingNetwork(args.in_feats, self.num_experts)
        
        # 预测层
        self.elu = nn.ELU()
        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(args.dropout)
        in_feat = 2 * args.out_feats
        for idx, out_feat in enumerate(args.mlp):
            if idx == 0:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('elu', nn.ELU())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat
            else:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('sigmoid', nn.Sigmoid())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat
                
        # 特征融合参数
        self.fuse_weight_m = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight_d = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight_md = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        
        self.fuse_weight_1 = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        
     
        self.fuse_weight_m.data.fill_(0.5)
        self.fuse_weight_d.data.fill_(0.5)
        self.fuse_weight_md.data.fill_(0.5)
        self.fuse_weight_1.data.fill_(0.5)
        self.fuse_weight_2.data.fill_(0.5)
        
    def forward(self, mm_graph, dd_graph, md_graph, miRNA, disease, samples):
        # 将数据移动到指定设备
        mm_graph = mm_graph.to(device)
        dd_graph = dd_graph.to(device)
        md_graph = md_graph.to(device)
        miRNA = miRNA.to(device)
        disease = disease.to(device)
        
        md = th.cat((self.lin_m(miRNA), self.lin_d(disease)), dim=0)
        
      
        m_weights = self.m_gate(miRNA)
        
   
        m_expert_outputs = []
        m_expert_res = []
        for i, expert in enumerate(self.m_experts):
            output, res = expert(mm_graph, miRNA)
            m_expert_outputs.append(output.unsqueeze(1))
            m_expert_res.append(res.unsqueeze(1))
        
     
        m_expert_outputs = th.cat(m_expert_outputs, dim=1)
        m_expert_res = th.cat(m_expert_res, dim=1)
        
  
        m_weights = m_weights.unsqueeze(2).expand(-1, -1, self.args.out_feats)
        emb_mm_sim_3 = (m_expert_outputs * m_weights).sum(dim=1)
        res_mi = (m_expert_res * m_weights).sum(dim=1)
        

        d_weights = self.d_gate(disease)
        

        d_expert_outputs = []
        d_expert_res = []
        for i, expert in enumerate(self.d_experts):
            output, res = expert(dd_graph, disease)
            d_expert_outputs.append(output.unsqueeze(1))
            d_expert_res.append(res.unsqueeze(1))
        
        d_expert_outputs = th.cat(d_expert_outputs, dim=1)
        d_expert_res = th.cat(d_expert_res, dim=1)
        
        d_weights = d_weights.unsqueeze(2).expand(-1, -1, self.args.out_feats)
        emb_dd_sim_3 = (d_expert_outputs * d_weights).sum(dim=1)
        res_di = (d_expert_res * d_weights).sum(dim=1)

        md_weights = self.md_gate(md)
        
        md_expert_outputs = []
        md_expert_res = []
        for i, expert in enumerate(self.md_experts):
            output, res = expert(md_graph, md)
            md_expert_outputs.append(output.unsqueeze(1))
            md_expert_res.append(res.unsqueeze(1))
        
        md_expert_outputs = th.cat(md_expert_outputs, dim=1)
        md_expert_res = th.cat(md_expert_res, dim=1)
        
        md_weights = md_weights.unsqueeze(2).expand(-1, -1, self.args.out_feats)
        emb_ass_3 = (md_expert_outputs * md_weights).sum(dim=1)
        res_mmdd = (md_expert_res * md_weights).sum(dim=1)
        

        emb_mm_sim_3 = self.fuse_weight_m * emb_mm_sim_3 + (1 - self.fuse_weight_m) * res_mi
        
        emb_dd_sim_3 = self.fuse_weight_d * emb_dd_sim_3 + (1 - self.fuse_weight_d) * res_di
        
        emb_ass_3 = self.fuse_weight_md * emb_ass_3 + (1 - self.fuse_weight_md) * res_mmdd
        
        emb_mm_ass = emb_ass_3[:self.args.miRNA_number, :]
        emb_dd_ass = emb_ass_3[self.args.miRNA_number:, :]
        
        emb_mm = self.fuse_weight_1 * emb_mm_sim_3 + (1 - self.fuse_weight_1) * emb_mm_ass
        emb_dd = self.fuse_weight_2 * emb_dd_sim_3 + (1 - self.fuse_weight_2) * emb_dd_ass
        
        # 预测miRNA-疾病关联
        emb = th.cat((emb_mm[samples[:, 0]], emb_dd[samples[:, 1]]), dim=1)
        result = self.mlp(emb)
        
        return result, emb_mm_sim_3, emb_mm_ass, emb_dd_sim_3, emb_dd_ass, emb

    