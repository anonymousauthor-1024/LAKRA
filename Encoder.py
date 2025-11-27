import torch
from torch.functional import Tensor
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import RGCNConv, FastRGCNConv
from utils import cconv, cconv_new, ccorr, ccorr_new, rotate
from torch_scatter import scatter_add
import math
from torch.nn import ModuleList, Sequential
from torch_geometric.utils import softmax as geometric_softmax

class CompGATv3(MessagePassing):
    def __init__(self, in_channels, out_channels, rel_dim, drop, bias, op, beta, num_heads=4):
        super(CompGATv3, self).__init__(aggr = "add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.bias = bias
        self.beta = beta
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        # self.w_loop = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_in = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_out = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_rel_update = torch.nn.Linear(in_channels, out_channels).cuda()
        
        # Multi-head attention mechanism
        self.w_query = torch.nn.Linear(in_channels, out_channels, bias=False).cuda()  # Use only entity embedding
        self.w_query_concat = torch.nn.Linear(2*in_channels, out_channels, bias=False).cuda()  # For concat version
        self.w_key = torch.nn.Linear(in_channels, out_channels, bias=False).cuda()
        #self.w_value = torch.nn.Linear(in_channels, out_channels, bias=False).cuda()
        
        # Improved attention network
        self.w_att = torch.nn.Linear(3*in_channels, out_channels).cuda()
        self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()
        
        # Relation bias and gating mechanism
        self.rel_bias = torch.nn.Linear(in_channels, 1, bias=False).cuda()
        self.gate = torch.nn.Linear(2*in_channels, 1, bias=False).cuda()
        
        # Transformer-style Layer Normalization
        #self.layer_norm1 = torch.nn.LayerNorm(out_channels).cuda()
        #self.layer_norm2 = torch.nn.LayerNorm(out_channels).cuda()
        
        # Position-aware attention (for graph structure)
        #self.pos_enc = torch.nn.Embedding(1000, in_channels).cuda()  # Adjustable size
        
        # Type Information Propagation components
        self.w_t = torch.nn.Linear(in_channels, out_channels, bias=False).cuda()
        
        self.drop_ratio = drop
        self.drop = torch.nn.Dropout(drop, inplace=False)
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        
        self.res_w = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.activation = torch.nn.Tanh() #torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    
    def compute_type_information_optimized(self, x, edge_index, edge_type, rel_emb):
        """
        Optimized computation of type information m_i^type for each entity
        Formula: m_i^type = Σ_{e_k ∈ N_E(i)} α_{ik} (W_t e_k)
        
        This version combines attention computation and aggregation for better efficiency
        """
        device = x.device
        num_ent = x.size(0)
        rel_num = rel_emb.size(0)
        original_rel_num = rel_num // 2
        
        # Get source and destination nodes from edge information
        src_nodes = edge_index[0]  # Source nodes (neighbors)
        dst_nodes = edge_index[1]  # Destination nodes
        
        # Vectorized computation of type attention scores
        # Find corresponding inverse relations for all edges at once
        inverse_relations = torch.where(
            edge_type < original_rel_num,
            edge_type + original_rel_num,  # Original -> Inverse
            edge_type - original_rel_num   # Inverse -> Original
        )
        
        # Get relation embeddings for current and inverse relations
        current_rel_embs = rel_emb[edge_type]      # [num_edges, rel_dim]
        inverse_rel_embs = rel_emb[inverse_relations]  # [num_edges, rel_dim]
        
        # Vectorized cosine similarity computation
        cos_sim = F.cosine_similarity(current_rel_embs, inverse_rel_embs, dim=1)
        type_attention_scores = (cos_sim + 1) / 2  # Map from [-1, 1] to [0, 1]
        
        # Normalize attention scores using scatter_softmax for better efficiency
        
        type_attention_normalized = geometric_softmax(type_attention_scores, dst_nodes)
        
        # Get neighbor embeddings and apply W_t transformation
        neighbor_embeddings = x[src_nodes]  # Shape: [num_edges, in_channels]
        transformed_neighbors = self.w_t(neighbor_embeddings)  # Shape: [num_edges, out_channels]
        
        # Apply attention weights
        weighted_neighbors = transformed_neighbors * type_attention_normalized.unsqueeze(1)
        
        # Aggregate to destination nodes using scatter_add
        type_info = torch.zeros(num_ent, self.out_channels, device=device, dtype=x.dtype)
        type_info.scatter_add_(0, dst_nodes.unsqueeze(1).expand(-1, self.out_channels), weighted_neighbors)
        
        return type_info
    def forward(self, x, edge_index, edge_type, rel_emb, pre_alpha=None, r_emb_triple=None):
        # rel_emb = torch.cat([rel_emb, self.loop_rel], dim=0)
        
        num_ent = x.size(0)

        # Original neighbor aggregation
        in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_emb, pre_alpha=pre_alpha, r_emb_triple=r_emb_triple)
        loop_res = self.res_w(x)        
        out = self.drop(in_res) + self.drop(loop_res)

        # Type Information Propagation (Optimized version)
        # Compute type information m_i^type for each entity in one step
        #type_info = self.compute_type_information_optimized(x, edge_index, edge_type, rel_emb)
        
        # Add type information to final entity embeddings
        #out = out + self.drop(type_info)  # Apply dropout to type information as well

        if self.bias:
            out = out + self.bias_value
        out = self.bn(out)
        #out = self.layer_norm1(out)
        out = self.activation(out)
        
        
        return out, self.w_rel_update(rel_emb), self.alpha.detach()

    def message(self,x_i, x_j, edge_type, rel_emb, ptr, index, size_i, pre_alpha, r_emb_triple=None):
        
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        num_edge = xj_rel.size(0)//2

        in_message = xj_rel[:num_edge]
        out_message = xj_rel[num_edge:]
                
        trans_in = self.w_in(in_message)
        trans_out = self.w_out(out_message)
        
        out = torch.cat((trans_in, trans_out), dim=0)
        

        # Multi-head attention, relation-aware, gating mechanism, layer normalization
        
        # 1. Multi-Head Attention mechanism
        batch_size = x_i.size(0)
        '''
        if r_emb_triple is not None:
            # Use concat method instead of addition
            query_input = torch.cat([x_i, r_emb_triple], dim=-1)
            query = self.w_query_concat(query_input).view(batch_size, self.num_heads, self.head_dim)
        else:
        '''
        #print("Using original query method")
        #query = self.w_query(x_i).view(batch_size, self.num_heads, self.head_dim)
        #key = self.w_key(x_j).view(batch_size, self.num_heads, self.head_dim)
        #value = self.w_value(x_j).view(batch_size, self.num_heads, self.head_dim)
        
        # 2. Relation-aware Query-Key adjustment
        #rel_emb_projected = self.w_rel(rel_emb).view(batch_size, self.num_heads, self.head_dim)
        
        # Integrate relation information into key
        #key_rel = key + rel_emb_projected
        
        # 3. Scaled Dot-Product Attention
        #attention_scores = torch.sum(query * key_rel, dim=-1) / math.sqrt(self.head_dim)  # [batch_size, num_heads]
        
        # 4. Relation-specific bias
        #rel_bias = self.rel_bias(rel_emb).squeeze(-1)  # [batch_size]
        #attention_scores = attention_scores + rel_bias.unsqueeze(1)  # Broadcasting
        
        # 5. Traditional MLP attention as supplement
        mlp_attention = self.leaky_relu(
            self.w_att(torch.cat((x_i, rel_emb, x_j), dim=1))
        ).cuda()
        mlp_attention = self.a(mlp_attention).squeeze(-1)  # [batch_size]
        #mlp_attention = self.a(mlp_attention).float()
        
        # 6. Gating mechanism to fuse multiple attention types
        #gate_input = torch.cat((x_i, x_j), dim=1)
        #gate_weight = torch.sigmoid(self.gate(gate_input)).squeeze(-1)  # [batch_size]
        
        # Fuse averaged multi-head attention with MLP attention
        #multi_head_avg = attention_scores.mean(dim=1)  # [batch_size]
        #final_attention = gate_weight * multi_head_avg + (1 - gate_weight) * mlp_attention
        
        # 7. Softmax normalization
        #alpha = softmax(final_attention.unsqueeze(-1), index, ptr, size_i)
        alpha = softmax(mlp_attention, index, ptr, size_i)
        #alpha = softmax(multi_head_avg.unsqueeze(-1), index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training, inplace=False)
        
        if pre_alpha!=None and self.beta != 0:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1,1)


        return out

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):
        
        if self.op == 'corr':
            trans_embed  = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        elif self.op == 'cross':
            trans_embed = ent_embed * rel_emb + ent_embed
        elif self.op == "corr_plus":
            trans_embed = ccorr_new(ent_embed, rel_emb) + ent_embed
        elif self.op == "rotate":
            trans_embed = rotate(ent_embed, rel_emb)
        else:
            raise NotImplementedError
        
        return trans_embed
