import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn import functional as F, Parameter
from Encoder import CompGATv3
from torch_geometric.nn import Sequential
#from Encoder import ARGAT
from  utils import *
from Evaluation import Evaluator

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # SimCLR loss
            mask = torch.eye(batch_size).float().to(device)
        elif labels is not None:
            # Supconloss
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # concat all contrast features at dim 0
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob

        # negative samples
        exp_logits = torch.exp(logits) * logits_mask
    
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        pos_per_sample=mask.sum(1) #B
        pos_per_sample[pos_per_sample<1e-6]=1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample #mask.sum(1)

        #mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class relation_contrast(torch.nn.Module):
    def __init__(self, temperature, num_neg):
        super(relation_contrast, self).__init__()
        self.temperature = temperature
        self.num_neg = num_neg
        # self.all_pos_triples = all_pos_triple

    def forward(self, pos_scores, neg_scores):
        neg_scores = neg_scores.view(-1, self.num_neg, 1)
        pos = torch.exp(torch.div(pos_scores, self.temperature))
        neg = torch.exp(torch.div(neg_scores, self.temperature)).sum(dim=1)
        loss = -torch.log(torch.div(pos, neg)).mean()
        return loss
    
    # def forward(self, aug_emb, ent_emb, hrt_batch):
    #     device = (torch.device('cuda')
    #               if aug_emb.is_cuda
    #               else torch.device('cpu'))
    #     num_ent = ent_emb.size(0)
    #     label = hrt_batch[:, 2].contiguous().view(-1, 1).to(device)

    #     mask = torch.eq(label, torch.arange(0, num_ent).view(-1, 1).T.to(device)).float().to(device)
        
    #     filter_batch = self.create_sparse_positive_filter_(hrt_batch, self.all_pos_triples)
    #     mask[filter_batch[:, 0], filter_batch[:, 1]] = 1.0
        
    #     aug_abs = aug_emb.norm(dim=1)
    #     emb_abs = ent_emb.norm(dim=1)
    #     sim_matrix = torch.einsum('ik,jk->ij', aug_emb, ent_emb) / torch.einsum('i,j->ij', aug_abs, emb_abs)
    #     sim_matrix = torch.exp(sim_matrix / self.temperature)

    #     # sim_matrix = torch.div(torch.matmul(aug_emb, ent_emb.T), self.temperature)
    #     pos_sim = (mask * sim_matrix).sum(dim=1)
    #     loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    #     loss = -(torch.log(loss) / mask.sum(dim=1)).mean()

    #     return loss
    
    # def create_sparse_positive_filter_(self, hrt_batch, all_pos_triples, filter_col=2):
        
    #     relations = hrt_batch[:, 1:2]
    #     relation_filter = (all_pos_triples[:, 1:2]).view(1, -1) == relations

    #     # Split batch
    #     other_col = 2 - filter_col
    #     entities = hrt_batch[:, other_col : other_col + 1]

    #     entity_filter_test = (all_pos_triples[:, other_col : other_col + 1]).view(1, -1) == entities
    #     filter_batch = (entity_filter_test & relation_filter).nonzero(as_tuple=False)
    #     filter_batch[:, 1] = all_pos_triples[:, filter_col : filter_col + 1].view(1, -1)[:, filter_batch[:, 1]]

    #     return filter_batch


class CLKG_compgatv3_convE(nn.Module):
    def __init__(self, args):
        super(CLKG_compgatv3_convE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h
        
        # Add dimension validation and adjustment like in DFconv3D
        if self.ent_h * self.ent_w != self.ent_dim:
            print(f"警告: ent_h ({self.ent_h}) * ent_w ({self.ent_w}) = {self.ent_h * self.ent_w} 不等于 ent_dim ({self.ent_dim})")
            print(f"建议的 ent_height 值:")
        
            # 找到所有可能的因子分解
            factors = []
            for h in range(1, int(self.ent_dim**0.5) + 1):
                if self.ent_dim % h == 0:
                    w = self.ent_dim // h
                    factors.append((h, w))
            
            print("可选的 (height, width) 组合:")
            for h, w in factors[-5:]:  # 显示最后5个较大的组合
                print(f"  ent_height={h}, ent_width={w}")
            
            # 使用最接近正方形的分解
            best_h = factors[-1][0] if factors else args.ent_height
            self.ent_h = best_h
            self.ent_w = self.ent_dim // self.ent_h
            print(f"自动选择: ent_h={self.ent_h}, ent_w={self.ent_w}")
        
        # Gaussian mapping parameters - set to ent_dim to avoid dimension mismatch
        #self.num_gaussian_kernels = self.ent_dim
        
        # Gaussian centers and standard deviations for entity mapping
        #self.ent_gaussian_centers = nn.Parameter(torch.randn(self.num_gaussian_kernels, self.ent_dim))
        #self.ent_gaussian_sigmas = nn.Parameter(torch.ones(self.num_gaussian_kernels))
        
        # Gaussian centers and standard deviations for relation mapping  
        #self.rel_gaussian_centers = nn.Parameter(torch.randn(self.num_gaussian_kernels, self.ent_dim))
        #self.rel_gaussian_sigmas = nn.Parameter(torch.ones(self.num_gaussian_kernels))
        
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            
        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
        
        elif args.gcn_layer == 4:
            self.layer = 4
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn2 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn3 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn4 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        # Calculate input features for projection layer based on fused feature matrix
        # Feature matrix from create_fused_feature_matrix has shape (batch, 1, 2*ent_h, ent_w)
        # After conv2d with filter_size, the output shape is:
        # (batch, filter_channel, 2*ent_h - filter_size + 1, ent_w - filter_size + 1)
        conv_out_h = 2 * self.ent_h - args.filter_size + 1
        conv_out_w = self.ent_w - args.filter_size + 1
        num_in_features = args.filter_channel * conv_out_h * conv_out_w
        
        print(f"Conv2D dimensions: filter_channel={args.filter_channel}, conv_out_h={conv_out_h}, conv_out_w={conv_out_w}")
        print(f"Expected input features to projection: {num_in_features}")

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def create_fused_feature_matrix(self, head_emb, r_emb, head_emb_gaussian, r_emb_gaussian, fusion_type='add'):
        """
        Create feature matrix from fused head and relation embeddings
        
        Args:
            head_emb: Head entity embedding (batch, ent_dim)
            r_emb: Relation embedding (batch, ent_dim)
            head_emb_gaussian: Gaussian mapped head embedding (batch, ent_dim)
            r_emb_gaussian: Gaussian mapped relation embedding (batch, ent_dim)
            fusion_type: Type of fusion to use
        
        Returns:
            Feature matrix ready for convolution (batch, 1, 2*ent_h, ent_w)
        """
        # Feature fusion: combine original embeddings with Gaussian mappings
        head_fused = self.feature_fusion(head_emb, head_emb_gaussian, fusion_type)
        r_fused = self.feature_fusion(r_emb, r_emb_gaussian, fusion_type)
        
        # Handle dimension mismatch for concat fusion
        if fusion_type == 'concat':
            # For concat, we need to adjust the dimensions
            ent_dim_fused = head_fused.shape[1]
            ent_h_fused = ent_dim_fused // self.ent_w
            if ent_h_fused * self.ent_w != ent_dim_fused:
                print(f"Warning: ent_h_fused ({ent_h_fused}) * ent_w ({self.ent_w}) = {ent_h_fused * self.ent_w} 不等于 ent_dim_fused ({ent_dim_fused})")
                # Pad to make it divisible
                padding = (ent_h_fused + 1) * self.ent_w - ent_dim_fused
                head_fused = F.pad(head_fused, (0, padding))
                r_fused = F.pad(r_fused, (0, padding))
                ent_h_fused = ent_h_fused + 1
                ent_dim_fused = head_fused.shape[1]  # Update ent_dim_fused after padding
            
            # Create feature matrix - both tensors should have same dimensions after padding
            x = head_fused.view(-1, 1, ent_dim_fused)
            r_fused_reshaped = r_fused.view(-1, 1, ent_dim_fused)
            x = torch.cat([x, r_fused_reshaped], 1)
            x = torch.transpose(x, 2, 1).reshape((-1, 1, 2*ent_h_fused, self.ent_w))
        else:
            # For other fusion types, use standard dimensions
            x = head_fused.view(-1, 1, self.ent_dim)
            x = torch.cat([x, r_fused], 1)
            x = torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
        
        return x
    
    def feature_fusion(self, original_emb, gaussian_emb, fusion_type='add'):
        """
        Fuse original embeddings with Gaussian mapped embeddings
        
        Args:
            original_emb: Original embedding tensor (batch, ent_dim)
            gaussian_emb: Gaussian mapped embedding tensor (batch, ent_dim)
            fusion_type: Type of fusion ('add', 'concat', 'weighted_add', 'multiply')
        
        Returns:
            Fused embedding tensor
        """
        if fusion_type == 'add':
            # Element-wise addition (current implementation)
            return original_emb + gaussian_emb
        
        elif fusion_type == 'concat':
            # Concatenation along feature dimension
            return torch.cat([original_emb, gaussian_emb], dim=1)
        
        elif fusion_type == 'weighted_add':
            # Weighted addition with learnable weights
            # You can add learnable parameters for alpha and beta
            alpha = 0.7  # Weight for original embedding
            beta = 0.3   # Weight for Gaussian embedding
            return alpha * original_emb + beta * gaussian_emb
        
        elif fusion_type == 'multiply':
            # Element-wise multiplication
            return original_emb * gaussian_emb
        
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb, save=False):
        r_emb_triple = torch.index_select(rel_emb, 0, r)

        if self.layer==2:
            ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb, r_emb_triple=r_emb_triple)
            ent_emb = self.hid_drop(ent_emb)
        
        elif self.layer==4:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
            ent_emb = self.hid_drop(ent_emb)
        '''
        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        # Apply Gaussian mapping
        head_emb_gaussian = self.gaussian_mapping(head_emb, self.ent_gaussian_centers, self.ent_gaussian_sigmas)
        r_emb_gaussian = self.gaussian_mapping(r_emb, self.rel_gaussian_centers, self.rel_gaussian_sigmas)
        
        # Feature fusion: combine original embeddings with Gaussian mappings using element-wise addition
        x = self.create_fused_feature_matrix(head_emb, r_emb, head_emb_gaussian, r_emb_gaussian, fusion_type='concat')
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        if save:
            return x, ent_emb
        x = self.bn2(x)
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
        '''
        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, r_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        if save:
            return x, ent_emb
        x = self.bn2(x)
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb


    def score_hrt(self, hrt_batch, ent_emb, rel_emb):
        '''
        h, r, t = hrt_batch.t()
        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        # Apply Gaussian mapping
        head_emb_gaussian = self.gaussian_mapping(head_emb, self.ent_gaussian_centers, self.ent_gaussian_sigmas)
        r_emb_gaussian = self.gaussian_mapping(r_emb, self.rel_gaussian_centers, self.rel_gaussian_sigmas)
        
        # Feature fusion: combine original embeddings with Gaussian mappings using element-wise addition
        x = self.create_fused_feature_matrix(head_emb, r_emb, head_emb_gaussian, r_emb_gaussian, fusion_type='concat')

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        # x = self.bn2(x)
        
        x = (F.normalize(x, dim=1)*F.normalize(tail_emb, dim=1)).sum(dim=1, keepdim=True)
        # x += self.b[t].unsqueeze(1)
        # x = torch.sigmoid(x)
        return x
        '''
        h,r,t = hrt_batch.t()
        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, rel_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        # x = self.bn2(x)
        
        x = (F.normalize(x, dim=1)*F.normalize(tail_emb, dim=1)).sum(dim=1, keepdim=True)
        # x += self.b[t].unsqueeze(1)
        # x = torch.sigmoid(x)
        return x
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        '''
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
            
            elif self.layer==4:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
                ent_emb = self.hid_drop(ent_emb)

            if save_emb:
                save = {"ent_emb": ent_emb,
                        "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            head_emb = torch.index_select(ent_emb, 0, e1)
            r_emb = torch.index_select(rel_emb, 0, rel)
            
            # Apply Gaussian mapping and feature fusion as in forward pass
            head_emb_gaussian = self.gaussian_mapping(head_emb, self.ent_gaussian_centers, self.ent_gaussian_sigmas)
            r_emb_gaussian = self.gaussian_mapping(r_emb, self.rel_gaussian_centers, self.rel_gaussian_sigmas)
            
            # Feature fusion: combine original embeddings with Gaussian mappings using element-wise addition
            x = self.create_fused_feature_matrix(head_emb, r_emb, head_emb_gaussian, r_emb_gaussian, fusion_type='concat')
            
            x = self.bn0(x)
            # x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return x
        '''
        rel = hr_batch[:, 1]
        r_emb_triple = torch.index_select(rel_emb, 0, rel)
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb, r_emb_triple=r_emb_triple)
                ent_emb = self.hid_drop(ent_emb)
            
            elif self.layer==4:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
                ent_emb = self.hid_drop(ent_emb)

            if save_emb:
                save = {"ent_emb": ent_emb,
                        "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            x = torch.cat([e1_embedded, rel_embedded], 1)
            x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(x)
            # x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return x


    def gaussian_mapping(self, embedding, gaussian_centers, gaussian_sigmas):
        """
        Apply Gaussian function mapping to embeddings
        φ_i(e) = exp(-1/2 * (|e - c_i| / σ_i)^2)
        """
        # embedding: (batch_size, ent_dim)
        # gaussian_centers: (num_kernels, ent_dim)  
        # gaussian_sigmas: (num_kernels,)
        
        batch_size = embedding.shape[0]
        num_kernels = gaussian_centers.shape[0]
        
        # Expand dimensions for broadcasting
        embedding_expanded = embedding.unsqueeze(1)  # (batch_size, 1, ent_dim)
        centers_expanded = gaussian_centers.unsqueeze(0)  # (1, num_kernels, ent_dim)
        sigmas_expanded = gaussian_sigmas.unsqueeze(0)  # (1, num_kernels)
        
        # Calculate distances
        distances = torch.norm(embedding_expanded - centers_expanded, dim=2)  # (batch_size, num_kernels)
        
        # Apply Gaussian function
        gaussian_features = torch.exp(-0.5 *                                                                                                                                                                                                                               (distances / sigmas_expanded)**2)
        
        return gaussian_features  # (batch_size, num_kernels)

    def reshape_for_stacking(self, tensor):
        """Reshape tensor from (batch, ent_dim) to (batch, ent_h, ent_w) for stacking"""
        batch_size = tensor.shape[0]
        return tensor.view(batch_size, self.ent_h, self.ent_w)

    def create_feature_stack_2d(self, head_emb, r_emb, head_emb_gaussian, r_emb_gaussian):
        """
        Create feature stack for 2D convolution by stacking features along height dimension
        F = stack([ρ(r), ρ(e_s), ρ(r^Φ), ρ(e_s^Φ), ρ(r)], axis=height)
        """
        # Reshape embeddings for stacking: (batch, height, width)
        r_orig_reshaped = self.reshape_for_stacking(r_emb)
        head_orig_reshaped = self.reshape_for_stacking(head_emb) 
        r_gauss_reshaped = self.reshape_for_stacking(r_emb_gaussian)
        head_gauss_reshaped = self.reshape_for_stacking(head_emb_gaussian)
        r_orig_reshaped2 = self.reshape_for_stacking(r_emb)  # Repeat relation for 5-layer structure
        
        # Stack features along height dimension to create 2D input
        feature_stack = torch.cat([
            r_orig_reshaped,      # ρ(r)
            head_orig_reshaped,   # ρ(e_s) 
            r_gauss_reshaped,     # ρ(r^Φ)
            head_gauss_reshaped,  # ρ(e_s^Φ)
            r_orig_reshaped2      # ρ(r)
        ], dim=1)  # Concatenate along height dimension: (batch, 5*height, width)
        
        return feature_stack


class CLKG_compgatv3_DFconv3D(nn.Module):
    def __init__(self, args):
        super(CLKG_compgatv3_DFconv3D, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        
        if self.ent_h * self.ent_w != self.ent_dim:
            print(f"警告: ent_h ({self.ent_h}) * ent_w ({self.ent_w}) = {self.ent_h * self.ent_w} 不等于 ent_dim ({self.ent_dim})")
            print(f"建议的 ent_height 值:")
        
            # 找到所有可能的因子分解
            factors = []
            for h in range(1, int(self.ent_dim**0.5) + 1):
                if self.ent_dim % h == 0:
                    w = self.ent_dim // h
                    factors.append((h, w))
            
            print("可选的 (height, width) 组合:")
            for h, w in factors[-5:]:  # 显示最后5个较大的组合
                print(f"  ent_height={h}, ent_width={w}")
            
            # 使用最接近正方形的分解
            best_h = factors[-1][0] if factors else args.ent_height
            self.ent_h = best_h
            self.ent_w = self.ent_dim // self.ent_h
            print(f"自动选择: ent_h={self.ent_h}, ent_w={self.ent_w}")
        
        
        # Gaussian mapping parameters - set to ent_dim to avoid dimension mismatch
        self.num_gaussian_kernels = self.ent_dim
        #self.num_gaussian_kernels = 16
        # Gaussian centers and standard deviations for entity mapping
        self.ent_gaussian_centers = nn.Parameter(torch.randn(self.num_gaussian_kernels, self.ent_dim))
        self.ent_gaussian_sigmas = nn.Parameter(torch.ones(self.num_gaussian_kernels))
        
        # Gaussian centers and standard deviations for relation mapping  
        self.rel_gaussian_centers = nn.Parameter(torch.randn(self.num_gaussian_kernels, self.ent_dim))
        self.rel_gaussian_sigmas = nn.Parameter(torch.ones(self.num_gaussian_kernels))
        
        if args.gcn_layer == 2:
            self.layer = 2
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            
        elif args.gcn_layer == 1:
            self.layer = 1
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
        
        elif args.gcn_layer == 4:
            self.layer = 4
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn2 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn3 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn4 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)

        # 3D Convolution setup - input has 5 channels in depth dimension
        self.conv3d = torch.nn.Conv3d(in_channels=1, out_channels=args.filter_channel, 
                                     kernel_size=(2, args.filter_size, args.filter_size), 
                                     stride=1, padding=0, bias=False)
        self.bn0 = torch.nn.BatchNorm3d(1)
        self.bn1 = torch.nn.BatchNorm3d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))        # Calculate output size after 3D convolution
        # Input: (batch, 1, 5, ent_h, ent_w)  - Note: NOT 2*ent_h like in 2D ConvE
        # After conv3d with kernel (2, filter_size, filter_size): 
        # Output: (batch, filter_channel, 5-2+1, ent_h-filter_size+1, ent_w-filter_size+1)
        conv_out_depth = 5 - 2 + 1  # 4
        #conv_out_depth = 3 - 1 + 1  # Corrected: kernel depth is 1, so output depth remains 5
        conv_out_h = self.ent_h - args.filter_size + 1  # Corrected: single ent_h, not 2*ent_h
        conv_out_w = self.ent_w - args.filter_size + 1
        num_in_features = args.filter_channel * conv_out_depth * conv_out_h * conv_out_w

        print(f"3D Conv dimensions: filter_channel={args.filter_channel}, depth={conv_out_depth}, h={conv_out_h}, w={conv_out_w}")
        print(f"Expected input features to projection: {num_in_features}")

        # Output projection layer
        self.proj_out = torch.nn.Linear(num_in_features, args.ent_dim)

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def gaussian_mapping(self, embedding, gaussian_centers, gaussian_sigmas):
        """
        Apply Gaussian function mapping to embeddings
        φ_i(e) = exp(-1/2 * (|e - c_i| / σ_i)^2)
        """
        # embedding: (batch_size, ent_dim)
        # gaussian_centers: (num_kernels, ent_dim)  
        # gaussian_sigmas: (num_kernels,)
        
        batch_size = embedding.shape[0]
        num_kernels = gaussian_centers.shape[0]
        
        # Expand dimensions for broadcasting
        embedding_expanded = embedding.unsqueeze(1)  # (batch_size, 1, ent_dim)
        centers_expanded = gaussian_centers.unsqueeze(0)  # (1, num_kernels, ent_dim)
        sigmas_expanded = gaussian_sigmas.unsqueeze(0)  # (1, num_kernels)
        
        # Calculate distances
        distances = torch.norm(embedding_expanded - centers_expanded, dim=2)  # (batch_size, num_kernels)
        
        # Apply Gaussian function
        gaussian_features = torch.exp(-0.5 * (distances / sigmas_expanded)**2)
        
        return gaussian_features  # (batch_size, num_kernels)

    def reshape_for_conv3d(self, tensor):
        """Reshape tensor from (batch, ent_dim) to (batch, ent_h, ent_w) for stacking"""
        batch_size = tensor.shape[0]
        return tensor.view(batch_size, self.ent_h, self.ent_w)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb, save=False):
        r_emb_triple = torch.index_select(rel_emb, 0, r)
        # GNN encoding
        if self.layer==2:
            ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb, r_emb_triple=r_emb_triple)
            ent_emb = self.hid_drop(ent_emb)
        
        elif self.layer==4:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
            ent_emb = self.hid_drop(ent_emb)

        # Extract head and relation embeddings
        head_emb = torch.index_select(ent_emb, 0, h)  # (batch_size, ent_dim)
        r_emb = torch.index_select(rel_emb, 0, r)      # (batch_size, ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)   # (batch_size, ent_dim)
        
        # Apply Gaussian mapping
        head_emb_gaussian = self.gaussian_mapping(head_emb, self.ent_gaussian_centers, self.ent_gaussian_sigmas)
        r_emb_gaussian = self.gaussian_mapping(r_emb, self.rel_gaussian_centers, self.rel_gaussian_sigmas)
        
        # Ensure Gaussian mapped features have same dimension as original embeddings
        if head_emb_gaussian.shape[1] != self.ent_dim:
            print(f"Warning: head_emb_gaussian shape {head_emb_gaussian.shape} does not match ent_dim {self.ent_dim}")
            head_emb_gaussian = F.linear(head_emb_gaussian, torch.randn(self.ent_dim, head_emb_gaussian.shape[1], device=head_emb_gaussian.device))
        if r_emb_gaussian.shape[1] != self.ent_dim:
            print(f"Warning: r_emb_gaussian shape {r_emb_gaussian.shape} does not match ent_dim {self.ent_dim}")
            r_emb_gaussian = F.linear(r_emb_gaussian, torch.randn(self.ent_dim, r_emb_gaussian.shape[1], device=r_emb_gaussian.device))
        
        # Reshape embeddings for 3D stacking: (batch, height, width)
        r_orig_reshaped = self.reshape_for_conv3d(r_emb)
        head_orig_reshaped = self.reshape_for_conv3d(head_emb) 
        r_gauss_reshaped = self.reshape_for_conv3d(r_emb_gaussian)
        head_gauss_reshaped = self.reshape_for_conv3d(head_emb_gaussian)
        r_orig_reshaped2 = self.reshape_for_conv3d(r_emb)  # Repeat relation for 5-layer structure
          # Stack features along depth dimension: F = stack([ρ(r), ρ(e_s), ρ(r^Φ), ρ(e_s^Φ), ρ(r)], axis=2)
        # 正确的特征堆叠应该在深度维度进行
        feature_stack = torch.stack([
            r_orig_reshaped,      # ρ(r)
            head_orig_reshaped,   # ρ(e_s) 
            r_gauss_reshaped,     # ρ(r^Φ)
            head_gauss_reshaped,  # ρ(e_s^Φ)
            r_orig_reshaped2      # ρ(r)
        ], dim=1)  # Stack along depth dimension: (batch, height, width, 5)
        
        # Rearrange for 3D convolution: (batch, channels=1, depth=5, height, width)
        x = feature_stack.unsqueeze(1)  # (batch, 1, 5, height, width)
        
        # Apply 3D convolution
        x = self.bn0(x)
        x = self.conv3d(x)  # Output: (batch, filter_channel, depth_out, height_out, width_out)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        
        # Flatten for projection
        x = x.view(x.shape[0], -1)  # (batch, features)
        x = self.proj(x)
        
        if save:
            return x, ent_emb
            
        x = self.bn2(x)
        cl_x = x
        
        # Score computation using sigmoid as in paper: ψ(e_s, r, e_o) = sigmoid((OutW_out)e_o)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        #x = torch.sigmoid(x)
        
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
    
    def score_hrt(self, hrt_batch, ent_emb, rel_emb):
        h, r, t = hrt_batch.t()
        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        # Apply Gaussian mapping
        head_emb_gaussian = self.gaussian_mapping(head_emb, self.ent_gaussian_centers, self.ent_gaussian_sigmas)
        r_emb_gaussian = self.gaussian_mapping(r_emb, self.rel_gaussian_centers, self.rel_gaussian_sigmas)
        
        # Ensure same dimensions
        if head_emb_gaussian.shape[1] != self.ent_dim:
            head_emb_gaussian = F.linear(head_emb_gaussian, torch.randn(self.ent_dim, head_emb_gaussian.shape[1], device=head_emb_gaussian.device))
        if r_emb_gaussian.shape[1] != self.ent_dim:
            r_emb_gaussian = F.linear(r_emb_gaussian, torch.randn(self.ent_dim, r_emb_gaussian.shape[1], device=r_emb_gaussian.device))
        
        # Reshape and stack features
        r_orig_reshaped = self.reshape_for_conv3d(r_emb)
        head_orig_reshaped = self.reshape_for_conv3d(head_emb)
        r_gauss_reshaped = self.reshape_for_conv3d(r_emb_gaussian)
        head_gauss_reshaped = self.reshape_for_conv3d(head_emb_gaussian)
        r_orig_reshaped2 = self.reshape_for_conv3d(r_emb)
        
        feature_stack = torch.stack([
            r_orig_reshaped,      # ρ(r)
            head_orig_reshaped,   # ρ(e_s) 
            r_gauss_reshaped,     # ρ(r^Φ)
            head_gauss_reshaped,  # ρ(e_s^Φ)
            r_orig_reshaped2      # ρ(r)
        ], dim=1)
        
        # Rearrange for 3D convolution: (batch, channels=1, depth=5, height, width)
        x = feature_stack.unsqueeze(1)  # (batch, 1, 5, height, width)
        
        # 3D convolution
        x = self.bn0(x)
        x = self.conv3d(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        
        # Score with tail entity
        x = (F.normalize(x, dim=1) * F.normalize(tail_emb, dim=1)).sum(dim=1, keepdim=True)
        #return torch.sigmoid(x)
        return x
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        rel = hr_batch[:, 1]
        r_emb_triple = torch.index_select(rel_emb, 0, rel)
        with torch.no_grad():
            # GNN encoding
            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb, r_emb_triple=r_emb_triple)
                ent_emb = self.hid_drop(ent_emb)
            
            elif self.layer==4:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
                ent_emb = self.hid_drop(ent_emb)

            if save_emb:
                save = {"ent_emb": ent_emb, "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            head_emb = torch.index_select(ent_emb, 0, e1)
            r_emb = torch.index_select(rel_emb, 0, rel)            # Apply Gaussian mapping and 3D convolution as in forward pass
            head_emb_gaussian = self.gaussian_mapping(head_emb, self.ent_gaussian_centers, self.ent_gaussian_sigmas)
            r_emb_gaussian = self.gaussian_mapping(r_emb, self.rel_gaussian_centers, self.rel_gaussian_sigmas)
            if head_emb_gaussian.shape[1] != self.ent_dim:
                head_emb_gaussian = F.linear(head_emb_gaussian, torch.randn(self.ent_dim, head_emb_gaussian.shape[1], device=head_emb_gaussian.device))
            if r_emb_gaussian.shape[1] != self.ent_dim:
                r_emb_gaussian = F.linear(r_emb_gaussian, torch.randn(self.ent_dim, r_emb_gaussian.shape[1], device=r_emb_gaussian.device))
            
            r_orig_reshaped = self.reshape_for_conv3d(r_emb)
            head_orig_reshaped = self.reshape_for_conv3d(head_emb)
            r_gauss_reshaped = self.reshape_for_conv3d(r_emb_gaussian)
            head_gauss_reshaped = self.reshape_for_conv3d(head_emb_gaussian)
            r_orig_reshaped2 = self.reshape_for_conv3d(r_emb)
            feature_stack = torch.stack([
                r_orig_reshaped,      # ρ(r)
                head_orig_reshaped,   # ρ(e_s) 
                r_gauss_reshaped,     # ρ(r^Φ)
                head_gauss_reshaped,  # ρ(e_s^Φ)
                r_orig_reshaped2      # ρ(r)
            ], dim=1)
            
            # Rearrange for 3D convolution: (batch, channels=1, depth=5, height, width)
            x = feature_stack.unsqueeze(1)  # (batch, 1, 5, height, width)
            
            x = self.bn0(x)
            x = self.conv3d(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            
            # Final prediction
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            #x = torch.sigmoid(x)
            
            return x