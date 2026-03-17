import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from maskrcnn_benchmark.modeling.utils import cat, concat_box_prediction_layers, permute_and_flatten
from timm.models.layers import DropPath

import sys

from transformers.activations import ACT2FN
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _make_conv(input_dim, output_dim, k, stride=1):
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, (k, k), padding=(pad, pad), stride=(stride, stride)),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True)
    )


def _make_mlp(input_dim, output_dim, drop):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True),
                         nn.Dropout(drop),
                         nn.Linear(output_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True))


def _make_coord(batch, height, width):
    # relative position encoding
    xv, yv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    xv_min = (xv.float() * 2 - width) / width
    yv_min = (yv.float() * 2 - height) / height
    xv_max = ((xv + 1).float() * 2 - width) / width
    yv_max = ((yv + 1).float() * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.ones(height, width) * (1. / height)
    wmap = torch.ones(height, width) * (1. / width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0), \
                                               xv_max.unsqueeze(0), yv_max.unsqueeze(0), \
                                               xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0), \
                                               hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0))
    coord = coord.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coord


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D
        self.clamp_min_for_underflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW
        self.clamp_max_for_overflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW

        self._reset_parameters()

        self._C = cfg

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None, size_per_level=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1) # [bsz * num_heads, src_len, tgt_len]


        # 进行我们的cross-attention和self-attention
        ##########################################################################################################################
        # if self._C.MODEL.USE_CROSS_SELF_ATTENTION:
        #     v_query = self.v_proj(v) * self.scale
        #     v_key = self.v_proj(v)
        #     v_query_heads = self._shape(v_query, tgt_len, bsz).view(bsz * self.num_heads, tgt_len, self.head_dim)
        #     v_key_heads = self._shape(v_key, tgt_len, bsz).view(bsz * self.num_heads, tgt_len, self.head_dim)
        #     v_self_attn = torch.bmm(v_query_heads, v_key_heads.transpose(1, 2))  # [bsz * num_heads, tgt_len, tgt_len]
            
        #     if self.stable_softmax_2d:
        #         v_self_attn = v_self_attn - v_self_attn.max()
        #     if self.clamp_min_for_underflow:
        #         v_self_attn = torch.clamp(v_self_attn, min=-50000)
        #     if self.clamp_max_for_overflow:
        #         v_self_attn = torch.clamp(v_self_attn, max=50000)
                
        #     v_self_attn = F.softmax(v_self_attn, dim=-1)  # [bsz * num_heads, tgt_len, tgt_len]

        #     # 4. 为每个token选择topk个最相关的视觉patch并融合自注意力
        #     k = min(5, tgt_len)
        #     topk_values, topk_indices = torch.topk(attn_weights_l, k, dim=-1)  # [bsz * num_heads, src_len, k]
            
        #     # 5. 构建融合自注意力
        #     enhanced_attn_weights_l = []  # 将储存L→V的增强注意力
            
        #     for head_idx in range(bsz * self.num_heads):
        #         head_topk_indices = topk_indices[head_idx]  # [src_len, k]
        #         head_topk_values = topk_values[head_idx]    # [src_len, k]
        #         head_v_self_attn = v_self_attn[head_idx]    # [tgt_len, tgt_len]
                
        #         token_enhanced_attns = []
        #         for token_idx in range(src_len):
        #             token_patch_indices = head_topk_indices[token_idx]  # [k]
        #             token_patch_weights = head_topk_values[token_idx]   # [k]
        #             token_patch_weights = F.softmax(token_patch_weights, dim=0)
                    
        #             selected_self_attns = head_v_self_attn[token_patch_indices]  # [k, tgt_len]
        #             token_fused_attn = torch.sum(selected_self_attns * token_patch_weights.unsqueeze(1), dim=0)  # [tgt_len]
        #             token_enhanced_attns.append(token_fused_attn)
                
        #         head_enhanced_attn = torch.stack(token_enhanced_attns, dim=0)  # [src_len, tgt_len]
        #         enhanced_attn_weights_l.append(head_enhanced_attn)
            
        #     enhanced_attn_weights_l = torch.stack(enhanced_attn_weights_l, dim=0)  # [bsz * num_heads, src_len, tgt_len]

        #     # print(enhanced_attn_weights_l.shape)
        #     # print("哈哈")
        #     # sys.exit()
            
        #     # 6. 融合注意力权重
        #     alpha = 0.3  # 权重因子
        #     final_attn_weights_l = alpha * enhanced_attn_weights_l + (1 - alpha) * attn_weights_l  # L→V
        #     # final_attn_weights_l = alpha * enhanced_attn_weights_l + attn_weights_l  # L→V
            
        #     # 对于V→L方向，不需要单独计算增强，可以直接用L→V的增强结果的转置
        #     final_attn_weights = final_attn_weights_l.transpose(1, 2)  # [bsz * num_heads, tgt_len, src_len]



        #     # # 6. 融合注意力权重（使用残差连接方式）
        #     # alpha = 0.5  # 权重因子
        #     # # 直接累加原始注意力和增强注意力
        #     # fused_attn_weights_l = alpha * enhanced_attn_weights_l + attn_weights_l  # L→V

        #     # # 数值稳定性处理
        #     # if self.clamp_min_for_underflow:
        #     #     fused_attn_weights_l = torch.clamp(fused_attn_weights_l, min=-50000)
        #     # if self.clamp_max_for_overflow:
        #     #     fused_attn_weights_l = torch.clamp(fused_attn_weights_l, max=50000)

        #     # # 对融合后的注意力重新应用softmax归一化
        #     # final_attn_weights_l = F.softmax(fused_attn_weights_l, dim=-1)  # [bsz * num_heads, src_len, tgt_len]

        if self._C.MODEL.USE_CROSS_SELF_ATTENTION:
            # 获取最后一个尺度的信息
            last_h, last_w = size_per_level[-1]
            last_level_size = last_h * last_w

            second_last_h, second_last_w = size_per_level[-2]
            second_last_level_size = second_last_h * second_last_w

            third_last_h, third_last_w = size_per_level[-3]
            third_last_level_size = third_last_h * third_last_w

            last_level_size = last_level_size + second_last_level_size + third_last_level_size
             
            # 计算最后一个尺度在整个tgt_len中的起始索引
            start_idx = tgt_len - last_level_size
            
            # 只取最后一个尺度的视觉特征
            v_last = v[:, start_idx:, :]  # [bsz, last_level_size, embed_dim]
            
            # 对最后一个尺度进行正常的投影和自注意力计算
            v_query_last = self.v_proj(v_last) * self.scale
            v_key_last = self.v_proj(v_last)
            
            v_query_heads_last = self._shape(v_query_last, last_level_size, bsz).view(bsz * self.num_heads, last_level_size, self.head_dim)
            v_key_heads_last = self._shape(v_key_last, last_level_size, bsz).view(bsz * self.num_heads, last_level_size, self.head_dim)
            
            # 计算自注意力矩阵（仅限于最后一个尺度）
            v_self_attn_last = torch.bmm(v_query_heads_last, v_key_heads_last.transpose(1, 2))  # [bsz * num_heads, last_level_size, last_level_size]
            
            if self.stable_softmax_2d:
                v_self_attn_last = v_self_attn_last - v_self_attn_last.max(dim=-1, keepdim=True)[0]
            if self.clamp_min_for_underflow:
                v_self_attn_last = torch.clamp(v_self_attn_last, min=-50000)
            if self.clamp_max_for_overflow:
                v_self_attn_last = torch.clamp(v_self_attn_last, max=50000)
            

            temperature = 0.3
            v_self_attn_last = v_self_attn_last / temperature
            v_self_attn_last = F.softmax(v_self_attn_last, dim=-1)  # [bsz * num_heads, last_level_size, last_level_size]
                        
            # 从attn_weights_l中提取与最后一个尺度相关的部分
            attn_weights_l_last = attn_weights_l[:, :, start_idx:]  # [bsz * num_heads, src_len, last_level_size]
            
            # 为每个token选择topk个最相关的视觉patch
            k = min(100, last_level_size)
            topk_values, topk_indices = torch.topk(attn_weights_l_last, k, dim=-1)  # [bsz * num_heads, src_len, k]
            
            # 构建增强注意力
            enhanced_attn_weights_l = torch.zeros_like(attn_weights_l)  # 初始化全零
            
            # 使用高效的批处理方式
            batch_indices = torch.arange(bsz * self.num_heads, device=v.device).view(-1, 1, 1)
            batch_indices = batch_indices.expand(-1, src_len, k)
            
            # 获取相关patches的自注意力行
            selected_patches_attn = v_self_attn_last[batch_indices, topk_indices]  # [bsz * num_heads, src_len, k, last_level_size]
            
            # 计算权重
            patch_weights = F.softmax(topk_values, dim=-1).unsqueeze(-1)  # [bsz * num_heads, src_len, k, 1]
            
            # 加权融合
            enhanced_attn_last = (selected_patches_attn * patch_weights).sum(dim=2)  # [bsz * num_heads, src_len, last_level_size]

            # print("哈哈")
            # print(enhanced_attn_last.shape) torch.Size([16, 256, 40])
            # sys.exit()
            
            # 只更新最后一个尺度的增强注意力
            enhanced_attn_weights_l[:, :, start_idx:] = enhanced_attn_last
            
            # 融合原始注意力和增强注意力
            # alpha = 1000.0
            alpha = 0.0
            # 对于最后一个尺度使用增强注意力，其他尺度保持原状
            final_attn_weights_l = attn_weights_l.clone()  # 复制原始注意力
            # final_attn_weights_l[:, :, start_idx:] = alpha * enhanced_attn_last + (1 - alpha) * attn_weights_l[:, :, start_idx:]
            final_attn_weights_l[:, :, start_idx:] = alpha * enhanced_attn_last + attn_weights_l[:, :, start_idx:]
            # final_attn_weights_l[:, :, start_idx:] = alpha * enhanced_attn_last + alpha * attn_weights_l[:, :, start_idx:]

            final_attn_weights_l = F.softmax(final_attn_weights_l, dim=-1)

            # print("哈哈")
            # print(final_attn_weights_l.shape) torch.Size([16, 256, 12280])
            # sys.exit()

            # 对于V→L方向，使用L→V的转置
            final_attn_weights = final_attn_weights_l.transpose(1, 2)  # [bsz * num_heads, tgt_len, src_len]

        ##########################################################################################################################


        if attention_mask_l is not None:

            # print("哈哈")
            # sys.exit()

            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        if False and self._C.MODEL.USE_CROSS_SELF_ATTENTION:
            # For the visual branch (V->L direction)
            # Get the last scale information
            last_h, last_w = size_per_level[-1]
            last_level_size = last_h * last_w
            
            second_last_h, second_last_w = size_per_level[-2]
            second_last_level_size = second_last_h * second_last_w
            
            third_last_h, third_last_w = size_per_level[-3]
            third_last_level_size = third_last_h * third_last_w
            
            last_level_size = last_level_size + second_last_level_size + third_last_level_size
            
            # Calculate the starting index of the last scale in the entire tgt_len
            start_idx = tgt_len - last_level_size
            
            # Only take the visual features of the last scale
            v_last = v[:, start_idx:, :]  # [bsz, last_level_size, embed_dim]
            
            # Normal projection and self-attention calculation for the last scale
            v_query_last = self.v_proj(v_last) * self.scale
            v_key_last = self.v_proj(v_last)
            
            v_query_heads_last = self._shape(v_query_last, last_level_size, bsz).view(bsz * self.num_heads, last_level_size, self.head_dim)
            v_key_heads_last = self._shape(v_key_last, last_level_size, bsz).view(bsz * self.num_heads, last_level_size, self.head_dim)
            
            # Calculate self-attention matrix (only for the last scale)
            v_self_attn_last = torch.bmm(v_query_heads_last, v_key_heads_last.transpose(1, 2))  # [bsz * num_heads, last_level_size, last_level_size]
            
            if self.stable_softmax_2d:
                v_self_attn_last = v_self_attn_last - v_self_attn_last.max(dim=-1, keepdim=True)[0]
            if self.clamp_min_for_underflow:
                v_self_attn_last = torch.clamp(v_self_attn_last, min=-50000)
            if self.clamp_max_for_overflow:
                v_self_attn_last = torch.clamp(v_self_attn_last, max=50000)
            
            temperature = 0.3
            v_self_attn_last = v_self_attn_last / temperature
            v_self_attn_last = F.softmax(v_self_attn_last, dim=-1)  # [bsz * num_heads, last_level_size, last_level_size]
            
            # Extract the part related to the last scale from attn_weights_v
            attn_weights_v_last = attn_weights_v[:, start_idx:, :]  # [bsz * num_heads, last_level_size, src_len]
            
            # Select topk most relevant visual patches for each token
            k = min(100, last_level_size)
            topk_values, topk_indices = torch.topk(attn_weights_v_last.transpose(1, 2), k, dim=-1)  # [bsz * num_heads, src_len, k]
            
            # Build enhanced attention
            enhanced_attn_weights_v = torch.zeros_like(attn_weights_v)  # Initialize with zeros
            
            # Use efficient batch processing
            batch_indices = torch.arange(bsz * self.num_heads, device=v.device).view(-1, 1, 1)
            batch_indices = batch_indices.expand(-1, src_len, k)
            
            # Get the self-attention rows of relevant patches
            selected_patches_attn = v_self_attn_last[batch_indices, topk_indices]  # [bsz * num_heads, src_len, k, last_level_size]
            
            # Calculate weights
            patch_weights = F.softmax(topk_values, dim=-1).unsqueeze(-1)  # [bsz * num_heads, src_len, k, 1]
            
            # Weighted fusion
            enhanced_attn_last = (selected_patches_attn * patch_weights).sum(dim=2)  # [bsz * num_heads, src_len, last_level_size]
            
            # Only update the enhanced attention for the last scale
            enhanced_attn_weights_v_t = torch.zeros_like(attn_weights_v.transpose(1, 2))
            enhanced_attn_weights_v_t[:, :, start_idx:] = enhanced_attn_last
            enhanced_attn_weights_v = enhanced_attn_weights_v_t.transpose(1, 2)
            
            # Merge original attention and enhanced attention
            alpha = 0.0  # Same alpha as used for language branch
            
            # For the last scale, use enhanced attention; for other scales, keep original
            final_attn_weights_v = attn_weights_v.clone()  # Copy original attention
            final_attn_weights_v[:, start_idx:, :] = alpha * enhanced_attn_weights_v[:, start_idx:, :] + attn_weights_v[:, start_idx:, :]
            
            final_attn_weights_v = F.softmax(final_attn_weights_v, dim=-1)
            
            # Use the enhanced attention for the V branch
            attn_probs_v = F.dropout(final_attn_weights_v, p=self.dropout, training=self.training)
        else:
            attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        
        # attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)

        # 先只对文本进行
        if self._C.MODEL.USE_CROSS_SELF_ATTENTION:
            attn_probs_l = F.dropout(final_attn_weights_l, p=self.dropout, training=self.training) 
            # print("哈哈已经操作")
        else:
            attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training) 

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

class BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

        self.cfg = cfg
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            if not self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                self.shrink_lang = FeatureResizer(l_dim * 5, l_dim, 0.1)

    def forward(self, q0, q1, q2, q3, q4, l, attention_mask_l=None, dummy_tensor=None):

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            visu_feat = []
            lang_feat = []
            for ii, feat in enumerate([q0, q1, q2, q3, q4]):
                bs, _, h, w = feat.shape
                q = feat.flatten(2).transpose(1, 2)
                
                new_v, new_l = self.single_attention_call(q, l, attention_mask_l=attention_mask_l)
                new_v = new_v.transpose(1, 2).contiguous().view(bs, -1, h, w)
                lang_feat.append(new_l)
                visu_feat.append(new_v)
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                pass
            else:
                lang_feat = self.shrink_lang(torch.cat(lang_feat, dim = -1)) # From multiple dimensions
                lang_feat = [lang_feat, None, None, None, None]
        else:
            visu_feat = []
            size_per_level, visual_features_flatten = [], []
            for ii, feat_per_level in enumerate([q0, q1, q2, q3, q4]):
                bs, c, h, w = feat_per_level.shape
                size_per_level.append([h, w])
                feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
                visual_features_flatten.append(feat)
            visual_features_flatten = cat(visual_features_flatten, dim=1)
            new_v, new_l = self.single_attention_call(visual_features_flatten, l, attention_mask_l=attention_mask_l, size_per_level=size_per_level)
            # [bs, N, C] -> [bs, C, N]
            new_v = new_v.transpose(1, 2).contiguous()

            start = 0
            for (h, w) in size_per_level:
                new_v_per_level = new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous()
                visu_feat.append(new_v_per_level)
                start += h * w
            
            lang_feat = [new_l, None, None, None, None]

        return visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], visu_feat[4], lang_feat[0], lang_feat[1], lang_feat[2], lang_feat[3], lang_feat[4]

    
    def single_attention_call(self, v, l, attention_mask_l=None, dummy_tensor=None, size_per_level=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l, size_per_level=size_per_level)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l


# Single Direction MHA
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz)
        value_states = self._shape(self.v_proj(v), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)


        return attn_output, attn_weights


class AttentionMLP(nn.Module):
    def __init__(self, q_dim, hidden_dim, dropout=0.1):
        super(AttentionMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(q_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, q_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AttentionT2I(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, mode="i2t", use_layer_scale = False,
                 clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(AttentionT2I, self).__init__()

        # pre_layer norm
        self.layer_norm_q_1 = nn.LayerNorm(q_dim)
        self.layer_norm_k_1 = nn.LayerNorm(k_dim)
        self.attn = MultiHeadAttention(q_dim=q_dim,
                                       k_dim=k_dim,
                                       embed_dim=embed_dim,
                                       num_heads=num_heads,
                                       clamp_min_for_underflow=clamp_min_for_underflow,
                                       clamp_max_for_overflow=clamp_max_for_overflow)
        self.mode = mode

        # add layer scale for training stability
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.gamma = nn.Parameter(init_values * torch.ones((q_dim)), requires_grad=True)


    def forward(self, q0, q1, q2, q3, q4, k, v, attention_mask, dummy_arg=None):
        qs = []
        for q_index, q in enumerate([q0, q1, q2, q3, q4]):
            bs, _, h, w = q.shape
            # (batch, seq_len, embed_size)
            q = q.flatten(2).transpose(1, 2)
            q = self.layer_norm_q_1(q)
            k, v = self.layer_norm_k_1(k), self.layer_norm_k_1(v)
            delta_q = self.attn(q, k, v, attention_mask=attention_mask)[0]
            if self.use_layer_scale:
                q = q + self.drop_path(self.gamma * delta_q)
            else:
                q = q + delta_q
            q = q.transpose(1, 2).contiguous().view(bs, -1, h, w)
            qs.append(q)


        return qs[0], qs[1], qs[2], qs[3], qs[4]
