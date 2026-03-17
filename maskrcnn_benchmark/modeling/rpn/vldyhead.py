import os
import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict

from .inference import make_atss_postprocessor
from .loss import make_atss_loss_evaluator
from .anchor_generator import make_anchor_generator_complex

from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.layers import Scale, DYReLU, SELayer, ModulatedDeformConv
from maskrcnn_benchmark.layers import NaiveSyncBatchNorm2d, FrozenBatchNorm2d
from maskrcnn_benchmark.modeling.backbone.fbnet import *
from maskrcnn_benchmark.engine.inference import create_positive_map_label_to_token_from_positive_map
from ..utils import cat, concat_box_prediction_layers, permute_and_flatten

from maskrcnn_benchmark.utils.fuse_helper import FeatureResizer, func_attention, _make_mlp, _make_conv, _make_coord, \
    BiAttentionBlock, AttentionT2I, BiAttentionBlockForCheckpoint, BertLMPredictionHead
from transformers.models.bert.modeling_bert import BertConfig, BertAttention, BertIntermediate, BertOutput, \
    BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
import torch.utils.checkpoint as checkpoint
import pdb

from maskrcnn_benchmark.modeling.language_backbone.clip_model import QuickGELU, LayerNorm, DropPath
from timm.models.layers import DropPath, trunc_normal_

import numpy as np
import sys

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class BoxCoder(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, gt_boxes, anchors):
        TO_REMOVE = 1  # TODO remove
        ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
        gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        anchors = anchors.to(preds.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = preds[:, 2::4] / ww
        dh = preds[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)

        return pred_boxes


class Conv3x3Norm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=1,
                 deformable=False,
                 bn_type=None):
        super(Conv3x3Norm, self).__init__()

        if deformable:
            self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                            groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)

        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]

        if bn_type == "bn":
            bn_op = nn.BatchNorm2d(out_channels)
        elif bn_type == "sbn":
            bn_op = nn.SyncBatchNorm(out_channels)
        elif bn_type == "nsbn":
            bn_op = NaiveSyncBatchNorm2d(out_channels)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=out_channels)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(out_channels)
        if bn_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    def forward(self, input, **kwargs):
        x = self.conv(input, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class DyConv(torch.nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 conv_func=nn.Conv2d,
                 use_dyfuse=True,
                 use_dyrelu=False,
                 use_deform=False
                 ):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList() # 1. 动态卷积层
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse: # 2. 注意力机制
            self.AttnConv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = h_sigmoid()
        else:
            self.AttnConv = None

        if use_dyrelu: # 3. 动态ReLU
            self.relu = DYReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_deform: # 4. 可变形卷积
            self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.AttnConv is not None:
            for m in self.AttnConv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs): # 可变形卷积核 自适应感受野 动态生成参数 根据输入自动调整参数
        visual_feats = inputs["visual"] 
        language_dict_features = inputs["lang"]

        next_x = []
        for level, feature in enumerate(visual_feats):

            conv_args = dict()
            if self.offset is not None: # 1. 可变形卷积处理
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)] # 2. 多尺度特征融合

            if level > 0:
                temp_fea.append(self.DyConv[2](visual_feats[level - 1], **conv_args)) # 上一层
            if level < len(visual_feats) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](visual_feats[level + 1], **conv_args), 
                                                    size=[feature.size(2), feature.size(3)])) # 下一层
            mean_fea = torch.mean(torch.stack(temp_fea), dim=0, keepdim=False) # 3. 特征聚合

            if self.AttnConv is not None: # 4. 注意力加权
                attn_fea = []
                res_fea = []
                for fea in temp_fea:
                    res_fea.append(fea)
                    attn_fea.append(self.AttnConv(fea))

                res_fea = torch.stack(res_fea)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))

                mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)

            next_x.append(mean_fea)

        next_x = [self.relu(item) for item in next_x]

        features_dict = {"visual": next_x,
                         "lang": language_dict_features}

        return features_dict


class BertEncoderLayer(BertPreTrainedModel):
    def __init__(self, config,  clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super().__init__(config)
        self.config = config

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        from maskrcnn_benchmark.modeling.rpn.modeling_bert import BertAttention, BertIntermediate, BertOutput

        self.attention = BertAttention(config,  clamp_min_for_underflow, clamp_max_for_overflow)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, inputs):
        language_dict_features = inputs["lang"] # 语言特征字典
        hidden_states = language_dict_features["hidden"] # 隐藏状态
        attention_mask = language_dict_features["masks"] # 注意力掩码

        device = hidden_states.device
        input_shape = hidden_states.size()[:-1]
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads. 
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device) # 1. 准备注意力掩码

        self_attention_outputs = self.attention( # 2. 自注意力处理
            hidden_states,
            extended_attention_mask,
            None,
            output_attentions=False,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward( # 3. 前馈网络处理
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        hidden_states = outputs[0]

        language_dict_features["hidden"] = hidden_states

        features_dict = {"visual": inputs["visual"], # 返回更新后的特征字典，# 保持视觉特征不变
                         "lang": language_dict_features # 更新后的语言特征
                         }

        return features_dict

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CLIPTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = self.config.MODEL.CLIP.WIDTH
        n_head = self.config.MODEL.CLIP.HEADS
        drop_path = self.config.MODEL.CLIP.DROP_PATH
        self.context_length = self.config.MODEL.CLIP.CONTEXT_LENGTH
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, inputs):
        language_dict_features = inputs["lang"]
        x = language_dict_features["hidden"]
        mask = language_dict_features["masks"]
        # get extended attention mask for nn.MultiHeadAttention
        key_padding_mask = (1.0 - mask).to(torch.bool)

        x = x.permute(1, 0, 2)
        x = x + self.drop_path(self.attention(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.permute(1, 0, 2)

        language_dict_features["hidden"] = x
        features_dict = {"visual": inputs["visual"],
                         "lang": language_dict_features
                         }
        return features_dict


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs

# 核心创新点:通过注意力机制融合
class VLFuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(self, cfg):
        super(VLFuse, self).__init__()
        self.init_configs(cfg)
        self.cfg = cfg

        self.use_checkpoint = False
        if hasattr(cfg.MODEL.DYHEAD, 'USE_CHECKPOINT'):
            self.use_checkpoint = cfg.MODEL.DYHEAD.USE_CHECKPOINT
            self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # early fusion module
        print("EARLY FUSION ON, USING {}".format(cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE))
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-S":
            # single-direction (text->image)
            # text -> image
            self.t2i_attn = AttentionT2I(q_dim=self.joint_embedding_size,
                                           k_dim=self.lang_dim,
                                           embed_dim=self.embed_dim,
                                           num_heads=self.n_head,
                                           hidden_dim=self.t2i_hidden_dim,
                                           dropout=0.1,
                                           drop_path=.0,
                                           init_values=1.0 / cfg.MODEL.DYHEAD.NUM_CONVS,
                                           mode="t2i",
                                           use_layer_scale=cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_LAYER_SCALE,
                                           clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW,
                                           clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW
                                           )

        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-B":
            # bi-direction (text->image, image->text)
            self.b_attn = BiAttentionBlockForCheckpoint(v_dim=self.joint_embedding_size,
                        l_dim=self.lang_dim,
                        embed_dim=self.embed_dim,
                        num_heads=self.n_head,
                        hidden_dim=self.i2t_hidden_dim,
                        dropout=0.1,
                        drop_path=.0,
                        init_values=1.0 / cfg.MODEL.DYHEAD.NUM_CONVS,
                        cfg=cfg
                        )
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL and self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                self.shrink_lang = FeatureResizer(self.lang_dim * 5,
                                self.lang_dim, 0.1)


        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "SCAN":
            # single-direction (text->image)
            self.mapping_lang = _make_mlp(self.lang_dim,
                                          self.joint_embedding_size,
                                          self.joint_embedding_dropout)
            self.joint_fusion = nn.ModuleList([_make_conv(self.joint_inp_dim, self.joint_out_dim, 1) \
                                               for _ in range(5)])

        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "FILM":
            # single-direction (text->image)
            self.mapping_lang = _make_mlp(self.lang_dim,
                                          self.joint_embedding_size,
                                          self.joint_embedding_dropout)
            self.gamma = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in range(5))
            self.beta = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in range(5))

            self.joint_fusion = nn.ModuleList([_make_conv(self.joint_inp_dim, self.joint_out_dim, 1) \
                                               for _ in range(5)])

        else:
            print("NO FUSION INVOLVED.")

    def init_configs(self, cfg):
        # common params
        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.joint_embedding_size = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        self.joint_embedding_dropout = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT
        self.joint_mlp_layers = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_MLP_LAYERS

        self.max_query_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        self.n_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS
        self.coord_dim = 8
        self.joint_inp_dim = self.coord_dim + self.joint_embedding_size
        self.joint_out_dim = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_OUT_SIZE

        # mha params
        self.n_head = 8
        self.embed_dim = 2048
        self.t2i_hidden_dim = 1024  # 256 * 4
        self.i2t_hidden_dim = 3072  # 768 * 4

        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024

    def forward(self, x):
        visual_features = x["visual"]
        language_dict_features = x["lang"]

        batch_size = visual_features[0].shape[0]
        device = visual_features[0].device

        fused_visual_features = None
        fused_language_dict_features = None

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-S":
            language_feature = language_dict_features['hidden']
            mask = language_dict_features['masks']
            # text -> image
            if self.use_checkpoint:
                q0, q1, q2, q3, q4 = checkpoint.checkpoint(
                    self.t2i_attn,
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3],
                    visual_features[4],
                    language_feature, language_feature,
                    mask,
                    self.dummy_tensor
                )
            else:
                q0, q1, q2, q3, q4 = self.t2i_attn(
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3], # 视觉特征作为query
                    visual_features[4], # 语言特征作为key
                    language_feature, language_feature,  # 语言特征作为value
                    attention_mask=mask
                )

            fused_visual_features = [q0, q1, q2, q3, q4]
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-B": # 论文公式(4)(5)(6)的完整实现
            # 论文中提到的两个好处：
            # 1. 提升短语定位性能
            if self.use_checkpoint:
                q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = checkpoint.checkpoint(self.b_attn,
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3],
                    visual_features[4],
                    language_dict_features['hidden'],
                    language_dict_features['masks'],
                    self.dummy_tensor
                )
            else:
                q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = self.b_attn(
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3], # 视觉特征
                    visual_features[4],
                    language_dict_features['hidden'], # 语言特征
                    language_dict_features['masks'], # 注意力掩码
                    self.dummy_tensor
                )

            # 融合后的视觉特征更适合定位
            fused_visual_features = [q0, q1, q2, q3, q4] # 公式(5): O^(i+1) = DyHeadModule(O^i + O'i2t) 融合后的视觉特征
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL and self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                # 语言特征指导视觉特征
                language_features = self.shrink_lang(torch.cat([l0, l1, l2, l3, l4], dim = -1))
                # 公式(6): P^(i+1) = BERTLayer(P^i + P'i2t) 语言特征更新
            else:
                language_features = l0

            language_dict_features['hidden'] = language_features
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "SCAN":
            # text -> image
            language_feature = language_dict_features['aggregate']
            language_feature = self.mapping_lang(language_feature)
            visu_feat = []
            for ii, feat in enumerate(visual_features):
                attn_feat = func_attention(feat, language_feature, smooth=1, raw_feature_norm="softmax")
                visu_feat.append(attn_feat)

            fused_visual_features = [fusion(feat) for feat, fusion in zip(visu_feat, self.joint_fusion)]
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "FILM":
            # text -> image
            # relative position embedding
            coord_feats = [_make_coord(batch_size, x.shape[2], x.shape[3]) for x in visual_features]
            # I only use a global representation of language
            # you can also use more complex modeling using word-level representations
            # Usage: lang_feat = lang_feat['words'] shape [seq_len, dim]
            language_feature = language_dict_features['aggregate']
            language_feature = self.mapping_lang(language_feature)

            # attention mechanism for fusion
            gamma = [F.tanh(gamma(language_feature)) for gamma in self.gamma]
            beta = [F.tanh(beta(language_feature)) for beta in self.beta]

            visu_feat = []
            for ii, feat in enumerate(visual_features):
                coord_feat = coord_feats[ii].to(device)
                feat = torch.cat([feat, coord_feat], dim=1)
                b = beta[ii].view(batch_size, -1, 1, 1).expand_as(feat)
                g = gamma[ii].view(batch_size, -1, 1, 1).expand_as(feat)
                feat = F.relu(g * feat + b)
                visu_feat.append(feat)

            fused_visual_features = [fusion(feat) for feat, fusion in zip(visu_feat, self.joint_fusion)]
            fused_language_dict_features = language_dict_features

        else:
            fused_visual_features = visual_features
            fused_language_dict_features = language_dict_features

        features_dict = {"visual": fused_visual_features,
                         "lang": fused_language_dict_features}

        return features_dict

class ShapeAdaptiveAdapter(nn.Module):
    def __init__(self, c_in, num_adapters=10, reduction=4):
        super().__init__()
        self.num_adapters = num_adapters

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c_in),
                nn.Linear(c_in, c_in // reduction, bias=False),
                nn.GELU(),
                # nn.Dropout(0.9),  # 添加dropout
                nn.Linear(c_in // reduction, c_in, bias=False),
            ) for _ in range(num_adapters)
        ])

        # self.adapter_weight = nn.Parameter(torch.tensor(0.0))
        # 使用Kaiming初始化
        for adapter in self.adapters:
            for m in adapter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def select_adapter_by_aspect_ratio(self, output_feat, aspect_ratio):
        # 1. 获取所有adapter的输出
        output_feat = output_feat.contiguous()
        aspect_ratio = aspect_ratio.contiguous()

        B, C, H, W = output_feat.shape
        _, num_anchors, _, _ = aspect_ratio.shape

        # print(aspect_ratio.shape) # torch.Size([2, 1, 72, 72])
        # sys.exit()

        # 1. 重塑特征以适应线性层
        output_feat_flat = output_feat.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        output_feat_flat = output_feat_flat.reshape(-1, C)  # [B*H*W, C]

        # 并行处理所有adapter的输出并添加残差连接
        adapter_outputs = []
        for adapter in self.adapters:
            # 计算适配器输出
            adapter_out = adapter(output_feat_flat)
            # 添加残差连接
            # out = adapter_out + output_feat_flat
            out = adapter_out
            # 重塑回原始维度
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            adapter_outputs.append(out)
        adapter_outputs = torch.stack(adapter_outputs)  # [num_adapters, B, C, H, W]

        aspect_ratio_flat = aspect_ratio.reshape(B, num_anchors, -1)  # [B, num_anchors, H*W]
        thresholds = torch.tensor([1/9, 1/7, 1/5, 1/3, 1, 3, 5, 7, 9], device=aspect_ratio.device)

        # 使用einsum进行比较操作，避免隐式广播
        aspect_expanded = aspect_ratio_flat.unsqueeze(-1)  # [B, num_anchors, H*W, 1]
        thresholds = thresholds.view(1, 1, 1, -1)  # [1, 1, 1, 9]

        # 显式执行比较，保持维度控制
        comparison = (aspect_expanded > thresholds)
        adapter_indices = torch.sum(comparison, dim=-1)  # [B, num_anchors, H*W]

        # print(aspect_ratio_flat.unsqueeze(-1).shape) torch.Size([2, 1, 5184, 1])
        # print(thresholds.shape) torch.Size([2, 1, 9])
        # sys.exit()

        # print(adapter_indices.shape) # torch.Size([2, 2, 5184])
        # print("哈哈")
        # sys.exit()

        # 5. 使用gather操作选择对应的adapter输出
        # 重塑adapter_outputs以便于gather操作
        adapter_outputs = adapter_outputs.permute(0, 1, 3, 4, 2).contiguous()  # [num_adapters, B, H, W, C]
        
        # 创建索引张量
        batch_indices = torch.arange(B, device=output_feat.device).view(B, 1, 1).expand(B, num_anchors, H*W)
        h_indices = torch.arange(H, device=output_feat.device).view(1, 1, H, 1).expand(B, num_anchors, H, W).reshape(B, num_anchors, -1)
        w_indices = torch.arange(W, device=output_feat.device).view(1, 1, 1, W).expand(B, num_anchors, H, W).reshape(B, num_anchors, -1)

        # print("哈哈3")

        # print(f"adapter_outputs shape: {adapter_outputs.shape}") # adapter_outputs shape: torch.Size([10, 2, 72, 72, 256])
        # print(f"adapter_indices shape: {adapter_indices.shape}") # adapter_indices shape: torch.Size([2, 2, 5184])
        # print(f"batch_indices shape: {batch_indices.shape}") # batch_indices shape: torch.Size([2, 1, 5184])
        # print(f"h_indices shape: {h_indices.shape}") # h_indices shape: torch.Size([2, 1, 5184])
        # print(f"w_indices shape: {w_indices.shape}") # w_indices shape: torch.Size([2, 1, 5184])

        # print("哈哈")
        # sys.exit()
        
        # 使用gather操作选择对应的adapter输出
        # 7. 使用gather操作选择对应的adapter输出
        selected_outputs = adapter_outputs[
            adapter_indices,  # [B, num_anchors, H*W]
            batch_indices,    # [B, num_anchors, H*W]
            h_indices,        # [B, num_anchors, H*W]
            w_indices,        # [B, num_anchors, H*W]
        ]  # [B, num_anchors, H*W, C]

        # print(selected_outputs.shape) # torch.Size([2, 2, 5184, 256])
        # print("哈哈")
        # print(B) # 2
        # print(num_anchors) # 1
        # print(H) # 72
        # print(W) # 72
        # print(C) # 256
        # sys.exit()
        # print("哈哈4")
        selected_outputs = selected_outputs.reshape(B, num_anchors, H, W, C)  # [B, num_anchors, H, W, C]

        # 对num_anchors维度取平均，得到最终输出
        selected_outputs = selected_outputs.mean(dim=1)  # [B, H, W, C]
        selected_outputs = selected_outputs.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # 6. 加权混合
        # ratio = torch.sigmoid(self.adapter_weight)  # 限制在0-1之间
        # output_feat = ratio * selected_outputs + (1 - ratio) * output_feat

        ratio = 1.0
        output_feat = ratio * selected_outputs + output_feat
        # output_feat = ratio * selected_outputs + (1 - ratio) * output_feat
        # output_feat = ratio * selected_outputs + output_feat
        # output_feat = ratio * selected_outputs + ratio * output_feat

        # print("哈哈2")
        # print(output_feat.shape)
        # sys.exit()
        
        return output_feat


class VLDyHead(torch.nn.Module):
    def __init__(self, cfg):
        super(VLDyHead, self).__init__()
        self.cfg = cfg
        # bert_cfg = BertConfig.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)
        if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":
            lang_cfg = BertConfig.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)
        elif cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "clip":
            lang_cfg = cfg
        else:
            lang_cfg = None
            raise NotImplementedError

        num_classes = cfg.MODEL.DYHEAD.NUM_CLASSES - 1
        num_tokens = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        num_anchors = len(cfg.MODEL.RPN.ASPECT_RATIOS) * cfg.MODEL.RPN.SCALES_PER_OCTAVE
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        channels = cfg.MODEL.DYHEAD.CHANNELS

        if cfg.MODEL.DYHEAD.USE_GN:
            bn_type = ['gn', cfg.MODEL.GROUP_NORM.NUM_GROUPS]
        elif cfg.MODEL.DYHEAD.USE_NSYNCBN:
            bn_type = 'nsbn'
        elif cfg.MODEL.DYHEAD.USE_SYNCBN:
            bn_type = 'sbn'
        else:
            bn_type = None

        use_dyrelu = cfg.MODEL.DYHEAD.USE_DYRELU
        use_dyfuse = cfg.MODEL.DYHEAD.USE_DYFUSE
        use_deform = cfg.MODEL.DYHEAD.USE_DFCONV

        if cfg.MODEL.DYHEAD.CONV_FUNC:
            conv_func = lambda i, o, s: eval(cfg.MODEL.DYHEAD.CONV_FUNC)(i, o, s, bn_type=bn_type)
        else:
            conv_func = lambda i, o, s: Conv3x3Norm(i, o, s, deformable=use_deform, bn_type=bn_type)

        dyhead_tower = []
        for i in range(cfg.MODEL.DYHEAD.NUM_CONVS):
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON:
                # cross-modality fusion
                dyhead_tower.append(
                    VLFuse(cfg)
                )
                # self language path
                if i < cfg.MODEL.DYHEAD.NUM_CONVS - 1 or cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:
                    # dyhead_tower.append(
                    #     BertEncoderLayer(
                    #     bert_cfg,
                    #     clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                    #     clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)
                    # )
                    if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":
                        dyhead_tower.append(
                            BertEncoderLayer(
                                lang_cfg,
                                clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                                clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)
                        )
                    elif cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "clip":
                        dyhead_tower.append(
                            CLIPTransformerLayer(lang_cfg)
                        )
                    else:
                        raise NotImplementedError

                else:
                    dyhead_tower.append(
                        DummyLayer()
                    )

            # self vision path
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=conv_func,
                    use_dyrelu=(use_dyrelu and in_channels == channels) if i == 0 else use_dyrelu, # 动态ReLU
                    use_dyfuse=(use_dyfuse and in_channels == channels) if i == 0 else use_dyfuse, # 动态特征融合
                    use_deform=(use_deform and in_channels == channels) if i == 0 else use_deform, # 可变形卷积
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        # 基础检测头
        self.cls_logits = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=1) # 分类
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1) # 边界框回归
        self.centerness = nn.Conv2d(channels, num_anchors * 1, kernel_size=1) # 中心度预测（中心度表示预测框中心点到目标中心的距离）

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DYHEAD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        log_scale = self.cfg.MODEL.DYHEAD.LOG_SCALE

        # 这几个分类头的作用：难道是grounding的？ 根据grounding引出的（这个领域通用的吧）

        # soft token head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            self.token_logits = nn.Conv2d(channels, num_anchors * num_tokens, kernel_size=1)
            # ABLATION
            # self.token_logits = nn.Conv2d(channels, num_anchors * num_tokens, kernel_size=1, bias=False)
            # self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
            # self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # contrastive alignment head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS == False
            contrastive_hdim = cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_HIDDEN_DIM
            self.contrastive_align_projection_image = nn.Conv2d(channels, num_anchors * contrastive_hdim, kernel_size=1)
            self.contrastive_align_projection_text = nn.Linear(channels, contrastive_hdim, bias=True)
            self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)

        # dot product soft token head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS == False
            self.dot_product_projection_image = nn.Identity()
            self.dot_product_projection_text = nn.Linear(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM,
                                                         num_anchors * channels, bias=True)
            self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
            # DEBUG
            # self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
            self.bias_lang = nn.Parameter(torch.zeros(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM), requires_grad=True)
            self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # initialization
        for modules in [self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # Grounding任务带来的额外损失:（只是利用了grounding的丰富文本，进行了数据扩充）
        # 通过多任务学习增强特征
        # if use soft token loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            for modules in [self.token_logits]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

            torch.nn.init.constant_(self.token_logits.bias, bias_value)
            # print(torch.norm(self.token_logits.weight))

        # if use contrastive loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            for modules in [self.contrastive_align_projection_image]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

        # if use dot product token loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            for modules in [self.dot_product_projection_image]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, bias_value)
        
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "clip":
                lang_cfg = BertConfig.from_pretrained("bert-base-uncased")
                lang_cfg.hidden_size = cfg.MODEL.CLIP.WIDTH
                lang_cfg.vocab_size = cfg.MODEL.CLIP.VOCAB_SIZE
            self.mlm_head = BertLMPredictionHead(
                lang_cfg
            ) #nn.Linear(hidden_size, config.vocab_size, bias=False)

        if self.cfg.MODEL.USE_Prototype_FUSE:
            # Separate prototypes for visual and text features
            self.visual_prototypes = {}  # Dictionary to store visual class prototypes
            self.text_prototypes = {}    # Dictionary to store text class prototypes
            self.visual_features_buffer = {}  # Buffer for visual features
            self.text_features_buffer = {}    # Buffer for text features
            self.prototype_counts = {}   # Counter for each class
            self.scale_counts = {}
        
        if self.cfg.MODEL.USE_ISA:

            # print(channels)
            # print("哈哈")
            # sys.exit()

            self.shape_adapter = ShapeAdaptiveAdapter(channels)

    def save_shape_adapter(self, path):
        """保存ShapeAdaptiveAdapter的参数"""
        if hasattr(self, 'shape_adapter'):
            print(f"Saving ShapeAdaptiveAdapter to {path}")
            torch.save({
                'shape_adapter_state_dict': self.shape_adapter.state_dict(),
            }, path)
            print(f"ShapeAdaptiveAdapter saved to {path}")
        else:
            print("No ShapeAdaptiveAdapter found to save")

    def load_shape_adapter(self, path):
        """加载ShapeAdaptiveAdapter的参数"""
        if hasattr(self, 'shape_adapter'):
            if os.path.exists(path):
                print(f"Loading ShapeAdaptiveAdapter from {path}")
                # checkpoint = torch.load(path)
                # self.shape_adapter.load_state_dict(checkpoint['shape_adapter_state_dict'])
                device = next(self.shape_adapter.parameters()).device
                checkpoint = torch.load(path, map_location=device)
                self.shape_adapter.load_state_dict(checkpoint['shape_adapter_state_dict'])
                print(f"ShapeAdaptiveAdapter loaded from {path}")
                return True
            else:
                print(f"No checkpoint found at {path}")
        else:
            print("No ShapeAdaptiveAdapter found to load")
        return False


    def update_prototypes(self, visual_features, targets):
        """Update visual prototypes during training - Optimized version"""
        if not self.training:
            return
        
        # 获取特征
        if isinstance(visual_features, (list, tuple)):
            visual_feats = list(visual_features)  # 确保转换为 list
        else:
            visual_feats = [visual_features]  # 转换为列表形式
        
        # 预处理所有目标以便批量计算
        valid_targets = [t for t in targets if t is not None]

        # print(f"Number of visual_features scales: {len(visual_feats)}")
        # for i, feat in enumerate(visual_feats):
        #     if isinstance(feat, tuple):
        #         feat = feat[0]
        #     print(f"  Scale {i} shape: {feat.shape}")
        
        # print(f"Number of targets (images): {len(targets)}")
        # print(f"Number of valid targets: {len(valid_targets)}")
        
        # for img_idx, target in enumerate(valid_targets):
        #     if hasattr(target, 'bbox'):
        #         num_boxes = len(target.bbox) if hasattr(target.bbox, '__len__') else 0
        #         print(f"  Image {img_idx} has {num_boxes} bounding boxes")
        
        # 第一阶段：收集所有特征到buffer
        for img_idx, target in enumerate(valid_targets):
            # 获取标签和框
            labels = target.get_field("labels")
            boxes = target.bbox
            
            if len(labels) == 0:
                continue
            
            # 获取特征图和图像的尺寸
            H_img, W_img = target.size
            
            # 为每个尺度处理特征
            for scale_idx, visual_feat in enumerate(visual_feats):
                if isinstance(visual_feat, tuple):
                    visual_feat = visual_feat[0]
                
                # 获取当前尺度的特征图尺寸
                _, _, H_feat, W_feat = visual_feat.shape
                
                # 计算缩放因子
                scale_h = H_feat / H_img
                scale_w = W_feat / W_img
                # print()
                # print(f"Feature: {H_feat}x{W_feat}, Image: {H_img}x{W_img}, Scale: {scale_h:.4f}x{scale_w:.4f}")
                # print("哈哈")
                
                # 批量转换所有框的坐标到特征图坐标
                boxes_np = boxes.cpu().numpy()
                boxes_feat = np.zeros_like(boxes_np)
                boxes_feat[:, 0] = boxes_np[:, 0] * scale_w  # x1
                boxes_feat[:, 1] = boxes_np[:, 1] * scale_h  # y1
                boxes_feat[:, 2] = boxes_np[:, 2] * scale_w  # x2
                boxes_feat[:, 3] = boxes_np[:, 3] * scale_h  # y2
                boxes_feat = boxes_feat.astype(int)
                
                # 为每个框提取特征并更新buffer
                for box_idx, (label, box_feat) in enumerate(zip(labels, boxes_feat)):
                    label_item = label.item()
                    x1_feat, y1_feat, x2_feat, y2_feat = box_feat
                    
                    # 确保坐标在有效范围内
                    x1_feat = max(0, min(x1_feat, W_feat-1))
                    y1_feat = max(0, min(y1_feat, H_feat-1))
                    x2_feat = max(x1_feat+1, min(x2_feat, W_feat))
                    y2_feat = max(y1_feat+1, min(y2_feat, H_feat))
                    
                    # 提取视觉特征
                    box_visual_features = visual_feat[img_idx, :, y1_feat:y2_feat, x1_feat:x2_feat]
                    
                    # 检查边界框是否有效
                    if box_visual_features.numel() == 0:
                        continue
                        
                    # 平均池化
                    box_visual_features = box_visual_features.mean(dim=(1, 2))
                    
                    # 如果这是新标签，初始化缓冲区
                    if label_item not in self.visual_features_buffer:
                        self.visual_features_buffer[label_item] = []
                        self.prototype_counts[label_item] = 0
                        self.scale_counts[label_item] = {}
                    
                    # 更新特征缓冲区，保存不同尺度的特征
                    self.visual_features_buffer[label_item].append({
                        'scale': scale_idx,
                        'features': box_visual_features
                    })
                    self.prototype_counts[label_item] += 1
                    
                    # 更新该尺度的计数器
                    if scale_idx not in self.scale_counts[label_item]:
                        self.scale_counts[label_item][scale_idx] = 0
                    self.scale_counts[label_item][scale_idx] += 1
        
        # # 输出提取的特征统计信息
        # print("Prototype statistics:")
        # for label_item, count in self.prototype_counts.items():
        #     print(f"  Class {label_item}: {count} total features across all scales")
        #     for scale, scale_count in self.scale_counts[label_item].items():
        #         print(f"    Scale {scale}: {scale_count} features")
        
        # print("哈哈")
        
        # 第二阶段：根据收集的buffer计算prototypes
        for label_item, features_list in self.visual_features_buffer.items():
            # 如果没有为这个类别初始化原型，则初始化
            if label_item not in self.visual_prototypes:
                self.visual_prototypes[label_item] = {'scales': {}}
            
            # 按照scale对特征进行分组
            scale_features = {}
            for feature_data in features_list:
                scale = feature_data['scale']
                if scale not in scale_features:
                    scale_features[scale] = []
                scale_features[scale].append(feature_data['features'])
            
            # 计算每个尺度的平均原型
            for scale, features in scale_features.items():
                # 先计算该尺度的所有特征的平均值
                if len(features) > 0:
                    # 如果该尺度还没有原型，直接计算平均值作为原型
                    stacked_features = torch.stack(features)
                    self.visual_prototypes[label_item]['scales'][scale] = stacked_features.mean(dim=0)

    def enhance_features_with_prototypes(self, features):
        """Enhance features using class prototypes during testing - New version with foreground focus"""
        if self.training or not self.visual_prototypes:
            return features
    
        # return features
        
        if isinstance(features, (list, tuple)):
            enhanced_features_list = []
            for scale_idx, features_scale in enumerate(features):
                enhanced_features = features_scale.clone()
                B, C, H, W = features_scale.shape

                # print(f"\nScale {scale_idx} - Feature map size: {H}x{W}")
                total_pixels = H * W
                
                # 收集当前尺度的所有原型
                prototype_features = []
                prototype_labels = []
                for label, prototype in self.visual_prototypes.items():
                    if scale_idx in prototype['scales']:
                        prototype_features.append(prototype['scales'][scale_idx])
                        prototype_labels.append(label)
                
                if not prototype_features:  # 如果没有原型，返回原始特征
                    enhanced_features_list.append(enhanced_features)
                    continue
                    
                prototype_features = torch.stack(prototype_features)  # [num_prototypes, C]
                # print(f"Number of prototypes at this scale: {len(prototype_features)}")
                
                # 对每个batch处理
                for b in range(B):
                    # 重塑特征以便计算相似度
                    current_features = features_scale[b]  # [C, H, W]

                    # print(current_features.shape)
                    # print("哈哈")
                    # # sys.exit()

                    features_reshaped = current_features.view(C, -1).t()  # [H*W, C]

                    # 进行归一化
                    prototype_features_norm = F.normalize(prototype_features, p=2, dim=1)

                    current_features_norm = F.normalize(features_reshaped, p=2, dim=1)
                    
                    # 计算每个像素与所有原型的相似度
                    similarities = F.cosine_similarity(
                        current_features_norm.unsqueeze(1),  # [H*W, 1, C]
                        prototype_features_norm.unsqueeze(0),  # [1, num_prototypes, C]
                        dim=2
                    )  # [H*W, num_prototypes]
                    
                    # 将相似度图重塑回空间维度
                    similarity_maps = similarities.view(H, W, -1)  # [H, W, num_prototypes]

                    # # 计算相似度阈值（可以根据需要调整）
                    # similarity_threshold = 0.85

                    thresholds = {
                        0: 0.93,  # P2 - 最大尺度，低级特征，使用较低阈值
                        1: 0.87, # P3
                        2: 0.81,   # P4
                        3: 0.80, # P5
                        4: 0.86   # P6 - 最小尺度，高级特征，使用较高阈值
                    }
                    similarity_threshold = thresholds[scale_idx]


                    # 创建前景概率图
                    foreground_map = torch.zeros(H, W, device=features_scale.device)

                    # 用于存储高相似度原型的特征
                    matched_prototypes = []
                    matched_similarities = []
                    
                    # 对每个原型的相似度图进行处理
                    # print(f"\nBatch {b} statistics:")
                    for proto_idx, similarity_map in enumerate(similarity_maps.permute(2, 0, 1)):
                        # 只保留高于阈值的相似度
                        valid_similarities = torch.where(similarity_map > similarity_threshold,
                                                    similarity_map,
                                                    torch.zeros_like(similarity_map))
                        
                        # valid_pixels = (similarity_map > similarity_threshold).sum().item()
                        # percentage = (valid_pixels / total_pixels) * 100
                        # print(f"Prototype {prototype_labels[proto_idx]}: {valid_pixels} pixels ({percentage:.2f}%) above threshold {similarity_threshold}")
                        
                        # 累加到前景概率图
                        foreground_map += valid_similarities

                        # 如果这个原型有足够高的相似度，保存下来
                        if valid_similarities.max() > similarity_threshold:
                            matched_prototypes.append(prototype_features[proto_idx])
                            matched_similarities.append(valid_similarities)
                    

                    # 更好的归一化处理
                    if foreground_map.max() > 0:
                        # # 1. 先用阈值过滤掉较弱的响应
                        # max_val = foreground_map.max()
                        # foreground_map = torch.where(foreground_map < 0.3 * max_val,
                        #                         torch.zeros_like(foreground_map),
                        #                         foreground_map)
                        
                        # # 2. 可选：使用更强的非线性变换来增强对比度
                        # foreground_map = torch.pow(foreground_map, 2.0)  # 或者其他指数，如1.5, 3.0等
                        
                        # 3. 最后再做归一化，但只对非零值进行
                        if foreground_map.max() > 0:  # 再次检查是否有非零值
                            # 最大最小值归一化
                            min_val = foreground_map.min()
                            max_val = foreground_map.max()
                            foreground_map = (foreground_map - min_val) / (max_val - min_val + 1e-6)


                    # # 统计选中的点的数量和百分比
                    # selected_points = (foreground_map > 0).sum().item()
                    # total_points = H * W
                    # selected_percentage = (selected_points / total_points) * 100
                    # print(f"Scale {scale_idx}, Batch {b}: Selected points: {selected_points}/{total_points} ({selected_percentage:.2f}%)")

                    # # 使用前景概率图作为权重来增强特征
                    # alpha = 1.0

                    alpha = 1.0

                    foreground_weight = foreground_map.unsqueeze(0).expand_as(current_features)
                    
                    # 根据前景概率图来决定特征增强的程度
                    enhanced_features[b] = current_features + alpha * foreground_weight * current_features
                    # enhanced_features[b] = (1 - alpha) * current_features + alpha * foreground_weight * current_features
                    # enhanced_features[b] = current_features *(1 + alpha * foreground_weight)
                    # enhanced_features[b] = current_features


                    # # 第二部分：使用匹配的prototype进行特征增强
                    # if matched_prototypes:
                    #     # prototype增强的强度
                    #     alpha2 = 10.0
                    #     matched_prototypes = torch.stack(matched_prototypes)  # [num_matched, C]
                    #     matched_similarities = torch.stack(matched_similarities)  # [num_matched, H, W]
                        
                    #     # 归一化相似度权重
                    #     matched_similarities = matched_similarities / (matched_similarities.sum(dim=0, keepdim=True) + 1e-6)
                        
                    #     # 计算加权的prototype特征
                    #     weighted_prototypes = (matched_prototypes.unsqueeze(-1).unsqueeze(-1) *  # [num_matched, C, 1, 1]
                    #                         matched_similarities.unsqueeze(1))  # [num_matched, 1, H, W]
                    #     prototype_enhancement = weighted_prototypes.sum(dim=0)  # [C, H, W]
                        
                    #     # 将prototype特征加到增强后的特征上
                    #     enhanced_features[b] = enhanced_features[b] + alpha2 * prototype_enhancement
                
                enhanced_features_list.append(enhanced_features)

            return enhanced_features_list


    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None, targets=None):
        logits = []
        bbox_reg = []
        centerness = []

        # 1. 特征融合

        # x是 来自FPN的多尺度特征图
        feat_inputs = {"visual": x, 
                       "lang": language_dict_features} 
        # language_dict_features: 语言特征字典，包含：
        # 'embedded': 语言嵌入特征,
        # 'hidden': 隐藏层特征,
        # 'masks': 掩码信息,
        # 'mlm_labels': MLM标签（如果使用） 自监督学习：不需要额外的标注数据。随机选择一些token进行掩码（mask）

        dyhead_tower = self.dyhead_tower(feat_inputs)

        # soft token
        t_logits = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            t_logits = []
        
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:
            embedding = dyhead_tower["lang"]["hidden"] # B, L, D
        

        # print(embedding.shape) # torch.Size([1, 256, 768])
        # print([v.shape for v in dyhead_tower["visual"]]) 
        # # [torch.Size([1, 256, 80, 80]), torch.Size([1, 256, 40, 40]), torch.Size([1, 256, 20, 20]), torch.Size([1, 256, 10, 10]), torch.Size([1, 256, 5, 5])]
        # print("哈哈")
        # sys.exit()
        
        # MLM loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            mlm_logits = self.mlm_head(embedding) # 预测被掩码的token
        else:
            mlm_logits = None

        # contrastive
        contrastive_logits = None
        proj_tokens = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS: # 一、对比学习
            contrastive_logits = []
            # follow MDETR's way
            proj_tokens = F.normalize( # 文本特征投影
                self.contrastive_align_projection_text(embedding), p=2, dim=-1
            )

        # dot product soft token
        dot_product_logits = None
        dot_product_proj_tokens = None
        dot_product_proj_tokens_bias = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS: # 二、点积
            dot_product_logits = []
            # norm
            embedding = F.normalize(embedding, p=2, dim=-1)
            dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0) # 文本特征投影
            # 输出维度: [B, 256, num_anchors * channels]
            # w/o norm
            # dot_product_proj_tokens = self.dot_product_projection_text(embedding / 28.0)

            dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0
            # 维度: [B, 256, 1]  # 每个token的偏置项

        # shallow contrastive (original feature from image & text encoder)
        shallow_img_emb_feats = None
        shallow_text_emb = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS \
                or self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            shallow_img_emb_feats = []
            shallow_text_emb = embedding

        # print([v.shape for v in x])
        # shallow contrastive: use the feature from swint backbone
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            for b, feature in enumerate(swint_feature_c4):
                # BF, CF, HF, WF = feat.shape
                # shallow_img_emb = permute_and_flatten(feat, BF, -1, CF, HF, WF)
                shallow_img_emb_feats.append(feature)

        fused_visual_features = None
        if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
            fused_visual_features = []

        if self.cfg.MODEL.USE_Prototype_FUSE:
            # Update prototypes during training
            if self.training and targets is not None:
                # self.update_prototypes(visual_features, language_dict_features, targets)
                self.update_prototypes(dyhead_tower["visual"], targets)

            # print("哈哈")
            if not self.training:
                # visual_features = self.enhance_features_with_prototypes(visual_features, is_visual=True)
                dyhead_tower["visual"] = self.enhance_features_with_prototypes(dyhead_tower["visual"])

        # use the feature from FPN
        for l, feature in enumerate(x): # [l]对应不同的特征层

            if self.cfg.MODEL.USE_ISA:
                # 1. 先获取预测框
                bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l])) # torch.Size([4, 4, 100, 124])

                # if l == 0 and self.training:  # 仅在训练时对第一个特征层进行检查，避免影响推理速度
                #     with torch.no_grad():
                #         dx = bbox_pred[:, 0::4].detach()
                #         dy = bbox_pred[:, 1::4].detach()
                #         dw = bbox_pred[:, 2::4].detach()
                #         dh = bbox_pred[:, 3::4].detach()
                #         print(f"dx 范围: [{dx.min().item():.4f}, {dx.max().item():.4f}], 均值: {dx.mean().item():.4f}")
                #         print(f"dy 范围: [{dy.min().item():.4f}, {dy.max().item():.4f}], 均值: {dy.mean().item():.4f}")
                #         print(f"dw 范围: [{dw.min().item():.4f}, {dw.max().item():.4f}], 均值: {dw.mean().item():.4f}")
                #         print(f"dh 范围: [{dh.min().item():.4f}, {dh.max().item():.4f}], 均值: {dh.mean().item():.4f}")
                
                # print("哈哈")
                # sys.exit()

                # print(f"1. bbox_pred shape: {bbox_pred.shape}")  # [B, num_anchors * 4, H, W]
                # print(f"   - Batch size: {bbox_pred.shape[0]}")
                # print(f"   - Number of anchors * 4: {bbox_pred.shape[1]}")
                # print(f"   - Height: {bbox_pred.shape[2]}")
                # print(f"   - Width: {bbox_pred.shape[3]}")

                # sys.exit()
                
                # 2. 计算宽高比
                dw = bbox_pred[:, 2::4]  # 宽度的对数偏移 .从索引2开始，每隔4个取一个值，即取所有 dw 值
                dh = bbox_pred[:, 3::4]  # 高度的对数偏移 .从索引3开始，每隔4个取一个值，即取所有 dh 值
                # 使用当前特征层的实际宽高比
                # B_sia, C_sia, H_sia, W_sia = dyhead_tower["visual"][l].shape
                # anchor_w = W_sia / H_sia  # 当前特征层的宽高比
                anchor_w = 1.0  # 当前特征层的宽高比
                anchor_h = 1.0    # 基准高度

                w = anchor_w * torch.exp(dw)  # 实际的宽度
                h = anchor_h * torch.exp(dh)  # 实际的高度

                aspect_ratio = w / h # aspect_ratio shape: torch.Size([4, 1, 100, 124])

                # print(f"2. aspect_ratio shape: {aspect_ratio.shape}")  # [B, num_anchors, H, W]
                # print(f"   - Batch size: {aspect_ratio.shape[0]}")
                # print(f"   - Number of anchors: {aspect_ratio.shape[1]}")
                # print(f"   - Height: {aspect_ratio.shape[2]}")
                # print(f"   - Width: {aspect_ratio.shape[3]}")
                
                # 3. 根据宽高比选择adapter处理分类特征
                cls_feature = dyhead_tower["visual"][l] # cls_feature shape: torch.Size([4, 256, 100, 124])

                # print(f"3. cls_feature shape: {cls_feature.shape}")  # [B, C, H, W]
                # print(f"   - Batch size: {cls_feature.shape[0]}")
                # print(f"   - Channels: {cls_feature.shape[1]}")
                # print(f"   - Height: {cls_feature.shape[2]}")
                # print(f"   - Width: {cls_feature.shape[3]}")

                # print("哈哈哈")

                # sys.exit()

                adapter_output = self.shape_adapter.select_adapter_by_aspect_ratio(
                    cls_feature, 
                    aspect_ratio
                )
                dyhead_tower["visual"][l] = adapter_output

                # print(f"4. adapter_output shape: {adapter_output.shape}")  # 应该与 cls_feature 相同
    
                # 可以再进行一次的，因为此时是经过adapter修正之后的
                bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l])) # torch.Size([4, 4, 100, 124])
                logits.append(self.cls_logits(dyhead_tower["visual"][l]))

                # print(f"5. logits shape: {logits[-1].shape}")  # [B, num_classes, H, W]
                
                # 5. 其他预测保持不变
                bbox_reg.append(bbox_pred)

                # print(f"6. bbox_reg shape: {bbox_reg[-1].shape}")  # [B, num_anchors * 4, H, W]

                # print("哈哈")
                # sys.exit()
            
            else:

                # 预测每个位置的类别
                logits.append(self.cls_logits(dyhead_tower["visual"][l])) # [B, num_anchors * num_classes, H, W]
                
                # 预测边界框的偏移量
                bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l])) # bbox_reg: [B, num_anchors * 4, H, W]
                bbox_reg.append(bbox_pred)

            # 测目标中心度
            centerness.append(self.centerness(dyhead_tower["visual"][l])) # [B, num_anchors * 1, H, W]

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS: # 三、token的
                t_logits.append(self.token_logits(dyhead_tower["visual"][l])) # [B, num_anchors * num_tokens, H, W]

                # ABLATION
                # b = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                # x = dyhead_tower["visual"][l]
                # B, C, H, W = x.shape
                # bias = b.repeat(B, 1, H, W)
                # t_logits.append(self.token_logits(dyhead_tower["visual"][l] + bias) + self.bias0)

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS: # 一、对比学习
                x = dyhead_tower["visual"][l]
                B, _, H, W = x.shape
                C = proj_tokens.shape[2]
                proj_queries = self.contrastive_align_projection_image(dyhead_tower["visual"][l]) # 2. 视觉特征投影
                proj_queries = permute_and_flatten(proj_queries, B, -1, C, H, W) # [B, H*W, D*A]
                # 是每个空间位置的特征和每个token的特征对齐。

                normalized_img_emb = F.normalize(proj_queries, p=2, dim=-1)
                normalized_text_emb = proj_tokens # [B, L, D]
                contrastive_logit = ( # 3. 计算对比损失
                        torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.log_scale.exp())
                contrastive_logits.append(contrastive_logit) # contrastive_logits: [B, L, H*W]

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS: # 二、点积
                x = dyhead_tower["visual"][l] # x  # 输入维度: [B, C, H, W]  # 多尺度特征图
                if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES: # 保存融合后的视觉特征用于后续处理
                    fused_visual_features.append(x) # fused_visual_features: [B, C, H, W]
                B, C, H, W = x.shape

                # add bias (language)
                dot_product_proj_queries = self.dot_product_projection_image(x) # 3. 视觉特征投影
                dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)
                # 同样是每个空间位置和每个token对齐。
                # 其中A = H*W，表示特征图上的所有空间位置

                A = dot_product_proj_queries.shape[1]
                bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)

                # dot_product_proj_queries: [B, A, C]
                # dot_product_proj_tokens: [B, 256, num_anchors * C]
                # 点积操作后：[B, A, 256]
                dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias
                if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                    dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                    dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
                dot_product_logits.append(dot_product_logit) # dot_product_logits: [B, num_anchors * L, H, W]

                # 每个空间位置的视觉特征都会与所有文本token计算相似度，形成一个相似度矩阵，这可以用于后续的损失计算和特征融合

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS: # 使用原始特征进行对比学习
                feat = feature
                BF, CF, HF, WF = feat.shape
                shallow_img_emb = permute_and_flatten(feat, BF, -1, CF, HF, WF) 
                shallow_img_emb_feats.append(shallow_img_emb) # shallow_img_emb_feats: [B, C, H, W]

        # no matter the feature is from backboone or from fpn, we use shallow_img_embs all the time
        if shallow_img_emb_feats is not None and shallow_text_emb is not None:
            # shallow_img_embs = torch.cat(shallow_img_embs, dim=1)
            proj_tokens = shallow_text_emb
        return logits, bbox_reg, centerness, t_logits, proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features


class VLDyHeadModule(torch.nn.Module):

    def __init__(self, cfg):
        super(VLDyHeadModule, self).__init__()
        self.cfg = cfg
        self.head = VLDyHead(cfg)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_atss_loss_evaluator(cfg, box_coder) # ATSS损失计算
        self.box_selector_train = make_atss_postprocessor(cfg, box_coder, is_train=True)
        self.box_selector_test = make_atss_postprocessor(cfg, box_coder, is_train=False)
        self.anchor_generator = make_anchor_generator_complex(cfg)

        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.joint_embedding_size = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        self.joint_embedding_dropout = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT
        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024

        # 对比对齐损失的特征调整器
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            self.resizer = FeatureResizer(
                input_feat_size=self.lang_dim,
                output_feat_size=self.joint_embedding_size,
                dropout=self.joint_embedding_dropout
            )
        
        # 额外的线性变换层
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
            self.tunable_linear = torch.nn.Linear(self.lang_dim, 1000, bias=False)
            self.tunable_linear.weight.data.fill_(0.0)

    def forward(self, images, features, targets=None,
                language_dict_features=None,
                positive_map=None,
                captions=None,
                swint_feature_c4=None,
                background_text_features=None  # 添加背景文本特征参数
                ):

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS: # 语言特征处理
            # resizer needed
            embedding = language_dict_features['embedded']
            embedding = self.resizer(embedding) # 调整特征维度
        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # no resizer needed
            embedding = language_dict_features['embedded']
        else:
            embedding = None

        if "masks" in language_dict_features:
            text_masks = language_dict_features["masks"]
        else:
            text_masks = None
        
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
            # print("哈哈3")
            embedding = self.tunable_linear.weight[:embedding.size(1), :].unsqueeze(0) + embedding
            language_dict_features['embedded'] = embedding
            language_dict_features['hidden'] = self.tunable_linear.weight[:embedding.size(1), :].unsqueeze(0) + language_dict_features['hidden']


            # 也对背景文本进行操作
            if self.cfg.MODEL.USE_BACKGROUND:

                # 背景文本特征处理 - 从原始结构中获取各层次的特征
                bg_embedded = background_text_features['embedded']  # [batch_size, seq_len, hidden_dim]
                bg_embedded = self.tunable_linear.weight[:bg_embedded.size(1), :].unsqueeze(0) + bg_embedded

                bg_mask = background_text_features['masks']
                adjusted_bg_embedded = bg_embedded * bg_mask.unsqueeze(-1).float()
                adjusted_bg_aggregate = adjusted_bg_embedded.sum(1) / (bg_mask.sum(-1).unsqueeze(-1).float())

                # print("哈哈")
                # print(adjusted_bg_aggregate.shape)
                # sys.exit()

                background_text_features = adjusted_bg_aggregate
        

        # print("哈哈4")
        # sys.exit()


        box_cls, box_regression, centerness, token_logits, \
        proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features = self.head(features,
                                                                        language_dict_features,
                                                                        embedding,
                                                                        swint_feature_c4,
                                                                        targets
                                                                        )
        anchors = self.anchor_generator(images, features) # 这是初始框，最终预测的是偏移量

        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors,
                                       captions,
                                       positive_map,
                                       token_logits,
                                       proj_tokens,
                                       contrastive_logits,
                                       dot_product_logits,
                                       text_masks,
                                       mlm_logits = mlm_logits,
                                       mlm_labels = language_dict_features["mlm_labels"],
                                       shallow_img_emb_feats=shallow_img_emb_feats,
                                       fused_visual_features=fused_visual_features,
                                       background_text_features=background_text_features  # 传入背景文本特征
                                       )
        else:
            return self._forward_test(box_regression, centerness, anchors,
                                      box_cls,
                                      token_logits,
                                      dot_product_logits,
                                      positive_map,
                                      fused_visual_features=fused_visual_features
                                      )
    
    def compute_background_contrastive_loss(self, background_visual_features, background_text_features, text_masks=None):
        """
        计算背景特征与背景文本特征之间的对比损失
        """
        # print(background_visual_features.shape) # torch.Size([2, 256])
        # print(background_text_features.shape) # torch.Size([44, 768])
        # print("哈哈")
        # sys.exit()
        # print("哈哈2")
        
        # 1. 特征归一化
        background_visual_features = F.normalize(background_visual_features, p=2, dim=-1)
        background_text_features = F.normalize(background_text_features, p=2, dim=-1)
        
        # 2. 文本特征投影
        dot_product_proj_tokens = self.head.dot_product_projection_text(background_text_features / 2.0)
        dot_product_proj_queries = self.head.dot_product_projection_image(background_visual_features)
        
        # 3. 计算文本偏置
        # dot_product_proj_tokens_bias = torch.matmul(background_text_features, self.head.bias_lang) + self.head.bias0
       
        # 再次归一化投影后的特征 - 确保相似度在[-1,1]范围内
        dot_product_proj_queries = F.normalize(dot_product_proj_queries, p=2, dim=-1)
        dot_product_proj_tokens = F.normalize(dot_product_proj_tokens, p=2, dim=-1)

        # [2, proj_dim] × [44, proj_dim]^T -> [2, 44]
        similarity = torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.t())
       
       
        # if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
        #     similarity = torch.clamp(similarity, max=50000)
        #     similarity = torch.clamp(similarity, min=-50000)
        
        # background_loss = 1 - similarity.mean()

        attention_weights = F.softmax(similarity * 10.0, dim=1)  # 温度系数可调整

        # similarity * attention_weights:[batch_size, num_text_features]
        # 计算加权平均的相似度
        weighted_similarity = (similarity * attention_weights).sum(dim=1).mean() 
        beta = 1.0
        background_loss = 1 - beta * weighted_similarity


        return background_loss


    def _forward_train(self, box_cls, box_regression, centerness, targets, anchors,
                       captions=None,
                       positive_map=None,
                       token_logits=None,
                       proj_tokens=None,
                       contrastive_logits=None,
                       dot_product_logits=None,
                       text_masks=None,
                       mlm_logits=None,
                       mlm_labels=None,
                       shallow_img_emb_feats=None,
                       fused_visual_features=None,
                       background_text_features=None  # 添加背景文本特征参数
                       ):

        # 在VLDyHeadModule的_forward_train方法中
        if self.cfg.MODEL.USE_BACKGROUND:

            loss_box_cls, loss_box_reg, loss_centerness, loss_token, loss_contrastive_align, loss_dot_product_token, loss_shallow_contrastive, background_visual_features, token_labels_stacked = self.loss_evaluator(
                box_cls, box_regression, centerness, targets, anchors,
                captions,
                positive_map,
                token_logits,
                proj_tokens,
                contrastive_logits,
                dot_product_logits,
                text_masks,
                shallow_img_emb_feats,
                background_text_features,
                fused_visual_features # 使用fused_visual_features替代shallow_img_emb_feats
            )
            # print(background_text_features)

            # background_contrastive_logits = self.compute_background_contrastive_logits(
            #     background_visual_features, 
            #     background_text_features
            # )

            # print()
            
            # # 计算背景对比损失
            # background_contrastive_loss = torch.tensor(0.0, device=loss_centerness.device)

            # print("哈哈1")

            background_contrastive_loss = self.compute_background_contrastive_loss(
                background_visual_features,
                background_text_features,
            )

        else:
            loss_box_cls, loss_box_reg, loss_centerness, loss_token, loss_contrastive_align, loss_dot_product_token, loss_shallow_contrastive = self.loss_evaluator(
                box_cls, box_regression, centerness, targets, anchors,
                captions,
                positive_map,
                token_logits,
                proj_tokens,
                contrastive_logits,
                dot_product_logits,
                text_masks,
                shallow_img_emb_feats
            )

        losses = {
            # "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }

        if mlm_labels is not None and mlm_logits is not None:
            losses["mlm_loss"] = nn.CrossEntropyLoss(ignore_index = -100)(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)) * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_COEF

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS:
            losses["loss_cls"] = loss_box_cls
        else:
            losses["loss_cls"] = 0.0 * loss_box_cls

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            losses["loss_token"] = loss_token * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            losses["loss_contrastive_align"] = loss_contrastive_align * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_ALIGN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            losses["loss_dot_product_token"] = loss_dot_product_token * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DOT_PRODUCT_TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS or \
                self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            losses["loss_shallow_contrastive"] = loss_shallow_contrastive * \
                                                 self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_LOSS_WEIGHT
        
        # 只在USE_BACKGROUND为True时添加背景对齐损失
        if self.cfg.MODEL.USE_BACKGROUND:
            losses["loss_background_align"] = background_contrastive_loss * self.cfg.MODEL.BACKGROUND_ALIGN_LOSS_WEIGHT

            # print(loss_background_align)
            # print("哈哈")
            # sys.exit()

        if self.cfg.MODEL.RPN_ONLY:
            return None, losses, None
        else:
            # Let's just use one image per batch
            assert (box_regression[0].shape[0]) == 1
            positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=1)
            boxes = self.box_selector_train(box_regression, centerness, anchors,
                                        box_cls,
                                        token_logits,
                                        dot_product_logits,
                                        positive_map=positive_map_label_to_token
                                        )
            train_boxes = [] # 通过atss进行初筛
            for b, t in zip(boxes, targets):
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                train_boxes.append(cat_boxlist([b, tb]))
            return train_boxes, losses, fused_visual_features

    def _forward_test(self, box_regression, centerness, anchors,
                      box_cls=None,
                      token_logits=None,
                      dot_product_logits=None,
                      positive_map=None,
                      fused_visual_features=None
                      ):

        boxes = self.box_selector_test(box_regression, centerness, anchors,
                                       box_cls,
                                       token_logits,
                                       dot_product_logits,
                                       positive_map,
                                       )
        return boxes, {}, fused_visual_features
