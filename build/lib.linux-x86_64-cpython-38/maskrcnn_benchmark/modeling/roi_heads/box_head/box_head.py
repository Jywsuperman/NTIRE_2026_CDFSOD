# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.utils.amp import custom_fwd, custom_bwd

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg) # 特征提取器
        self.predictor = make_roi_box_predictor(cfg) # 预测器
        self.post_processor = make_roi_box_post_processor(cfg) # 后处理器
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg) # 损失计算器
        self.onnx = cfg.MODEL.ONNX

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets) # 1. 采样proposals

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals) # 2. 特征提取
        # 从RPN获得的特征可能是：1. 融合特征 (fused_visual_features) 2. 原始视觉特征 (visual_features)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x) # 3. 预测

        if self.onnx:
            return x, (class_logits, box_regression, [box.bbox for box in proposals]), {}

        if not self.training: # 测试时的处理:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator( # 4. 损失计算
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
