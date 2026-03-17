# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import itertools

from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR, WarmupReduceLROnPlateau
import sys

def make_optimizer(cfg, model):

    # 设置随机种子
    torch.manual_seed(cfg.DATASETS.SHUFFLE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.DATASETS.SHUFFLE_SEED)
        torch.cuda.manual_seed_all(cfg.DATASETS.SHUFFLE_SEED)
    
    # # 确保CUDA操作是确定的
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    params = []
    # adapter_params_count = 0  # 统计adapter参数数量
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR # 0.0001
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        # print("哈哈10")

        # different lr schedule
        if "language_backbone" in key:
            lr = cfg.SOLVER.LANG_LR # 0.00001
            # print("哈哈11")

        if "backbone.body" in key and "language_backbone.body" not in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_BODY_LR_FACTOR # 1.0

        if "bias" in key:
            # 通常偏置参数的权重衰减设置得更小，因为偏置参数对过拟合的贡献较小
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            # print("哈哈13")

        if 'norm' in key or 'Norm' in key:
            # 对于BatchNorm等归一化层，通常设置较小的权重衰减，甚至设为0，因为:
            weight_decay *= cfg.SOLVER.WEIGHT_DECAY_NORM_FACTOR
            # print("Setting weight decay of {} to {}".format(key, weight_decay))

            # print("哈哈14")
        
        # 为 ShapeAdaptiveAdapter 设置特殊学习率
        if "shape_adapter" in key: # BASE_LR:0.0001
            lr = 1e-6
            # adapter_params_count += value.numel()  # 统计参数数量
            # print(f"Adapter parameter: {key}, shape: {value.shape}, learning rate: {lr}")
        if "tunable_linear" in key: # BASE_LR:0.0001
            lr = 1e-5
            # adapter_params_count += value.numel()  # 统计参数数量
            print(f"tunable_linear parameter: {key}, shape: {value.shape}, learning rate: {lr}")

        # 不同的参数设置不同的学习率
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER == "ADAMW":

        # print("哈哈16")
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, lr)


    # # 打印adapter参数统计信息
    # print(f"\nShape Adapter Statistics:")
    # print(f"Total number of adapter parameters: {adapter_params_count}") # Total number of adapter parameters: 327680
    # print(f"Base learning rate: {cfg.SOLVER.BASE_LR}") # Base learning rate: 0.0001
    # print(f"Adapter learning rate: {cfg.SOLVER.BASE_LR * 0.1}") # Adapter learning rate: 1e-05

    # sys.exit()

    return optimizer


def make_lr_scheduler(cfg, optimizer):

    # 设置随机种子
    torch.manual_seed(cfg.DATASETS.SHUFFLE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.DATASETS.SHUFFLE_SEED)
        torch.cuda.manual_seed_all(cfg.DATASETS.SHUFFLE_SEED)
    
    # # 确保CUDA操作是确定的
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    
    if cfg.SOLVER.MULTI_MAX_EPOCH:
        assert len(cfg.SOLVER.MULTI_MAX_EPOCH) == len(cfg.SOLVER.STEPS)
        lr_scheduler = []

        for stage_step, stage_max_epoch in zip(cfg.SOLVER.STEPS, cfg.SOLVER.MULTI_MAX_ITER):
            milestones = []
            for step in stage_step:
                milestones.append(round(step * stage_max_epoch))
            lr_scheduler.append(WarmupMultiStepLR(optimizer,
                                                  milestones,
                                                  cfg.SOLVER.GAMMA,
                                                  warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                                                  warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                                                  warmup_method=cfg.SOLVER.WARMUP_METHOD, )
                                )
        return lr_scheduler

    elif cfg.SOLVER.USE_COSINE:
        max_iters = cfg.SOLVER.MAX_ITER
        return WarmupCosineAnnealingLR(
            optimizer,
            max_iters,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            eta_min=cfg.SOLVER.MIN_LR
        )

    elif cfg.SOLVER.USE_AUTOSTEP:
        max_iters = cfg.SOLVER.MAX_ITER
        return WarmupReduceLROnPlateau(
            optimizer,
            max_iters,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            eta_min=cfg.SOLVER.MIN_LR,
            patience=cfg.SOLVER.STEP_PATIENCE,
            verbose=False # 这里设置为不输出，方便查看信息
        )

    else:
        milestones = []
        for step in cfg.SOLVER.STEPS:
            if step < 1:
                milestones.append(round(step * cfg.SOLVER.MAX_ITER))
            else:
                milestones.append(step)
        return WarmupMultiStepLR(
            optimizer,
            milestones,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
