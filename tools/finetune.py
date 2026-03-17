# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import random
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import sys

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob

import pdb
import torch
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.alter_trainer import do_train as alternative_train
from maskrcnn_benchmark.engine.stage_trainer import do_train as multi_stage_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import all_gather, synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
import shutil

import torch.distributed as dist

from collections import defaultdict

import numpy as np

def removekey(d, prefix):
    r = dict(d)
    listofkeys = []
    for key in r.keys():
        if key.startswith(prefix):
            listofkeys.append(key)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r

def print_seeds(cfg):
    import random
    import numpy as np
    """
    打印配置中的随机种子信息
    Args:
        cfg: 配置对象
    """
    print("\n" + "="*50)
    print("Random Seeds Configuration:")
    print("-"*50)
    if hasattr(cfg, 'DATASETS'):
        print(f"Dataset Shuffle Seed: {cfg.DATASETS.SHUFFLE_SEED}")
    if hasattr(cfg, 'SOLVER'):
        print(f"Solver Seed: {cfg.SOLVER.SEED}")
    print("Current Random States:")
    print(f"Python random seed: {random.getstate()[1][0]}")
    print(f"Numpy random seed: {np.random.get_state()[1][0]}")
    print(f"Torch CPU random seed: {torch.initial_seed()}")
    if torch.cuda.is_available():
        print(f"Torch GPU random seed: {torch.cuda.initial_seed()}")
    print("="*50 + "\n")

def train(cfg, local_rank, distributed, zero_shot, skip_optimizer_resume=False, save_config_path = None, current_exp_performance=None, current_exp_iterations=None, dataset_name=None, output_txt_name=None):

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0 #<TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )
    if cfg.TEST.DURING_TRAINING:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None
    
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)


    # 这些全部不冻结
    if cfg.MODEL.LINEAR_PROB:
        # print("哈哈1")
        assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
        if hasattr(model.backbone, 'fpn'):
            assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
    if cfg.MODEL.BACKBONE.FREEZE: # 这个
        # print("swin backbone被冻结")
        for p in model.backbone.body.parameters():
            p.requires_grad = False
    if cfg.MODEL.FPN.FREEZE: # 这个
        # print("FPN被冻结")
        for p in model.backbone.fpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.RPN.FREEZE: # 这个
        # print("RPN被冻结")
        for p in model.rpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.LINEAR_PROB:
        # print("哈哈5")
        if model.rpn is not None:
            # print("哈哈6")
            for key, p in model.rpn.named_parameters():
                if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                    p.requires_grad = False
        if model.roi_heads is not None:
            # print("哈哈7")
            for key, p in model.roi_heads.named_parameters():
                if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                    p.requires_grad = False
    if cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER: # 这个
        # print("使用了线性层")
        if model.rpn is not None:
            # print("哈哈9")
            for key, p in model.rpn.named_parameters():
                if 'tunable_linear' in key:
                    p.requires_grad = True
    # if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
    #     print("language backbone被冻结")

    # sys.exit()
    
    optimizer = make_optimizer(cfg, model)

    # print("嘻嘻")
    # sys.exit()

    scheduler = make_lr_scheduler(cfg, optimizer)

    # 设置
    set_global_deterministic(42)
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
            broadcast_buffers=True,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS,
            process_group=None  # 使用默认进程组
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(skip_optimizer=skip_optimizer_resume)
        arguments.update(extra_checkpoint_data)
    else:
        state_dict = checkpointer._load_file(try_to_find(cfg.MODEL.WEIGHT))
        checkpointer._load_model(state_dict)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    meters = MetricLogger(delimiter="  ")

    if zero_shot:
        return model
    
    if cfg.DATASETS.ALTERNATIVE_TRAINING:
        alternative_train(
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
        )
    elif cfg.DATASETS.MULTISTAGE_TRAINING:
        arguments['epoch_per_stage'] = cfg.SOLVER.MULTI_MAX_EPOCH
        multi_stage_train(
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
        )
    else:

        # print("哈哈哈1")
        # sys.exit()
        meters = MetricLogger(delimiter="  ")
        do_train(
            cfg,
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            data_loaders_val,
            meters=meters,
            current_exp_performance=current_exp_performance,
            current_exp_iterations=current_exp_iterations,
            dataset_name=dataset_name,
            output_txt_name=output_txt_name
        )

    return model

def test(cfg, model, distributed, verbose=False):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    log_dir = cfg.OUTPUT_DIR
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST
    if isinstance(dataset_names[0], (list, tuple)):
        dataset_names = [dataset for group in dataset_names for dataset in group]
    output_folders = [None] * len(dataset_names)
    if log_dir:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(log_dir, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    
    # 用于存储AP值
    ap_value = None

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        results = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY and cfg.MODEL.RPN_ARCHITECTURE=="RPN",
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            cfg=cfg
        )

        # 提取AP值
        if cfg.MODEL.MultiHH and is_main_process() and results:
            try:
                ap_value = results[0].results['bbox']['AP']
            except:
                pass

        synchronize()
    if verbose:
        with open(os.path.join(output_folder, "bbox.csv")) as f:
            print(f.read())

    
     # 如果需要返回AP值
    if cfg.MODEL.MultiHH:
        # 使用all_gather确保所有进程都能获得AP值
        if distributed:
            ap_values = all_gather([ap_value])
            if is_main_process() and ap_values:
                ap_value = ap_values[0]  # 只取第一个进程的值
        return ap_value
        
    return None

def tuning_highlevel_override(cfg,):
    if cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "full":
        cfg.MODEL.BACKBONE.FREEZE = False
        cfg.MODEL.FPN.FREEZE = False
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "linear_prob":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = True
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True
        cfg.MODEL.DYHEAD.USE_CHECKPOINT = False # Disable checkpoint
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v1":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v2":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True # 对进行特征融合前，语言特征通过线性变换层
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v3":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = True # Turn on linear probe
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False # Turn on language backbone
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "language_prompt_v4":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True
        cfg.MODEL.LINEAR_PROB = True # Turn on linear probe
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True # Turn off language backbone
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "Go_rpn":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = False # 没有对rpn特定参数以外的进行冻结
        # cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False # 没有对语言特征额外的线性层处理
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True # 对语言特征额外的线性层处理
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True # Turn off language backbone
    
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "Go_rpn_swin":
        # cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.BACKBONE.FREEZE = False
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = False # 没有对rpn特定参数以外的进行冻结
        # cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False # 没有对语言特征额外的线性层处理
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True # 对语言特征额外的线性层处理
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True # Turn off language backbone
    
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "Go_rpn_new":
        # cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = False # 没有对rpn特定参数以外的进行冻结
        # cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False # 没有对语言特征额外的线性层处理
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True # 对语言特征额外的线性层处理
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False # Turn off language backbone
    
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "Go_rpn_lang":
        # cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.BACKBONE.FREEZE = False
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = False # 没有对rpn特定参数以外的进行冻结
        # cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False # 没有对语言特征额外的线性层处理
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False # 对语言特征额外的线性层处理
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True # Turn off language backbone
    
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "train_all":
        cfg.MODEL.BACKBONE.FREEZE = False
        cfg.MODEL.FPN.FREEZE = False
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = False # 没有对rpn特定参数以外的进行冻结
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False # 没有对语言特征额外的线性层处理
        # cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True # 对语言特征额外的线性层处理
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False # Turn off language backbone
    
    elif cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "train_rpn":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = False
        cfg.MODEL.LINEAR_PROB = False # 没有对rpn特定参数以外的进行冻结
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True # 没有对语言特征额外的线性层处理
        # cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = True # 对语言特征额外的线性层处理
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True # Turn off language backbone


    return cfg

def report_freeze_options(cfg):
    print("Backbone Freeze:", cfg.MODEL.BACKBONE.FREEZE)
    print("FPN Freeze:", cfg.MODEL.FPN.FREEZE)
    print("RPN Freeze:", cfg.MODEL.RPN.FREEZE)
    print("Linear Probe:", cfg.MODEL.LINEAR_PROB)
    print("Language Freeze:", cfg.MODEL.LANGUAGE_BACKBONE.FREEZE)
    print("Linear Layer (True Prmopt Tuning):", cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER)
    print("High Level Override:", cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--ft-tasks",
        default="",
        metavar="FILE",
        help="path to fine-tune configs",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-train",
        dest="skip_train",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--skip_optimizer_resume", action="store_true")

    parser.add_argument("--custom_shot_and_epoch_and_general_copy", default=None, type=str)

    parser.add_argument("--shuffle_seeds", default=None, type=str)

    parser.add_argument("--evaluate_only_best_on_test", action="store_true") # just a dummpy parameter; only used in eval_all.py, add it here so it does not complain...
    parser.add_argument("--push_both_val_and_test", action="store_true") # just a dummpy parameter; only used in eval_all.py, add it here so it does not complain...

    parser.add_argument('--use_prepared_data', action='store_true')


    parser.add_argument("--keep_testing", action="store_true")

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--num_experiments", default=2, type=int)

    parser.add_argument("--output_txt_name", default="B1",type=str)

    # test得到的json路径
    parser.add_argument("--output_json_save_path", default="B1",type=str)

    args = parser.parse_args()


    #####################################
    # set_random_seed(int(args.seed))
    set_global_deterministic(int(args.seed))

    #####################################


    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    # print("哈哈0")
    # sys.exit()

    if args.distributed:
        # print("哈哈1")
        # sys.exit()
        torch.cuda.set_device(args.local_rank)
        # print("哈哈2")
        # sys.exit()
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    # dist.barrier()
    # print("哈哈3")
    # sys.exit()

    # 设置分布式训练环境
    # cfg.local_rank = setup_distributed_training(seed=42)

    cfg.local_rank = args.local_rank

    cfg.num_gpus = num_gpus
    cfg.output_json_save_path=args.output_json_save_path

    cfg.merge_from_file(args.config_file) # （1）加载基础配置文件 (--config-file)
    # print(cfg)
    # print("args.opts", args.opts)
    cfg.merge_from_list(args.opts) # （2）加载命令行参数 (args.opts)


    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    #logger.info("Collecting env info (might take some time)")
    #logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config_prompts.yml')
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    ft_configs = []
    if args.ft_tasks:
        for ft_file in args.ft_tasks.split(","):
            for file in sorted(glob.glob(ft_file)):
                ft_configs.append(file)
    else:
        ft_configs = [args.config_file]

    shuffle_seeds = []
    if args.shuffle_seeds:
        shuffle_seeds = [int(seed) for seed in args.shuffle_seeds.split(',')]
    else:
        shuffle_seeds = [None]

    # print(args.num_experiments)
    # sys.exit()
    
    # 在定义shuffle_seeds后添加
    num_experiments = args.num_experiments

    # Initialize lists to track experiment results
    all_ap_values = []
    if is_main_process():
        all_exp_history = []  # List of (performance_list, iteration_list) tuples
        all_exp_max_performance = []  # List of (max_performance, max_iteration) tuples
        all_exp_final_performance = []  # List of (final_performance, final_iteration) tuples

    
    for experiment_id in range(num_experiments):
        if is_main_process():
            print(f"\n======== 开始第 {experiment_id+1}/{num_experiments} 次实验 ========\n")
            set_global_deterministic(int(args.seed))
        
        # Create a list to store this experiment's performance history
        if is_main_process():
            current_exp_performance = []
            current_exp_iterations = []

    # print("哈哈1")
    # print(shuffle_seeds)
    # sys.exit()
    # print("哈哈1")
    # sys.exit()
        model = None
        for task_id, ft_cfg in enumerate(ft_configs, 1):
            # print("哈哈1")
            # sys.exit()
            for shuffle_seed in shuffle_seeds:

                actual_seed = shuffle_seed

                cfg_ = cfg.clone()
                cfg_.defrost()
                cfg_.merge_from_file(ft_cfg) # （3）加载任务配置文件 (--task_config)
                cfg_.merge_from_list(args.opts) # （4）再次加载命令行参数，会被覆盖


                # 从配置文件名称中提取数据集名称
                dataset_name = os.path.basename(ft_cfg).split('.')[0]  # 例如从 'FISH.yaml' 提取 'FISH'
                
                # 从 custom_shot_and_epoch_and_general_copy 中提取 shot 数
                shot_num = args.custom_shot_and_epoch_and_general_copy.split('_')[0]  # 从 "10_200_1" 提取 "10"
                
                ###########################
                dataset_shot_name = os.path.normpath(args.output_json_save_path).split(os.sep)[-2:]
                ###########################

                from datetime import datetime
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 构建新的输出路径
                # ft_output_dir = os.path.join(output_dir, "finetuing", f'{dataset_name}_{shot_num}shot', os.path.splitext(os.path.basename(cfg_.MODEL.WEIGHT))[0], f'_exp_{experiment_id}')
                ft_output_dir = os.path.join(
                    output_dir,
                    "finetuing_GLIP_Large",
                    # f"{dataset_name}_{shot_num}shot",
                    *dataset_shot_name,
                    # os.path.splitext(os.path.basename(cfg_.MODEL.WEIGHT))[0],
                    f"exp{experiment_id}_time_{time_str}"
                )


                if args.custom_shot_and_epoch_and_general_copy:
                    custom_shot = int(args.custom_shot_and_epoch_and_general_copy.split("_")[0])
                    custom_epoch = int(args.custom_shot_and_epoch_and_general_copy.split("_")[1])
                    custom_copy = int(args.custom_shot_and_epoch_and_general_copy.split("_")[2])
                    cfg_.SOLVER.MAX_EPOCH = custom_epoch
                    cfg_.DATASETS.GENERAL_COPY = custom_copy
                    if args.use_prepared_data:
                        if custom_shot != 0: # 0 means full data training
                            cfg_.DATASETS.TRAIN = ("{}_{}_{}".format(cfg_.DATASETS.TRAIN[0], custom_shot, cfg_.DATASETS.SHUFFLE_SEED), )
                            try:
                                custom_shot_val = int(args.custom_shot_and_epoch_and_general_copy.split("_")[3])
                            except:
                                custom_shot_val = custom_shot
                            cfg_.DATASETS.TEST = ("{}_{}_{}".format(cfg_.DATASETS.TEST[0], custom_shot_val, cfg_.DATASETS.SHUFFLE_SEED), )
                            if custom_shot_val == 1 or custom_shot_val == 3:
                                cfg_.DATASETS.GENERAL_COPY_TEST = 4 # to avoid less images than GPUs
                    else:
                        cfg_.DATASETS.FEW_SHOT = custom_shot
                else:
                    custom_shot = None
                    custom_epoch = None

                # if shuffle_seed is not None:
                #     cfg_.DATASETS.SHUFFLE_SEED = shuffle_seed
                #     ft_output_dir = ft_output_dir + '_seed_{}'.format(shuffle_seed)
                # if actual_seed is not None:
                #     cfg_.DATASETS.SHUFFLE_SEED = actual_seed
                #     ft_output_dir = ft_output_dir + f'_seed_{actual_seed}_exp_{experiment_id}'

                if actual_seed is not None:
                    # cfg_.DATASETS.SHUFFLE_SEED = actual_seed
                    # time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # ft_output_dir = ft_output_dir + f'_seed_{actual_seed}_exp_{experiment_id}_{time_str}'

                    cfg_.DATASETS.SHUFFLE_SEED = actual_seed
                    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    ft_output_dir = ft_output_dir + f'_seed_{actual_seed}_exp_{experiment_id}_time_{time_str}'
                    os.makedirs(ft_output_dir, exist_ok=True)
                    

                # Remerge to make sure that the command line arguments are prioritized
                cfg_.merge_from_list(args.opts)
                if "last_checkpoint" in cfg_.MODEL.WEIGHT:
                    with open(cfg_.MODEL.WEIGHT.replace("model_last_checkpoint.pth", "last_checkpoint"), "r") as f:
                        last_checkpoint = f.read()
                    cfg_.MODEL.WEIGHT = cfg_.MODEL.WEIGHT.replace("model_last_checkpoint.pth", last_checkpoint)
                    print("cfg.MODEL.WEIGHT ", cfg_.MODEL.WEIGHT)
                
                # print("哈哈1")
                # sys.exit()

                mkdir(ft_output_dir)
                cfg_.OUTPUT_DIR = ft_output_dir
                
                tuning_highlevel_override(cfg_)
                cfg_.freeze()

                # logger.info("Loaded fine-tune configuration file {}".format(ft_cfg))
                with open(ft_cfg, "r") as cf:
                    config_str = "\n" + cf.read()
                    # logger.info(config_str)

                output_config_path = os.path.join(ft_output_dir, 'config.yml')
                # print("Saving config into: {}".format(output_config_path))
                # save config here because the data loader will make some changes
                save_config(cfg_, output_config_path)
                # logger.info("Training {}".format(ft_cfg))
                if custom_shot == 10000:
                    if is_main_process():
                        print("Copying pre-training checkpoint")
                        shutil.copy(try_to_find(cfg_.MODEL.WEIGHT), os.path.join(ft_output_dir, "model_best.pth"))
                else:
                    model = train(
                        cfg_, 
                        args.local_rank, 
                        args.distributed, 
                        args.skip_train or custom_shot == 10000, 
                        skip_optimizer_resume=args.skip_optimizer_resume,
                        save_config_path=output_config_path,
                        current_exp_performance=current_exp_performance if is_main_process() else None,
                        current_exp_iterations=current_exp_iterations if is_main_process() else None,
                        dataset_name=dataset_name,
                        output_txt_name=args.output_txt_name
                        )
                    
                    # if not args.skip_test:
                    #     ap_value = test(cfg_, model, args.distributed)

                    #     # 只在主进程中收集AP值
                    #     if ap_value is not None:
                    #         # 确保ap_value是一个数字而不是列表或其他类型
                    #         try:
                    #             # 如果ap_value是列表，尝试获取第一个元素
                    #             if isinstance(ap_value, list):
                    #                 if ap_value and len(ap_value) > 0:
                    #                     ap_value = ap_value[0]
                                
                    #             # 如果ap_value是字典或类似结构，尝试提取AP值
                    #             if hasattr(ap_value, 'results'):
                    #                 ap_value = ap_value.results['bbox']['AP']
                    #             elif isinstance(ap_value, dict) and 'results' in ap_value:
                    #                 ap_value = ap_value['results']['bbox']['AP']
                                    
                    #             # 确保最终是一个数字
                    #             ap_value = float(ap_value)
                                
                    #             # 现在可以安全地格式化和添加到结果列表
                    #             all_ap_values.append(ap_value)
                    #             if is_main_process():
                    #                 print(f"第{experiment_id+1}次实验 AP = {ap_value:.4f}")
                    #         except Exception as e:
                    #             if is_main_process():
                    #                 print(f"处理AP值时出错: {e}")
                    #                 print(f"AP值类型: {type(ap_value)}, 值: {ap_value}")
                    
                    if args.keep_testing:
                        # for manual testing
                        cfg_.defrost()
                        cfg_.DATASETS.TEST = ("test", )
                        test(cfg_, model, args.distributed, verbose=True)
                        print(cfg_.DATASETS.OVERRIDE_CATEGORY)
                        pdb.set_trace()
                        # test(cfg_, model, args.distributed, verbose=True)
                        continue
        # 
        # 强制清理内存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # 等待一段时间确保内存释放
        import time
        time.sleep(5)
        
        # Record initial memory state
        if is_main_process():
            print(f"Memory usage before experiment {experiment_id+1}:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
            
            # Store the experiment results
            if current_exp_performance:
                # Find max performance and its iteration
                max_performance = max(current_exp_performance)
                max_iter_idx = current_exp_performance.index(max_performance)
                max_iter = current_exp_iterations[max_iter_idx]
                
                # Get final performance and iteration
                final_performance = current_exp_performance[-1]
                final_iter = current_exp_iterations[-1]
                
                # Store in experiment tracking lists
                all_exp_max_performance.append((max_performance, max_iter))
                all_exp_final_performance.append((final_performance, final_iter))
                all_exp_history.append((current_exp_performance, current_exp_iterations))

                print("\n===== Performance History =====")
                for i, (iter_num, perf) in enumerate(zip(current_exp_iterations, current_exp_performance)):
                    print(f"Iteration {iter_num}: {perf:.5f}")
                print("=============================\n")


    # After all experiments, print summary results
    if is_main_process():
        print("\n" + "="*50)
        print("\n" + "="*50)
        
        # Print detailed history for each experiment
        for exp_idx, (performance_history, iteration_history) in enumerate(all_exp_history):
            print(f"\n===== Experiment {exp_idx+1} Performance History =====")
            for perf, iter_num in zip(performance_history, iteration_history):
                print(f"Iteration {iter_num}: {perf:.4f}")
                
            max_perf, max_iter = all_exp_max_performance[exp_idx]
            final_perf, final_iter = all_exp_final_performance[exp_idx]
            print(f"Max: {max_perf:.4f} (iter {max_iter})")
            print(f"Final: {final_perf:.4f} (iter {final_iter})")
            print("=============================")
        
        # Print overall summary
        print("\n===== Overall Results =====")
        for exp_idx, ((max_perf, max_iter), (final_perf, final_iter)) in enumerate(zip(all_exp_max_performance, all_exp_final_performance)):
            print(f"Exp {exp_idx+1} - Max: {max_perf:.4f} (iter {max_iter}), Final: {final_perf:.4f} (iter {final_iter})")
        
        # Calculate and print averages
        max_perfs = [x[0] for x in all_exp_max_performance]
        final_perfs = [x[0] for x in all_exp_final_performance]
        
        mean_max = np.mean(max_perfs)
        std_max = np.std(max_perfs)
        
        mean_final = np.mean(final_perfs)
        std_final = np.std(final_perfs)
        
        print(f"Avg Max: {mean_max:.4f} ± {std_max:.4f}")
        print(f"Avg Final: {mean_final:.4f} ± {std_final:.4f}")
        print(f"{mean_max:.4f} {mean_final:.4f}")
        print("=============================")

def set_all_seeds(seed):
    """
    设置所有随机种子
    """
    import random
    import numpy as np
    import torch
    import os
    
    # 设置 Python、Numpy 种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 设置 PyTorch 种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置 cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置 Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_gpu_info():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print(f"Current device: {torch.cuda.current_device()}")

def set_global_deterministic(seed):
    """
    一次性设置所有的随机种子和确定性选项
    Args:
        seed (int): 随机种子值
    """
    import random
    import numpy as np
    import torch
    import os
    
    # 确保seed是有效的整数
    seed = int(seed)
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # 设置Python随机种子（使用更安全的方式）
    random.seed(seed)
    # random.seed(random.randint(0, 2**32 - 1))  # 重置一次以确保状态正确
    
    # 设置Numpy随机种子
    np.random.seed(seed)
    
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    
    # 设置CUDA相关选项
    if torch.cuda.is_available():
        # 设置CUDA随机种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU设置
        
        # 设置CUDA确定性选项
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # torch.use_deterministic_algorithms(True)  # 强制使用确定性算法
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 更严格的CUBLAS配置
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
       

if __name__ == "__main__":

    # print_gpu_info()
    # sys.exit()

    # seed = 42

    # set_global_deterministic(seed)
    
    # # 验证随机性
    # print_python_seed()
    
    # sys.exit()

    main()
