# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import sys
import os
import math
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, broadcast_data, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.ema import ModelEma
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from .inference import inference
import pdb

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_data_loader=None,
        meters=None,
        zero_shot=False,
        current_exp_performance=None,
        current_exp_iterations=None,
        dataset_name=None,
        output_txt_name=None,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)

    # print(max_iter) 3000
    # print("哈哈")
    # sys.exit()

    start_iter = arguments["iteration"]
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH

    # print(checkpoint_period) #FISH 2.0
    # print("哈哈")
    # sys.exit()

    skip_iteration = 0
    checkpoint_period = 2

    if dataset_name == "d1":
        skip_iteration = 2
        checkpoint_period = 2
    
    if dataset_name == "d2":
        skip_iteration = 7
        checkpoint_period = 2
    
    if dataset_name == "d3":
        skip_iteration = 10
        checkpoint_period = 2


    if dataset_name == "NEUDET":
        skip_iteration = 60
        # skip_iteration = 0
        checkpoint_period = 2
    elif dataset_name == "UODD":
        skip_iteration = 10
        checkpoint_period = 2
    elif dataset_name == "Clipart1k":
        skip_iteration = 29
        checkpoint_period = 2
    elif dataset_name == "FISH":
        skip_iteration = 0
        checkpoint_period = 1
    elif dataset_name == "ArTaxOr":
        skip_iteration = 66
        checkpoint_period = 2
    elif dataset_name == "DIOR":
        skip_iteration = 50
        checkpoint_period = 4


    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        patience_counter = 0
        previous_best = -2

    eval_results_history = []
    eval_iterations = []

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1
    

    # iterations_per_epoch = len(data_loader) // cfg.SOLVER.MAX_EPOCH # 15
    # print("哈哈")
    # print(iterations_per_epoch)
    # epoch_data = {}  # 用于存储每个epoch的数据顺序

    # output_txt_name

    log_dir = "/home/jyw/CDFSOD/GLIP/tools/outputs"
    os.makedirs(log_dir, exist_ok=True)
    # log_file_path = os.path.join(log_dir, f"B2_training_log.txt")
    log_file_path = os.path.join(log_dir, f"{output_txt_name}_training_log.txt")
    if is_main_process():
        with open(log_file_path, "w") as f:
            f.write(f"Training started\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write("="*50 + "\n\n")

    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):

        # print(positive_map.shape)
        # # print(positive_map)
        # print("哈哈")
        # sys.exit()
                
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        captions = None
        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        except:
            pass
        # Freeze language backbone
        if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if hasattr(model, "module"):
                model.module.language_backbone.eval()
            else:
                model.language_backbone.eval()


        # cfg.SOLVER.USE_AMP = False

        if cfg.SOLVER.USE_AMP: # 这个分支

            # print("哈哈1")
            with autocast():
                # 自动将某些操作从float32(FP32)降为float16(FP16)
                # - 大多数卷积、矩阵乘法等使用FP16
                # - 一些数值敏感的操作保持FP32
                # - 自动选择合适的精度
                if len(captions) > 0:
                    torch.cuda.synchronize()
                    loss_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map)
                else:
                    torch.cuda.synchronize()
                    loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map)
            else:
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

        # Adapt the weight decay: only support multiStepLR
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):

            # print("哈哈2")
            if milestone_target < len(scheduler.milestones):
                next_milestone = list(scheduler.milestones)[milestone_target]
            else:
                next_milestone = float('inf')
            if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                gamma = scheduler.gamma
                logger.info("Drop the weight decay by {}!".format(gamma))
                for param in optimizer.param_groups:
                    if 'weight_decay' in param:
                        param['weight_decay'] *= gamma
                # move the target forward
                milestone_target += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # if iteration % 20 == 0 or iteration == max_iter:
        if iteration % 1 == 0 or iteration == max_iter:
            #logger.info(
            if global_rank <= 0:
                print(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "wd: {wd:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        
        if iteration % 1 == 0 or iteration == max_iter:
            if global_rank <= 0:
                # Original print statement stays as is
                log_str = meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "wd: {wd:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    wd=optimizer.param_groups[0]["weight_decay"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
                # Add logging to file
                with open(log_file_path, "a") as f:
                    f.write(log_str + "\n")
        

        if iteration <= skip_iteration:
            continue

        # 直接不进行test
        if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
            # print(cfg.output_txt_path) 
            # print("哈哈")
            # sys.exit()
            # filename=str(cfg.output_txt_path) # 保存的路径
            # filename=cfg.output_json_save_path
            ####################################################
            # from datetime import datetime
            # from datetime import datetime


            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{cfg.output_json_save_path}_{time_str}_iteration_{iteration}"

            os.makedirs(filename, exist_ok=True)
            ####################################################

            # print("哈哈3")
            if is_main_process():
                print("Evaluating")
            eval_result = 0.0
            model.eval()
            if cfg.SOLVER.TEST_WITH_INFERENCE:
                with torch.no_grad():
                    try:
                        _model = model.module
                    except:
                        _model = model
                    _result = inference(
                        model = _model,
                        data_loader = val_data_loader,
                        dataset_name="val",
                        device=device,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        # output_folder=None,
                        output_folder=filename,
                        cfg=cfg,
                        verbose=False
                    )
                    if is_main_process():
                        eval_result = _result[0].results['bbox']['AP']
            else:
                # print("哈哈4")
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, *_ = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    print("哈哈5")
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                            box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']

            model.train()

            if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
                # print("哈哈6")
                model_ema.ema.eval()
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, positive_map_eval = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model_ema.ema(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model_ema.ema(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    # print("哈哈7")
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                              box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
                
            arguments.update(eval_result=eval_result)


            if is_main_process() and current_exp_performance is not None and current_exp_iterations is not None:
                current_exp_performance.append(eval_result)
                current_exp_iterations.append(iteration)

                with open(log_file_path, "a") as f:
                    f.write(f"Iteration {iteration}: Accuracy = {eval_result:.4f}\n\n")


            # 记录性能结果和对应迭代次数
            if is_main_process():
                eval_results_history.append(eval_result)
                eval_iterations.append(iteration)
                # print(f"Performance at iteration {iteration}: {eval_result:.4f}")


            if cfg.SOLVER.USE_AUTOSTEP: # 这个分支
                # print("哈哈8")
                eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)
            
            if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1: # 这个分支
                # print("哈哈9")
                if eval_result < previous_best:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    previous_best = eval_result

                    checkpointer.save("model_best", **arguments) # 保存最优权重

                print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)

                if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
                    if is_main_process():
                        print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))

                        # # 打印所有性能历史记录
                        # print("\n===== Performance History =====")
                        # for i, (iter_num, perf) in enumerate(zip(eval_iterations, eval_results_history)):
                        #     print(f"Iteration {iter_num}: {perf:.4f}")
                        # print("=============================\n")

                    break


        # if iteration % checkpoint_period == 0:
        #     checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            # checkpointer.save("model_final", **arguments)
            print("达到了最大迭代次数")
            break

        torch.cuda.synchronize()  # 确保所有CUDA操作完成

    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    if is_main_process():
        with open(log_file_path, "a") as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"Training complete\n")
            f.write(f"Total training time: {total_time_str} ({total_training_time / (max_iter):.4f} s / it)\n")


