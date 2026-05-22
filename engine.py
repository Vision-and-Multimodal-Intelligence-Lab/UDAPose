import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable
import copy
import numpy as np
import torch
import time
import util.misc as utils
from util import box_ops, keypoint_ops
from util.utils import to_device


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    wo_class_error=False,
    lr_scheduler=None,
    args=None,
    logger=None,
    ema_m=None,
):
    scaler = torch.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn  # True
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if not wo_class_error:
        metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    _cnt = 0
    for batch_idx, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, logger=logger)
    ):
        samples = samples.to(device)
        targets = [
            {k: v.to(device) if k != "img_light_level" else v for k, v in t.items()} for t in targets
        ]  # list of dicts, each dict for one batch
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            if need_tgt_for_training:  # True
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break
    if getattr(criterion, "loss_weight_decay", False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, "tuning_matching", False):
        criterion.tuning_matching(epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, "loss_weight_decay", False):
        resstat.update({f"weight_{k}": v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.inference_mode()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    wo_class_error=False,
    args=None,
    logger=None,
):
    output_path = Path(output_dir)
    dataset_len = len(data_loader.dataset)
    expected_predictions_per_image = postprocessors["bbox"].num_select
    saved_predictions = []
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Test:"
    iou_types = tuple(k for k in ("bbox", "keypoints"))
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    if args.dataset_file == "coco":
        from datasets.coco_eval import CocoEvaluator

        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    elif args.dataset_file == "crowdpose":
        from datasets.crowdpose_eval import CocoEvaluator

        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    elif args.dataset_file == "humanart":
        from datasets.humanart_eval import CocoEvaluator

        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    elif args.dataset_file == "exlpose":
        from datasets.exlpose_eval import CocoEvaluator

        coco_evaluator = CocoEvaluator(base_ds, iou_types, args, useCats=useCats)
    elif args.dataset_file == "ehpt":
        from datasets.ehpt_eval import CocoEvaluator

        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    else:
        raise NotImplementedError(f"Dataset {args.dataset_file} not supported")
    _cnt = 0
    times = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) if k != "img_light_level" else v for k, v in t.items()} for t in targets]
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            start_time = time.time()
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            end_time = time.time()
            times.append(end_time - start_time)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {target["image_id"].item(): copy.deepcopy(output) for target, output in zip(targets, results)}

        for image_id, output in copy.deepcopy(res).items():
            scores_tensor = output["scores"]
            prediction_count = scores_tensor.shape[0]
            if prediction_count != expected_predictions_per_image:
                raise ValueError(
                    f"Expected {expected_predictions_per_image} predictions per image, "
                    f"but received {prediction_count} for image_id {image_id}."
                )
            for idx in range(expected_predictions_per_image):
                serialized_output = {"image_id": image_id}
                for key, value in copy.deepcopy(output).items():
                    if isinstance(value, torch.Tensor):
                        item = value[idx]
                        serialized_output[key] = item.item() if item.ndim == 0 else item.tolist()
                    else:
                        serialized_output[key] = value
                saved_predictions.append(serialized_output)
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Averaged time: {:.4f} s".format(np.mean(times)))
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
            stats["coco_eval_keypoints_detr"] = coco_evaluator.coco_eval["keypoints"].stats.tolist()
    gathered_predictions = utils.all_gather(saved_predictions)
    if utils.is_main_process():
        flattened_predictions = [
            prediction for predictions_per_rank in gathered_predictions for prediction in predictions_per_rank
        ]
        expected_total_predictions = dataset_len * expected_predictions_per_image
        if len(flattened_predictions) < expected_total_predictions:
            raise ValueError(
                f"Gathered {len(flattened_predictions)} predictions, "
                f"but expected at least {expected_total_predictions} "
                f"({expected_predictions_per_image} per image for {dataset_len} images)."
            )
        if len(flattened_predictions) > expected_total_predictions:
            warning_msg = (
                f"Gathered {len(flattened_predictions)} predictions, exceeding the "
                f"expected {expected_total_predictions}. Truncating to remove padding."
            )

            print(warning_msg)
            flattened_predictions = flattened_predictions[:expected_total_predictions]
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_name = getattr(args, "dataset_file", None)
        predictions_filename = f"{dataset_name}_predictions.json" if dataset_name else "predictions.json"
        predictions_path = output_path / predictions_filename
        with predictions_path.open("w") as f:
            json.dump(flattened_predictions, f, indent=4)
        print(f"Saved predictions to {predictions_path}")
    return stats, coco_evaluator
