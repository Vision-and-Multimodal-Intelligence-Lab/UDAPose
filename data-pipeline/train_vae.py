import copy
import os
import shutil
import sys

import cv2
import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from aim import Image, Run
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from tqdm import tqdm

from dataset import get_dataloader
from util.count_param import count_parameters
from util.fft import reconstruct_high_freq
from util.get_model import get_discriminator, get_model
from util.get_optim import get_optim
from util.get_scheduler import get_scheduler
from util.loss_factory import LossFactory, PerceptualLoss
from util.torch_flags import set_torch_flags


@hydra.main(config_path="config", config_name="train_vae.yaml", version_base=None)
def main(cfg):
    torch.cuda.empty_cache()
    logger = get_logger(__name__)
    cv2.setNumThreads(0)
    set_seed(cfg.seed)
    set_torch_flags(cfg)

    accelerator = Accelerator(
        mixed_precision="bf16",
        project_dir=cfg.output_dir,
        gradient_accumulation_steps=cfg.train.accumulate,
        step_scheduler_with_optimizer=False,
        # fsdp_plugin=FullyShardedDataParallelPlugin(
        #     fsdp_version=2,
        #     cpu_ram_efficient_loading=True,
        # ),  # https://huggingface.co/docs/accelerate/en/package_reference/fsdp#accelerate.FullyShardedDataParallelPlugin
    )
    output_dir = cfg.output_dir

    logger.info(f"{os.cpu_count()=}")
    logger.info(f"Using {accelerator.num_processes} GPUs")
    logger.info("Using automatic mixed precision: bf16")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Gradient accumulation steps: {cfg.train.accumulate}")

    if accelerator.is_main_process:
        run = Run(
            repo=output_dir,
            experiment=cfg.name,
        )

    modules = []
    model = get_model(cfg)
    modules.append(model)
    if cfg.train.loss.perceptual_loss:
        disc = get_discriminator(cfg)
        modules.append(disc)

    optim = get_optim(cfg, modules, accelerator)
    scheduler = get_scheduler(cfg, optim, accelerator)
    train_dataloader = get_dataloader(cfg)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    enc_total, enc_train = count_parameters([model.encoder, model.quant_conv])
    dec_total, dec_train = count_parameters([model.decoder, model.post_quant_conv])
    _, total_train = count_parameters(model.train_modules)
    logger.info(f"{len(model.train_modules)=}")
    logger.info(f"{model.train_modules=}")

    logger.info(f"Encoder total parameters: {enc_total / 1e6:.2f}M")
    logger.info(f"Encoder trainable parameters: {enc_train / 1e6:.2f}M")
    logger.info(f"Decoder total parameters: {dec_total / 1e6:.2f}M")
    logger.info(f"Decoder trainable parameters: {dec_train / 1e6:.2f}M")
    logger.info(f"Total trainable parameters: {total_train / 1e6:.2f}M")
    if cfg.train.loss.perceptual_loss:
        _, disc_train = count_parameters(disc)
        logger.info(f"Discriminator trainable parameters: {disc_train / 1e6:.2f}M")

    train_dataloader, model, ema_model, optim, scheduler = accelerator.prepare(
        train_dataloader, model, ema_model, optim, scheduler
    )
    if cfg.train.loss.perceptual_loss:
        disc = accelerator.prepare(disc)

    criterion = LossFactory(cfg)
    if cfg.train.loss.perceptual_loss:
        perc_crit = PerceptualLoss()

    # Note: Resume training if checkpoint is provided
    epoch_start: int = 0
    iter_start: int = 0
    resume_start_epoch: int = 0
    if cfg.train.resume:
        if cfg.train.checkpoint:
            accelerator.load_state(cfg.train.checkpoint, strict=True)
            logger.info(f"Loading weights from state: {cfg.train.checkpoint}")
            f, i = cfg.train.checkpoint.split("/")[-1].split("_")
            epoch_start = int(f.split("-")[-1])
            iter_start = int(i.split("-")[-1])
            logger.info(
                f"Resuming training from epoch {epoch_start} and iteration {iter_start}"
            )
            resume_start_epoch = epoch_start

        else:
            logger.error("No checkpoint provided, cannot resume training.")
            exit(1)

    cur_epoch = epoch_start
    cur_iter = iter_start
    iter_loss = {
        "total_loss": 0.0,
        "mse_loss": 0.0,
        "fft_loss": 0.0,
        "perc_loss": 0.0,
    }
    epoch_loss = copy.deepcopy(iter_loss)
    eval_loss = copy.deepcopy(iter_loss)
    prev_loss = float("inf")
    prev_path = None
    # Note: Start training loop
    while cur_epoch < cfg.train.epochs + resume_start_epoch:
        accelerator.wait_for_everyone()
        cur_epoch += 1
        ema_model.train()

        for k in epoch_loss.keys():
            epoch_loss[k] = 0.0
        for k in eval_loss.keys():
            eval_loss[k] = 0.0

        # update selected training images
        if cfg.data.dataset == "exlpose+ocn":
            train_dataloader.dataset.shuffle_img_ids()

        for idx, (z0_img, skip_img, tgt_img) in enumerate(
            tqdm(
                train_dataloader,
                desc=f"Train Epoch {cur_epoch}",
                leave=False,
                disable=not accelerator.is_main_process,
            )
        ):
            cur_iter += 1  # iter counts forward process

            skip = reconstruct_high_freq(skip_img)

            # [0,255] -> [-1,1]
            z0_img = z0_img.float() / 127.5 - 1.0
            skip = skip.float() / 127.5 - 1.0
            tgt_img = tgt_img.float() / 127.5 - 1.0

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    pred = model(z0_img, skip_img=skip)
                    mse_loss, fft_loss = criterion(pred, tgt_img)

                    perc_loss = torch.tensor(
                        0.0, device=mse_loss.device, dtype=mse_loss.dtype
                    ).detach()
                    if cfg.train.loss.perceptual_loss:
                        disc_input = torch.cat([pred, tgt_img], dim=0)
                        disc_output = disc(disc_input)
                        logits_fake, logits_real = disc_output.chunk(2)
                        perc_loss = perc_crit(logits_fake, logits_real)
                        perc_loss = perc_loss * cfg.train.loss.perceptual_weight

                    total_loss = mse_loss + fft_loss + perc_loss

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

            if (idx + 1) % cfg.train.accumulate == 0 or idx == len(train_dataloader) - 1:
                accelerator.unwrap_model(ema_model).update_parameters(model)

            # record loss
            mse_loss = accelerator.gather(mse_loss).mean()
            fft_loss = accelerator.gather(fft_loss).mean()
            perc_loss = accelerator.gather(perc_loss).mean()
            total_loss = accelerator.gather(total_loss).mean()
            iter_loss["mse_loss"] = mse_loss.item()
            iter_loss["fft_loss"] = fft_loss.item()
            iter_loss["perc_loss"] = perc_loss.item()
            iter_loss["total_loss"] = total_loss.item()
            epoch_loss["mse_loss"] += mse_loss.item()
            epoch_loss["fft_loss"] += fft_loss.item()
            epoch_loss["perc_loss"] += perc_loss.item()
            epoch_loss["total_loss"] += total_loss.item()
            if accelerator.is_main_process:
                run.track(
                    {**iter_loss},
                    context={"train": "per iteration"},
                    step=cur_iter,
                    epoch=cur_epoch,
                )

        # calculate epoch loss
        for k, v in epoch_loss.items():
            epoch_loss[k] /= len(train_dataloader)

        # log epoch loss to console
        logger.info(f"==== Train epoch {cur_epoch} ====")
        for k, v in epoch_loss.items():
            logger.info(f"{k}: {v}")
        logger.info(f"LR: {scheduler.get_last_lr()[0]:.9f}")

        # log epoch loss to aim
        if accelerator.is_main_process:
            run.track(
                {**epoch_loss},
                context={"train": "per epoch"},
                step=cur_iter,
                epoch=cur_epoch,
            )
            run.track(
                {"lr": scheduler.get_last_lr()[0]},
                context={"train": "per epoch"},
                step=cur_iter,
                epoch=cur_epoch,
            )

        # scheduler step
        if cfg.train.scheduler.name.lower() == "reducelronplateau":
            scheduler.step(epoch_loss["total_loss"])
        else:
            scheduler.step()

        if cur_epoch % 20 == 0 or cur_epoch > cfg.train.epochs * 0.95:
            update_bn(train_dataloader, ema_model)
            ema_model.eval()
            with torch.inference_mode():
                for idx, (z0_img, skip_img, tgt_img) in enumerate(
                    tqdm(
                        train_dataloader,
                        desc=f"Validate Epoch {cur_epoch}",
                        leave=False,
                        disable=not accelerator.is_main_process,
                    )
                ):
                    skip = reconstruct_high_freq(skip_img)

                    # [0,255] -> [-1,1]
                    z0_img = z0_img.float() / 127.5 - 1.0
                    skip = skip.float() / 127.5 - 1.0
                    tgt_img = tgt_img.float() / 127.5 - 1.0

                    with accelerator.autocast():
                        pred = ema_model(z0_img, skip_img=skip)
                        mse_loss, fft_loss = criterion(pred, tgt_img)
                        total_loss = mse_loss + fft_loss

                        mse_loss = accelerator.gather(mse_loss).mean()
                        fft_loss = accelerator.gather(fft_loss).mean()
                        total_loss = accelerator.gather(total_loss).mean()
                        eval_loss["mse_loss"] += mse_loss.item()
                        eval_loss["fft_loss"] += fft_loss.item()
                        eval_loss["total_loss"] += total_loss.item()

                    # log images
                    if idx % len(train_dataloader) < 3:
                        tgt_img = convert_to_uint8(tgt_img[0])
                        pred_img = convert_to_uint8(pred[0])
                        log_img_top = torch.cat([tgt_img, pred_img], dim=-1)
                        skip = convert_to_uint8(skip[0])
                        skip_img = convert_to_uint8(skip_img[0])
                        log_img_bottom = torch.cat([skip_img, skip], dim=-1)
                        log_img = torch.cat([log_img_top, log_img_bottom], dim=-2)
                        if accelerator.is_main_process:
                            run.track(
                                Image(log_img),
                                name="Imgs",
                                step=idx + cur_epoch * len(train_dataloader),
                                epoch=cur_epoch,
                            )

                for k, v in eval_loss.items():
                    eval_loss[k] /= len(train_dataloader)
                logger.info(f"==== Validate epoch {cur_epoch} ====")
                for k, v in eval_loss.items():
                    logger.info(f"{k}: {v}")
                if accelerator.is_main_process:
                    run.track(
                        {**eval_loss},
                        context={"validate": "per epoch"},
                        epoch=cur_epoch,
                    )

        # ---- save checkpoint ----
        if (
            cur_epoch > cfg.train.epochs * 0.95
            or cur_epoch % cfg.train.save_interval == 0
        ):
            accelerator.wait_for_everyone()
            accelerator.save_state(
                os.path.join(output_dir, "state", f"Epoch-{cur_epoch}_Iter-{cur_iter}")
            )
        elif eval_loss["total_loss"] < prev_loss and cur_epoch > 200:
            accelerator.wait_for_everyone()
            accelerator.save_state(
                os.path.join(output_dir, "state", f"Epoch-{cur_epoch}_Iter-{cur_iter}")
            )
            if prev_path is not None and accelerator.is_main_process:
                shutil.rmtree(prev_path)
                logger.info(f"Removed previous checkpoint: {prev_path}")
            prev_loss = eval_loss["total_loss"]
            prev_path = os.path.join(
                output_dir, "state", f"Epoch-{cur_epoch}_Iter-{cur_iter}"
            )

        accelerator.wait_for_everyone()

    # Note: Clear cache and finish training
    accelerator.wait_for_everyone()
    accelerator.print("All done!")
    torch.cuda.empty_cache()
    if accelerator.state.num_processes > 1:
        torch.distributed.destroy_process_group()


def convert_to_uint8(img: torch.Tensor):
    img = img.detach()
    img = (img + 1.0) * 127.5
    img = img * (0.3460 * 255.0 / img.mean())
    img = img.round().clamp(0, 255).to(torch.uint8)
    return img


if __name__ == "__main__":
    if int(os.environ.get("RANK", "0")) != 0:
        sys.argv += ["hydra.output_subdir=null"]

    main()
