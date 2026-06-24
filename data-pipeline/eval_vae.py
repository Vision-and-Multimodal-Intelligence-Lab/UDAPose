import copy
import os
import sys

import cv2
import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from safetensors.torch import load_file
from torchvision import io
from tqdm import tqdm
import einops

from dataset import get_val_dataloader
from util.fft import reconstruct_high_freq
from util.get_model import get_model
from util.loss_factory import LossFactory
from util.torch_flags import set_torch_flags


@hydra.main(config_path="config", config_name="train_vae.yaml", version_base=None)
def main(cfg):
    torch.cuda.empty_cache()
    logger = get_logger(__name__)
    cv2.setNumThreads(0)
    set_seed(cfg.seed)
    set_torch_flags(cfg)

    os.makedirs("debug", exist_ok=True)

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

    model = get_model(cfg)
    train_dataloader = get_val_dataloader(cfg)

    # Note: Resume training if checkpoint is provided
    if cfg.train.checkpoint:
        state_dict = load_file(cfg.train.checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module.") :]] = v
            elif k == "n_averaged":
                continue
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)

        accelerator.print(f"Loaded weights from {cfg.train.checkpoint}")

    else:
        logger.error("No checkpoint provided, cannot resume training.")
        exit(1)

    train_dataloader, model = accelerator.prepare(train_dataloader, model)
    criterion = LossFactory(cfg)

    iter_loss = {
        "total_loss": 0.0,
        "mse_loss": 0.0,
        "fft_loss": 0.0,
        "perc_loss": 0.0,
    }
    eval_loss = copy.deepcopy(iter_loss)
    # Note: Start training loop

    model.eval()
    with torch.inference_mode():
        for idx, (z0_img, skip_img, tgt_img) in enumerate(
            tqdm(
                train_dataloader,
                desc="Validate",
                leave=False,
                disable=not accelerator.is_main_process,
            )
        ):
            skip = reconstruct_high_freq(skip_img)
            # skip = skip_img

            # [0,255] -> [-1,1]
            z0_img = z0_img.float() / 127.5 - 1.0
            skip = skip.float() / 127.5 - 1.0
            tgt_img = tgt_img.float() / 127.5 - 1.0

            with accelerator.autocast():
                pred = model(z0_img, skip_img=skip)
                mse_loss, fft_loss = criterion(pred, tgt_img, skip)
                total_loss = mse_loss + fft_loss

                mse_loss = accelerator.gather(mse_loss).mean()
                fft_loss = accelerator.gather(fft_loss).mean()
                total_loss = accelerator.gather(total_loss).mean()
                eval_loss["mse_loss"] += mse_loss.item()
                eval_loss["fft_loss"] += fft_loss.item()
                eval_loss["total_loss"] += total_loss.item()

            tgt_img = convert_to_uint8(tgt_img)
            pred_img = convert_to_uint8(pred)
            log_img_top = torch.cat([tgt_img, pred_img], dim=-1)
            skip = convert_to_uint8(skip)
            skip_img = convert_to_uint8(skip_img)
            log_img_bottom = torch.cat([skip_img, skip], dim=-1)
            log_img = torch.cat([log_img_top, log_img_bottom], dim=-2)
            log_imgs = torch.split(log_img, 1, dim=0)

            for i, log_img in enumerate(log_imgs):
                io.write_png(
                    log_img.detach().cpu()[0],
                    f"debug/{idx * accelerator.num_processes + accelerator.process_index * accelerator.num_processes + i}.png",
                )

        for k, v in eval_loss.items():
            eval_loss[k] /= len(train_dataloader)
        logger.info("==== Validate epoch ====")
        for k, v in eval_loss.items():
            logger.info(f"{k}: {v}")

    # Note: Clear cache and finish training
    accelerator.wait_for_everyone()
    accelerator.print("All done!")
    torch.cuda.empty_cache()
    if accelerator.state.num_processes > 1:
        torch.distributed.destroy_process_group()


def convert_to_uint8(img: torch.Tensor):
    img = img.detach()
    img = (img + 1.0) * 127.5
    rgb_mean = einops.reduce(img, "b c h w -> b 1 1 1", "mean")
    img = img * (0.3460 * 255.0 / rgb_mean)
    img = img.round().clamp(0, 255).to(torch.uint8)
    return img


if __name__ == "__main__":
    sys.argv += ["hydra.output_subdir=null"]

    main()
