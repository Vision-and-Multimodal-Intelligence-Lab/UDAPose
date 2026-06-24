"""
In this file, we would use a combination of LL-N, LL-H, LL-E, A7M3, RICOH3 to generate low-light images.
To ensafe our claim, we would only use test set images.
"""

import copy
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import hydra
import torch
from accelerate import (
    Accelerator,
    DDPCommunicationHookType,
    DistributedDataParallelKwargs,
    DistributedType,
)
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torchvision.transforms.v2 import functional as F

from dataset.ExLPoseDataset import get_dataloader
from styleid_module.StyleTransferModule import StyleTransferModule
from styleid_module.util import (
    decode_latent,
    decode_latent_vanilla,
    encode_latent,
    encode_latent_vanilla,
    load_stable_diffusion,
    load_vae,
)
from util.fft import reconstruct_high_freq
from util.io import save_image
from util.rescale import rescale_stats
from util.torch_flags import set_torch_flags

DTYPE = torch.bfloat16


@hydra.main(config_path="config", config_name="style_transfer.yaml", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    set_torch_flags(cfg)
    torch.set_default_dtype(DTYPE)
    cv2.setNumThreads(0)

    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.BF16)
        ],
    )

    style_text = content_text = None

    result_path = save_dir = cfg.save_dir

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if os.path.exists(result_path):
            confirm = (
                input(
                    f"Directory '{result_path}' already exists. Do you want to delete it and all its contents? (yes/no): "
                )
                .strip()
                .lower()
            )
            if confirm == "yes":
                import shutil

                shutil.rmtree(result_path)
                os.makedirs(result_path)
            else:
                print("Aborted: Directory not deleted. Exiting.")
                exit(1)
        else:
            os.makedirs(result_path)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    # get SD modules
    cnt_vae, sty_vae, gen_vae = load_vae(cfg, accelerator)
    tokenizer, text_encoder, unet, scheduler = load_stable_diffusion(cfg)
    scheduler.set_timesteps(cfg.styleid_module.ddim_steps)

    dataloader = get_dataloader(cfg)

    dataloader, tokenizer, text_encoder, unet, cnt_vae, sty_vae, gen_vae = (
        accelerator.prepare(
            dataloader, tokenizer, text_encoder, unet, cnt_vae, sty_vae, gen_vae
        )
    )

    # Init style transfer module
    unet_wrapper = StyleTransferModule(
        unet
        if accelerator.state.distributed_type == DistributedType.NO
        else accelerator.unwrap_model(unet),
        text_encoder
        if accelerator.state.distributed_type == DistributedType.NO
        else accelerator.unwrap_model(text_encoder),
        tokenizer,
        scheduler,
        cfg=cfg,
        accelerator=accelerator,
    )

    # Get style image tokens
    style_denoise_kwargs = unet_wrapper.get_text_condition(style_text)
    content_denoise_kwargs = unet_wrapper.get_text_condition(content_text)

    with ThreadPoolExecutor(
        os.cpu_count() // 2 // accelerator.state.num_processes,
        thread_name_prefix="StyleID transfer save images",
    ) as executor:
        for _ in tqdm(
            range(cfg.run_epochs),
            disable=not accelerator.is_main_process,
            desc="Run Epochs",
        ):
            for count, packs in enumerate(
                tqdm(
                    dataloader,
                    disable=not accelerator.is_main_process,
                    leave=False,
                    desc="Content img",
                )
            ):
                accelerator.wait_for_everyone()
                (
                    wl_pack,
                    normal_pack,
                    hard_pack,
                    extreme_pack,
                    a7m3_pack,
                    ricoh3_pack,
                ) = packs

                # jump out for last process for duplicate joined data
                supposed_length = len(dataloader.dataset.wl_img_ids)
                cur_pos = (
                    count * accelerator.state.num_processes
                    + accelerator.state.local_process_index
                )
                if cur_pos >= supposed_length:
                    break

                wl_h, wl_w = wl_pack["h"][0], wl_pack["w"][0]

                # get cnt latent
                (
                    cnt_features,
                    cnt_latent,
                    final_path,
                    cnt_mean,
                    cnt_std,
                ) = get_cnt_latent(
                    cfg,
                    result_path,
                    wl_pack,
                    cnt_vae,
                    unet_wrapper,
                    content_denoise_kwargs,
                    accelerator,
                    executor,
                )

                packs = {
                    "normal": normal_pack,
                    "hard": hard_pack,
                    "extreme": extreme_pack,
                    "a7m3": a7m3_pack,
                    "ricoh3": ricoh3_pack,
                }

                for partition in tqdm(
                    cfg.partitions,
                    disable=not accelerator.is_main_process,
                    leave=False,
                    desc="Style ref.",
                ):
                    # get sty latent
                    (
                        sty_features,
                        sty_latent,
                        sty_filename,
                        sty_mean,
                        sty_std,
                        sty_img,
                        sty_hs,
                        sty_means,
                        sty_stds,
                    ) = get_sty_latent(
                        cfg,
                        final_path,
                        cnt_std,
                        packs[partition],
                        sty_vae,
                        unet_wrapper,
                        style_denoise_kwargs,
                        partition,
                        accelerator,
                        executor,
                    )

                    if os.path.exists(
                        os.path.join(
                            final_path, f"stylized_{partition}_{sty_filename}.webp"
                        )
                    ):
                        continue

                    # !: Apply corresponding QKV features for later use
                    for layer_name in sty_features.keys():
                        unet_wrapper.attn_features_modify[layer_name] = {}
                        for t_norm in scheduler.timesteps:
                            t_norm = t_norm.item()
                            unet_wrapper.attn_features_modify[layer_name][t_norm] = (
                                # content as q, style as kv
                                cnt_features[layer_name][t_norm][0],
                                sty_features[layer_name][t_norm][1],
                                sty_features[layer_name][t_norm][2],
                            )

                    sty_convert(
                        cfg,
                        final_path,
                        partition,
                        sty_filename,
                        sty_mean,
                        sty_std,
                        sty_img,
                        content_denoise_kwargs,
                        cnt_latent,
                        sty_latent,
                        gen_vae,
                        unet_wrapper,
                        accelerator,
                        executor,
                        sty_hs,
                        sty_means,
                        sty_stds,
                        wl_h,
                        wl_w,
                    )

            # update epoch counter
            dataloader.dataset.cur_epoch += 1

    accelerator.wait_for_everyone()
    accelerator.print("All done!")
    torch.cuda.empty_cache()
    if accelerator.state.num_processes > 1:
        torch.distributed.destroy_process_group()


@torch.inference_mode()
def sty_convert(
    cfg,
    final_path,
    partition,
    sty_filename,
    sty_mean,
    sty_std,
    sty_img,
    content_denoise_kwargs,
    cnt_latent,
    sty_latent,
    gen_vae,
    unet_wrapper,
    accelerator,
    executor,
    sty_hs=None,
    sty_means=None,
    sty_stds=None,
    wl_h=None,
    wl_w=None,
):
    # !: Inject QKV features (enable modify hook)
    unet_wrapper.trigger_get_qkv = False
    unet_wrapper.trigger_modify_qkv = True

    latent_cs = cnt_latent

    with accelerator.autocast():
        _, latents = unet_wrapper.reverse_process(
            latent_cs,
            content_denoise_kwargs,
            cfg.styleid_module.reverse_diffusion_align_latent_stats,
            sty_means,
            sty_stds,
        )

        # for t, lnt in enumerate(reversed(latents)):
        #     torch.save(lnt, os.path.join(final_path, f"sty-{sty_filename}_t-{t}.pt"))

        latent = latents[-1]
        gen_vae = (
            gen_vae
            if accelerator.state.distributed_type == DistributedType.NO
            else accelerator.unwrap_model(gen_vae)
        )
        if cfg.styleid_module.generated_decoder.lower() == "autoencoderkl":
            final_img = decode_latent_vanilla(latent, gen_vae)
        else:
            final_img = decode_latent(latent, gen_vae, sty_hs)

    if wl_h is not None and wl_w is not None:
        if final_img.shape[-2] != wl_h or final_img.shape[-1] != wl_w:
            final_img = F.resize(final_img, (wl_h, wl_w))

    executor.submit(
        save_image,
        final_img,
        os.path.join(final_path, f"ori_stylized_{partition}_{sty_filename}.webp"),
    )

    # Note: Adjust to original style distribution
    final_img = rescale_stats(final_img, sty_mean, sty_std)
    executor.submit(
        save_image,
        final_img,
        os.path.join(final_path, f"stylized_{partition}_{sty_filename}.webp"),
    )

    # Note: brightup for visualization
    final_img = rescale_stats(final_img, 0.3460 * 255)
    executor.submit(
        save_image,
        final_img,
        os.path.join(final_path, f"norm_stylized_{partition}_{sty_filename}.webp"),
    )


@torch.inference_mode()
def get_sty_latent(
    cfg,
    final_path,
    cnt_std,
    sty_pack,
    sty_vae,
    unet_wrapper,
    style_denoise_kwargs,
    partition: str,
    accelerator: Accelerator,
    executor: ThreadPoolExecutor,
):
    assert partition in ["normal", "hard", "extreme", "a7m3", "ricoh3"], f"{partition=}"

    # !: get style attention features (key, value)
    unet_wrapper.trigger_get_qkv = True
    unet_wrapper.trigger_modify_qkv = False

    pack = repack(sty_pack)

    img = pack["img"]
    sty_mean = img[0].float().mean(dim=(-1, -2))
    sty_std = img[0].float().std(dim=(-1, -2))
    if cfg.dataset.style_brightness_adjust == "align_mean_per_channel":
        img = rescale_stats(
            img,
            torch.tensor(
                cfg.dataset.target_mean_factor,
            )
            * 255,
        )
    elif cfg.dataset.style_brightness_adjust == "align_mean":
        img = rescale_stats(
            img,
            cfg.dataset.target_mean_factor * 255,
        )
    else:
        raise ValueError(
            f"Unsupported style brightness adjust: {cfg.dataset.style_brightness_adjust}"
        )

    sty_vae = (
        sty_vae
        if accelerator.state.distributed_type == DistributedType.NO
        else accelerator.unwrap_model(sty_vae)
    )
    with accelerator.autocast():
        if cfg.styleid_module.style_encoder.lower() == "autoencoderkl":
            sty_latent = encode_latent_vanilla(img, sty_vae)
            hs = None
        else:
            sty_latent, _ = encode_latent(img, sty_vae)
            img = reconstruct_high_freq(img)
            _, hs = encode_latent(img, sty_vae)

        _, latents, means, stds = unet_wrapper.invert_process(
            sty_latent, denoise_kwargs=style_denoise_kwargs
        )
        assert len(latents) == 50, f"{len(latents)=}"
        sty_latent = latents[-1]
        unet_wrapper.style_latents = latents

    if not os.path.exists(os.path.join(final_path, f"sty_{partition}_inter")):
        os.makedirs(os.path.join(final_path, f"sty_{partition}_inter"))

    # for t, lnt in enumerate(latents):
    #         torch.save(
    #             lnt,
    #             os.path.join(
    #                 final_path,
    #                 f"sty_{partition}_inter",
    #                 f"sty-{pack['fname']}_t-{t}.pt",
    #             ),
    #         )

    # Note: also save ori style image and normalized style image
    if not os.path.exists(
        os.path.join(final_path, f"sty_{partition}_inter", f"sty_{pack['fname']}.webp")
    ):
        bright_sty_img = rescale_stats(pack["img"], 0.3460 * 255)
        to_save = bright_sty_img

        executor.submit(
            save_image,
            to_save.cpu(),
            os.path.join(
                final_path,
                f"sty_{partition}_inter",
                f"ori_sty_{pack['fname']}.webp",
            ),
        )

    sty_features = copy.deepcopy(unet_wrapper.attn_features)
    return (
        sty_features,
        sty_latent,
        pack["fname"],
        sty_mean,
        sty_std,
        pack["img"],
        hs,
        means,
        stds,
    )


@torch.inference_mode()
def get_cnt_latent(
    cfg,
    result_path,
    cnt_pack,
    cnt_vae,
    unet_wrapper,
    content_denoise_kwargs,
    accelerator: Accelerator,
    executor: ThreadPoolExecutor,
):
    # !: get content attention features (key, value)
    unet_wrapper.trigger_get_qkv = True
    unet_wrapper.trigger_modify_qkv = False

    pack = repack(cnt_pack)
    img = pack["img"]
    cnt_mean = img[0].float().mean()
    cnt_std = img[0].float().std()

    final_path = os.path.join(result_path, pack["fname"])
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    cnt_vae = (
        cnt_vae
        if accelerator.state.distributed_type == DistributedType.NO
        else accelerator.unwrap_model(cnt_vae)
    )

    with accelerator.autocast():
        if cfg.styleid_module.content_encoder.lower() == "autoencoderkl":
            cnt_latent = encode_latent_vanilla(img, cnt_vae)
        else:
            cnt_latent, _ = encode_latent(img, cnt_vae)

        _, latents, _, _ = unet_wrapper.invert_process(
            cnt_latent, content_denoise_kwargs
        )
        assert len(latents) == 50, f"{len(latents)=}"
        cnt_latent = latents[-1]
        unet_wrapper.content_latents = latents

    if not os.path.exists(os.path.join(final_path, "cnt_inter")):
        os.makedirs(os.path.join(final_path, "cnt_inter"))

    # for t, lnt in enumerate(latents):
    #     torch.save(lnt, os.path.join(final_path, "cnt_inter", f"cnt-{pack['fname']}_t-{t}.pt"))

    # Note: save content image
    if not os.path.exists(
        os.path.join(final_path, "cnt_inter", f"cnt_{pack['fname']}.webp")
    ):
        executor.submit(
            save_image,
            pack["img"].cpu(),
            os.path.join(final_path, "cnt_inter", f"cnt_{pack['fname']}.webp"),
        )

    cnt_features = copy.deepcopy(unet_wrapper.attn_features)
    return (
        cnt_features,
        cnt_latent,
        final_path,
        cnt_mean,
        cnt_std,
    )


def repack(package: dict):
    img = package["img"]
    fname = str(package["fname"].item())  # image id
    h = package["h"][0]
    w = package["w"][0]

    return {
        "img": img,
        "fname": fname,
        "h": h,
        "w": w,
    }


if __name__ == "__main__":
    sys.argv += ["hydra.output_subdir=null"]
    main()
