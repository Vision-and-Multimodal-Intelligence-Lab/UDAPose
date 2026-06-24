import json
from os import path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from safetensors.torch import load_file


def load_stable_diffusion(cfg):
    model_key = get_sd_hf_pretrained_model_name(
        cfg.styleid_module.stable_diffusion_version
    )

    # Create model
    pipe = StableDiffusionPipeline.from_pretrained(model_key)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    del pipe

    # Use DDIM scheduler
    scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

    return tokenizer, text_encoder, unet, scheduler


def load_vae(cfg, accelerator):
    hf_pretrained_model_name = get_sd_hf_pretrained_model_name(
        cfg.styleid_module.stable_diffusion_version
    )

    if (
        cfg.styleid_module.content_encoder.lower()
        == cfg.styleid_module.style_encoder.lower()
        == cfg.styleid_module.generated_decoder.lower()
    ):
        cnt_vae = sty_vae = gen_vae = load_vae_from_pretrained(
            cfg,
            cfg.styleid_module.content_encoder.lower(),
            accelerator,
            hf_pretrained_model_name,
        )
        accelerator.print("Loaded same VAE for content, style, and generated")
    else:
        cnt_vae = load_vae_from_pretrained(
            cfg,
            cfg.styleid_module.content_encoder.lower(),
            accelerator,
            hf_pretrained_model_name,
        )
        sty_vae = load_vae_from_pretrained(
            cfg,
            cfg.styleid_module.style_encoder.lower(),
            accelerator,
            hf_pretrained_model_name,
        )
        gen_vae = load_vae_from_pretrained(
            cfg,
            cfg.styleid_module.generated_decoder.lower(),
            accelerator,
            hf_pretrained_model_name,
        )

    return cnt_vae, sty_vae, gen_vae


def get_sd_hf_pretrained_model_name(sd_ver: str):
    sd_ver = str(sd_ver).lower()
    if sd_ver == "2.1":
        hf_pretrained_model_name = "sd2-community/stable-diffusion-2-1"
    elif sd_ver == "2.1-base":
        hf_pretrained_model_name = "sd2-community/stable-diffusion-2-1-base"
    else:
        raise ValueError(f"Invalid model key: {sd_ver}")

    return hf_pretrained_model_name


def load_vae_from_pretrained(
    cfg,
    cls,
    accelerator,
    hf_pretrained_model_name: str = "",
):
    if cls == "autoencoderkl":
        vae = AutoencoderKL.from_pretrained(
            hf_pretrained_model_name,
            subfolder="vae",
        )
        accelerator.print(
            f"Loaded pretrained AutoencoderKL from {hf_pretrained_model_name}"
        )
    elif cls == "skip_high_freq":
        from model.vae.skip_high_freq.SkipHighVAE import SkipHighVAE

        ext = path.splitext(cfg.vae.skip_high_freq.pretrained_config)[1]
        if ext == ".json":
            with open(cfg.vae.skip_high_freq.pretrained_config, "r") as f:
                vae_config_dict = json.load(f)
        else:
            raise ValueError(
                f"Invalid config file: {cfg.vae.skip_high_freq.pretrained_config}, ext: {ext}"
            )

        vae = SkipHighVAE().from_config(vae_config_dict)
        vae.load_state_dict(
            load_file(cfg.vae.skip_high_freq.pretrained_model),
            strict=True,
        )
        accelerator.print(
            f"Loaded custom VAE from {cfg.vae.skip_high_freq.pretrained_model}"
        )
    elif cls == "skip_v6":
        from model.vae.skip_v6.SkipVAEv6 import SkipVAEv6

        ext = path.splitext(cfg.vae.skip_v6.pretrained_config)[1]
        if ext == ".json":
            with open(cfg.vae.skip_v6.pretrained_config, "r") as f:
                vae_config_dict = json.load(f)
        else:
            raise ValueError(
                f"Invalid config file: {cfg.vae.skip_v6.pretrained_config}, ext: {ext}"
            )

        vae = SkipVAEv6().from_config(vae_config_dict)

        state_dict = load_file(cfg.vae.skip_v6.pretrained_model)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            elif k == "n_averaged":
                continue
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        vae.load_state_dict(
            state_dict,
            strict=True,
        )
        accelerator.print(f"Loaded custom VAE from {cfg.vae.skip_v6.pretrained_model}")
    else:
        raise NotImplementedError(f"VAE class {cls} is not implemented.")

    vae.eval()
    return vae


@torch.inference_mode()
def encode_latent_vanilla(imgs, vae):
    """
    Encode images to latent space
    Args:
        imgs: torch.Tensor, range: [0, 255]
        vae: AutoencoderKL
    """
    ndim = imgs.ndim
    if ndim == 3:
        imgs = imgs.unsqueeze(0)
    elif ndim != 4:
        raise ValueError(f"imgs.ndim must be 3 or 4, but got {ndim}")

    imgs = imgs.to(device=vae.device, dtype=vae.dtype)
    imgs = imgs / 127.5 - 1.0  # [0, 255] -> [-1, 1]
    latents = vae.encode(imgs).latent_dist.mode() * 0.18215
    if ndim == 3:
        latents = latents.squeeze(0)
    return latents


@torch.inference_mode()
def encode_latent(imgs, vae):
    """
    Args:
        imgs: torch.Tensor, range: [0, 255]
        vae: CustomVAE
    """
    ndim = imgs.ndim
    if ndim == 3:
        imgs = imgs.unsqueeze(0)
    elif ndim != 4:
        raise ValueError(f"imgs.ndim must be 3 or 4, but got {ndim}")

    imgs = imgs.to(device=vae.device, dtype=vae.dtype)
    imgs = imgs / 127.5 - 1.0  # [0, 255] -> [-1, 1]
    latents, hs = vae.encode(imgs)
    latents = latents.latent_dist.mode() * 0.18215
    if ndim == 3:
        latents = latents.squeeze(0)
        hs = [h.squeeze(0) for h in hs]
    return latents, hs


@torch.inference_mode()
def decode_latent_vanilla(latents, vae):
    """
    Decode latent space to images
    Args:
        latents: torch.Tensor
        vae: AutoencoderKL
    Returns:
        imgs: torch.Tensor, range: [0, 255]
    """
    ndim = latents.ndim
    if ndim == 3:
        latents = latents.unsqueeze(0)
    elif ndim != 4:
        raise ValueError(f"latents.ndim must be 3 or 4, but got {ndim}")

    latents = latents.to(device=vae.device, dtype=vae.dtype) / 0.18215
    imgs = vae.decode(latents).sample
    imgs = (imgs + 1.0) * 127.5
    imgs = imgs.clamp(0, 255).type(torch.uint8)
    if ndim == 3:
        imgs = imgs.squeeze(0)
    return imgs


@torch.inference_mode()
def decode_latent(latents, vae, hs):
    """
    Decode latent space to images
    Args:
        latents: torch.Tensor, shape: (B, 4, H // 8, W // 8)
        vae: VQGANVAE
    Returns:
        imgs: torch.Tensor, shape: (B, 3, H, W), range: [0, 1]
    """
    ndim = latents.ndim
    if ndim == 3:
        latents = latents.unsqueeze(0)
    elif ndim != 4:
        raise ValueError(f"latents.ndim must be 3 or 4, but got {ndim}")

    latents = latents.to(device=vae.device, dtype=vae.dtype) / 0.18215
    imgs = vae.decode(latents, hs=hs).sample
    imgs = (imgs + 1.0) * 127.5
    imgs = imgs.clamp(0, 255).type(torch.uint8)
    if ndim == 3:
        imgs = imgs.squeeze(0)
    return imgs
