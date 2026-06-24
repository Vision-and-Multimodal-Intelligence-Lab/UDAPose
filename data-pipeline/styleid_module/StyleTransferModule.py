from typing import Optional

import torch
from tqdm import tqdm


class StyleTransferModule:
    def __init__(
        self,
        unet,
        text_encoder,
        tokenizer,
        scheduler,
        cfg=None,
        style_transfer_params=None,
        accelerator=None,
        device=torch.device("cuda"),
    ):
        self.unet = unet  # SD unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.device = self.accelerator.device if accelerator else device
        self.cfg = cfg

        self.style_transfer_params: dict = (
            cfg.styleid_module.style_transfer_params
            if cfg is not None
            else style_transfer_params
        )
        assert self.style_transfer_params is not None
        if accelerator is not None:
            accelerator.print(self.style_transfer_params)
        else:
            print(self.style_transfer_params)

        # where to save key value (attention block feature)
        self.attn_features = {}
        # where to save key value to modify (attention block feature)
        self.attn_features_modify = {}

        # self.attn_results = {}
        self.attn_scores_sum = {}
        self.attn_scores_sq_sum = {}
        self.attn_scores_n = {}

        self.content_latents = []
        self.style_latents = []

        self.cur_t = None

        # Get residual and attention block in decoder
        # [0 ~ 11], total 12 layers
        _, attn = get_unet_layers(unet)

        # where to inject key and value
        qkv_injection_layer_num = self.style_transfer_params["injection_layers"]

        # Note: we only hijack self-attention block here.
        # Note: In UNet, the image embedding is first self-attention then cross-attention with text embedding.
        # Note: In transformers library, attn1 means self-attention block, attn2 means cross-attention block.
        for i in qkv_injection_layer_num:
            self.attn_features["layer{}_attn".format(i)] = {}
            attn[i].transformer_blocks[0].attn1.register_forward_hook(
                self.__get_query_key_value("layer{}_attn".format(i))
            )

        # Modify hook (if you change query key value)
        for i in qkv_injection_layer_num:
            attn[i].transformer_blocks[0].attn1.register_forward_hook(
                self.__modify_self_attn_qkv("layer{}_attn".format(i))
            )

        for i in qkv_injection_layer_num:
            self.attn_scores_sum["layer{}_attn".format(i)] = {}
            self.attn_scores_sq_sum["layer{}_attn".format(i)] = {}
            self.attn_scores_n["layer{}_attn".format(i)] = {}

        # for i in qkv_injection_layer_num:
        #     self.attn_results["layer{}_attn".format(i)] = {}
        #     attn[i].transformer_blocks[0].attn2.register_forward_hook(
        #         self.__get_after_qkv("layer{}_attn".format(i))
        #     )

        # triggers for obtaining or modifying features
        self.trigger_get_qkv = (
            False  # if set True --> save attn qkv in self.attn_features
        )
        # if set True --> save attn qkv by self.attn_features_modify
        self.trigger_modify_qkv = False

        self.modify_num = None  # ignore
        self.modify_num_sa = None  # ignore

    def get_text_condition(self, text):
        max_length = self.tokenizer.model_max_length
        device = self.device

        # Determine batch size based on whether text is provided
        if text is None:
            batch_size = 1
        elif isinstance(text, str):
            batch_size = 1
        else:  # list[str]
            batch_size = len(text)

        # ---- Compute Unconditional Embeddings (Shared Logic) ---
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0].to(
            device
        )

        # ---- Early return if no text prompt (Unconditional only) ---
        if text is None:
            return {"encoder_hidden_states": uncond_embeddings}

        # ---- Compute Conditional Embeddings (Only if text is provided) ---
        text_input = self.tokenizer(
            text,  # Handles str or list[str]
            padding="max_length",
            max_length=max_length,
            # truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0].to(
            device
        )

        # ---- Concatenate for CFG ---
        # text_cond = torch.cat([text_embeddings, uncond_embeddings])
        # !: Only use text_embeddings for conditional diffusion
        text_cond = text_embeddings
        denoise_kwargs = {"encoder_hidden_states": text_cond}
        return denoise_kwargs

    @torch.inference_mode()
    def reverse_process(
        self, input, denoise_kwargs, align_stats="init", sty_means=None, sty_stds=None
    ):
        """
        Reverse diffusion process, i.e. x_t -> x_{t-1} -> ... -> x_0
        """
        pred_images = []
        pred_latents = []

        sty_means = sty_means[::-1]
        sty_stds = sty_stds[::-1]

        # Reverse diffusion process
        for idx, t in enumerate(
            tqdm(
                self.scheduler.timesteps,
                leave=False,
                desc="Reverse Process",
                disable=not self.accelerator.is_main_process
                if self.accelerator
                else False,
            )
        ):
            if (align_stats == "init" and idx == 0) or align_stats == "all" and idx < self.cfg.styleid_module.align_stats_threshold:
                input_mean = input.mean(dim=(-1, -2), keepdim=True)
                input_std = input.std(dim=(-1, -2), keepdim=True)
                sty_mean = sty_means[idx]
                sty_std = sty_stds[idx]
                input = (input - input_mean) / (input_std + 1.0e-4) * sty_std + sty_mean
            elif align_stats == "init" or align_stats == "no" or align_stats == "all":
                pass
            else:
                raise ValueError(f"Invalid align_stats: {align_stats}")
            # setting t (for saving time step)
            self.cur_t = t.item()

            # For text condition on stable diffusion
            if "encoder_hidden_states" in denoise_kwargs.keys():
                bs = denoise_kwargs["encoder_hidden_states"].shape[0]
                input = torch.cat([input] * bs)

            noisy_residual = self.unet(
                input, t.to(self.device), **denoise_kwargs
            ).sample

            # For text condition on stable diffusion
            if noisy_residual.shape[0] == 2:
                # perform guidance
                noise_pred_text, noise_pred_uncond = noisy_residual.chunk(2)
                noisy_residual = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                input, _ = input.chunk(2)

            # coef * P_t(e_t(x_t)) + D_t(e_t(x_t))
            res = self.scheduler.step(noisy_residual, t, input)
            prev_noisy_sample = res.prev_sample
            pred_original_sample = res.pred_original_sample

            input = prev_noisy_sample

            pred_latents.append(pred_original_sample.clone())
            # pred_images.append(
            #     decode_latent(pred_original_sample, vae=self.vae, hs=hs)
            # )

        return pred_images, pred_latents

    # Inversion (https://github.com/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)
    @torch.inference_mode()
    def invert_process(self, input, denoise_kwargs):
        """
        DDIM Inversion, i.e. x_0 -> x_1 -> ... -> x_t
        """
        pred_images = []
        pred_latents = []
        pred_means = []
        pred_stds = []

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.scheduler.timesteps)
        num_inference_steps = len(self.scheduler.timesteps)
        cur_latent = input.clone()

        for i in tqdm(
            range(0, num_inference_steps),
            leave=False,
            desc="Inverse Process",
            disable=not self.accelerator.is_main_process if self.accelerator else False,
        ):
            t = timesteps[i]

            self.cur_t = t.item()

            # For text condition on stable diffusion
            if "encoder_hidden_states" in denoise_kwargs.keys():
                bs = denoise_kwargs["encoder_hidden_states"].shape[0]
                cur_latent = torch.cat([cur_latent] * bs)

            # Predict the noise residual
            noise_pred = self.unet(
                cur_latent, t.to(self.device), **denoise_kwargs
            ).sample

            # For text condition on stable diffusion
            if noise_pred.shape[0] == 2:
                # perform guidance
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                cur_latent, _ = cur_latent.chunk(2)

            current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
            # min(999, t.item() + (1000//num_inference_steps)) # t+1
            next_t = t
            alpha_t = self.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.scheduler.alphas_cumprod[next_t]

            # !: v-prediction
            if self.scheduler.config.prediction_type == "v_prediction":
                beta_t = 1 - alpha_t
                pred_original_sample = (
                    alpha_t.sqrt() * cur_latent - beta_t.sqrt() * noise_pred
                )
                pred_epsilon = alpha_t.sqrt() * noise_pred + beta_t.sqrt() * cur_latent
                pred_sample_direction = (1 - alpha_t_next).sqrt() * pred_epsilon
                cur_latent = (
                    alpha_t_next.sqrt() * pred_original_sample + pred_sample_direction
                )
            # !: ε-prediction
            elif self.scheduler.config.prediction_type == "epsilon":
                # Inverted update step (re-arranging the update step to get x_t (new latents) as a function of x_{t-1} (current latents)
                # modified version of Eq.12 in https://arxiv.org/pdf/2010.02502
                cur_latent = (cur_latent - (1 - alpha_t).sqrt() * noise_pred) * (
                    alpha_t_next.sqrt() / alpha_t.sqrt()
                ) + (1 - alpha_t_next).sqrt() * noise_pred
            else:
                raise NotImplementedError(
                    f"Unsupported prediction type: {self.scheduler.config.prediction_type}"
                )

            pred_latents.append(cur_latent.clone())
            pred_means.append(cur_latent.mean(dim=(-1, -2), keepdim=True))
            pred_stds.append(cur_latent.std(dim=(-1, -2), keepdim=True))
            # pred_images.append(decode_latent(cur_latent, vae=self.vae, hs=hs))

        return pred_images, pred_latents, pred_means, pred_stds

    # ============================ hook operations ===============================

    # save key value in self.original_kv[name]
    def __get_query_key_value(self, name):
        def hook(model, input, output):
            if self.trigger_get_qkv:
                _, attn_scores, query, key, value, _ = attention_op(model, input[0])

                self.attn_features[name][int(self.cur_t)] = (
                    query.detach(),
                    key.detach(),
                    value.detach(),
                )

                # attn_scores = attn_scores.type(torch.float64)

                # self.attn_scores_sum[name][int(self.cur_t)] = attn_scores.sum()
                # self.attn_scores_sq_sum[name][int(self.cur_t)] = (attn_scores ** 2).sum()
                # self.attn_scores_n[name][int(self.cur_t)] = attn_scores.numel()

        return hook

    def __modify_self_attn_qkv(self, name):
        def hook(model, input, output):
            if self.trigger_modify_qkv:
                module_name = name  # TODO

                # Note: q_cs -> Q_cs, linear projection & reshape
                _, _, Q_cs, _, _, _ = attention_op(model, input[0])

                Q_c, K_s, V_s = self.attn_features_modify[name][int(self.cur_t)]

                # style injection
                Q_hat_cs = Q_c * self.style_transfer_params["gamma"] + Q_cs * (
                    1 - self.style_transfer_params["gamma"]
                )
                K_cs, V_cs = K_s, V_s

                # Replace query key and value
                _, attn_scores, _, _, _, modified_output = attention_op(
                    model,
                    input[0],
                    query=Q_hat_cs,
                    key=K_cs,
                    value=V_cs,
                    temperature=self.style_transfer_params["tau"],
                )

                # attn_scores = attn_scores.type(torch.float64)

                # self.attn_scores_sum[name][int(self.cur_t)] = attn_scores.sum()
                # self.attn_scores_sq_sum[name][int(self.cur_t)] = (attn_scores ** 2).sum()
                # self.attn_scores_n[name][int(self.cur_t)] = attn_scores.numel()

                return modified_output

        return hook

    # def __get_after_qkv(self, name):
    #     def hook(model, input, output):
    #         self.attn_results[name][int(self.cur_t)] = output

    #     return hook


def get_unet_layers(unet):
    layer_num = [i for i in range(12)]
    resnet_layers = []
    attn_layers = []

    for idx, ln in enumerate(layer_num):
        up_block_idx = idx // 3
        layer_idx = idx % 3

        resnet_layers.append(
            getattr(unet, "up_blocks")[up_block_idx].resnets[layer_idx]
        )
        if up_block_idx > 0:
            attn_layers.append(
                getattr(unet, "up_blocks")[up_block_idx].attentions[layer_idx]
            )
        else:
            attn_layers.append(None)

    return resnet_layers, attn_layers


# Diffusers attention code for getting query, key, value and attention map
def attention_op(
    attn,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    query=None,
    key=None,
    value=None,
    attention_probs=None,
    temperature=1.0,
):
    residual = hidden_states

    if attn.spatial_norm is not None:
        raise NotImplementedError("Spatial norm is not implemented")
        # hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size, channel, height * width
        ).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    if query is None:
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    if key is None:
        key = attn.to_k(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
    if value is None:
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)

    if key.shape[0] != query.shape[0]:
        key, value = key[: query.shape[0]], value[: query.shape[0]]

    # apply temperature scaling
    query = query * temperature  # same as applying it on qk matrix

    if attention_probs is None:
        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attention_probs, attention_scores = get_attention_scores(
            query,
            key,
            attention_mask,
            attn.upcast_attention,
            attn.upcast_softmax,
            attn.scale,
        )

    batch_heads, img_len, txt_len = attention_probs.shape

    # h = w = int(img_len ** 0.5)
    # attention_probs_return = attention_probs.reshape(batch_heads // attn.heads, attn.heads, h, w, txt_len)

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch_size, channel, height, width
        )

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return (
        attention_probs,
        attention_scores,
        query,
        key,
        value,
        hidden_states,
    )


# Modified from `diffusers.models.attention_processor.Attention.get_attention_scores`.
# Now it returns attention probabilities along with attention scores (QK^T).
def get_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    upcast_attention: bool = False,
    upcast_softmax: bool = False,
    scale: float = 0.125,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.
        upcast_attention (`bool`, *optional*, defaults to `False`): Whether to upcast the attention scores.
        upcast_softmax (`bool`, *optional*, defaults to `False`): Whether to upcast the softmax.
        scale (`float`, *optional*, defaults to `0.125`): The scale to use for the attention scores.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    if upcast_attention:
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0],
            query.shape[1],
            key.shape[1],
            dtype=query.dtype,
            device=query.device,
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=scale,
    )
    del baddbmm_input

    if upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = attention_scores.softmax(dim=-1)
    # del attention_scores

    attention_probs = attention_probs.to(dtype)

    return attention_probs, attention_scores
