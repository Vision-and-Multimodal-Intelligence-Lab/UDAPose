from typing import Optional, Tuple

import torch
from diffusers.models.autoencoders.vae import (
    Decoder,
)
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
from diffusers.models.upsampling import Upsample2D
from diffusers.utils import deprecate, is_torch_version
from torch import nn
from torch.nn import functional as F

from model.vae.skip_v6.Merge import Merge


# ==== Modify from diffusers ====
class CustomDecoder(Decoder):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = CustomUpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                resolution_idx=None,
                dropout=0.0,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=norm_type,
                temb_channels=temb_channels,
                idx=i,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
            )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

        # TODO: Remove hardcoding
        # hidden + skip = 128 + 128
        self.merge = Merge(256, block_out_channels[0])
        # hidden + skip = 3 + 128
        # self.final_out = Merge(131, 3, 512)

    def forward(
        self,
        sample: torch.Tensor,
        latent_embeds: Optional[torch.Tensor] = None,
        skip: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""
        assert len(skip) == 4, f"{len(skip)=}"

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            assert is_torch_version(">=", "1.11.0")
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                sample,
                latent_embeds,
                use_reentrant=False,
            )
            sample = sample.to(upscale_dtype)

            # Note: Do not use gradient checkpointing for the up blocks
            # Note: As it would lead to unequivalent forward
            # up
            for idx, up_block in enumerate(self.up_blocks):
                sample = up_block(sample, latent_embeds, skip[idx])
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for idx, up_block in enumerate(self.up_blocks):
                sample = up_block(sample, latent_embeds, skip[idx])

        # !: we move final merge here.
        sample = self.merge(sample, skip[-1]) + sample

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # sample = self.final_out(sample, skip[-1]) + sample

        return sample


# ==== Modify from diffusers ====
class CustomUpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
        idx=None,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.idx = idx

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    CustomUpsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        idx=idx,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        skip_h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, skip_h=skip_h)

        return hidden_states


class CustomUpsample2D(Upsample2D):
    def __init__(
        self,
        channels: int,  # !: Specified as `out_channels`
        use_conv: bool = False,  # !: Specified as True
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,  # !: Specified as `out_channels`
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
        idx=None,
    ):
        super().__init__(
            channels=channels,
            use_conv=use_conv,
            use_conv_transpose=use_conv_transpose,
            out_channels=out_channels,
            name=name,
            kernel_size=kernel_size,
            padding=padding,
            norm_type=norm_type,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            interpolate=interpolate,
        )

        # TODO: Remove hardcoding
        merged_channels = [1024, 768, 384]
        # hidden+skip = 512+512, 512+256, 256+128
        self.merge = Merge(merged_channels[idx], out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_size: Optional[int] = None,
        skip_h: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        assert hidden_states.shape[1] == self.channels

        # Pytorch bug fix, see:
        # https://github.com/huggingface/diffusers/issues/984
        # https://github.com/pytorch/pytorch/issues/141831
        hidden_states = hidden_states.contiguous()

        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

        hidden_states = self.conv(hidden_states)

        hidden_states = self.merge(hidden_states, skip_h) + hidden_states

        return hidden_states
