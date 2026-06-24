from typing import Tuple

import torch
from diffusers.models.downsampling import Downsample2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
from diffusers.utils import deprecate, is_torch_version
from torch import nn


# ==== Modify from diffusers ====
class CustomEncoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = CustomDownEncoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=0.0,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_padding=0,
                resnet_time_scale_shift="default",
            )

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], conv_out_channels, 3, padding=1
        )

        self.gradient_checkpointing = False

    def forward(
        self, sample: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        hs = [sample]

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            assert is_torch_version(">=", "1.11.0")
            for down_block in self.down_blocks:
                sample, h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), sample, use_reentrant=False
                )
                hs.append(h)
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), sample, use_reentrant=False
            )

        else:
            # down
            for down_block in self.down_blocks:
                sample, h = down_block(sample)
                hs.append(h)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # from low-res to high-res, z1, z2, z3, z4
        hs = tuple(hs[:-1][::-1])
        """
        hs[0].shape=torch.Size([B, 512, 128, 128]) -> z1
        hs[1].shape=torch.Size([B, 256, 256, 256]) -> z2
        hs[2].shape=torch.Size([B, 128, 512, 512]) -> z3
        hs[3].shape=torch.Size([B, 128, 512, 512]) -> z4
        """
        return sample, hs  # (z0, (z1, z2, z3, z4))


# ==== Modify from diffusers ====
class CustomDownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
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

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self, hidden_states: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        h = hidden_states

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states, h
