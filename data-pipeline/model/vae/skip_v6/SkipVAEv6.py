from typing import Optional, Tuple, Union

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from torch import nn

from model.vae.skip_v6.Decoder import CustomDecoder
from model.vae.skip_v6.Encoder import CustomEncoder
from model.vae.skip_v6.Merge import Merge


class SkipVAEv6(AutoencoderKL):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = CustomEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        # pass init params to Decoder
        self.decoder = CustomDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = (
            nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(latent_channels, latent_channels, 1)
            if use_post_quant_conv
            else None
        )

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(
            sample_size / (2 ** (len(self.config.block_out_channels) - 1))
        )
        self.tile_overlap_factor = 0.25

        self.frozen_modules = [
            self.encoder,
            self.quant_conv,
            self.post_quant_conv,
            self.decoder,
        ]
        self.train_modules = [
            module for module in self.decoder.modules() if isinstance(module, Merge)
        ]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape

        if self.use_tiling and (
            width > self.tile_sample_min_size or height > self.tile_sample_min_size
        ):
            return self._tiled_encode(x)

        enc, hs = self.encoder(x)

        if self.quant_conv is not None:
            enc = self.quant_conv(enc)

        return enc, hs

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h, hs = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior, hs)

        return AutoencoderKLOutput(latent_dist=posterior), hs

    def _decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
        hs: Optional[list] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
        ):
            return self.tiled_decode(z, return_dict=return_dict)

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z, skip=hs)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self,
        z: torch.FloatTensor,
        return_dict: bool = True,
        generator=None,
        hs: Optional[list] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, hs=hs).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        skip_img: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior, _ = self.encode(x)
        _, hs = self.encode(skip_img)

        posterior = posterior.latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z, hs=hs).sample

        return dec
