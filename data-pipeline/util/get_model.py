from safetensors.torch import load_file

from model.discriminator.model import NLayerDiscriminator, init_weights


def get_model(cfg):
    if cfg.model.arch == "skip_v2":
        from model.vae.skip_v2.SkipVAEv2 import SkipVAEv2

        cls = SkipVAEv2
    elif cfg.model.arch == "skip_v3":
        from model.vae.skip_v3.SkipVAEv3 import SkipVAEv3

        cls = SkipVAEv3
    elif cfg.model.arch == "skip_v4":
        from model.vae.skip_v4.SkipVAEv4 import SkipVAEv4

        cls = SkipVAEv4
    elif cfg.model.arch == "skip_v5":
        from model.vae.skip_v5.SkipVAEv5 import SkipVAEv5

        cls = SkipVAEv5
    elif cfg.model.arch == "skip_v6":
        from model.vae.skip_v6.SkipVAEv6 import SkipVAEv6

        cls = SkipVAEv6
    elif cfg.model.arch == "skip_v7":
        from model.vae.skip_v7.SkipVAEv7 import SkipVAEv7

        cls = SkipVAEv7
    elif cfg.model.arch == "skip_v8":
        from model.vae.skip_v8.SkipVAEv8 import SkipVAEv8

        cls = SkipVAEv8
    elif cfg.model.arch == "skip_v9":
        from model.vae.skip_v9.SkipVAEv9 import SkipVAEv9

        cls = SkipVAEv9
    else:
        raise ValueError(f"Model architecture {cfg.model.arch} not supported")

    from model.vae.skip_v9.SkipVAEv9 import SkipVAEv9

    model = cls.from_config(cfg.pretrained_config_path)
    if isinstance(model, SkipVAEv9):
        state_dict = load_file(cfg.pretrained_model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_state_dict["control_encoder." + k[len("encoder.") :]] = v
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(load_file(cfg.pretrained_model_path), strict=False)

    for m in model.frozen_modules:
        for param in m.parameters():
            param.requires_grad = False
        m.eval()

    for m in model.train_modules:
        for param in m.parameters():
            param.requires_grad = True
        m.train()

    if cfg.train.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    return model


def get_discriminator(cfg):
    disc = NLayerDiscriminator().apply(init_weights)
    disc.train()

    return disc
