from safetensors.torch import load_file


def get_model(cfg):

    from model.vae.skip_v6.SkipVAEv6 import SkipVAEv6

    cls = SkipVAEv6


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
