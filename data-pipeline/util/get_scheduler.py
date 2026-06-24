from torch.optim import lr_scheduler


def get_scheduler(cfg, optim, accelerator=None):
    scheduler_kwargs = {**cfg.train.scheduler}
    name = scheduler_kwargs.pop("name").lower()
    name = str(cfg.train.scheduler.name).lower()

    if name == "multisteplr":
        return lr_scheduler.MultiStepLR(optim, **scheduler_kwargs)

    elif name == "reducelronplateau":
        return lr_scheduler.ReduceLROnPlateau(optim, **scheduler_kwargs)

    else:
        raise NotImplementedError(f"Unsupported scheduler: {name}")
