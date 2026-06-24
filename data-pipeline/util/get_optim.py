import math

from accelerate.utils import DistributedType
from torch.optim import AdamW


def get_optim(cfg, modules, accelerator):
    optim_kwargs = {**cfg.train.optimizer}

    lr = optim_kwargs.pop("base_lr")
    bs = cfg.data.batch_size
    if accelerator.distributed_type == DistributedType.FSDP:
        # For FSDP, we should scale the learning rate by the sqrt of the number of processes.
        lr *= math.sqrt(
            bs * accelerator.gradient_accumulation_steps * accelerator.num_processes
        )
    elif accelerator.distributed_type == DistributedType.MULTI_GPU:
        # For DDP, it's common to scale the learning rate by the sqrt of the number of processes.
        lr *= math.sqrt(
            bs * accelerator.gradient_accumulation_steps * accelerator.num_processes
        )
    elif accelerator.distributed_type == DistributedType.NO:
        # Not a distributed environment, no scaling needed.
        lr *= math.sqrt(bs * accelerator.gradient_accumulation_steps)
    else:
        raise ValueError(
            f"Distributed type {accelerator.distributed_type} not supported."
        )

    name = optim_kwargs.pop("name").lower()
    optim_kwargs["lr"] = lr

    all_params = []
    for module in modules:
        all_params.extend(list(filter(lambda p: p.requires_grad, module.parameters())))

    if name == "adamw":
        return AdamW(all_params, **optim_kwargs)

    else:
        raise ValueError(f"Optimizer {name} not supported")
