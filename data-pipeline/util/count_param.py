from torch import nn


def count_parameters(model: list[nn.Module] | nn.Module) -> tuple[int, int]:
    if isinstance(model, list):
        total_params = 0
        trainable_params = 0

        for m in model:
            total_params += sum(p.numel() for p in m.parameters())
            trainable_params += sum(
                p.numel() for p in m.parameters() if p.requires_grad
            )

    else:
        # Total parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params
