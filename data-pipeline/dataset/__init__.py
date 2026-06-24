from torch.utils.data import DataLoader


def get_dataloader(cfg):
    if cfg.data.dataset == "exlpose":
        from .TrainExLPoseDataset import TrainExLPoseDataset

        return DataLoader(
            TrainExLPoseDataset(cfg),
            cfg.data.batch_size,
            cfg.data.shuffle,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    elif cfg.data.dataset == "ocn":
        from .TrainOCNDataset import TrainOCNDataset

        return DataLoader(
            TrainOCNDataset(cfg),
            cfg.data.batch_size,
            cfg.data.shuffle,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    elif cfg.data.dataset == "exlpose+ocn":
        from .TrainDataset import TrainDataset

        return DataLoader(
            TrainDataset(cfg),
            cfg.data.batch_size,
            cfg.data.shuffle,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.dataset}")


def get_val_dataloader(cfg):

    if cfg.data.dataset == "exlpose":
        from .TrainExLPoseDataset import ValExLPoseDataset

        return DataLoader(
            ValExLPoseDataset(cfg),
            cfg.data.batch_size,
            False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    elif cfg.data.dataset == "ocn":
        from .TrainOCNDataset import TrainOCNDataset

        return DataLoader(
            TrainOCNDataset(cfg),
            cfg.data.batch_size,
            cfg.data.shuffle,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    elif cfg.data.dataset == "exlpose+ocn":
        from .TrainDataset import TrainDataset

        return DataLoader(
            TrainDataset(cfg),
            cfg.data.batch_size,
            cfg.data.shuffle,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.dataset}")
