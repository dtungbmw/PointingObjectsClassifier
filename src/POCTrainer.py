from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from pc_dataset import *
from pathlib import Path
import logging
import torch
import os
from pathlib import Path
from YoloPointingTransformer import TransformerPointingTrainer


class POCTrainer:

    def setup_dataloader(self, cfg: DictConfig) -> DataLoader:

        DEVICE = cfg.DEVICE
        if DEVICE == "cpu" and cfg.task == "train":
            logging.warning("Training DeePoint with CPU takes a long time.")
        if cfg.task == "test" and cfg.verbose is not True and cfg.shrink_rate != 1:
            logging.warning(
                "Using only part of test dataset. You should set `shrink_rate=1` except for speeding up the performance for visualization"
            )

        assert not (
            cfg.task == "test" and cfg.ckpt is None
        ), "checkpoint should be specified for evaluation"
        
        keypoints_path = Path(os.getcwd()).parent.parent / cfg.keypoints_root
        train_ds = PCDataset(keypoints_path, cfg)
        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.hardware.bs,
            #sampler=train_sampler,
            num_workers=cfg.hardware.nworkers,
            persistent_workers=True if cfg.hardware.nworkers != 0 else False,
        )
        return train_dl
    
    def launch_training(self, dataloader: DataLoader):
        
        transformer_trainer = TransformerPointingTrainer()
        
        transformer_trainer.train(dataloader)
        pass