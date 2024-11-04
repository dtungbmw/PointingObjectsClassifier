from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from src.pc_dataset import *
from pathlib import Path
import logging
import torch
import os
from pathlib import Path
from src.YoloPointingTransformer import TransformerPointingTrainer
import logging


class POCTrainer:
    
    def __init__(self) -> None:
        
        logging.basicConfig(level=logging.DEBUG,  # Set the minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
                    handlers=[logging.StreamHandler()])  # Default handler (outputs to console)


    def setup_dataloader(self, cfg: DictConfig) -> DataLoader:

        DEVICE = cfg.DEVICE
        if DEVICE == "cpu" and cfg.task == "train":
            logging.warning("Training POC with CPU takes a long time.")
        if cfg.task == "test" and cfg.verbose is not True and cfg.shrink_rate != 1:
            logging.warning(
                "Using only part of test dataset. You should set `shrink_rate=1` except for speeding up the performance for visualization"
            )

        assert not (
            cfg.task == "test" and cfg.ckpt is None
        ), "checkpoint should be specified for evaluation"
        
        keypoints_path = Path(os.getcwd()).parent / cfg.keypoints_root
        print("Set up training dataset...")
        train_ds = PCDataset(keypoints_path, cfg)
        print("Create training dataloader...")
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