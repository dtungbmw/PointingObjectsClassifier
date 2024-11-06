import torch
import dataset
from pathlib import Path
import random
import hydra
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from omegaconf import DictConfig, OmegaConf
import logging
from POCTrainer import *
from omegaconf import DictConfig
from hydra import initialize_config_dir, compose
from pathlib import Path
import os
from poc_timer import *



@hydra.main(version_base=None, config_path="../conf", config_name="base")
def main(cfg: DictConfig) -> None:

    logging.info(
        "Successfully loaded settings:\n"
        + "==================================================\n"
        f"{OmegaConf.to_yaml(cfg)}"
        + "==================================================\n"
    )

    trainer = POCTrainer()
    dl = trainer.setup_dataloader(cfg)
    print(cfg)    
    
    print(f"Starting {cfg.task}...")
    if cfg.task == "train":
        trainer.launch_training(dl)
    elif cfg.task == "test":
        trainer.test(module, test_dl, ckpt_path=cfg.ckpt)
    else:
        raise NotImplementedError

    return


if __name__ == "__main__":
    main()
