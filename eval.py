import torch
from torch.utils.data import DataLoader
# from lightning.pytorch.callbacks import ModelSummary
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra, wandb, os, sys
# from hydra.core.hydra_config import HydraConfig
from scripts.network.dataloader import VODH5Dataset
from pathlib import Path
from scripts.raliflow_pl_model import Fusion_ModelWrapper
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@hydra.main(version_base=None, config_path="conf", config_name="fusion_eval") 
def main(cfg):
    # Random Seed
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = "/data/fjy/" + cfg.model_name + "-eval/" # 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
    
    
    checkpoint_params = DictConfig(torch.load(cfg.checkpoint)["hyper_parameters"])
    cfg.output = checkpoint_params.cfg.output + f"-{cfg.av2_mode}"
    # cfg.model.update(checkpoint_params.cfg.model)

    if cfg.dataset_name == "VOD":
        if cfg.model_name == "FusionFlow":
            if cfg.estimate_sf == "radar_and_lidar" :
                mymodel = Fusion_ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)
        else:
            raise ValueError("Unavailable cfg.model_name")

    wandb_logger = WandbLogger(save_dir=output_dir,
                               entity="kth-rpl",
                               project= cfg.model_name + "-eval",   # f"fastflow3d-eval"
                               name=f"{cfg.output}",
                               offline=(cfg.wandb_mode == "offline"))
    
    
    trainer = pl.Trainer(logger=wandb_logger, accelerator='gpu',devices=cfg.available_gpus)

    

    if cfg.dataset_name == "VOD":
        trainer.validate(model = mymodel, \
                    dataloaders = DataLoader(VODH5Dataset(cfg.dataset_path + f"/{cfg.av2_mode}"), batch_size=1, num_workers=cfg.num_workers, shuffle=False))

    wandb.finish()

if __name__ == "__main__":
    main()