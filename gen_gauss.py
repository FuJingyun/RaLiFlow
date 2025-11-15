import torch
from torch.utils.data import DataLoader
# from lightning.pytorch.callbacks import ModelSummary
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import hydra, wandb, os, sys

from scripts.network.dataloader import VODH5Dataset
from pathlib import Path
from scripts.raliflow_pl_model import gen_gauss_ModelWrapper


@hydra.main(version_base=None, config_path="conf", config_name="gen_gauss") 
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = "/data/fjy/" + cfg.model_name + "-eval/" 
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mymodel = gen_gauss_ModelWrapper(cfg)
    wandb_logger = WandbLogger(save_dir=output_dir,
                               entity="kth-rpl",
                               project= cfg.model_name + "-eval",  
                               name=f"{cfg.output}",
                               offline=(cfg.wandb_mode == "offline"))
    
    trainer = pl.Trainer(logger=wandb_logger, accelerator='gpu',devices=cfg.available_gpus)

    trainer.test(model = mymodel, \
                dataloaders = DataLoader(VODH5Dataset(cfg.dataset_path + f"/{cfg.av2_mode}"),  batch_size=1,  shuffle=False))
            
    wandb.finish()

if __name__ == "__main__":
    main()