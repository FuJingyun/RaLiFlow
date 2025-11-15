"""

# Created: 2023-11-05 10:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Model Wrapper for Pytorch Lightning

"""

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path

from lightning import LightningModule
from hydra.utils import instantiate
from omegaconf import OmegaConf,open_dict

import os, sys, time, h5py
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from scripts.utils.mics import import_func, weights_init, zip_res
from scripts.network.models.basic import cal_pose0to1
from scripts.network.official_metric import my_OfficialMetrics, evaluate_leaderboard, evaluate_leaderboard_v2






class gen_gauss_ModelWrapper(LightningModule):
    def __init__(self, cfg, eval=False):
        super().__init__()
        cfg.model.target['voxel_size'] = cfg.voxel_size
        cfg.model.target['point_cloud_range'] = cfg.point_cloud_range
        if ('voxel_size' in cfg.model.target) and ('point_cloud_range' in cfg.model.target) and not eval and 'point_cloud_range' in cfg:
            OmegaConf.set_struct(cfg.model.target, True)
                
        self.model = instantiate(cfg.model.target)
        self.model.apply(weights_init)

        self.epochs = cfg.epochs if 'epochs' in cfg else None
        

        if "vod_mode" in cfg:
            self.vod_mode = cfg.vod_mode

        if 'dataset_path' in cfg:
            self.dataset_path = cfg.dataset_path
            if 'av2_mode' in cfg:
                self.save_vis_path = cfg.dataset_path + f"{cfg.av2_mode}/"

        # self.vis_name = "RaLiFlow"    
        self.save_hyperparameters()

   
    def test_step(self, batch, batch_idx):
        key = str(batch['timestamp'][0])
        scene_id = str(batch['scene_id'][0])

        if self.vod_mode == "radar_lidar":
            batch['origin_pc0'] = batch['lidar_pc0'].clone()
            batch['pc0'] = batch['lidar_pc0']
            batch['pc1'] = batch['lidar_pc1']
            batch['radar_pc0'] = torch.cat([batch['radar_pc0'], batch['radar0_RCS'], batch['radar0_vr_compensated'], batch['radar0_time']], dim=2)
            batch['radar_pc1'] = torch.cat([batch['radar_pc1'], batch['radar1_RCS'], batch['radar1_vr_compensated'], batch['radar1_time']], dim=2)

            batch['origin_radar_pc0'] = batch['radar_pc0'].clone()
            self.model.timer[12].start("One Scan")
            res_dict = self.model(batch)
            self.model.timer[12].stop()

            batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
            res_dict = {key: res_dict[key][0] for key in res_dict if len(res_dict[key])>0}

            lidar_pc0_gauss_score = res_dict["lidar_pc0_gauss_score"]           
            lidar_pc1_gauss_score = res_dict["lidar_pc1_gauss_score"]
            radar_pc0_gauss_score = res_dict["radar_pc0_gauss_score"]
            radar_pc1_gauss_score = res_dict["radar_pc1_gauss_score"]

            

            with h5py.File(self.save_vis_path+ scene_id + ".h5", 'r+') as f:
                if "lidar_pc0_gauss_score" in f[key]:
                    del f[key]["lidar_pc0_gauss_score"]
                f[key].create_dataset("lidar_pc0_gauss_score", data=lidar_pc0_gauss_score.cpu().detach().numpy().astype(np.float32))
                if "radar_pc0_gauss_score" in f[key]:
                    del f[key]["radar_pc0_gauss_score"]
                f[key].create_dataset("radar_pc0_gauss_score", data=radar_pc0_gauss_score.cpu().detach().numpy().astype(np.float32))
                if "lidar_pc1_gauss_score" in f[key]:
                    del f[key]["lidar_pc1_gauss_score"]
                f[key].create_dataset("lidar_pc1_gauss_score", data=lidar_pc1_gauss_score.cpu().detach().numpy().astype(np.float32))
                if "radar_pc1_gauss_score" in f[key]:
                    del f[key]["radar_pc1_gauss_score"]
                f[key].create_dataset("radar_pc1_gauss_score", data=radar_pc1_gauss_score.cpu().detach().numpy().astype(np.float32))
                    
    def on_test_epoch_end(self):
        print(f"Successfully genearte gaussian scores!!!\n")




class Fusion_ModelWrapper(LightningModule):
    def __init__(self, cfg, eval=False):
        super().__init__()
        cfg.model.target['voxel_size'] = cfg.voxel_size
        cfg.model.target['point_cloud_range'] = cfg.point_cloud_range

        if ('voxel_size' in cfg.model.target) and ('point_cloud_range' in cfg.model.target) and not eval and 'point_cloud_range' in cfg:
            OmegaConf.set_struct(cfg.model.target, True)
                
        self.model = instantiate(cfg.model.target)
        self.model.apply(weights_init)

        self.loss_fn = import_func("scripts.network.loss_func."+cfg.loss_fn) if 'loss_fn' in cfg else None
        self.ra_loss_fn = import_func("scripts.network.loss_func."+cfg.loss_fn) if 'loss_fn' in cfg else None
        
        self.add_seloss = cfg.add_seloss if 'add_seloss' in cfg else None
        self.cfg_loss_name = cfg.loss_fn if 'loss_fn' in cfg else None

        self.batch_size = int(cfg.batch_size) if 'batch_size' in cfg else 1
        self.lr = cfg.lr if 'lr' in cfg else None
        self.epochs = cfg.epochs if 'epochs' in cfg else None
        
        self.metrics = my_OfficialMetrics(class_name= "Lidar")
        self.radar_metrics = my_OfficialMetrics(class_name= "Radar")

        self.box_loss = True
        if self.box_loss:
            self.dy_seg = import_func("scripts.network.loss_func.dy_seg_Loss")


        if 'checkpoint' in cfg:
            self.load_checkpoint_path = cfg.checkpoint

        if "vod_mode" in cfg:
            self.vod_mode = cfg.vod_mode

        if 'av2_mode' in cfg:
            self.av2_mode = cfg.av2_mode
        else:
            self.av2_mode = None
            if 'pretrained_weights' in cfg:
                if cfg.pretrained_weights is not None:
                    self.model.load_from_checkpoint(cfg.pretrained_weights)

        if 'dataset_path' in cfg:
            self.dataset_path = cfg.dataset_path
            if 'av2_mode' in cfg:
                self.save_vis_path = cfg.dataset_path + f"{cfg.av2_mode}/"

        self.vis_name = "RaLiFlow"    

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        res_dict = self.model(batch)

        # compute loss
        total_loss = 0.0
        batch_sizes = len(batch["pose0"])
        pose_flows = res_dict['pose_flow']
        radar_pose_flows = res_dict['radar_pose_flow']
        est_flow = res_dict['flow']
        radar_est_flow = res_dict['radar_flow']
        pc0_valid_idx = res_dict['pc0_valid_point_idxes'] # since padding
        radar_pc0_valid_idx = res_dict['radar_pc0_valid_point_idxes'] # since padding


        gt_lidar_dynamics = batch['lidar_motion_mask']
        gt_radar_dynamics = batch['radar_motion_mask']
        gt_lidar_ids = batch['lidar_id']
        gt_radar_ids = batch['radar_id']

      
        gt_flow = batch['flow']
        radar_gt_flow = batch['radar_flow']
        clean_radar_mask = batch['clean_radar_mask']
    
        for batch_id in range(batch_sizes):
            # Lidar part
            pc0_valid_from_pc2res = pc0_valid_idx[batch_id]
            pose_flow_ = pose_flows[batch_id][pc0_valid_from_pc2res]
            est_flow_ = est_flow[batch_id]  

            # compensate ego motion
            gt_flow_ = gt_flow[batch_id][pc0_valid_from_pc2res]
            gt_flow_ = gt_flow_ - pose_flow_         
            lidar_res_dict = {'est_flow': est_flow_, 
                        'gt_flow': gt_flow_, 
                        'gt_classes': None if 'flow_category_indices' not in batch else batch['flow_category_indices'][batch_id][pc0_valid_from_pc2res],
                        }
            
            gt_lidar_dynamic_mask = gt_lidar_dynamics[batch_id][pc0_valid_from_pc2res]
            gt_lidar_id = gt_lidar_ids[batch_id][pc0_valid_from_pc2res]
            
            # Radar part
            radar_pc0_valid_from_pc2res = radar_pc0_valid_idx[batch_id]
            # only calculate loss for high quality radar points
            clean_radar_mask_ = clean_radar_mask[batch_id][radar_pc0_valid_from_pc2res]

            radar_pose_flow_ = radar_pose_flows[batch_id][radar_pc0_valid_from_pc2res]
            radar_est_flow_ = radar_est_flow[batch_id]
            radar_gt_flow_ = radar_gt_flow[batch_id][radar_pc0_valid_from_pc2res]
            radar_gt_flow_ = radar_gt_flow_ - radar_pose_flow_
            gt_classes_ = batch['radar_flow_category_indices'][batch_id][radar_pc0_valid_from_pc2res]
            
            # radar_res_dict = {'est_flow': radar_est_flow_, 
            #             'gt_flow': radar_gt_flow_, 
            #             'gt_classes': None if 'radar_flow_category_indices' not in batch else gt_classes_,
            #             }
            radar_res_dict = {'est_flow': radar_est_flow_[clean_radar_mask_], 
                        'gt_flow': radar_gt_flow_[clean_radar_mask_], 
                        'gt_classes': None if 'radar_flow_category_indices' not in batch else gt_classes_[clean_radar_mask_],
                        }
            

            
            gt_radar_dynamic_mask = gt_radar_dynamics[batch_id][radar_pc0_valid_from_pc2res]
            gt_radar_id = gt_radar_ids[batch_id][radar_pc0_valid_from_pc2res]


            loss1 = self.loss_fn(lidar_res_dict) + self.ra_loss_fn(radar_res_dict)
            total_loss += loss1

            if self.box_loss:
                if (gt_lidar_dynamic_mask.shape[0]+gt_radar_dynamic_mask.shape[0])> 128:
                    dy_seg_res = {'est_flow': torch.cat((est_flow_[gt_lidar_dynamic_mask], radar_est_flow_[gt_radar_dynamic_mask]),dim=0),
                                'gt_flow':torch.cat((gt_flow_[gt_lidar_dynamic_mask], radar_gt_flow_[gt_radar_dynamic_mask]),dim=0),
                                'track_id':torch.cat((gt_lidar_id[gt_lidar_dynamic_mask], gt_radar_id[gt_radar_dynamic_mask]),dim=0)
                                }
            
                    loss2 = self.dy_seg(dy_seg_res)
                    total_loss += loss2
                   
   
        self.log("trainer/loss", total_loss/batch_sizes, sync_dist=True, batch_size=self.batch_size)
        return total_loss

    def train_validation_step_(self, batch, res_dict):
        if (batch['flow'][0].shape[0] > 0) & (batch['radar_flow'][0].shape[0] > 0):
            pose_flows = res_dict['pose_flow']
            radar_pose_flows = res_dict['radar_pose_flow']
            # Lidar 
            for batch_id, gt_flow in enumerate(batch["flow"]):
                valid_from_pc2res = res_dict['pc0_valid_point_idxes'][batch_id]
                pose_flow = pose_flows[batch_id][valid_from_pc2res]

                final_flow_ = pose_flow.clone() + res_dict['flow'][batch_id]
                v1_dict= evaluate_leaderboard(final_flow_, pose_flow, batch['pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                           batch['flow_is_valid'][batch_id][valid_from_pc2res], batch['flow_category_indices'][batch_id][valid_from_pc2res])
                v2_dict = evaluate_leaderboard_v2(final_flow_, pose_flow, batch['pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                        batch['flow_is_valid'][batch_id][valid_from_pc2res], batch['flow_category_indices'][batch_id][valid_from_pc2res])
                
                self.metrics.step(v1_dict, v2_dict)

            # Radar 
            for batch_id, gt_flow in enumerate(batch["radar_flow"]):
                valid_from_pc2res = res_dict['radar_pc0_valid_point_idxes'][batch_id]
                pose_flow = radar_pose_flows[batch_id][valid_from_pc2res]

                final_flow_ = pose_flow.clone() + res_dict['radar_flow'][batch_id]
                v1_dict= evaluate_leaderboard(final_flow_, pose_flow, batch['radar_pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                           batch['radar_flow_is_valid'][batch_id][valid_from_pc2res], batch['radar_flow_category_indices'][batch_id][valid_from_pc2res])
                v2_dict = evaluate_leaderboard_v2(final_flow_, pose_flow, batch['radar_pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                        batch['radar_flow_is_valid'][batch_id][valid_from_pc2res], batch['radar_flow_category_indices'][batch_id][valid_from_pc2res])
                
                self.radar_metrics.step(v1_dict, v2_dict)
        else:
            pass
        
    def on_validation_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)
        
        if self.av2_mode == 'val':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
            print(f"More details parameters and training status are in checkpoints")        

        self.metrics.normalize()
        self.radar_metrics.normalize()

        # wandb log things:
        for key in self.metrics.bucketed:
            for type_ in 'Static', 'Dynamic':
                self.log(f"Lidar val/{type_}/{key}", self.metrics.bucketed[key][type_],sync_dist=True)
        for key in self.metrics.epe_3way:
            self.log(f"Lidar val/{key}", self.metrics.epe_3way[key],sync_dist=True)

        for key in self.radar_metrics.bucketed:
            for type_ in 'Static', 'Dynamic':
                self.log(f"Radar val/{type_}/{key}", self.radar_metrics.bucketed[key][type_],sync_dist=True)
        for key in self.radar_metrics.epe_3way:
            self.log(f"Radar val/{key}", self.radar_metrics.epe_3way[key],sync_dist=True)
        
        self.metrics.print()
        self.radar_metrics.print()

        # For model monitor 
        Total_EPE_3way = self.metrics.epe_3way['Three-way'] + self.radar_metrics.epe_3way['Three-way']
        self.log(f"Total_EPE_3way", Total_EPE_3way,sync_dist=True)
        
        # Renew
        self.metrics = my_OfficialMetrics(class_name= "Lidar")
        self.radar_metrics = my_OfficialMetrics(class_name= "Radar")
        
    def eval_only_step_(self, batch, res_dict):
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if len(res_dict[key])>0}


        if self.vod_mode == "radar_lidar":
            pc0 = batch['origin_pc0']
            radar_pc0 = batch['origin_radar_pc0']
            pose_flow = res_dict['pose_flow']
            radar_pose_flow = res_dict['radar_pose_flow']

            if 'pc0_valid_point_idxes' in res_dict:
                valid_from_pc2res = res_dict['pc0_valid_point_idxes']

                # flow in the original pc0 coordinate
                pred_flow = pose_flow.clone()
                pred_flow[valid_from_pc2res] = pose_flow[valid_from_pc2res] + res_dict['flow']

                final_flow = pred_flow

            if 'radar_pc0_valid_point_idxes' in res_dict:
                radar_valid_from_pc2res = res_dict['radar_pc0_valid_point_idxes']

                # flow in the original pc0 coordinate
                radar_pred_flow = radar_pose_flow.clone()
                radar_pred_flow[radar_valid_from_pc2res] = radar_pose_flow[radar_valid_from_pc2res] + res_dict['radar_flow']

                radar_final_flow = radar_pred_flow


            if self.av2_mode == 'val': # since only val we have ground truth flow to eval
                gt_flow = batch["lidar_flow"]
                v1_dict = evaluate_leaderboard(final_flow[valid_from_pc2res], pose_flow[valid_from_pc2res], pc0[valid_from_pc2res], \
                                                gt_flow[valid_from_pc2res], batch['lidar_flow_valid'][valid_from_pc2res], \
                                                batch['lidar_flow_category'][valid_from_pc2res])
                
                v2_dict = evaluate_leaderboard_v2(final_flow[valid_from_pc2res], pose_flow[valid_from_pc2res], pc0[valid_from_pc2res], \
                                                gt_flow[valid_from_pc2res], batch['lidar_flow_valid'][valid_from_pc2res], \
                                                    batch['lidar_flow_category'][valid_from_pc2res])                
                self.metrics.step(v1_dict, v2_dict)

                
                gt_flow = batch["radar_flow"]
                v1_dict = evaluate_leaderboard(radar_final_flow[radar_valid_from_pc2res], radar_pose_flow[radar_valid_from_pc2res],\
                                                   radar_pc0[radar_valid_from_pc2res], gt_flow[radar_valid_from_pc2res], batch['radar_flow_valid'][radar_valid_from_pc2res],\
                                                      batch['radar_flow_category'][radar_valid_from_pc2res])
                
                v2_dict = evaluate_leaderboard_v2(radar_final_flow[radar_valid_from_pc2res], radar_pose_flow[radar_valid_from_pc2res],\
                                                   radar_pc0[radar_valid_from_pc2res], gt_flow[radar_valid_from_pc2res], batch['radar_flow_valid'][radar_valid_from_pc2res],\
                                                      batch['radar_flow_category'][radar_valid_from_pc2res])
                
                self.radar_metrics.step(v1_dict, v2_dict)
                    
    def validation_step(self, batch, batch_idx):
        if self.av2_mode == 'val' or self.av2_mode == 'test':
            if self.vod_mode == "lidar_only":
                batch['origin_pc0'] = batch['lidar_pc0'].clone()
                batch['pc0'] = batch['lidar_pc0']
                batch['pc1'] = batch['lidar_pc1']
                self.model.timer[12].start("One Scan")
                res_dict = self.model(batch)
                self.model.timer[12].stop()
                self.eval_only_step_(batch, res_dict)
            
            if self.vod_mode == "radar_lidar":
                batch['origin_pc0'] = batch['lidar_pc0'].clone()
                batch['pc0'] = batch['lidar_pc0']
                batch['pc1'] = batch['lidar_pc1']
                # batch['radar_pc0'] = torch.cat([batch['radar_pc0'], batch['radar0_RCS'],  batch['radar0_vr_compensated'], batch['radar0_time']], dim=2)
                # batch['radar_pc1'] = torch.cat([batch['radar_pc1'], batch['radar1_RCS'],  batch['radar1_vr_compensated'], batch['radar1_time']], dim=2)
                batch['radar_pc0'] = torch.cat([batch['radar_pc0'], batch['radar0_vr'],  batch['radar0_vr_compensated'], batch['radar0_time']], dim=2)
                batch['radar_pc1'] = torch.cat([batch['radar_pc1'], batch['radar1_vr'],  batch['radar1_vr_compensated'], batch['radar1_time']], dim=2)
                batch['origin_radar_pc0'] = batch['radar_pc0'].clone()
                self.model.timer[12].start("One Scan")
                res_dict = self.model(batch)
                self.model.timer[12].stop()
                self.eval_only_step_(batch, res_dict)
        else:
            res_dict = self.model(batch)
            self.train_validation_step_(batch, res_dict)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer


    # for VIS
    def test_step(self, batch, batch_idx):
        if self.av2_mode == 'val' or self.av2_mode == 'test':
            key = str(batch['timestamp'][0])
            scene_id = str(batch['scene_id'][0])

            if self.vod_mode == "radar_lidar":
                batch['origin_pc0'] = batch['lidar_pc0'].clone()
                batch['pc0'] = batch['lidar_pc0']
                batch['pc1'] = batch['lidar_pc1']
                batch['radar_pc0'] = torch.cat([batch['radar_pc0'], batch['radar0_RCS'], batch['radar0_vr_compensated'], batch['radar0_time']], dim=2)
                batch['radar_pc1'] = torch.cat([batch['radar_pc1'], batch['radar1_RCS'], batch['radar1_vr_compensated'], batch['radar1_time']], dim=2)
                batch['origin_radar_pc0'] = batch['radar_pc0'].clone()
                self.model.timer[12].start("One Scan")
                res_dict = self.model(batch)
                self.model.timer[12].stop()

                batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
                res_dict = {key: res_dict[key][0] for key in res_dict if len(res_dict[key])>0}

                lidar_pc0_gauss_score = res_dict["lidar_pc0_gauss_score"]
                radar_pc0_gauss_score = res_dict["radar_pc0_gauss_score"]
                lidar_pc1_gauss_score = res_dict["lidar_pc1_gauss_score"]
                radar_pc1_gauss_score = res_dict["radar_pc1_gauss_score"]


                with h5py.File(self.save_vis_path+ scene_id + ".h5", 'r+') as f:
                    if "lidar_pc0_gauss_score" in f[key]:
                        del f[key]["lidar_pc0_gauss_score"]
                    f[key].create_dataset("lidar_pc0_gauss_score", data=lidar_pc0_gauss_score.cpu().detach().numpy().astype(np.float32))
                    if "radar_pc0_gauss_score" in f[key]:
                        del f[key]["radar_pc0_gauss_score"]
                    f[key].create_dataset("radar_pc0_gauss_score", data=radar_pc0_gauss_score.cpu().detach().numpy().astype(np.float32))
                    if "lidar_pc1_gauss_score" in f[key]:
                        del f[key]["lidar_pc1_gauss_score"]
                    f[key].create_dataset("lidar_pc1_gauss_score", data=lidar_pc1_gauss_score.cpu().detach().numpy().astype(np.float32))
                    if "radar_pc1_gauss_score" in f[key]:
                        del f[key]["radar_pc1_gauss_score"]
                    f[key].create_dataset("radar_pc1_gauss_score", data=radar_pc1_gauss_score.cpu().detach().numpy().astype(np.float32))
                    
                  
                 
                    

    def on_test_epoch_end(self):
        print(f"\n\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
        print(f"We already write the estimate flow: {self.vis_name} into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
        print(f"python tests/scene_flow.py --flow_mode '{self.vis_name}' --data_dir {self.dataset_path}")
        print(f"Enjoy! ^v^ ------ \n")




