
"""
# Created: 2023-07-18 15:08
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""

import torch.nn as nn
# import torch.nn.functional as F
import dztimer, torch

from .basic.unet import vod_FastFlow3DUNet # FastFlow3DUNet
from .basic.my_embedder_model import pillar_DynamicEmbedder
from .basic.decoder import LinearDecoder, ConvGRUDecoder
from .basic import cal_pose0to1
import copy
from .basic.gauss_map import  gauss_map, gauss_attention

INDEX_SHIFT = [ [0,0], [-1,0],[1,0], [0,1],[-1,1],[1,1],[0,-1],[-1,-1],[1,-1] ]


class gen_gauss(nn.Module):
    def __init__(self, 
                voxel_size = [0.1, 0.1, 6],
                point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 3],
                 ):
        super().__init__()
        grid_feature_size = [abs(int((point_cloud_range[0] - point_cloud_range[3]) / voxel_size[0])),
                    abs(int((point_cloud_range[1] - point_cloud_range[4]) / voxel_size[1]))]
        
        self.gauss = gauss_map(voxel_size=voxel_size,
                                        pseudo_image_dims=grid_feature_size,
                                        point_cloud_range=point_cloud_range)
        self.embedder = pillar_DynamicEmbedder(voxel_size=voxel_size,
                                        pseudo_image_dims=grid_feature_size,
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=32,
                                        input_channels=3)
        
        self.radar_embedder = pillar_DynamicEmbedder(voxel_size=voxel_size,
                                        pseudo_image_dims=grid_feature_size,
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=32,
                                        input_channels=6)
        
        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")

        self.load_state_dict(state_dict=state_dict, strict=False)
        return self

    def forward(self, batch):
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """
        self.timer[0].start("Data Preprocess")
        batch_sizes = len(batch["pose0"])

        pose_flows = []
        radar_pose_flows = []
        transform_pc0s = []
        transform_radar_pc0s = []

        for batch_id in range(batch_sizes):
            selected_pc0 = batch["pc0"][batch_id]
            radar_pc0 = batch["radar_pc0"][batch_id]

            self.timer[0][0].start("pose")
            # Ego Motion
            with torch.no_grad():
                if 'ego_motion' in batch:
                    pose_0to1 = batch['ego_motion'][batch_id]
                else:
                    pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])
            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            # transform selected_pc0 to pc1
            # lidar
            transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            # radar
            transform_radar_pc0_xyz = radar_pc0[:,:3]  @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            transform_radar_pc0 = copy.deepcopy(radar_pc0)
            transform_radar_pc0[:,:3] = transform_radar_pc0_xyz

            self.timer[0][1].stop()
            # Ego motion compensated
            pose_flows.append(transform_pc0 - selected_pc0)
            radar_pose_flows.append(transform_radar_pc0_xyz - radar_pc0[:,:3])
            transform_pc0s.append(transform_pc0)
            transform_radar_pc0s.append(transform_radar_pc0)

        pc0s = torch.stack(transform_pc0s, dim=0)
        radar_pc0s = torch.stack(transform_radar_pc0s, dim=0)
        pc1s = batch["pc1"]
        radar_pc1s = batch["radar_pc1"]

        self.timer[0].stop()


        # For SEG train:
        self.timer[1].start("Voxelization")

        pc0_before_pseudoimages, pc0_voxel_infos_lst, pc0_point_feas , pc0_bev_feats , pc0_bev_coors = self.embedder(pc0s) # pc0_point_feas
        pc1_before_pseudoimages, pc1_voxel_infos_lst, _ , pc1_bev_feats , pc1_bev_coors = self.embedder(pc1s)

        radar_pc0_before_pseudoimages, radar_pc0_voxel_infos_lst,  \
            radar_pc0_point_feas , radar_pc0_bev_feats , radar_pc0_bev_coors = self.radar_embedder(radar_pc0s) # radar_pc0_point_feas
        radar_pc1_before_pseudoimages, radar_pc1_voxel_infos_lst,  \
            _ , radar_pc1_bev_feats , radar_pc1_bev_coors = self.radar_embedder(radar_pc1s)


        lidar_pc0_gauss_score, radar_pc0_gauss_score = self.gauss(pc0_bev_coors, radar_pc0_bev_coors, radar_pc0_voxel_infos_lst)
        lidar_pc1_gauss_score, radar_pc1_gauss_score = self.gauss(pc1_bev_coors, radar_pc1_bev_coors, radar_pc1_voxel_infos_lst)

   

        model_res = {
            "lidar_pc0_gauss_score": lidar_pc0_gauss_score,
            "radar_pc0_gauss_score": radar_pc0_gauss_score,
            "lidar_pc1_gauss_score": lidar_pc1_gauss_score,
            "radar_pc1_gauss_score": radar_pc1_gauss_score,
        }


        return model_res



class RaLiFlow(nn.Module):
    def __init__(self, 
                voxel_size = [0.1, 0.1, 6],
                point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 3],
                decoder_option = "gru",
                num_iters = 4,
                 ):
        super().__init__()
        grid_feature_size = [abs(int((point_cloud_range[0] - point_cloud_range[3]) / voxel_size[0])),
                     abs(int((point_cloud_range[1] - point_cloud_range[4]) / voxel_size[1]))]
        

        self.embedder = pillar_DynamicEmbedder(voxel_size=voxel_size,
                                        pseudo_image_dims=grid_feature_size,
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=32,
                                        input_channels=3)
        
        self.radar_embedder = pillar_DynamicEmbedder(voxel_size=voxel_size,
                                        pseudo_image_dims=grid_feature_size,
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=32,
                                        input_channels=6)
        
        
        
        self.bi_fusion_att = True   #False  True 
        if self.bi_fusion_att:
            self.bi_att = gauss_attention(pseudo_image_dims=grid_feature_size,
                                        feat_channels=32)     
      
        self.backbone = vod_FastFlow3DUNet()

        if decoder_option == "gru":
            self.head = ConvGRUDecoder(num_iters = num_iters)
            self.radar_head = ConvGRUDecoder(num_iters = num_iters)
        elif decoder_option == "linear":
            self.head = LinearDecoder()
            self.radar_head = LinearDecoder()

       
        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")

        self.load_state_dict(state_dict=state_dict, strict=False)
        return self

    def forward(self, batch):
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """
        self.timer[0].start("Data Preprocess")
        batch_sizes = len(batch["pose0"])

        pose_flows = []
        radar_pose_flows = []
        transform_pc0s = []
        transform_radar_pc0s = []

        for batch_id in range(batch_sizes):
            selected_pc0 = batch["pc0"][batch_id]
            radar_pc0 = batch["radar_pc0"][batch_id]

            self.timer[0][0].start("pose")
            with torch.no_grad():
                if 'ego_motion' in batch:
                    pose_0to1 = batch['ego_motion'][batch_id]
                else:
                    pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])
            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            # transform selected_pc0 to pc1
            # ego motion compensation lidar
            transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            # ego motion compensation radar
            transform_radar_pc0_xyz = radar_pc0[:,:3]  @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            transform_radar_pc0 = copy.deepcopy(radar_pc0)
            transform_radar_pc0[:,:3] = transform_radar_pc0_xyz

            self.timer[0][1].stop()
            pose_flows.append(transform_pc0 - selected_pc0)
            radar_pose_flows.append(transform_radar_pc0_xyz - radar_pc0[:,:3])
            transform_pc0s.append(transform_pc0)
            transform_radar_pc0s.append(transform_radar_pc0)

        pc0s = torch.stack(transform_pc0s, dim=0)
        radar_pc0s = torch.stack(transform_radar_pc0s, dim=0)
        pc1s = batch["pc1"]
        radar_pc1s = batch["radar_pc1"]
        self.timer[0].stop()
        

        self.timer[1].start("Voxelization")
        pc0_before_pseudoimages, pc0_voxel_infos_lst, pc0_point_feas , pc0_bev_feats , pc0_bev_coors = self.embedder(pc0s) 
        pc1_before_pseudoimages, pc1_voxel_infos_lst, _ , pc1_bev_feats , pc1_bev_coors = self.embedder(pc1s)

        radar_pc0_before_pseudoimages, radar_pc0_voxel_infos_lst,  \
            radar_pc0_point_feas , radar_pc0_bev_feats , radar_pc0_bev_coors = self.radar_embedder(radar_pc0s)
        radar_pc1_before_pseudoimages, radar_pc1_voxel_infos_lst,  \
            _ , radar_pc1_bev_feats , radar_pc1_bev_coors = self.radar_embedder(radar_pc1s)


        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_voxel_infos_lst]
        radar_pc0_valid_point_idxes = [e["point_idxes"] for e in radar_pc0_voxel_infos_lst]

        pc0_points_lst = [e["points"] for e in pc0_voxel_infos_lst]
        radar_pc0_points_lst = [e["points"] for e in radar_pc0_voxel_infos_lst]

        self.timer[1].stop()



        self.timer[2].start("Fusion_Attention")
        if self.bi_fusion_att:
            # [B 32 H W]
            pc0_ra2li_aug, pc0_li2ra_aug = self.bi_att(pc0_bev_feats , pc0_bev_coors, radar_pc0_bev_feats , radar_pc0_bev_coors,\
                                                        batch["lidar_pc0_gauss_score"], batch["radar_pc0_gauss_score"]) 
            pc1_ra2li_aug, pc1_li2ra_aug = self.bi_att(pc1_bev_feats , pc1_bev_coors, radar_pc1_bev_feats , radar_pc1_bev_coors,\
                                                        batch["lidar_pc1_gauss_score"], batch["radar_pc1_gauss_score"])
            
            # [B 32 H W]
            pc0_before_pseudoimages = pc0_before_pseudoimages + pc0_ra2li_aug
            pc1_before_pseudoimages = pc1_before_pseudoimages + pc1_ra2li_aug
            radar_pc0_before_pseudoimages = radar_pc0_before_pseudoimages + pc0_li2ra_aug
            radar_pc1_before_pseudoimages = radar_pc1_before_pseudoimages + pc1_li2ra_aug
        
        self.timer[2].stop()



        self.timer[3].start("Encoder")

        fusion_pc0_before_pseudoimages = torch.cat([pc0_before_pseudoimages, radar_pc0_before_pseudoimages], dim=1) 
        fusion_pc1_before_pseudoimages = torch.cat([pc1_before_pseudoimages, radar_pc1_before_pseudoimages], dim=1) 

        grid_flow_pseudoimage = self.backbone(fusion_pc0_before_pseudoimages,
                                            fusion_pc1_before_pseudoimages)

        self.timer[3].stop()


        self.timer[4].start("Decoder")
        flows = self.head(
            torch.cat((pc0_before_pseudoimages, pc1_before_pseudoimages),
                    dim=1), grid_flow_pseudoimage, pc0_voxel_infos_lst)
        
        radar_flows = self.radar_head(
            torch.cat((radar_pc0_before_pseudoimages, radar_pc1_before_pseudoimages),
                    dim=1), grid_flow_pseudoimage, radar_pc0_voxel_infos_lst)
        
        self.timer[4].stop()
        

        model_res = {

        "flow": flows,   
        'pose_flow': pose_flows,

        "radar_flow": radar_flows, 
        'radar_pose_flow': radar_pose_flows,

        "pc0_valid_point_idxes": pc0_valid_point_idxes,
        "radar_pc0_valid_point_idxes": radar_pc0_valid_point_idxes,

        "pc0_points_lst": pc0_points_lst,
        "radar_pc0_points_lst": radar_pc0_points_lst
        }

        return model_res
    
