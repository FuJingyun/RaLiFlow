import torch
import torch.nn as nn

from .make_voxels import  DynamicVoxelizer  # HardVoxelizer,
from .process_voxels import DynamicPillarFeatureNet
from .scatter import rali_PointPillarsScatter



INDEX_SHIFT = [ [0,0], [-1,0],[1,0], [0,1],[-1,1],[1,1],[0,-1],[-1,-1],[1,-1] ]


class pillar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, input_channels: int) -> None:
        super().__init__()

        self.pseudo_image_dims = pseudo_image_dims
        
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=input_channels,
            feat_channels=(feat_channels,),  #  int(0.5*feat_channels), feat_channels,  # feat_channels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg') # 'avg'
        
        self.scatter = rali_PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []
        out_bev_feats_lst = []
        out_bev_coors_lst = []

        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
           
            # [ 0, 113, 325],
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0, 464,  99 ]
            # point_feats : torch.Size([ N_points , 32])
            pt_fea_lst.append(point_feats)

                   
            # [N, 2]
            bev_coors = voxel_coors[:,1:]
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)

            out_bev_feats_lst.append(voxel_feats)
            out_bev_coors_lst.append(bev_coors)
            
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst


