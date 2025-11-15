"""
# Created: 2023-11-04 15:52
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Torch dataloader for the dataset we preprocessed.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py, os, pickle, argparse
# from tqdm import tqdm
# import numpy as np


def fusion_vod_collate_fn_pad(batch):
    # padding the data
    pc0_after_mask_ground, pc1_after_mask_ground, radar_pc0, radar_pc1= [], [], [], []
    for i in range(len(batch)):
        pc0_after_mask_ground.append(batch[i]['lidar_pc0'])
        pc1_after_mask_ground.append(batch[i]['lidar_pc1'])
        
        # temp_radar_pc0 = torch.cat([batch[i]['radar_pc0'], batch[i]['radar0_RCS'], batch[i]['radar0_vr_compensated'], batch[i]['radar0_time']], dim=1)
        # radar_pc0.append(temp_radar_pc0) 
        # temp_radar_pc1 = torch.cat([batch[i]['radar_pc1'], batch[i]['radar1_RCS'], batch[i]['radar1_vr_compensated'], batch[i]['radar1_time']], dim=1) 
        # radar_pc1.append(temp_radar_pc1)
        temp_radar_pc0 = torch.cat([batch[i]['radar_pc0'], batch[i]['radar0_vr'], batch[i]['radar0_vr_compensated'], batch[i]['radar0_time']], dim=1)
        radar_pc0.append(temp_radar_pc0) 
        temp_radar_pc1 = torch.cat([batch[i]['radar_pc1'], batch[i]['radar1_vr'], batch[i]['radar1_vr_compensated'], batch[i]['radar1_time']], dim=1) 
        radar_pc1.append(temp_radar_pc1)

    pc0_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_after_mask_ground, batch_first=True, padding_value=torch.nan)
    radar_pc0 = torch.nn.utils.rnn.pad_sequence(radar_pc0, batch_first=True, padding_value=torch.nan)
    radar_pc1 = torch.nn.utils.rnn.pad_sequence(radar_pc1, batch_first=True, padding_value=torch.nan)

    res_dict =  {
        'pc0': pc0_after_mask_ground,
        'pc1': pc1_after_mask_ground,
        'radar_pc0' : radar_pc0,
        'radar_pc1' : radar_pc1,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))]
    }


    if 'lidar_flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['lidar_flow'] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['lidar_flow_valid'] for i in range(len(batch))], batch_first=True)
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['lidar_flow_category'] for i in range(len(batch))], batch_first=True)
        lidar_motion_mask = torch.nn.utils.rnn.pad_sequence([batch[i]['lidar_motion_mask'] for i in range(len(batch))], batch_first=True)
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices
        res_dict['lidar_motion_mask'] = lidar_motion_mask

    if 'radar_flow' in batch[0]:
        radar_flow = torch.nn.utils.rnn.pad_sequence([batch[i]['radar_flow'] for i in range(len(batch))], batch_first=True)
        radar_flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['radar_flow_valid'] for i in range(len(batch))], batch_first=True)
        radar_flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['radar_flow_category'] for i in range(len(batch))], batch_first=True)
        radar_motion_mask = torch.nn.utils.rnn.pad_sequence([batch[i]['radar_motion_mask'] for i in range(len(batch))], batch_first=True)
        clean_radar_mask = torch.nn.utils.rnn.pad_sequence([batch[i]['clean_radar_mask'] for i in range(len(batch))], batch_first=True)
        res_dict['radar_flow'] = radar_flow
        res_dict['radar_flow_is_valid'] = radar_flow_is_valid
        res_dict['radar_flow_category_indices'] = radar_flow_category_indices
        res_dict['radar_motion_mask'] = radar_motion_mask
        res_dict['clean_radar_mask'] = clean_radar_mask

    if ("lidar_dy" in batch[0]) and ("radar_dy" in batch[0]):
        lidar_dy = torch.nn.utils.rnn.pad_sequence([batch[i]['lidar_dy'] for i in range(len(batch))], batch_first=True)
        radar_dy = torch.nn.utils.rnn.pad_sequence([batch[i]['radar_dy'] for i in range(len(batch))], batch_first=True)
        res_dict['lidar_dy'] = lidar_dy
        res_dict['radar_dy'] = radar_dy

    if "lidar_id" in batch[0]:
        lidar_id = torch.nn.utils.rnn.pad_sequence([batch[i]['lidar_id'] for i in range(len(batch))], batch_first=True)
        res_dict['lidar_id'] = lidar_id
    if "radar_id" in batch[0]:
        radar_id = torch.nn.utils.rnn.pad_sequence([batch[i]['radar_id'] for i in range(len(batch))], batch_first=True)      
        res_dict['radar_id'] = radar_id
        
    if "pc1_lidar_id" in batch[0]:
        pc1_lidar_id = torch.nn.utils.rnn.pad_sequence([batch[i]['pc1_lidar_id'] for i in range(len(batch))], batch_first=True)
        res_dict['pc1_lidar_id'] = pc1_lidar_id
    if "pc1_radar_id" in batch[0]:
        pc1_radar_id = torch.nn.utils.rnn.pad_sequence([batch[i]['pc1_radar_id'] for i in range(len(batch))], batch_first=True)
        res_dict['pc1_radar_id'] = pc1_radar_id

    
    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    if 'lidar_pc0_gauss_score' in batch[0]:
        res_dict['lidar_pc0_gauss_score'] = [batch[i]['lidar_pc0_gauss_score'] for i in range(len(batch))]
        res_dict['radar_pc0_gauss_score'] = [batch[i]['radar_pc0_gauss_score'] for i in range(len(batch))]
        res_dict['lidar_pc1_gauss_score'] = [batch[i]['lidar_pc1_gauss_score'] for i in range(len(batch))]
        res_dict['radar_pc1_gauss_score'] = [batch[i]['radar_pc1_gauss_score'] for i in range(len(batch))]

    return res_dict






class VODH5Dataset(Dataset):
    def __init__(self, directory):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(VODH5Dataset, self).__init__()
        self.directory = directory
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
               
        self.scene_id_bounds = {}  
        # scene_id max timestamp
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": int(timestamp),
                    "max_timestamp": int(timestamp),
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # min timestamp
                if int(timestamp) < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = int(timestamp)
                    bounds["min_index"] = idx
                # max timestamps
                if int(timestamp) > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = int(timestamp)
                    bounds["max_index"] = idx

    def __len__(self):
        return len(self.data_index)

    
    def __getitem__(self, index_):
        scene_id, timestamp = self.data_index[index_]
        # to make sure we have continuous frames
        if self.scene_id_bounds[scene_id]["max_index"] == index_:
            index_ = index_ - 1
        # get the data again
        scene_id, timestamp = self.data_index[index_]


        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            lidar_pc0 = torch.tensor(f[key]['lidar_pc'][:,:3])
            radar_pc0 = torch.tensor(f[key]['radar_pc'][:,:3])
            radar0_RCS = torch.tensor(f[key]['radar_pc'][:,3].reshape(-1,1)) # Radar Cross Section
            radar0_vr = torch.tensor(f[key]['radar_pc'][:,4].reshape(-1,1))
            radar0_vr_compensated = torch.tensor(f[key]['radar_pc'][:,5].reshape(-1,1))
            radar0_time = torch.tensor(f[key]['radar_pc'][:,6].reshape(-1,1))
            pose0 = torch.tensor(f[key]['pose'][:])

            next_timestamp = str(self.data_index[index_+1][1])   # next_scene_id, next_timestamp = self.data_index[index_+1]
            lidar_pc1 = torch.tensor(f[next_timestamp]['lidar_pc'][:,:3])
            radar_pc1 = torch.tensor(f[next_timestamp]['radar_pc'][:,:3])
            radar1_RCS = torch.tensor(f[next_timestamp]['radar_pc'][:,3].reshape(-1,1)) # Radar Cross Section
            radar1_vr = torch.tensor(f[next_timestamp]['radar_pc'][:,4].reshape(-1,1))
            radar1_vr_compensated = torch.tensor(f[next_timestamp]['radar_pc'][:,5].reshape(-1,1))
            radar1_time = torch.tensor(f[next_timestamp]['radar_pc'][:,6].reshape(-1,1))
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])


            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,

                'lidar_pc0': lidar_pc0,
                'radar_pc0': radar_pc0,
                'radar0_RCS': radar0_RCS,
                'radar0_vr_compensated': radar0_vr_compensated,
                'radar0_vr': radar0_vr,
                'radar0_time' : radar0_time,
                'pose0': pose0,

                'lidar_pc1': lidar_pc1,
                'radar_pc1': radar_pc1,
                'radar1_RCS': radar1_RCS,
                'radar1_vr_compensated': radar1_vr_compensated,
                'radar1_vr': radar1_vr,
                'radar1_time' : radar1_time,
                'pose1': pose1,
            }

            if 'lidar_flow' in f[key]:
                lidar_flow = torch.tensor(f[key]['lidar_flow'][:])
                lidar_flow_valid = torch.tensor(f[key]['lidar_flow_valid'][:])
                lidar_flow_category = torch.tensor(f[key]['lidar_flow_category'][:])
                lidar_motion_mask = torch.tensor(f[key]['lidar_motion_mask'][:])

                res_dict['lidar_flow'] = lidar_flow
                res_dict['lidar_flow_valid'] = lidar_flow_valid
                res_dict['lidar_flow_category'] = lidar_flow_category
                res_dict['lidar_motion_mask'] = lidar_motion_mask

            
            if 'radar_flow' in f[key]:
                radar_flow = torch.tensor(f[key]['radar_flow'][:])
                radar_flow_valid = torch.tensor(f[key]['radar_flow_valid'][:])
                radar_flow_category = torch.tensor(f[key]['radar_flow_category'][:])
                radar_motion_mask = torch.tensor(f[key]['radar_motion_mask'][:])
                clean_radar_mask = torch.tensor(f[key]['clean_radar_mask'][:])

                res_dict['radar_flow'] = radar_flow
                res_dict['radar_flow_valid'] = radar_flow_valid
                res_dict['radar_flow_category'] = radar_flow_category
                res_dict['radar_motion_mask'] = radar_motion_mask
                res_dict['clean_radar_mask'] = clean_radar_mask


            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion

            if "lidar_id" in f[key]:
                res_dict['lidar_id'] = torch.tensor(f[key]["lidar_id"][:])
            if "radar_id" in f[key]:
                res_dict['radar_id'] = torch.tensor(f[key]["radar_id"][:])
            
            if "lidar_id" in f[next_timestamp]:
                res_dict['pc1_lidar_id'] = torch.tensor(f[next_timestamp]["lidar_id"][:])
            if "radar_id" in f[next_timestamp]:
                res_dict['pc1_radar_id'] = torch.tensor(f[next_timestamp]["radar_id"][:])


            if ("lidar_pc0_gauss_score" in f[key]) and ("radar_pc0_gauss_score" in f[key]):
                res_dict['lidar_pc0_gauss_score'] = torch.tensor(f[key]['lidar_pc0_gauss_score'][:])
                res_dict['radar_pc0_gauss_score'] = torch.tensor(f[key]['radar_pc0_gauss_score'][:])
            
            if ("lidar_pc1_gauss_score" in f[key]) and ("radar_pc1_gauss_score" in f[key]):
                res_dict['lidar_pc1_gauss_score'] = torch.tensor(f[key]['lidar_pc1_gauss_score'][:])
                res_dict['radar_pc1_gauss_score'] = torch.tensor(f[key]['radar_pc1_gauss_score'][:])

        return res_dict








