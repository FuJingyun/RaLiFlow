import numpy as np
import open3d as o3d
# from sklearn.neighbors import BallTree
# import time


def cal_grid_index(row,col,grid_size):
    mesh_size = int (51.2 / grid_size)
    grid_index = int(row * mesh_size + col)
    return grid_index


def is_in_mesh(row,col,grid_size):
    mesh_size = int (51.2 / grid_size)
    check =  (row > (mesh_size-1) or row < 0 or col < 0 or col > (mesh_size-1))
    return ~check

def get_neighbor(grid_index,grid_size):
    mesh_size = int (51.2 / grid_size)
    neighbor = [grid_index]
    row = int(grid_index / mesh_size)
    col = int(grid_index % mesh_size)
    # 1 
    temp_row = row-1
    temp_col = col-1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 2 
    temp_row = row
    temp_col = col-1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 3 
    temp_row = row+1
    temp_col = col-1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 4 
    temp_row = row-1
    temp_col = col
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 5 
    temp_row = row+1
    temp_col = col
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 6 
    temp_row = row-1
    temp_col = col+1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 7 
    temp_row = row
    temp_col = col+1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 8 
    temp_row = row+1
    temp_col = col+1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    return neighbor


def remove_under_ground(xyz, ground_h):
    val_inds = (xyz[:, 2] > ground_h-1)
    velo_valid = xyz[val_inds, :]
    return velo_valid

def remove_height(xyz):
    val_inds = (xyz[:, 2] > -3)
    val_inds = val_inds &  (xyz[:, 2] < 3)
    velo_valid = xyz[val_inds, :]
    return velo_valid


def in_region(xyz):
    val_inds = (xyz[:, 0] > 0 )
    val_inds = val_inds & (xyz[:, 0] < 51.2 )
    val_inds = val_inds & (xyz[:, 1] > -25.6 )
    val_inds = val_inds & (xyz[:, 1] < 25.6 )
    val_inds = val_inds & (xyz[:, 2] < 3 )
    val_inds = val_inds & (xyz[:, 2] > -3 )
    # remove_near_car
    invalid_inds = (xyz[:, 0] <3) & (xyz[:, 0] >0) & (xyz[:, 1] <1.5) & (xyz[:, 1] >-1.5)
    val_inds = val_inds & (~invalid_inds)
    velo_valid = xyz[val_inds, :]
    return velo_valid



#cam00
def remove_xy(xyz):
    value = 51.2
    val_inds = (xyz[:, 1] > -value )
    val_inds = val_inds & (xyz[:, 1] < value )
    val_inds = val_inds & (xyz[:, 0] > -value )
    val_inds = val_inds & (xyz[:, 0] < value )
    velo_valid = xyz[val_inds, :]
    velo_invalid = xyz[~val_inds, :]

    return velo_valid, velo_invalid


def remove_z(xyz):
    val_inds =  (xyz[:, 2] <0.5)
    velo_valid = xyz[val_inds, :]
    velo_invalid = xyz[~val_inds, :]

    return velo_valid, velo_invalid

def read_calib(file):
    with open(file, "r") as f:
        lines = f.readlines()
        intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
        extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
        extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)

    return intrinsic, extrinsic

def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    This function applies the homogenous transform using the dot product.
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    return transform.dot(points.T).T




def lidar_bin_filter(valid_radar, valid_lidar, grid_size):
    mesh={}
    for lidar_index in range(valid_lidar.shape[0]):
        row = int((valid_lidar[lidar_index,0]) / grid_size)
        col = int((25.6 + valid_lidar[lidar_index,1])/ grid_size)
        grid_index = cal_grid_index(row,col,grid_size)
        if (str(grid_index) not in mesh.keys()):
            mesh[str(grid_index)] = 1
        else:
            mesh[str(grid_index)]+=1
    filtered_by_lidar =  np.zeros(valid_radar.shape[0])
    for radar_index in range(valid_radar.shape[0]):
        row = int((valid_radar[radar_index,0]) / grid_size)
        col = int((25.6 + valid_radar[radar_index,1])/ grid_size)
        grid_index = cal_grid_index(row,col,grid_size)
        neighbor = get_neighbor(grid_index,grid_size)
        count_neighbor = 0
        for neighbor_grid in neighbor:
            if(str(neighbor_grid)in mesh.keys()):
                count_neighbor += mesh[str(neighbor_grid)]
        if count_neighbor>1:
            filtered_by_lidar[radar_index] = 1
    lidar_filtered_mask = np.array(filtered_by_lidar, dtype= bool)
    lidar_filtered_radar = valid_radar[lidar_filtered_mask,:]
    return lidar_filtered_radar



