""" common utility functions 
"""

import os 
import sys 
import ctypes
import torch 
import numpy as np 
import multiprocessing as mp 
from .losses import MultiHeadClfLoss

def setup_multiprocessing():
    mp.set_start_method('spawn')
    if sys.platform.startswith("linux"):
        try:
            libgcc_s = ctypes.CDLL("/usr/lib64/libgcc_s.so.1")
            print("loaded /usr/lib64/libgcc_s.so.1")
        except:
            pass 

def make_save_dir(base_dir, subdir_name, return_folder=False):
    assert os.path.isdir(base_dir), "base dir does not exits"
    subdir = base_dir + "/" + subdir_name
    if not os.path.exists(subdir):
        print("making folder at {}".format(subdir))
        os.mkdir(subdir)
    return subdir

def to_relative_coord(all_poses):
    """convert the key points from absolute coordinate to center+relative coordinate
        all_poses shape (n_samples, n_keypoints, n_dim)
        
    COCO_KEYPOINTS = [
        'nose',            # 0
        'left_eye',        # 1
        'right_eye',       # 2
        'left_ear',        # 3
        'right_ear',       # 4
        'left_shoulder',   # 5
        'right_shoulder',  # 6
        'left_elbow',      # 7
        'right_elbow',     # 8
        'left_wrist',      # 9
        'right_wrist',     # 10
        'left_hip',        # 11
        'right_hip',       # 12
        'left_knee',       # 13
        'right_knee',      # 14
        'left_ankle',      # 15
        'right_ankle',     # 16
    ]

    Args:
        all_poses (np.ndarray): pose array, size (N, V, C)

    Returns:
        converted_poses: size (N, V+1, C)
    """
    is_tensor = isinstance(all_poses, torch.Tensor)
    left_shoulder = all_poses[:, 5, :]
    right_shoulder = all_poses[:, 6, :]
    left_hip = all_poses[:, 11, :]
    right_hip = all_poses[:, 12, :]
    top_mid = 0.5*(left_shoulder + right_shoulder)
    bottom_mid = 0.5*(left_hip + right_hip)
    mid = 0.5*(top_mid + bottom_mid)
    if is_tensor:
        mid = mid.unsqueeze(1)
        relative_coord = all_poses - mid
        converted_poses = torch.cat((relative_coord, mid), dim=1)
    else:
        mid = np.expand_dims(mid, axis=1)
        relative_coord = all_poses - mid 
        converted_poses = np.concatenate((relative_coord, mid), axis=1)
    
    return converted_poses

def search_key(obj_dict:dict, value):
    """ search the name of action
    """
    for key in obj_dict.keys():
        if obj_dict[key] == value:
            return key
    raise ValueError("Unrecognized dict value")
