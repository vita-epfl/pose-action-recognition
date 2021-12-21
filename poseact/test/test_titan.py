import os
import json
import glob
import torch
import pickle
import argparse 

import numpy as np
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from typing import List, Set, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset
from poseact.models import MultiHeadMonoLoco, MonolocoModel
from poseact.utils.iou import get_iou_matches
from poseact.utils.losses import MultiHeadClfLoss
from multiprocessing import Pool

from poseact.utils.titan_dataset import (TITANDataset, 
                                         TITANSimpleDataset, 
                                         TITANSeqDataset,
                                         folder_names, 
                                         construct_from_pifpaf_results,
                                         get_titan_att_types,
                                         pickle_all_sequences,
                                         calc_anno_distribution,
                                         Sequence,
                                         Frame,
                                         Person,
                                         Vehicle)

def test_construct_dataset(args):
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir)
    dataset = TITANDataset(pifpaf_out, dataset_dir, split="train")
    construct_from_pifpaf_results(pifpaf_out, dataset_dir, save_dir, debug=True)
    dataset = TITANDataset(dataset_dir=dataset_dir, pickle_dir=save_dir, use_pickle=True, split="test")
    
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir, mode="track")
    construct_from_pifpaf_results(pifpaf_out, dataset_dir, save_dir, debug=True, mode="track")
    dataset = TITANDataset(dataset_dir=dataset_dir, pickle_dir=save_dir, use_pickle=True, split="test", mode="track")

def test_forward(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir)

    dataset = TITANDataset(dataset_dir=dataset_dir, pickle_dir=save_dir, use_pickle=True)
    simple_dataset = TITANSimpleDataset(dataset, merge_cls=args.merge_cls)
    dataloader = DataLoader(simple_dataset, batch_size=2, shuffle=True, collate_fn=TITANSimpleDataset.collate)

    model = MultiHeadMonoLoco(input_size=17*2, output_size=simple_dataset.n_cls).to(device)
    criterion = MultiHeadClfLoss(n_tasks=len(simple_dataset.n_cls), imbalance="focal", gamma=2, device=device)
    for poses, labels in dataloader:
        poses, labels = poses.to(device), labels.to(device)
        pred = model(poses)
        loss = criterion(pred, labels)
        print(loss)
        break

def test_seq_dataset(args):
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir)
    dataset = TITANDataset(dataset_dir=dataset_dir, pickle_dir=save_dir, use_pickle=True)
    seq_dataset = TITANSeqDataset(dataset)
    dataloader = DataLoader(seq_dataset, batch_size=2, shuffle=True, collate_fn=TITANSeqDataset.collate)
    
    for poses, labels in dataloader:
        print(poses.shape, labels.shape)
        break

def test_pad_pose():
    zeros = np.zeros((1, 17, 2))
    ones = np.ones((1, 17, 2))
    padded_pose = np.concatenate((zeros, zeros, 1*ones, 2*ones, zeros, zeros, 3*ones, 4*ones, zeros, zeros),axis=0)
    padded_label = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    is_not_zero = np.logical_not(np.all(padded_pose==0, axis=(1,2)))
    valid_frames = np.sum(is_not_zero) # now many none zero poses are there (non-empty)
    cumsum = np.cumsum(is_not_zero)
    is_leading_zeros = cumsum == 0
    is_trailing_zeros = cumsum == valid_frames
    last_pose = np.where(is_trailing_zeros)[0][0] # the last valid pose 
    is_trailing_zeros[last_pose] = False
    # remove the leading zeros and trailing zeros 
    not_leading_or_trailing = np.logical_not(np.logical_or(is_leading_zeros, is_trailing_zeros))
    # pad the mid zeros with previous poses before removing any zeros 
    is_zero = np.logical_not(is_not_zero)
    is_mid_zeros = np.logical_and(is_zero, not_leading_or_trailing)
    for mid_idx in np.where(is_mid_zeros)[0]: # find out the location of mid zeros
        padded_pose[mid_idx] = padded_pose[max(mid_idx-1, 0)] # set it to the previous pose
        padded_label[mid_idx] = -100
    filtered_pose = padded_pose[not_leading_or_trailing]
    filtered_label = padded_label[not_leading_or_trailing]
    correct_pose = np.concatenate((1*ones, 2*ones, 2*ones, 2*ones, 3*ones, 4*ones),axis=0)
    correct_label = np.array([3, 4, -100, -100, 7, 8])
    assert np.allclose(filtered_pose, correct_pose)
    assert np.allclose(filtered_label, correct_label)

def test_seq_forward(args):
    pass 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--base_dir", type=str, default="../", help="default root working directory")
    parser.add_argument("--function", type=str, default="None", help="which function to call")
    parser.add_argument("--merge-cls", action="store_true", help="remove unlearnable, merge 5 labels into one")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "track"],help="for making the pickle file")
    args = parser.parse_args() # ["--base_dir", "poseact/"]
    
    # test_pad_pose()
    test_construct_dataset(args)
    # test_forward(args)
    # test_seq_dataset(args)