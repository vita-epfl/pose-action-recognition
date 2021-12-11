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
                                         test_construct_dataset,
                                         test_forward,
                                         test_seq_dataset,
                                         Sequence,
                                         Frame,
                                         Person,
                                         Vehicle)

def test_construct_dataset(args):
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir)
    
    dataset = TITANDataset(pifpaf_out, dataset_dir, split="train")
    construct_from_pifpaf_results(pifpaf_out, dataset_dir, save_dir, debug=True)
    dataset = TITANDataset(dataset_dir=dataset_dir, pickle_dir=save_dir, use_pickle=True, split="test")

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
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--base_dir", type=str, default="../", help="default root working directory")
    parser.add_argument("--function", type=str, default="None", help="which function to call")
    parser.add_argument("--merge-cls", action="store_true", help="remove unlearnable, merge 5 labels into one")
    args = parser.parse_args() # ["--base_dir", "poseact/"]

    test_construct_dataset(args)
    test_forward(args)
    # test_seq_dataset(args)