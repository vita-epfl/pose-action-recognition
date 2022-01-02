""" Intention Recognition of Pedestrians and Cyclists by 2D Pose Estimation 
    https://ieeexplore.ieee.org/abstract/document/8876650 arXiv:1910.03858 
"""

import os
import re 
import json
import glob
import torch
import pickle
import argparse 
import itertools

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from typing import List, Set, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset
from poseact.utils import to_relative_coord, search_key
from poseact.utils.iou import get_iou_matches
from poseact.utils.losses import IGNORE_INDEX
from multiprocessing import Pool

np.set_printoptions(precision=3, suppress=True)

class Person:
    
    action_dict = {"left":0, "right":1, "stop":2, "none":3}
    
    def __init__(self, pred, gt_anno) -> None:
        self.frame = gt_anno["frame_idx"]
        self.action = gt_anno["left_or_right"]
        self.gt_bbox = gt_anno["bbox_gt"]
        if pred is None:
            self.key_points = np.zeros((17, 2))
            self.kp_confidence = np.zeros(17)
            self.pred_box = [0, 0, 0, 0] 
        else:
            x_y_conf = np.array(pred['keypoints']).reshape(-1, 3)
            self.key_points = x_y_conf[:, :2] # x, y
            self.kp_confidence = x_y_conf[:, 2]
            self.pred_box = pred['bbox']
    
class Sequence(object):
    
    def __init__(self, seq_name:str) -> None:
        super().__init__()
        self.seq_name = seq_name
        video_info = [int(num) for num in re.findall(r"\d+\.?\d*", self.seq_name)]
        if len(video_info) == 3:
            self.cyclist, self.style, self.num = video_info
            self.is_yt = False
        else:
            self.is_yt = True
            self.num = video_info[0]
        self.persons: List[Person] = []
        
    @property
    def seq_len(self):
        return len(self.persons)
    
    def to_tensor(self):
        
        padded_pose = np.zeros((len(self.persons), 17, 2), dtype=np.float32)
        padded_label = IGNORE_INDEX*np.ones(len(self.persons), dtype=np.int)
        
        for idx, person in enumerate(self.persons):
            padded_pose[idx] = person.key_points
            padded_label[idx] = person.action
            
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
        # (the person is out of view for some reason and appear again sometime afterwards)
        is_zero = np.logical_not(is_not_zero)
        is_mid_zeros = np.logical_and(is_zero, not_leading_or_trailing)
        for mid_idx in np.where(is_mid_zeros)[0]: # find out the location of mid zeros
            padded_pose[mid_idx] = padded_pose[max(mid_idx-1, 0)] # set it to the previous pose (or the first)
            padded_label[mid_idx] = IGNORE_INDEX # set to default ignore index of cross entropy loss 
        filtered_pose = torch.tensor(padded_pose[not_leading_or_trailing], dtype=torch.float32)
        filtered_label = torch.tensor(padded_label[not_leading_or_trailing], dtype=torch.long)
        filtered_pose = to_relative_coord(filtered_pose)
        
        return (filtered_pose, filtered_label)
    
class CASRDataset(Dataset):
    
    def __init__(self, save_dir=None, run_id=0, split="train") -> None:
        super().__init__()
        self.split = split
        self.run_id = run_id
        self.train_val_test = self.make_combs()
        raw_seqs: List[Sequence] = self.load_prepared_seqs(save_dir)
        self.seqs = self.enforce_eval_protocol(raw_seqs, split, run_id)
        self.data_statistics()
        self.n_feature = np.prod(self.seqs[0][0].shape[1:]) # should be 36 
        self.n_cls = len(np.unique(list(Person.action_dict.values()))) # should be 4
        
    def enforce_eval_protocol(self, raw_seqs, split, run_id):
        train, val, test = self.train_val_test[run_id]
        split_dict = {"all":range(4), "train":train, "val":val, "test":test, "yt":"yt"}
        chosen_split = split_dict.get(split)
        if chosen_split == "yt":
            filtered_seqs = [seq for seq in raw_seqs if seq.is_yt==True]
        else:
            filtered_seqs = [seq for seq in raw_seqs if seq.is_yt==False]
            filtered_seqs = [seq for seq in filtered_seqs if seq.cyclist in chosen_split] 
        print("run id {} train on cyclists {} validate on {} test on {} and yt chosen split is {}".format(
            run_id, train, val, test, split))
        # remove empty seqs
        filtered_seqs = [seq.to_tensor() for seq in filtered_seqs if len(seq.persons)>0]
        return filtered_seqs
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index):
        return self.seqs[index]
    
    @staticmethod
    def load_prepared_seqs(pickle_dir):
        file_name = "{}/CASR_pifpaf.pkl".format(pickle_dir)
        with open(file_name, "rb") as f:
            processed_seqs = pickle.load(f)
        return processed_seqs 
    
    @staticmethod
    def make_combs():
        """ From CASR paper: We use the videos of two cyclists for training, the videos of the other two 
            cyclists are used for validation (training time) and testing, respectively.
        """
        from itertools import combinations, permutations

        all_cyclists = [1, 2, 3, 4]
        all_combs = [] 
        for train in combinations(all_cyclists,2):
            test_val = set(all_cyclists).difference(set(train))
            for test, val in permutations(test_val):
                all_combs.append((train, (test,), (val,)))
                
        return all_combs
        
    def data_statistics(self):
        seq_lengths = [len(pose) for pose, label in self.seqs]
        num_seqs, edge = np.histogram(seq_lengths, bins=[0, 100, 250, 400, 600])
        print("In the {} split".format(self.split))
        for idx, (n, _) in enumerate(zip(num_seqs, edge)):
            print("{} seqs have lengths between {} and {}".format(n, edge[idx], edge[idx+1]))
        all_labels = torch.cat([label for pose, label in self.seqs])
        label, count = np.unique(all_labels, return_counts=True)
        stat_dict = {search_key(Person.action_dict, l):c for l, c in zip(label, count) if l in Person.action_dict.values()}
        print(stat_dict)
        print("\n")

    @staticmethod
    def collate_fn(list_of_seqs, padding_mode="replicate", pad_value=0):
        """
            list_of_seqs is a list of (data_sequence, label_sequence), the tuple is the output of dataset.__getitem__

            sort the sequences (decreasing order), pad and combine them into a tensor
            these sequences will be suitable for batched forward (even for some pure rnn solutions
            that uses packed sequences here
            https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence)
            list_of_seqs: List[Tuple[pose_seq, label_seq]]
        """
        sorted_seqs = sorted(list_of_seqs, key=lambda seq: len(seq[0]), reverse=True)
        pose_seq, pose_seq_len, label_seq = [], [], []
        for pose, label in sorted_seqs:
            pose_seq.append(pose)
            pose_seq_len.append(len(pose))
            label_seq.append(label)
        padded_pose = pad_seqs(pose_seq, mode=padding_mode, pad_value=pad_value)
        padded_label = pad_seqs(label_seq, mode=padding_mode, pad_value=IGNORE_INDEX, is_label_seq=True)
        return padded_pose, padded_label
    
def pad_seqs(list_of_seqs: List[torch.Tensor], mode="replicate", pad_value=0, is_label_seq=False):
    """
        pad the sequence with the last pose and label and combine them into a tensor 

        input:
            list_of_seqs (sorted): List[Tuple[pose_seq | label_seq]] 

        returns: 
            padded_seq (tensor): size (N, T, V, C), which means batch size, maximum sequence length, 
            number of skeleton joints and feature channels (3 for 3d skeleton, 2 for 2D)
    """
    max_seq_len = len(list_of_seqs[0])

    padded_list = []
    for seq in list_of_seqs:

        if is_label_seq:
            pad_value = seq[-1] if mode == "replicate" else pad_value
            seq = F.pad(seq, (0, max_seq_len - seq.shape[-1]), mode="constant", value=pad_value)
        else:
            # for pose sequences (T, V, C) => size (V, C, T) because padding works from the last dimension
            seq = seq.permute(1, 2, 0)
            seq = F.pad(seq, (0, max_seq_len - seq.shape[-1]), mode=mode)
            # back to size (T, V, C)
            seq = seq.permute(2, 0, 1)
        padded_list.append(seq.unsqueeze(0))
    padded_seq = torch.cat(padded_list, dim=0)

    return padded_seq.type(torch.long) if is_label_seq else padded_seq

class CASRSimpleDataset(Dataset):
    
    def __init__(self, dataset:CASRDataset) -> None:
        self.split = dataset.split
        self.run_id = dataset.run_id
        self.n_feature = dataset.n_feature
        self.n_cls = dataset.n_cls
        self.all_poses = torch.cat([seq[0] for seq in dataset.seqs], dim=0)
        self.all_labels = torch.cat([seq[1] for seq in dataset.seqs], dim=0)
        
    def __len__(self):
        return len(self.all_poses)
    
    def __getitem__(self, index):
        return self.all_poses[index], self.all_labels[index]
    
    @staticmethod
    def collate_fn(list_of_pairs):
        pose_list, label_list = [], []
        for pose, label in list_of_pairs:
            pose_list.append(pose.unsqueeze(0))
            label_list.append(label)
        return torch.cat(pose_list,dim=0), torch.tensor(label_list, dtype=torch.long)
    
def construct_dataset(data_dir, save_dir):
    
    pifpaf_out_dir = "{}/{}".format(data_dir, "pifpaf_poses")
    all_annos = preload_annos(data_dir)
    processed_seqs = []
    
    for seq_anno in all_annos:
        
        seq_data = Sequence(seq_name=seq_anno[0]["video_folder"])
        print("extracting a slice from video {}".format(seq_data.seq_name))
        if not os.path.exists("{}/{}/".format(pifpaf_out_dir, seq_data.seq_name)):
            print("sequence {} doesn't exists".format(seq_data.seq_name))
            continue
        
        for frame_anno in seq_anno:
            
            frame_pred_file = "{}/{}/{}.predictions.json".format(
                pifpaf_out_dir, frame_anno["video_folder"], frame_anno["frame_idx"])
            
            try: # person 2 style 3 006, the edited video has 84 frames, but the annotation has 500+ frames ... 
                with open(frame_pred_file) as f:
                    frame_pifpaf_pred = json.load(f)
            except:
                print("something bad happened, give up frame {} in {}".format(frame_anno["frame_idx"], frame_anno["video_folder"]))
                continue
             
            if len(frame_pifpaf_pred) == 0:
                seq_data.persons.append(Person(pred=None, gt_anno=frame_anno))
            else:
                gt_bbox = np.array(frame_anno["bbox_gt"]) # x, y, w, h
                gt_bbox[2:4] = gt_bbox[0:2] + gt_bbox[2:4] # (x1, y1, x2, y2)
                gt_bbox = [gt_bbox.tolist()] # get_iou_matches needs a list of bbox
                pred_bbox = np.array([pred["bbox"]+[pred["score"]] for pred in frame_pifpaf_pred])
                pred_bbox[:, 2:4] = pred_bbox[:, 0:2] + pred_bbox[:, 2:4] # (x1, y1, x2, y2)
                for pred_id, _ in get_iou_matches(pred_bbox.tolist(), gt_bbox):
                    seq_data.persons.append(Person(pred=frame_pifpaf_pred[pred_id], gt_anno=frame_anno)) 

        processed_seqs.append(seq_data)
    
    file_save_path = "{}/CASR_pifpaf.pkl".format(save_dir)
    with open(file_save_path, "wb") as f:
        pickle.dump(processed_seqs, f)
    
    return processed_seqs

def preload_annos(data_dir):
    
    file_name = "{}/{}".format(data_dir, "casr_annos.pkl")
    with open(file_name, "rb") as f:
        all_annos = pickle.load(f, encoding="bytes")
        
    yt_file_name = "{}/{}".format(data_dir, "casr_yt_annos.pkl")
    with open(yt_file_name, "rb") as f:
        all_annos.extend(pickle.load(f, encoding="bytes"))
        
    for i, seq_anno in enumerate(all_annos):
        for j, frame_anno in enumerate(seq_anno):
            all_annos[i][j] = bytes_to_str(frame_anno)

    return all_annos
    
def bytes_to_str(input_dict:dict):
    out = {} 
    for key, value in input_dict.items():
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        out.update({key:value})
    return out 

if __name__ == "__main__":
    pickle_dir = "./poseact/out/casrdata"
    save_dir = "./poseact/out/"
    # construct_dataset(pickle_dir, save_dir)
    seq_dataset = CASRDataset(save_dir, run_id=0, split="all")
    simple_dataset = CASRSimpleDataset(seq_dataset)
    seq_loader = DataLoader(seq_dataset, batch_size=5, shuffle=False, drop_last=True, collate_fn=CASRDataset.collate_fn)
    for pose, label in seq_loader:
        print(pose.shape, label.shape)
    
    simple_loader = DataLoader(simple_dataset, batch_size=128, shuffle=False, drop_last=True, collate_fn=CASRSimpleDataset.collate_fn)
    for pose, label in simple_loader:
        print(pose.shape, label.shape)
    