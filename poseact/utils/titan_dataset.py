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
from poseact.utils import to_relative_coord, search_key
from poseact.utils.iou import get_iou_matches
from poseact.utils.losses import IGNORE_INDEX
from multiprocessing import Pool

np.set_printoptions(precision=3, suppress=True)

class Person(object):
    
    # five categories of actions in TITAN dataset 
    action_category = ["communicative", "complex_context", "atomic", "simple_context", "transporting"]

    # obtained with get_titan_att_types
    communicative_dict = {'looking into phone': 0,
                          'talking in group': 1,
                          'talking on phone': 2,
                          'none of the above': 3}
    complex_context_dict = {'getting in 4 wheel vehicle': 0,
                            'getting off 2 wheel vehicle': 1,
                            'getting on 2 wheel vehicle': 2,
                            'getting out of 4 wheel vehicle': 3,
                            'loading': 4,
                            'unloading': 5,
                            'none of the above': 6}
    atomic_dict = {'bending': 0,
                   'jumping': 1,
                   'laying down': 2,
                   'running': 3,
                   'sitting': 4,
                   'squatting': 5,
                   'standing': 6,
                   'walking': 7,
                   'none of the above': 8}
    simple_context_dict = {'biking': 0,
                           'cleaning an object': 1,
                           'closing': 2,
                           'crossing a street at pedestrian crossing': 3,
                           'entering a building': 4,
                           'exiting a building': 5,
                           'jaywalking (illegally crossing NOT at pedestrian crossing)': 6,
                           'motorcycling': 7,
                           'opening': 8,
                           'waiting to cross street': 9,
                           'walking along the side of the road': 10,
                           'walking on the road': 11,
                           'none of the above': 12}
    transporting_dict = {'carrying with both hands': 0,
                         'pulling': 1,
                         'pushing': 2,
                         'none of the above': 3}
    
    # we also created a simpler set of actions with the actions that we think is learnable
    # and we use this set of actions to replace the original hierarchical labels if args.merge_cls is True
    # we merge biking and motorcycling together as cycling, and categorize looking into phone and talking on the phone as 
    # using phone
    valid_action_dict = {"walking":0, "standing":1, "sitting":2, "bending":3, "biking":4, 
                         "motorcycling":4, "looking into phone":5, "talking on phone":6}
    
    def __init__(self, pred, gt_anno) -> None:
        """ pred: pifpaf prediction for this person 
            gt_anno: corresponding ground truth annotation 
        """
        super().__init__()
        # locations
        self.object_track_id = int(gt_anno['obj_track_id'])
        x_y_conf = np.array(pred['keypoints']).reshape(-1, 3)
        self.key_points = x_y_conf[:, :2] # x, y
        self.kp_confidence = x_y_conf[:, 2]
        self.gt_box = [gt_anno[key] for key in ["left", "top", "width", "height"]]
        self.pred_box = pred['bbox']
        self.confidence = pred['score']
        if "id_" in pred.keys():
            self.pifpaf_track_id = pred.get("id_", None)
        # attributes provided by titan annotation
        self.age = gt_anno['attributes.Age']
        self.communicative = self.communicative_dict.get(gt_anno['attributes.Communicative'], None) 
        self.complex_context = self.complex_context_dict.get(gt_anno['attributes.Complex Contextual'], None)
        self.atomic = self.atomic_dict.get(gt_anno['attributes.Atomic Actions'], None)
        self.simple_context = self.simple_context_dict.get(gt_anno['attributes.Simple Context'], None)
        self.transporting = self.transporting_dict.get(gt_anno['attributes.Transporting'], None) 
    
    @classmethod
    def get_attr_dict(cls, query_type):
        # mapping = {"communicative":cls.communicative_dict, 
        #            "complex_context":cls.complex_context_dict,
        #            "atomic":cls.atomic_dict,
        #            "simple_context":cls.simple_context_dict,
        #            "transporting":cls.transporting_dict}
        mapping = {category: getattr(cls, category + "_dict") for category in cls.action_category}
        mapping.update({"valid_action":getattr(cls, "valid_action_dict")})
        original_dict = mapping[query_type]
        simplified = {cls.simplify_key(key) :value for key, value in original_dict.items()}
        return simplified

    @classmethod
    def pred_list_to_str(cls, list_of_action):
        """ convert a list of actions like [0, 0, ...] into list of strings ["action1", "action2", ...]
        """
        action_str = [] 
        if len(list_of_action) == 1:
            mappings = [cls.valid_action_dict]
        elif (list_of_action) == 5:
            mappings = [cls.communicative_dict, 
                        cls.complex_context_dict, 
                        cls.atomic_dict, 
                        cls.simple_context_dict, 
                        cls.transporting_dict]
        for action, mapping in zip(list_of_action, mappings):
            for key, value in mapping.items():
                if action == value:
                    action_str.append(cls.simplify_key(key))
                    break # find the first and then do next 
        return action_str
    
    def action_hierarchy(self):
        return [self.communicative, 
                self.complex_context, 
                self.atomic, 
                self.simple_context, 
                self.transporting]
    
    def merge_labels(self):
        
        atomic_action = self.search_key("atomic")
        simple_action = self.search_key("simple_context")
        com_action = self.search_key("communicative")
        pose = self.key_points
        
        # if the person is using the phone (looking into phone or talking on the phone)
        # this action will have highest priority 
        if com_action in self.valid_action_dict.keys():
            valid = True
            label = self.valid_action_dict.get(com_action)
        # if the person is biking (or motocycling), then simple context action 
        # will override atomic actions
        elif simple_action in self.valid_action_dict.keys():
            valid = True
            label = self.valid_action_dict.get(simple_action)
        # record the person's atomic action if it's learnable 
        elif atomic_action in self.valid_action_dict.keys():
            valid = True
            label = self.valid_action_dict.get(atomic_action)
        else:
            valid = False
            label = None
        
        return valid, pose, [label]
    
    def search_key(self, category):
        """ search the name of action
        """
        obj_dict = getattr(self, category + "_dict")
        value = getattr(self, category)
        return search_key(obj_dict, value)
    
        # label_dict = getattr(self, category + "_dict")
        # for key in label_dict.keys():
        #     if label_dict[key] == getattr(self, category):
        #         return key
        # raise ValueError("Unrecognized action")
            
    @classmethod
    def simplify_key(cls, key:str):
        """ convert the original long label into a shorter one, 
            better for printing 
        """
        simplify_dict = {'getting in 4 wheel vehicle': 'getting in 4 wv',
                        'getting off 2 wheel vehicle': "getting off 2 wv",
                        "getting on 2 wheel vehicle":'getting on 2 wv',
                        'getting out of 4 wheel vehicle':'getting out of 4 wv',
                        "crossing a street at pedestrian crossing":"crossing legally",
                        "jaywalking (illegally crossing NOT at pedestrian crossing)":"crossing illegally",
                        "waiting to cross street":"waiting to cross",
                        "walking along the side of the road": 'walking on the side',
                        'carrying with both hands':"carrying",
                        "none of the above":"none"
                        }
        if key in simplify_dict.keys():
            return simplify_dict[key]
        else:
            return key
        
class Vehicle(object):
    
    def __init__(self, pred, gt_anno) -> None:
        super().__init__()
        # locations
        self.object_track_id = None
        self.key_points = None
        self.gt_box = None
        self.pred_box = None
        self.confidence = None
        
        # attributes provided by titan annotation
        self.trunk_open = None 
        self.motion_status = None
        self.doors_open = None
        self.atomic = None
        self.simple_context = None
        self.transporting = None
        
class Frame(object):
    """ a frame contains multiple persons and vehicles (possibly other objects pifpaf detects?)

    """
    def __init__(self, seq_name, frame_name) -> None:
        super().__init__()
        self.seq_name = seq_name
        self.frame_name = frame_name
        self.persons: List[Person] = []
        self.vehicles:List[Vehicle] = []
        
    def collect_objects(self, merge_label=True):
        
        all_poses, pifpaf_box, all_labels = [], [], []
        for person in self.persons:
            if merge_label:
                valid, pose, label = person.merge_labels()
                if not valid:
                    continue
            else:
                pose = person.key_points
                label = person.action_hierarchy()
            pifpaf_box.append(person.pred_box)
            all_poses.append(pose)
            all_labels.append(label)
            
        box_array = np.array(pifpaf_box)
        pose_array = np.array(all_poses)
        label_array = np.array(all_labels)
        
        return pose_array, box_array, label_array
    
    def read_img(self, base_dir):
        img_path = "{}/data/TITAN/images_anonymized/{}/images/{}".format(base_dir, self.seq_name, self.frame_name)
        pil_img = Image.open(img_path).convert('RGB')
        return pil_img, img_path
    
    def unique_obj(self, method="gt"):
        if method=="gt":
            return set([person.object_track_id for person in self.persons])
        elif method=="pifpaf":
            return set([person.pifpaf_track_id for person in self.persons])

class Sequence(object):
    
    def __init__(self, seq_name:str) -> None:
        super().__init__()
        self.seq_name:str = seq_name
        self.frames:List[Frame] = []
        
    def seq_obj_ids(self, method="gt"):
        obj_ids = set()
        for frame in self.frames:
            obj_ids.update(frame.unique_obj(method))
        return obj_ids
    
    def to_tensor(self, method="gt"):
        obj_ids = self.seq_obj_ids(method)
        # map the object track id into N continuous number (for this sequence only)
        index_mapping = {obj_id:idx for obj_id, idx in zip(obj_ids, range(len(obj_ids)))}
        # a padded array of size (N, T, V, C), enough to contain all the poses for all persons in this sequence
        padded_pose_array = np.zeros((len(obj_ids), len(self.frames), 17, 2), dtype=np.float32)
        padded_label_array = IGNORE_INDEX*np.ones((len(obj_ids), len(self.frames)), dtype=np.int)
        # put the poses into the padded pose array frame by frame, person by person
        for fid, frame in enumerate(self.frames):
            for person in frame.persons:
                valid, pose, label = person.merge_labels()
                if not valid:
                    continue
                # choose whether we use GT or pifpaf track id 
                track_id = person.object_track_id if method=="gt" else person.pifpaf_track_id
                position = index_mapping[track_id]
                padded_pose_array[position, fid, :, :] = pose
                padded_label_array[position, fid] = label[0]
        
        # remove heading and trailing zeros https://stackoverflow.com/questions/11188364/remove-zero-lines-2-d-numpy-array
        poses_and_labels = []
        for idx, (padded_pose, padded_label) in enumerate(zip(padded_pose_array, padded_label_array)):
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
            filtered_label = torch.tensor(padded_label[not_leading_or_trailing], dtype=torch.float32)
            if filtered_label.numel() == 0: # skip all empty (invalid) sequences
                continue
            poses_and_labels.append((filtered_pose, filtered_label))

        return poses_and_labels
         

class TITANDataset(Dataset):
    
    def __init__(self, pifpaf_out=None, dataset_dir=None, pickle_dir=None, use_pickle=True, split="train", mode="single"):
        """

        Args:
            pifpaf_out ([str], optional): [pifpaf result dir]. Defaults to None.
            dataset_dir ([str], optional): [titan dataset root dir]. Defaults to None.
            pickle_dir ([str], optional): [the folder that stores the preprocessed pickle file, from `construct_from_pifpaf_results`]. Defaults to None.
            use_pickle (bool, optional): [description]. Defaults to True.
            split: 'all', 'train', 'val' or 'test'
        """
        super().__init__()
        self.pifpaf_out = pifpaf_out
        self.dataset_dir = dataset_dir
        self.pickle_dir = pickle_dir
        self.use_pickle = use_pickle
        self.split = split
        self.mode=mode
        self.seqs:List[Sequence] = []
        if pickle_dir is not None and use_pickle:
            print("loading preprocessed titan dataset from pickle file")
            self.seqs = self.load_from_pickle(pickle_dir=pickle_dir)
        elif os.path.exists(pifpaf_out) and os.path.exists(dataset_dir):
            print("constructing dataset from pifpaf detection results and original annotation")
            processed_seqs = construct_from_pifpaf_results(pifpaf_out, dataset_dir)
            self.seqs = processed_seqs
        else:
            print("Failed to load pose sequences")
        
        if split == "all": # keep all sequences
            return 
        elif len(self.seqs) > 0: # get one split from it
            self.seqs = self.get_data_split()
        
    def __len__(self):
        return len(self.seqs) 
    
    def __getitem__(self, index):
        return self.seqs[index]
    
    def load_from_pickle(self, pickle_dir):
        file_name = "TITAN_pifpaf.pkl" if self.mode=="single" else "TITAN_pifpaf_track.pkl"
        pickle_file = pickle_dir + "/" + file_name
        with open(pickle_file, "rb") as f:
            processed_seqs = pickle.load(f)
        return processed_seqs
    
    def get_data_split(self):
        name_mapping = {"train": "train_set", "val": "val_set", "test":"test_set"}
        split_name = name_mapping.get(self.split, "train_set")
        print("loading the {} of TITAN".format(split_name))
        split_file = "{}/splits/{}.txt".format(self.dataset_dir, split_name)
        with open(split_file, "r") as f:
            valid_seqs = f.readlines()
        valid_seqs = [name.rstrip() for name in valid_seqs]
        valid_seqs = sorted(valid_seqs, key=lambda item: int(item.split(sep="_")[-1]))
        split_seqs = [seq for seq in self.seqs if seq.seq_name in valid_seqs]
        return split_seqs
    
    @staticmethod
    def collate(list_of_sequences: List[Sequence]):
        pass 
    
class TITANSimpleDataset(Dataset):
    
    """ simple version of titan dataset, feed the 2D poses absolutely independently,
        without considering temporal and spatial relations 
    """
    
    def __init__(self, 
                 titan_dataset:TITANDataset, 
                 merge_cls=False, 
                 inflate:float=None, 
                 use_img=False, 
                 relative_kp=False,
                 rm_center=False,
                 normalize=False) -> None:
        super().__init__()
        """ merge_cls: remove the unlearnable classes, and merge the hierarchical labels into one set,
                       see `person.merge_labels` for details 
            inflate: copy the samples of minority classes, so the training process won't be overwhelmed by majority class
            use_img: for each person, crop an image patch from the frame, don't use poses
            relative_kp: convert a keypoint from absolute coordinate to center + relative coordinate
        """
        self.dataset_dir = titan_dataset.dataset_dir
        self.split = titan_dataset.split
        self.merge_cls = merge_cls
        self.inflate = inflate
        self.use_img = use_img
        self.relative_kp = relative_kp
        self.rm_center = rm_center
        self.normalize = normalize
        self.n_feature = None # number of features, will be set in self.get_poses/patches_from_frames(frames) 
        frames = self.form_frames(titan_dataset.seqs)
        # use keypoints by default, also possible to use patches 
        if not self.use_img:
            self.all_poses, self.all_labels = self.get_poses_from_frames(frames) 
        else:
            # all_poses is actually images here, use this name for convenience
            print("Cropping image patchs from the original frame")
            self.all_poses, self.all_labels = self.get_patches_from_frames(frames)
            
        if self.merge_cls:
            self.n_cls = [len(np.unique(list(Person.valid_action_dict.values())))]
            self.print_statistics()
        else:
            self.n_cls = [len(getattr(Person, category + "_dict")) for category in Person.action_category]
            
        if self.split=="train" and isinstance(self.inflate, float): # only inflate samples in training
            self.all_poses, self.all_labels = self.inflate_minority_classes()
            print("after inflating the minority classes")
            self.print_statistics()

    def __getitem__(self, index):
        if not self.use_img:
            pose = self.all_poses[index]
            label = self.all_labels[index]
            return pose, label
        else:
            pose = np.array(self.all_poses[index], dtype=np.float32) / 255
            label = self.all_labels[index]
            return pose, label
    
    def __len__(self):
        return len(self.all_poses)
    
    @staticmethod
    def form_frames(list_of_sequences: List[Sequence]) -> List[Frame]:
        all_frames = []
        for seq in list_of_sequences:
            all_frames.extend(seq.frames)
        return all_frames
    
    def get_poses_from_frames(self, frames:List[Frame]):
        all_poses, all_labels, all_wh = [], [], []
        # communicative, complex_context, atomic, simple_context, transporting
        
        for frame in frames:
            for person in frame.persons:
                if self.merge_cls:
                    valid, pose, label = person.merge_labels()
                    if not valid:
                        continue
                else:
                    pose = person.key_points
                    label = person.action_hierarchy()
                x, y, w, h = person.pred_box
                if self.normalize:
                    all_wh.append(np.array([w,h]).reshape(-1, 2))
                all_poses.append(pose)
                all_labels.append(label)
        if self.normalize:
            wh_array = np.array(all_wh)
        pose_array = np.array(all_poses)
        label_array = np.array(all_labels)
        
        if self.relative_kp:
            print("Converting the original coordinates to center+relative")
            pose_array = self.to_relative_coord(pose_array)
            if self.rm_center:
                pose_array = pose_array[:,:17,:]
            if self.normalize:
                pose_array = pose_array / wh_array
        self.n_feature = np.prod(pose_array.shape[1:])
        
        return pose_array, label_array
    
    @staticmethod
    def to_relative_coord(all_poses):

        return to_relative_coord(all_poses)
    
    def process_one_frame(self, frame:Frame):
        
        all_poses, all_labels = [], []
        img_file_path = "./{}/images_anonymized/{}/images/{}".format(self.dataset_dir, frame.seq_name, frame.frame_name)
        # like the example in pifpaf doc https://openpifpaf.github.io/predict_api.html
        frame_img = Image.open(img_file_path).convert("RGB") 
        for person in frame.persons:

            # obtain labels 
            if self.merge_cls:
                valid, _, label = person.merge_labels()
                if not valid:
                    continue
            else:
                label = person.action_hierarchy()
                
            # use ground truth box for training, use detection box for validation and testing 
            bbox = person.gt_box if self.split=="train" else person.pred_box # both are x y w h
            bbox = enlarge_bbox(bbox, enlarge=1) # make the box larger than the person, still x y w h
            # crop box has to be (x1, y1, x2, y2)
            crop_box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            patch = frame_img.crop(box=crop_box)
            # the shape 224, 224 is required in resnet pytorch example, see the link below
            # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
            patch = patch.resize((224, 224)) 
            pose = np.asarray(patch).astype(np.uint8) # use this name "pose" for convenience ... 
            pose = pose.transpose((2, 0, 1)) # size (224, 224, 3) => (3, 224, 224)
            all_poses.append(pose)
            all_labels.append(label)
            
        return all_poses, all_labels
    
    def get_patches_from_frames_mp(self, frames:List[Frame]):
        
        with Pool(processes=8) as p:
            combined_results = p.map(self.process_one_frame, frames)
            
        final_poses, final_labels = [], []
        for frame_result in combined_results:
            final_poses.extend(frame_result[0])
            final_labels.extend(frame_result[1])
        
        return final_poses, final_labels
        
    def get_patches_from_frames(self, frames:List[Frame]):
        """ obtain image patches from frames

        Args:
            frames (List[Frame]): [description]

        Returns:
            pose_array: image array of shape (n_samples, 3, 224, 224), basically (N, C, H, W)
            label_array: shape (n_samples, n_cls)
        """
        all_poses, all_labels = [], []
        for frame in frames:
            img_file_path = "./{}/images_anonymized/{}/images/{}".format(self.dataset_dir, frame.seq_name, frame.frame_name)
            # like the example in pifpaf doc https://openpifpaf.github.io/predict_api.html
            frame_img = Image.open(img_file_path).convert("RGB") 
            for person in frame.persons:

                # obtain labels 
                if self.merge_cls:
                    valid, _, label = person.merge_labels()
                    if not valid:
                        continue
                else:
                    label = person.action_hierarchy()
                    
                # use ground truth box for training, use detection box for validation and testing 
                bbox = person.gt_box if self.split=="train" else person.pred_box # both are x y w h
                bbox = enlarge_bbox(bbox, enlarge=1) # make the box larger than the person, still x y w h
                # crop box has to be (x1, y1, x2, y2)
                crop_box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                patch = frame_img.crop(box=crop_box)
                # the shape 224, 224 is required in resnet pytorch example, see the link below
                # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
                patch = patch.resize((224, 224)) 
                pose = np.asarray(patch).astype(np.uint8) # use this name "pose" for convenience ... 
                pose = pose.transpose((2, 0, 1)) # size (224, 224, 3) => (3, 224, 224)
                all_poses.append(pose)
                all_labels.append(label)
        
        pose_array = np.array(all_poses)
        label_array = np.array(all_labels)
        
        return pose_array, label_array
            

    def inflate_minority_classes(self):
        
        assert self.merge_cls, "must merge classes and then inflate, use --merge_cls in commandline"
        print("duplicating the minority class to approximately {:.1f}% of the majority class".format(self.inflate*100))
        
        original_poses, original_labels = self.all_poses, self.all_labels
        sample_count = self.data_statistics()
        max_val = max(sample_count.values())
        total = sum(sample_count.values())
        
        copy_pose_list, copy_label_list = [], []
        for key, value in sample_count.items():
            cls_idx = (original_labels == Person.valid_action_dict[key])
            percentage = value / total
            if percentage > 0.15: # don't inflate if the original data accounts for more than 15%
                continue
            num_duplicates = round(self.inflate*max_val / (value+10)) - 1
            if num_duplicates < 1:
                continue
            
            copied_poses = np.repeat(original_poses[cls_idx.flatten()], num_duplicates, axis=0)
            copied_labels = np.repeat(original_labels[cls_idx.flatten()], num_duplicates, axis=0)
            copy_pose_list.append(copied_poses)
            copy_label_list.append(copied_labels)
        
        copy_pose_array = np.concatenate(copy_pose_list, axis=0)
        copy_label_array = np.concatenate(copy_label_list, axis=0)
        
        inflated_pose = np.concatenate((original_poses, copy_pose_array), axis=0)
        inflated_labels = np.concatenate((original_labels, copy_label_array), axis=0)
        
        return inflated_pose, inflated_labels
            
        
            
    @staticmethod
    def collate(list_of_pairs):
        pose_list, label_list = [], []
        for pose, label in list_of_pairs:
            pose_list.append(pose)
            label_list.append(label)
        return torch.tensor(pose_list, dtype=torch.float32), torch.tensor(label_list, dtype=torch.long)
    
    def data_statistics(self):
        """ count the number of instances 
        """ 
        if self.merge_cls:
            label, count = np.unique(self.all_labels, return_counts=True)
            stat_dict = {search_key(Person.valid_action_dict, l):c for l, c in zip(label, count)}
            return stat_dict
        else:
            raise NotImplementedError
        
    def print_statistics(self):
        print("The simplified {} dataset consists of: \n {}".format(self.split, self.data_statistics()))
        print("In percentage it would be")
        print_dict_in_percentage(self.data_statistics())

class TITANSeqDataset(Dataset):
    
    def __init__(self, titan_dataset:TITANDataset, method="gt") -> None:
        super().__init__()
        self.seqs = []
        self.split = titan_dataset.split
        self.method=method # "gt" for using ground truth track ID, "pifpaf"
        for seq in titan_dataset.seqs:
            self.seqs.extend(seq.to_tensor(method=self.method))
        self.seqs = [(TITANSimpleDataset.to_relative_coord(padded_pose), padded_label) 
                     for padded_pose, padded_label in self.seqs]
        self.n_feature = torch.prod(torch.tensor(self.seqs[0][0].shape[1:])) # center point + relative coordinate
        self.n_cls = [len(np.unique(list(Person.valid_action_dict.values())))] # merged classes, list representations
        self.data_statistics()
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index):
        return self.seqs[index]
    
    @staticmethod
    def collate(list_of_pairs):
        return titan_seq_collate_fn(list_of_pairs, padding_mode="constant", pad_value=0)

    def data_statistics(self):
        seq_lengths = [len(pose) for pose, label in self.seqs]
        num_seqs, edge = np.histogram(seq_lengths, bins=[0, 20, 50, 100, 200, 300])
        print("\nIn the {} split".format(self.split))
        for idx, (n, _) in enumerate(zip(num_seqs, edge)):
            print("{} seqs have lengths between {} and {}".format(n, edge[idx], edge[idx+1]))
    
def titan_seq_pad_seqs(list_of_seqs: List[torch.Tensor], mode="constant", pad_value=0, is_label_seq=False):
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


def titan_seq_collate_fn(list_of_seqs, padding_mode="constant", pad_value=0):
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
    padded_pose = titan_seq_pad_seqs(pose_seq, mode=padding_mode, pad_value=pad_value)
    padded_label = titan_seq_pad_seqs(label_seq, mode=padding_mode, pad_value=IGNORE_INDEX, is_label_seq=True)
    return padded_pose, padded_label

def get_all_clip_names(dataset_dir):
    all_clip_dirs = glob.glob("{}/titan_0_4/*.csv".format(dataset_dir), recursive=True)
    clips = [name.replace("\\", "/").split(sep="/")[-1].rstrip(".csv") for name in all_clip_dirs]
    clips = sorted(clips, key=lambda item: int(item.split(sep="_")[-1]))
    return clips

def enlarge_bbox(bb, enlarge=1):
    """
    Convert the bounding box from Pifpaf to an enlarged version of it 
    """
    delta_h = (bb[3]) / (7 * enlarge)
    delta_w = (bb[2]) / (3.5 * enlarge)
    bb[0] -= delta_w
    bb[1] -= delta_h
    bb[2] += delta_w
    bb[3] += delta_h
    return bb

def construct_from_pifpaf_results(pifpaf_out, dataset_dir, save_dir=None, debug=True, mode="single"):
    """ read the pifpaf prediction results (json files), match the detections with ground truth,
        and pack the detections first by frame, then by sequence into a pickle file
        
    Args:
        pifpaf_out ([type]): [location of the pifpaf detection results]
        dataset_dir ([type]): [titan dataset folder]
        save_dir ([type], optional): [where to save the pickle file]. Defaults to None.
        debug (bool, optional): [if true, then test with 3 sequences]. Defaults to True.

    Raises:
        NotImplementedError: [description]

    Returns:
        processed sequences [list]: [a list of sequences]
    """
    processed_seqs = []
    total_number_list, detected_list = [], [] # how many annotated persons, how many detected 
    clips = get_all_clip_names(dataset_dir)

    # process all sequences 
    for clip_idx, clip in enumerate(clips):
        
        person_in_seq, detected_in_seq = 0, 0
        print("Processing {}".format(clip))
        # create a container for frame data in a sequence 
        seq_container = Sequence(seq_name=clip)
        
        anno_file = dataset_dir + "titan_0_4/" + clip + ".csv"
        seq_annos = pd.read_csv(anno_file)
        groups = seq_annos.groupby("frames")
        frame_names = sorted(list(groups.indices.keys())) 
        
        # process each frame in a sequence
        for frame in frame_names:
            # create a container for object data in a frame 
            frame_container = Frame(seq_name=clip, frame_name=frame)
            
            frame_gt_annos = groups.get_group(frame)
            frame_gt_annos = frame_gt_annos[frame_gt_annos["label"]=="person"] # just keep the annotation of person 
            if mode=="single":
                frame_pred_file = "{}/{}/{}.predictions.json".format(pifpaf_out, clip, frame)
                with open(frame_pred_file) as f:
                    frame_pifpaf_pred = json.load(f)
            elif mode=="track":
                seq_pred_file = "{}/TITAN_{}_track.json".format(pifpaf_out, clip)
                with open(seq_pred_file) as f:
                    seq_pred = f.readlines()
                for line in seq_pred:
                    # frame numbers are not consecutive, some frames may not have labels
                    # and thus they are not present in the pandas dataframe 
                    # so we need to match the detection and ground truth with frame ID
                    raw = json.loads(line)
                    if int(frame.split(".")[0]) == 6*raw['frame']:
                        break
                assert int(frame.split(".")[0]) == 6*raw['frame'], "frame {} is not the {} th frame in sequence {}".format(frame, raw['frame'], clip)
                frame_pifpaf_pred = raw["predictions"]
            else:
                raise NotImplementedError

            matches = [] # try to find matches if pifpaf detects some person 
            if len(frame_pifpaf_pred) > 0:          
                # print(frame_gt_annos)
                # print(frame_pifpaf_pred)
                gt_bbox = frame_gt_annos[["left", "top", "width", "height"]].to_numpy() # (x, y, w, h) 
                # print(gt_bbox)
                gt_bbox[:, 2:4] = gt_bbox[:, 0:2] + gt_bbox[:, 2:4] # (x1, y1, x2, y2)
                # print(gt_bbox)
                frame_pifpaf_pred = sorted(frame_pifpaf_pred, key=lambda item: item["score"], reverse=True)
                # (x, y, w, h) 
                pred_bbox = np.array([pred["bbox"]+[pred["score"]] for pred in frame_pifpaf_pred])
                pred_bbox[:, 2:4] = pred_bbox[:, 0:2] + pred_bbox[:, 2:4] # (x1, y1, x2, y2)
                matches = get_iou_matches(pred_bbox.tolist(), gt_bbox.tolist())
            
                # process each objects in a frame 
                for pred_id, gt_id in matches:
                    obj_pifpaf_pred = frame_pifpaf_pred[pred_id]
                    obj_gt_anno = frame_gt_annos.iloc[gt_id].to_dict()
                    if obj_gt_anno["label"] == "person":
                        person_container = Person(pred=obj_pifpaf_pred, gt_anno=obj_gt_anno)
                        frame_container.persons.append(person_container) # vars will turn the container into a dict 
                    elif obj_gt_anno["label"].startswith("vehicle"):
                        raise NotImplementedError
                    
            print("pifpaf detects {} person in {} {} and ground truth has {}".format(
                    len(matches), clip, frame, len(frame_gt_annos)))
            person_in_seq += len(frame_gt_annos)
            detected_in_seq += len(matches)

            seq_container.frames.append(frame_container) # append an empty 
        
        processed_seqs.append(seq_container)
        total_number_list.append(person_in_seq)
        detected_list.append(detected_in_seq)
        
        if debug and clip_idx == 3:
            break
    
    print("#Total Annotated persons: {} #Total detected persons: {}".format(sum(total_number_list), sum(detected_list)))
    
    if not debug and save_dir is not None:
        file_name = "TITAN_pifpaf.pkl" if mode=="single" else "TITAN_pifpaf_track.pkl"
        save_path = "{}/{}".format(save_dir, file_name)
        with open(save_path, "wb") as f:
            pickle.dump(processed_seqs, f)
        return processed_seqs
    else:
        return processed_seqs

def folder_names(base_dir, mode="single"):
    if mode=="single":
        pifpaf_out = "{}/out/pifpaf_results/".format(base_dir)
    elif mode=="track":
        pifpaf_out = "{}/out/pifpaf_track_results/".format(base_dir)
    dataset_dir = "{}/data/TITAN/".format(base_dir)
    save_dir = "{}/out/".format(base_dir)
    return pifpaf_out, dataset_dir, save_dir
    
def get_titan_att_types(args):
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir)
    clips = get_all_clip_names(dataset_dir)
    communicative, complex_context, atomic, simple_context, transporting = [set() for _ in range(5)]
    
    for clip in clips:
        anno_file = dataset_dir + "titan_0_4/" + clip + ".csv"
        seq_annos = pd.read_csv(anno_file)
        seq_annos = seq_annos[seq_annos["label"]=="person"] # just keep the annotation of person 
        communicative.update(set(seq_annos['attributes.Communicative'].to_list()))
        complex_context.update(set(seq_annos['attributes.Complex Contextual'].to_list()))
        atomic.update(set(seq_annos['attributes.Atomic Actions'].to_list()))
        simple_context.update(set(seq_annos['attributes.Simple Context'].to_list()))
        transporting.update(set(seq_annos['attributes.Transporting'].to_list()))

    att_hierarchy = [communicative, complex_context, atomic, simple_context, transporting]
    att_dict_list = []
    for att_set in att_hierarchy:
        # print(att_set)
        att_set.remove("none of the above")
        att_list = sorted(list(att_set))
        att_list.append("none of the above") # move 'none of the above' to the end 
        att_dict = {att_list[idx]:idx for idx in range(len(att_list))}
        att_dict_list.append(att_dict)
        print(att_dict)
    
    return att_dict_list
        
def pickle_all_sequences(args):
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir, mode=args.mode)
    construct_from_pifpaf_results(pifpaf_out, dataset_dir, save_dir, debug=False, mode=args.mode)

def print_dict_in_percentage(dict_record:Dict[str, int]):
    total_count = sum(list(dict_record.values()))
    for key, value in dict_record.items():
        print("\'{}\':{:.2f}%".format(Person.simplify_key(key), value/total_count*100))
    print()
    
def calc_anno_distribution(args):
    """count the number of each attribute

    Args:
        args ([type]): [description]
    """
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir)
    att_hierarchy = get_titan_att_types(args)

    for split in ["all", "train", "val", "test"]:
        dataset = TITANDataset(dataset_dir=dataset_dir, pickle_dir=save_dir, use_pickle=True, split=split)
        att_count = [dict.fromkeys(category.keys(), 0) for category in att_hierarchy]
        communicative, complex_context, atomic, simple_context, transporting = att_count
        for seq in dataset.seqs:
            for frame in seq.frames:
                for person in frame.persons:
                    communicative[person.search_key("communicative")] += 1
                    complex_context[person.search_key("complex_context")] += 1
                    atomic[person.search_key("atomic")] += 1
                    simple_context[person.search_key("simple_context")] += 1
                    transporting[person.search_key("transporting")] += 1
                    
                    
        print("For the {} set:".format(split), sep="\n")
        all_records = [communicative, complex_context, atomic, simple_context, transporting]
        all_names = ['communicative', 'complex_context', 'atomic', 'simple_context', 'transporting']
        for record, name in zip(all_records, all_names):
            print("for {} actions".format(name))
            print_dict_in_percentage(record)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--base_dir", type=str, default="./", help="default root working directory")
    parser.add_argument("--function", type=str, default="None", help="which function to call")
    parser.add_argument("--merge-cls", action="store_true", help="remove unlearnable, merge 5 labels into one")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "track"],help="for making the pickle file")
    args = parser.parse_args() # ["--base_dir", "poseact/", "--function", "pickle", "--mode", "track"]

    function_dict = {"annotation": get_titan_att_types, 
                     "pickle": pickle_all_sequences, 
                     "dist":calc_anno_distribution,
                     "label_stats": calc_anno_distribution}

    if args.function in function_dict.keys():
        function_dict.get(args.function)(args)


