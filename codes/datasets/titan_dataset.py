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

from typing import List, Set, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset
from utils.iou import get_iou_matches
from models import MultiHeadMonoLoco, MonolocoModel
from utils.losses import MultiHeadClfLoss

np.set_printoptions(precision=3, suppress=True)

class Person(object):
    
    action_category = ["communicative", "complex_context", "atomic", "simple_context", "transporting"]
    # 4, 7, 9, 13, 4
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
        
    valid_action_dict = {"walking":0, "standing":1, "sitting":2, "bending":3, "biking":4, "motorcycling":4}
    
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
        mappings = [cls.communicative_dict, cls.complex_context_dict, cls.atomic_dict, 
                    cls.simple_context_dict, cls.transporting_dict]
        for action, mapping in zip(list_of_action, mappings):
            for key, value in mapping.items():
                if action == value:
                    action_str.append(cls.simplify_key(key))
        return action_str
    
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
        

def search_key(obj_dict:dict, value):
    """ search the name of action
    """
    for key in obj_dict.keys():
        if obj_dict[key] == value:
            return key
    raise ValueError("Unrecognized dict value")

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
        
    def collect_objects(self):
        pass 
    
    def unique_obj(self):
        return set([person.object_track_id for person in self.persons])

class Sequence(object):
    
    def __init__(self, seq_name:str) -> None:
        super().__init__()
        self.seq_name:str = seq_name
        self.frames:List[Frame] = []
        
    def seq_obj_ids(self):
        obj_ids = set()
        for frame in self.frames:
            obj_ids.update(frame.unique_obj())
        return obj_ids
    
    def to_tensor(self):
        obj_ids = self.seq_obj_ids()
        pass 
class TITANDataset(Dataset):
    
    def __init__(self, pifpaf_out=None, dataset_dir=None, pickle_dir=None, use_pickle=True, split="train") -> None:
        """

        Args:
            pifpaf_out ([str], optional): [pifpaf result dir]. Defaults to None.
            dataset_dir ([str], optional): [titan dataset root dir]. Defaults to None.
            pickle_dir ([str], optional): [the folder that stores the preprocessed pickle file, from `construct_from_pifpaf_results`]. Defaults to None.
            use_pickle (bool, optional): [description]. Defaults to True.
            split: 'all', 'train', 'val' or 'test'
        """
        super().__init__()
        self.split = split
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
        
        if len(self.seqs) > 0: # get one split from it
            name_mapping = {"train": "train_set", "val": "val_set", "test":"test_set"}
            split_name = name_mapping.get(self.split, "train_set")
            print("loading the {} of TITAN".format(split_name))
            split_file = "{}/splits/{}.txt".format(dataset_dir, split_name)
            with open(split_file, "r") as f:
                valid_seqs = f.readlines()
            valid_seqs = [name.rstrip() for name in valid_seqs]
            valid_seqs = sorted(valid_seqs, key=lambda item: int(item.split(sep="_")[-1]))
            self.seqs = [seq for seq in self.seqs if seq.seq_name in valid_seqs]
        
    def __len__(self):
        return len(self.seqs) 
    
    def __getitem__(self, index):
        return self.seqs[index]
    
    def load_from_pickle(self, pickle_dir):
        pickle_file = pickle_dir + "/" + "TITAN_pifpaf.pkl"
        with open(pickle_file, "rb") as f:
            processed_seqs = pickle.load(f)
        return processed_seqs
    
    @staticmethod
    def collate(list_of_sequences: List[Sequence]):
        pass 
    
class TITANSimpleDataset(Dataset):
    
    """ simple version of titan dataset, feed the 2D poses absolutely independently,
        without considering temporal and spatial relations 
    """
    
    def __init__(self, titan_dataset:TITANDataset, merge_cls=False) -> None:
        super().__init__()
        """ merge_cls: remove the unlearnable classes, and merge the hierarchical labels into one,
                       see `self.merge_labels` for details 
        """
        self.split = titan_dataset.split
        self.merge_cls = merge_cls
        frames = self.form_frames(titan_dataset.seqs)
        self.all_poses, self.all_labels = self.get_poses_from_frames(frames)
        if self.merge_cls:
            self.n_cls = [len(np.unique(list(Person.valid_action_dict.values())))]
            print("The simplified {} dataset consists of: \n {}".format(self.split, self.data_statistics()))
            print("In percentage it would be")
            print_dict_in_percentage(self.data_statistics())
        else:
            self.n_cls = [len(getattr(Person, category + "_dict")) for category in Person.action_category]

    def __getitem__(self, index):
        return self.all_poses[index], self.all_labels[index]
    
    def __len__(self):
        return len(self.all_poses)
    
    @staticmethod
    def form_frames(list_of_sequences: List[Sequence]) -> List[Frame]:
        all_frames = []
        for seq in list_of_sequences:
            all_frames.extend(seq.frames)
        return all_frames
    
    def get_poses_from_frames(self, frames:List[Frame]):
        all_poses, all_labels = [], []
        # communicative, complex_context, atomic, simple_context, transporting
        
        for frame in frames:
            for person in frame.persons:
                if self.merge_cls:
                    valid, pose, label = self.merge_labels(person)
                    if not valid:
                        continue
                else:
                    pose = person.key_points
                    label = [person.communicative, 
                            person.complex_context, 
                            person.atomic, 
                            person.simple_context, 
                            person.transporting]
                all_poses.append(pose)
                all_labels.append(label)
        # TODO convert to numpy array
        return np.array(all_poses), np.array(all_labels)

    @staticmethod
    def collate(list_of_pairs):
        pose_list, label_list = [], []
        for pose, label in list_of_pairs:
            pose_list.append(pose)
            label_list.append(label)
        return torch.tensor(pose_list, dtype=torch.float32), torch.tensor(label_list, dtype=torch.long)
    
    def merge_labels(self, person:Person):
        
        atomic_action = person.search_key("atomic")
        simple_action = person.search_key("simple_context")
        pose = person.key_points
        
        # if the person is biking (or motocycling), then simple context action 
        # will override atomic actions
        if simple_action in person.valid_action_dict.keys():
            valid = True
            label = person.valid_action_dict.get(simple_action)
        # record the person's atomic action if it's learnable 
        elif atomic_action in person.valid_action_dict.keys():
            valid = True
            label = person.valid_action_dict.get(atomic_action)
        else:
            valid = False
            label = None
        
        return valid, pose, [label]
    
    def data_statistics(self):
        """ count the number of instances 
        """ 
        if self.merge_cls:
            label, count = np.unique(self.all_labels, return_counts=True)
            stat_dict = {search_key(Person.valid_action_dict, l):c for l, c in zip(label, count)}
            return stat_dict
        else:
            raise NotImplementedError
    
    
class TITANPatchDataset(Dataset):
    """ crop patches from titan images

    """
    def __init__(self) -> None:
        super().__init__()

class TITANSeqDataset(Dataset):
    
    def __init__(self, titan_dataset:TITANDataset) -> None:
        super().__init__()
        self.seqs = [seq.to_tensor() for seq in titan_dataset.seqs]

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index):
        return self.seqs[index]
    
    @staticmethod
    def collate(list_of_pairs):
        
        pass 

def get_all_clip_names(pifpaf_out):
    all_clip_dirs = glob.glob(pifpaf_out+"*", recursive=True)
    clips = [name.replace("\\", "/").split(sep="/")[-1] for name in all_clip_dirs]
    clips = sorted(clips, key=lambda item: int(item.split(sep="_")[-1]))
    return clips

def construct_from_pifpaf_results(pifpaf_out, dataset_dir, save_dir=None, debug=True):
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
    clips = get_all_clip_names(pifpaf_out)
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
            frame_pred_file = "{}/{}/{}.predictions.json".format(pifpaf_out, clip, frame)
            with open(frame_pred_file) as f:
                frame_pifpaf_pred = json.load(f)

            matches = [] # try to find matches if pifpaf detects some person 
            if len(frame_pifpaf_pred) != 0:          
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
        save_path = save_dir + "/" + "TITAN_pifpaf.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(processed_seqs, f)
        return processed_seqs
    else:
        return processed_seqs

def folder_names(base_dir):
    pifpaf_out = "{}/out/pifpaf_results/".format(base_dir)
    dataset_dir = "{}/data/TITAN/".format(base_dir)
    save_dir = "{}/out/".format(base_dir)
    return pifpaf_out, dataset_dir, save_dir
    
def get_titan_att_types(args):
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir)
    clips = get_all_clip_names(pifpaf_out)
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
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir)
    construct_from_pifpaf_results(pifpaf_out, dataset_dir, save_dir, debug=False)

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

def test_seq_dataset(args):
    pifpaf_out, dataset_dir, save_dir = folder_names(args.base_dir)
    dataset = TITANDataset(dataset_dir=dataset_dir, pickle_dir=save_dir, use_pickle=True)
    seq_dataset = TITANSeqDataset(dataset)
    dataloader = DataLoader(seq_dataset, batch_size=2, shuffle=True, collate_fn=TITANSeqDataset.collate)
    
    for poses, labels in dataloader:
        print(poses.shape, labels.shape)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--base_dir", type=str, default="./", help="default root working directory")
    parser.add_argument("--function", type=str, default="None", help="which function to call")
    parser.add_argument("--merge-cls", action="store_true", help="remove unlearnable, merge 5 labels into one")
    args = parser.parse_args(["--base_dir", "codes/", "--function", "forward", "--merge-cls"])

    function_dict = {"annotation": get_titan_att_types, 
                     "pickle": pickle_all_sequences, 
                     "dist":calc_anno_distribution,
                     "label_stats": calc_anno_distribution, 
                     "forward":test_forward}

    if args.function in function_dict.keys():
        function_dict.get(args.function)(args)

    # test_construct_dataset(args)
    # test_forward(args)
    # test_seq_dataset(args)

