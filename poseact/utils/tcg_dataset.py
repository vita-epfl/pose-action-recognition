import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Set, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


class TCGDataset(Dataset):
    
    # definition from tcg repo https://github.com/againerju/tcg_recognition/blob/master/TCGDB.py
    maj_cls = {"inactive": 0, "stop": 1, "go": 2, "clear": 3}
    sub_cls = {"inactive_normal-pose": 0, "inactive_out-of-vocabulary": 0, "inactive_transition": 0,
                    "stop_both-static": 1, "stop_both-dynamic": 2, "stop_left-static": 3,
                    "stop_left-dynamic": 4, "stop_right-static": 5, "stop_right-dynamic": 6,
                    "clear_left-static": 7, "clear_right-static": 8, "go_both-static": 9,
                    "go_both-dynamic": 10, "go_left-static": 11, "go_left-dynamic": 12,
                    "go_right-static": 13, "go_right-dynamic": 14}

    sampling_factor = 5  # to subsample 100 Hz to 20 Hz
    train_test_sets = {"xs":[[[1, 2, 3, 4], [5]],
                             [[1, 3, 4, 5], [2]],
                             [[1, 2, 4, 5], [3]],
                             [[1, 2, 3, 5], [4]],
                             [[2, 3, 4, 5], [1]]],
                    "xv": [[["right", "bottom", "left"], ["top"]],
                            [["right", "bottom", "top"], ["left"]],
                            [["bottom", "left", "top"], ["right"]],
                            [["right", "left", "top"], ["bottom"]]]}

    def __init__(self, data_path, label_type="major", eval_type=None, eval_id=None, training:bool=True, 
                 relative_kp=False, use_velocity=False):
        """
        a pytorch dataset class for TCG dataset https://arxiv.org/abs/2007.16072
        code adopted from https://github.com/againerju/tcg_recognition/blob/master/TCGDB.py

        Args:
            data_path ([type]): [description]
            label_type (str, optional): use "major" to get 4 class labels, "sub" for 15 class labels. Defaults to "major".
            eval_type (str, optional): "xs" for cross subject evaluation, "xv" for cross view. Defaults to "xs".
            eval_id (int): when eval_type is specified, also choose a split id (0~4)
        """

        self.raw_seq = np.load(data_path + "/" + "tcg_data.npy", allow_pickle=True)
        with open(data_path + "/" + "tcg.json") as f:
            self.raw_label = json.load(f)
            
        # just to indicate the data type, can be commented 
        self.seqs = []
        self.labels = []
        self.view_points = []
        self.subject_idx = []
        self.relative_kp = relative_kp
        self.use_velocity = use_velocity
        
        self.process_sequences()
        self.process_labels(label_type)
        self.enforce_eval_protocol(eval_type, eval_id, training)
        self.down_sample_seqs()
        
        if self.relative_kp:
            self.convert_to_relative_coordinate()
        if self.use_velocity:
            self.add_velocity_to_kp()
            
        self.n_feature = np.prod(self.seqs[0].shape[1:])
        del self.raw_seq, self.raw_label

    def process_sequences(self):
        """ convert the sequences from numpy arrays to a list of pytorch tensors 
        """
        seqs = [torch.tensor(seq, dtype=torch.float32) for seq in self.raw_seq]
        self.seqs = seqs

    def convert_to_relative_coordinate(self):
        print("converting the keypoint coordinates into center+relative")
        for idx, seq in enumerate(self.seqs):
            # seq is a tensor with shape (N, V, C) V=17, C=3
            center_point = seq[:, 3, :].unsqueeze(1)
            relative = seq - center_point
            center_and_relative = torch.cat((relative, center_point), dim=1)
            self.seqs[idx] = center_and_relative
    
    def add_velocity_to_kp(self):
        print("concatinating joint velocity with location")
        for idx, seq in enumerate(self.seqs):
            # seq is a tensor with shape (N, V, C) V=18 if use relative coordinates
            prev_pose = seq.clone()
            prev_pose[1:, :, :] = seq[:-1, :, :] # shift the original sequence by 1 step 
            velocity = seq - prev_pose # velocity will be 0 at first
            pose_and_velo = torch.cat((seq, velocity), dim=1)
            self.seqs[idx] = pose_and_velo
    
    def process_labels(self, label_type="major"):
        """ turn the label into list of tensors 
            generate class label for each frame, based on the original annotation

        Args:
            label_type (str, optional): type of required label, "major" for 4 classes, "sub" for 15 classes. Defaults to "major".

        return: 
        """
        seq_labels = self.raw_label['sequences']
        all_maj_cls, all_sub_cls = [], []
        # annotation for one sequence
        for one_label in seq_labels:
            # record subject and view point
            # adopted from https://github.com/againerju/tcg_recognition/blob/master/TCGDB.py#L168
            # cross subject means training on the recordings of some people, and doing tests on the data from someone else
            # cross view means the same action have different meanings to the vehicles in different directions
            self.subject_idx.append(one_label['subject_id'])
            self.view_points.append(one_label['scene_agents'][one_label['agent_number']]['position'])

            seq_len = one_label['num_frames']
            seq_maj_label = torch.zeros(seq_len, dtype=torch.float32)
            seq_sub_label = torch.zeros(seq_len, dtype=torch.float32)
            # annotation for one phase
            anno = one_label['annotation']
            for state, substate, start, stop in anno:
                maj_cls = self.maj_cls[state]
                sub_cls = self.sub_cls[state + "_" + substate]
                seq_maj_label[max(0, start):min(stop, seq_len-1)] = maj_cls
                seq_sub_label[max(0, start):min(stop, seq_len-1)] = sub_cls

            all_maj_cls.append(seq_maj_label)
            all_sub_cls.append(seq_sub_label)

        self.labels = all_sub_cls if label_type == "sub" else all_maj_cls

    def enforce_eval_protocol(self, eval_type, eval_id, training):

        # make sure the evaluation type and running id are valid
        if eval_type==None or eval_id==None:
            raise ValueError("Please specify the evaluation type and index")
        if eval_type == "xs":
            assert eval_id in [0, 1, 2, 3, 4], "valid choices are 0, 1, 2, 3 and 4"
        elif eval_type == "xv":
            assert eval_id in [0, 1, 2, 3], "valid choices are 0, 1, 2 and 3"

        # get the subject/view id according to preset protocol
        train, test = self.train_test_sets[eval_type][eval_id]

        # get a boolean array that indicates whether the sequence belong to training or testing
        if eval_type == "xs":
            train_seq_idx = np.in1d(self.subject_idx, train)
            test_seq_idx = np.in1d(self.subject_idx, test)
        elif eval_type == "xv":
            train_seq_idx = np.in1d(self.view_points, train)
            test_seq_idx = np.in1d(self.view_points, test)

        # take training or test set
        valid_idx = test_seq_idx if training == False else train_seq_idx

        # filter the
        self.seqs = [seq for seq, valid in zip(self.seqs, valid_idx) if valid]
        self.labels = [label for label, valid in zip(self.labels, valid_idx) if valid]
        self.view_points = [view for view, valid in zip(self.view_points, valid_idx) if valid]
        self.subject_idx = [subject for subject, valid in zip(self.subject_idx, valid_idx) if valid]

    def down_sample_seqs(self):

        if self.sampling_factor == 1:
            return

        self.seqs = [seq[::self.sampling_factor] for seq in self.seqs]
        self.labels = [label[::self.sampling_factor] for label in self.labels]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.seqs[index]
        label = self.labels[index]
        return (seq, label)

    def __len__(self):
        return len(self.seqs)

    @classmethod
    def get_num_split(cls, eval_type):
        return len(cls.train_test_sets[eval_type])

    @classmethod
    def get_output_size(cls, label_type):
        assert label_type in ["major", "sub"]
        label_dict = cls.maj_cls if label_type=="major" else cls.sub_cls
        return len(np.unique(list(label_dict.values())))
    
class TCGSingleFrameDataset(Dataset):
    """
        single frame version of TCG dataset, then a network processes each frame individually, 
        just to test an earlier idea

        we can't randomly choose frames from all sequences, because in that scenario, we will take frames from
        (almost) all sequences, and the trainset and testset will be highly correlated 
        Instead, we should first split different sequences, and then get frames from the two sets of sequences
    """

    def __init__(self, tcg_seq_dataset: TCGDataset):

        super().__init__()
        assert isinstance(tcg_seq_dataset, TCGDataset)
        seq_list = tcg_seq_dataset.seqs
        label_list = tcg_seq_dataset.labels
        self.frames = torch.cat(seq_list, dim=0)
        self.labels = torch.cat(label_list, dim=0).type(torch.long)
        self.n_feature = tcg_seq_dataset.n_feature
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.frames[index]
        label = self.labels[index]
        return (seq, label)

    def __len__(self):
        return len(self.frames)


def tcg_one_hot_encoding(n_cls, label):
    """ 
        n_cls (int): number of classes
        label (tensor): true label of the data, tensor format 
    """
    one_hot_label = torch.eye(n_cls)[label]
    return one_hot_label


def tcg_pad_seqs(list_of_seqs: List[torch.Tensor], mode="replicate", pad_value=0):
    """
        pad the sequence with the last pose and label and combine them into a tensor 

        input:
            list_of_seqs (sorted): List[Tuple[pose_seq | label_seq]] 

        returns: 
            padded_seq (tensor): size (N, T, V, C), which means batch size, maximum sequence length, 
            number of skeleton joints and feature channels (3 for 3d skeleton, 2 for 2D)
    """
    max_seq_len = len(list_of_seqs[0])
    is_label_seq = True if len(list_of_seqs[0].shape) == 1 else False

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


def tcg_collate_fn(list_of_seqs, padding_mode="replicate", pad_value=0):
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
    padded_pose = tcg_pad_seqs(pose_seq, mode=padding_mode, pad_value=pad_value)
    padded_label = tcg_pad_seqs(label_seq, mode=padding_mode, pad_value=pad_value)
    return padded_pose, padded_label

def test_init_and_get_item():
    datapath, label_type = "poseact/data", "major"
    trainset = TCGDataset(datapath, label_type, eval_type="xs", eval_id=1, training=True)
    one_seq, seq_label = trainset[13]
    trainset_frame = TCGSingleFrameDataset(trainset)
    one_frame, frame_label = trainset_frame[10086]
    
    

def test_pad_seqs():
    long_input = torch.ones(5, 3, 3)
    short_input = torch.tensor([1, 2, 3]).reshape(-1, 1, 1)*torch.ones(3, 3, 3)
    padded_long, padded_short = tcg_pad_seqs([long_input, short_input])
    padded_long = padded_long.squeeze()
    padded_short = padded_short.squeeze()
    assert torch.allclose(padded_long, long_input) and torch.allclose(padded_short[-1, :, :], torch.tensor([3.0]))

    long_input = torch.tensor([1, 2, 3, 4, 5])
    short_input = torch.tensor([1, 2, 3])
    padded_long, padded_short = tcg_pad_seqs([long_input, short_input], mode="replicate")
    assert torch.allclose(padded_long, long_input) and torch.allclose(padded_short, torch.tensor([1, 2, 3, 3, 3]))

    padded_long, padded_short = tcg_pad_seqs([long_input, short_input], mode="constant", pad_value=666)
    assert torch.allclose(padded_long, long_input) and torch.allclose(padded_short, torch.tensor([1, 2, 3, 666, 666]))


def test_seq_forward():
    datapath, label_type = "poseact/data", "major"
    trainset = TCGDataset(datapath, label_type, eval_type="xs", eval_id=1, training=True)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=tcg_collate_fn)
    very_simple_net = torch.nn.Linear(17*3, 128)
    for pose, label in trainloader:
        # pose has shape (N, T, V, C), out will be (N, T, out)
        out = very_simple_net(torch.flatten(pose, -2, -1))
        break


def test_reshape_and_forward():
    """ for the sequence input, we can reshape the input, put it to network and then reshape back
        results will be the same as batched processing 
    """
    from models import MonolocoModel
    net = MonolocoModel(51, 4, 256, 0.2, 3)
    net.eval()
    rand_data = torch.rand(3, 100, 17, 3)
    out1 = net(rand_data.reshape(300, 17, 3))
    out1 = out1.reshape(3, 100, 4)

    out2 = torch.zeros(3, 100, 4)
    for idx in range(3):
        out2[idx, :, :] = net(rand_data[idx, :, :])
    assert torch.allclose(out1, out2)


def test_single_frame_set():
    datapath, label_type = "poseact/data", "major"
    trainset = TCGDataset(datapath, label_type, eval_type="xs", eval_id=1, training=True)
    trainset = TCGSingleFrameDataset(trainset)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    very_simple_net = torch.nn.Linear(17*3, 128)
    for pose, label in trainloader:
        # pose has shape (N, V, C), out will be (N, out)
        out = very_simple_net(torch.flatten(pose, -2, -1))
        break


def test_save_log():
    filename = 'poseact/data/results/TCGSingleFrame_2021-10-13_20.16.55.txt'
    with open(filename, "w") as f:
        for epoch, (loss, acc) in enumerate(zip([0.3, 0.5, 0.7, 0.9], [0.3, 0.5, 0.7, 0.9])):
            f.write("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}\n".format(epoch, loss, acc))


def test_seq_model_forward():
    from models import TempMonolocoModel
    net = TempMonolocoModel(51, 4, 256, 0.2, 3)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    rand_data = torch.rand(3, 100, 17, 3)
    rand_label = torch.randint(0, 4, (3, 100))
    pred = net(rand_data)
    loss = criterion(pred.reshape(-1, 4), rand_label.reshape(-1))


def test_compute_accuracy():
    
    from models import MonolocoModel, TempMonolocoModel
    from utils import compute_accuracy
    
    model = MonolocoModel(51, 4, 128, 0.2, 3)
    datapath, label_type = "poseact/data", "major"
    tcg_testset = TCGDataset(datapath, label_type, eval_type="xs", eval_id=1, training=True)
    
    single_testset = TCGSingleFrameDataset(tcg_testset)
    testloader = DataLoader(single_testset, batch_size=128, shuffle=False)
    acc = compute_accuracy(model, testloader)

    model = TempMonolocoModel(51, 4, 128, 0.2, 3)
    testloader = DataLoader(tcg_testset, batch_size=1, shuffle=False, collate_fn=tcg_collate_fn)
    acc = compute_accuracy(model, testloader)

def test_class_utility():
    
    assert TCGDataset.get_num_split("xv")==4 and TCGDataset.get_num_split("xs")==5
    assert TCGDataset.get_output_size("major")==4 and TCGDataset.get_output_size("sub")==15
    
if __name__ == "__main__":

    test_init_and_get_item()
    test_class_utility()
    test_pad_seqs()
    test_seq_forward()
    test_reshape_and_forward()
    test_single_frame_set()
    test_seq_model_forward()
    test_compute_accuracy()
    test_save_log()
