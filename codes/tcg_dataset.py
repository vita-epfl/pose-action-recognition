import json
import torch
import numpy as np
import torch.nn.functional as F

from typing import List, Set, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


class TCGDataset(Dataset):

    def __init__(self, data_path, label_type="major"):

        self.raw_seq = np.load(data_path + "/" + "tcg_data.npy", allow_pickle=True)
        with open(data_path + "/" + "tcg.json") as f:
            self.raw_label = json.load(f)

        # definition from tcg repo https://github.com/againerju/tcg_recognition/blob/master/TCGDB.py
        self.maj_cls = {"inactive": 0, "stop": 1, "go": 2, "clear": 3}
        self.sub_cls = {"inactive_normal-pose": 0, "inactive_out-of-vocabulary": 0, "inactive_transition": 0,
                        "stop_both-static": 1, "stop_both-dynamic": 2, "stop_left-static": 3,
                        "stop_left-dynamic": 4, "stop_right-static": 5, "stop_right-dynamic": 6,
                        "clear_left-static": 7, "clear_right-static": 8, "go_both-static": 9,
                        "go_both-dynamic": 10, "go_left-static": 11, "go_left-dynamic": 12,
                        "go_right-static": 13, "go_right-dynamic": 14}

        self.seqs: list = self.process_sequences()
        self.labels: list = self.process_labels(label_type)
        del self.raw_seq, self.raw_label
        
    def process_sequences(self) -> List[torch.Tensor]:
        seqs = [torch.tensor(seq, dtype=torch.float32) for seq in self.raw_seq]
        return seqs

    def process_labels(self, label_type="major") -> List[torch.Tensor]:
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
        # TODO verify with matplotlib
        processed_label = all_sub_cls if label_type == "sub" else all_maj_cls

        return processed_label

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.seqs[index]
        label = self.labels[index]
        return (seq, label)

    def __len__(self):
        return len(self.seqs)


class TCGSingleFrameDataset(Dataset):
    """
        single frame version of TCG dataset, then a network processes each frame individually, 
        just to test an earlier idea
        
        we can't randomly choose frames from all sequences, because in that scenario, we will take frames from
        (almost) all sequences, and the trainset and testset will be highly correlated 
        Instead, we should first split different sequences, and then get frames from the two sets of sequences
    """

    def __init__(self, tcg_seq_dataset:TCGDataset):
        super().__init__()
        seq_list, label_list = [], []
        if isinstance(tcg_seq_dataset, Subset):
            for seq, label in tcg_seq_dataset:
                seq_list.append(seq)
                label_list.append(label)
        elif isinstance(tcg_seq_dataset, TCGDataset):
            seq_list = tcg_seq_dataset.seqs
            label_list = tcg_seq_dataset.labels
        self.frames = torch.cat(seq_list, dim=0)
        self.labels = torch.cat(label_list, dim=0).type(torch.long)    
        
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
        pose_seq_len.append(len(pose_seq))
        label_seq.append(label)
    padded_pose = tcg_pad_seqs(pose_seq, mode=padding_mode, pad_value=pad_value)
    padded_label = tcg_pad_seqs(label_seq, mode=padding_mode, pad_value=pad_value)
    return padded_pose, torch.tensor(pose_seq_len), padded_label


def tcg_train_test_split(dataset: TCGDataset, train_size=0.7, seed=42):
    # for reproducibility, permute the indexes with a fixed random number generator
    rng = torch.Generator().manual_seed(seed)
    rand_idx = torch.randperm(len(dataset), generator=rng)
    split_idx = int(len(dataset)*train_size)
    train_idx = rand_idx[0:split_idx]
    test_idx = rand_idx[split_idx+1:]
    trainset = Subset(dataset, train_idx)
    testset = Subset(dataset, test_idx)

    return trainset, testset


def test_init_and_get_item():
    datapath, label_type = "codes/data", "major"
    trainset = TCGDataset(datapath, label_type)
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
    datapath, label_type = "codes/data", "major"
    tcg_dataset = TCGDataset(datapath, label_type)
    trainset, testset = tcg_train_test_split(tcg_dataset)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=tcg_collate_fn)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=tcg_collate_fn)
    very_simple_net = torch.nn.Linear(17*3, 128)
    for pose, lens, label in trainloader:
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
    rand_data = torch.rand(3, 100, 17*3)
    out1 = net(rand_data.flatten(0, 1))
    out1 = out1.reshape(3, 100, 4)

    out2 = torch.zeros(3, 100, 4)
    for idx in range(3):
        out2[idx, :, :] = net(rand_data[idx, :, :])
    assert torch.allclose(out1, out2)

def test_single_frame_set():
    datapath, label_type = "codes/data", "major"
    tcg_dataset = TCGDataset(datapath, label_type)
    tcg_trainset, tcg_testset = tcg_train_test_split(tcg_dataset)
    trainset, testset = TCGSingleFrameDataset(tcg_trainset), TCGSingleFrameDataset(tcg_testset)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    very_simple_net = torch.nn.Linear(17*3, 128)
    for pose, label in trainloader:
        # pose has shape (N, V, C), out will be (N, T, out)
        out = very_simple_net(torch.flatten(pose, -2, -1))
        break

if __name__ == "__main__":
    test_init_and_get_item()
    test_pad_seqs()
    test_seq_forward()
    test_reshape_and_forward()
    test_single_frame_set()