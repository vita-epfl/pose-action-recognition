import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
torch.autograd.set_detect_anomaly(True)

from threading import Lock
from torch.multiprocessing import Pool
from poseact.models import MonolocoModel, TempMonolocoModel
from poseact.utils.tcg_dataset import TCGDataset, TCGSingleFrameDataset, tcg_collate_fn, tcg_pad_seqs
from poseact.utils.tcg_metrics import compute_accuracy

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
    
    net = MonolocoModel(51, 4, 256, 0.2, 3)
    net.eval()
    rand_data = torch.rand(3, 100, 17, 3)
    out1 = net(rand_data.reshape(300, 17, 3))
    out1 = out1.reshape(3, 100, 4)

    out2 = torch.zeros(3, 100, 4)
    for idx in range(3):
        out2[idx, :, :] = net(rand_data[idx, :, :])
    assert torch.allclose(out1, out2, rtol=0.001)


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
    
    net = TempMonolocoModel(51, 4, 256, 0.2, 3)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    rand_data = torch.rand(3, 100, 17, 3)
    rand_label = torch.randint(0, 4, (3, 100))
    pred = net(rand_data)
    loss = criterion(pred.reshape(-1, 4), rand_label.reshape(-1))


def test_compute_accuracy():
    
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