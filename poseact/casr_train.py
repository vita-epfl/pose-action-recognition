import os
import sys 
import numpy as np
import copy 
import argparse
import datetime
import matplotlib.pyplot as plt

import ctypes
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils import model_zoo
from torch.utils.data import DataLoader, Subset
torch.autograd.set_detect_anomaly(True)

from poseact.utils import setup_multiprocessing
from poseact.utils.losses import MultiHeadClfLoss, IGNORE_INDEX, FocalLoss
from poseact.models import MultiHeadMonoLoco, TempMonolocoModel, MonolocoModel
from poseact.utils.casr_metrics import compute_accuracy, get_all_predictions, get_eval_metrics, summarize_results
from poseact.utils.casr_dataset import CASRDataset, CASRSimpleDataset, Person, Sequence

# define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# set value for some arguments 
parser = argparse.ArgumentParser() 

parser.add_argument("--base_dir", type=str, default=".", help="root directory of the codes")
# model and training related arguments 
parser.add_argument("--model_type", type=str, default="single", choices=["single", "sequence"],help="single frame or sequential model")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--num_epoch", type=int, default=50, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate") 
parser.add_argument("--workers", type=int, default=0, help="number of workers for dataloader") 
parser.add_argument("--linear_size", type=int, default=128, help="size of hidden linear layer")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--n_stage", type=int, default=3, help="number of stages in a monoloco model")
parser.add_argument("--num_runs", type=int, default=12, help="number of different train-val-test setup")

# loss related arguments 
parser.add_argument("--gamma", type=float, default=1.5, help="the gamma parameter for focal loss, should be a positive integer")

# logging related arguments 
parser.add_argument("--task_name", type=str, default="Baseline", help="a name for this training task, used in save name")
parser.add_argument("--debug", action="store_true", help="debug mode, use a small fraction of datset")
parser.add_argument("--save_model", action="store_true", help="store trained network")

def manual_add_args(args):
    base_dir = args.base_dir
    args.pickle_dir = "{}/out/casrdata".format(base_dir)
    args.save_dir = "{}/out/".format(base_dir)
    args.weight_dir = "{}/out/trained/".format(base_dir)
    return args  

def train_model(args):
    
    is_sequence = True if args.model_type == "sequence" else False
    
    # prepare datasets and dataloaders
    trainset = CASRDataset(args.save_dir, run_id=args.run_id, split="train")
    valset = CASRDataset(args.save_dir, run_id=args.run_id, split="val")
    testset = CASRDataset(args.save_dir, run_id=args.run_id, split="test")
    ytset = CASRDataset(args.save_dir, run_id=args.run_id, split="yt")
    
    if args.debug:
        print("using a 2 epochs and first 2 sequences for debugging")
        args.num_epoch = 2
        trainset.seqs = trainset.seqs[:2]
        valset.seqs = valset.seqs[:2]
        testset.seqs = testset.seqs[:2]
        ytset.seqs = ytset.seqs[:2]
        args.batch_size = 2 if args.model_type == "sequence" else 32
            
            
    if is_sequence:
        collate_fn = CASRDataset.collate_fn
    else:
        trainset, valset = CASRSimpleDataset(trainset), CASRSimpleDataset(valset), 
        testset, ytset = CASRSimpleDataset(testset), CASRSimpleDataset(ytset)
        collate_fn = CASRSimpleDataset.collate_fn
        
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    ytloader = DataLoader(ytset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # prepare models and loss 
    input_size, output_size = trainset.n_feature, trainset.n_cls
    if is_sequence:
        model = TempMonolocoModel(input_size, output_size, args.linear_size, args.dropout, args.n_stage).to(device) 
    else:
        model = MonolocoModel(input_size, output_size, args.linear_size, args.dropout, args.n_stage).to(device)
        
    criterion = FocalLoss(gamma=args.gamma, device=device, ignore_index=IGNORE_INDEX)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    
    best_test_acc = -1
    train_loss_list, test_acc_list = [], [] 
    for epoch in range(args.num_epoch):
        model.train()
        batch_loss = [] 
        for pose, label in trainloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            # for single frame model, the output shape is (N, C), for sequence model it's (N, T, C)
            if is_sequence:
                N, T, C = pred.shape 
                pred = pred.view(N*T, C)
                label = label.view(N*T, 1)
            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item()) 
        
        train_loss = sum(batch_loss)/len(batch_loss)
        
        test_acc = compute_accuracy(model, valloader, is_sequence)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
        # scheduler.step(train_loss)
        print("Epoch {} Avg Loss {:.4f} test Acc {:.4f}".format(epoch, train_loss, test_acc))

        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        
    print("loading the parameters with best validation accuracy")
    model.load_state_dict(best_weights)
    
    if args.save_model:
        task_name = args.task_name
        slurm_job_id = os.environ.get("SLURM_JOBID", None)
        if slurm_job_id is not None:
            task_name = task_name + str(slurm_job_id)
        time_suffix = "{}".format(datetime.datetime.now()).replace(" ", "_").replace(":", ".")
        filename = "{}/CASR_{}_{}.pth".format(args.weight_dir, task_name, time_suffix)
        torch.save(model.state_dict(), filename)
        print("model saved to {}".format(filename))
        
    test_results = get_all_predictions(model, testloader, is_sequence)
    yt_results = get_all_predictions(model, ytloader, is_sequence)

    return test_results, yt_results
    
if __name__ == "__main__":
    # ["--base_dir", "poseact", "--model_type", "sequence", "--batch_size", "32"]
    args = parser.parse_args() # ["--base_dir", "poseact", "--debug"]
    args = manual_add_args(args)
    num_splits = len(CASRDataset.make_combs())
    
    print("Starting to run the experiment in serial")
    all_test_results, all_yt_results = [], []
    for idx in range(num_splits):
        print("\nThis is the {} th split in the CASR dataset\n".format(idx))
        args.run_id = idx
        split_test_results, split_yt_results = train_model(args)
        all_test_results.append(split_test_results)
        all_yt_results.append(split_yt_results)
    
    print("Summarizing results on the test splits over all train-val-test setup")
    acc, f1, jac, cfx = get_eval_metrics(all_test_results)
    summarize_results(acc, f1, jac, cfx)
    
    print("Summarizing results on the youtuber splits over all train-val-test setup")
    acc, f1, jac, cfx = get_eval_metrics(all_yt_results)
    summarize_results(acc, f1, jac, cfx)