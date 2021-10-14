import os
import numpy as np
import copy 
import argparse
import datetime
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from models import MonolocoModel, TempMonolocoModel
from utils import compute_accuracy
from tcg_dataset import TCGDataset, TCGSingleFrameDataset, tcg_train_test_split, tcg_collate_fn

# define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# # set value for some arguments 
parser = argparse.ArgumentParser() 
parser.add_argument("--data_dir", type=str, default="./data/tcg_dataset/", help="dataset folder")
parser.add_argument("--fig_dir", type=str, default="./figs/", help="path to save figures")
parser.add_argument("--weight_dir", type=str, default="./models/trained/", help="path to save trained models")
parser.add_argument("--result_dir", type=str, default="./out/", help="training logs")
parser.add_argument("--label_type", type=str, default="major", help="major for 4 classes, sub for 15 classes")
parser.add_argument("--batch_size", type=int, default=128, help="batch size, use a small value (1) for sequence model")
parser.add_argument("--num_epoch", type=int, default=50, help="number of training epochs")
parser.add_argument("--input_size", type=int, default=51, help="input size, number of joints times feature dimension, 51 for TCG")
parser.add_argument("--linear_size", type=int, default=256, help="size of hidden linear layer")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--n_stage", type=int, default=3, help="number of stages in a monoloco model")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate") 
parser.add_argument("--model_type", type=str, choices=["single", "sequence"], default="single", 
                    help="train single frame model or sequenced model")

def train_single_frame(args):
    
    tcg_dataset = TCGDataset(args.data_dir, args.label_type)
    tcg_trainset, tcg_testset = tcg_train_test_split(tcg_dataset)
    # split sequences first and then take frames from two sets of sequences, to avoid data correlation 
    # previously when I didn't notice this, the testing accuracy was about 85% 
    trainset, testset = TCGSingleFrameDataset(tcg_trainset), TCGSingleFrameDataset(tcg_testset)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    model = MonolocoModel(args.input_size, args.output_size, args.linear_size, args.dropout, args.n_stage).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    best_test_acc = -1
    train_loss_list, test_acc_list = [], [] 
    for epoch in range(args.num_epoch):
        model.train()
        batch_loss = [] 
        for pose, label in trainloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item()) 
        
        train_loss = sum(batch_loss)/len(batch_loss)
        # scheduler.step(train_loss)
        test_acc = compute_accuracy(model, testloader)
        if test_acc > best_test_acc:
            best_weights = copy.deepcopy(model.state_dict())
        
        print("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}".format(epoch, train_loss, test_acc))
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
    

    # save training records  
    task_prefix = "TCGSingleFrame_"
    time_prefix = "{}".format(datetime.datetime.now()).replace(" ", "_").replace(":", ".")
    save_model_path = args.weight_dir + task_prefix + time_prefix + ".pth"
    save_file_path  = args.result_dir + task_prefix + time_prefix + ".txt"
    
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), save_model_path)
    
    with open(save_file_path, "w") as f:
        f.write("label_{}_lr_{}_hidden_{}_drop_{}_stage_{}\n".format(args.label_type,
                                args.lr, args.linear_size, args.dropout, args.n_stage))
        for epoch, (loss, acc) in enumerate(zip(train_loss_list, test_acc_list)):
            f.write("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}\n".format(epoch, loss, acc))

def train_seq_model(args):
    
    tcg_dataset = TCGDataset(args.data_dir, args.label_type)
    trainset, testset = tcg_train_test_split(tcg_dataset)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=tcg_collate_fn)
    # always run one sequence for testing, so no need for padding 
    testloader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=tcg_collate_fn)

    model = TempMonolocoModel(args.input_size, args.output_size, args.linear_size, args.dropout, args.n_stage).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    best_test_acc = -1
    train_loss_list, test_acc_list = [], [] 
    for epoch in range(args.num_epoch):
        model.train()
        batch_loss = [] 
        for pose, label in trainloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            loss = criterion(pred.reshape(-1, args.output_size), label.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item()) 
        
        train_loss = sum(batch_loss)/len(batch_loss)
        # scheduler.step(train_loss)
        test_acc = compute_accuracy(model, testloader)
        if test_acc > best_test_acc:
            best_weights = copy.deepcopy(model.state_dict())
        
        print("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}".format(epoch, train_loss, test_acc))
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
    

    # save training records  
    task_prefix = "TCGSeqModel_"
    time_prefix = "{}".format(datetime.datetime.now()).replace(" ", "_").replace(":", ".")
    save_model_path = args.weight_dir + task_prefix + time_prefix + ".pth"
    save_file_path  = args.result_dir + task_prefix + time_prefix + ".txt"
    
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), save_model_path)
    
    with open(save_file_path, "w") as f:
        f.write("label_{}_lr_{}_hidden_{}_drop_{}_stage_{}\n".format(args.label_type, args.lr, args.linear_size, args.dropout, args.n_stage))
        for epoch, (loss, acc) in enumerate(zip(train_loss_list, test_acc_list)):
            f.write("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}\n".format(epoch, loss, acc))

if __name__ == "__main__":
    args = parser.parse_args()
    args.output_size = 15 if args.label_type == "sub" else 4
    if args.model_type == "single":
        print("training a single frame model")
        train_single_frame(args)
    elif args.model_type == "sequence":
        print("training a sequence model")
        train_seq_model(args)