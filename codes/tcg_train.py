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
from tcg_dataset import TCGDataset, TCGSingleFrameDataset, tcg_collate_fn

# define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# set value for some arguments 
parser = argparse.ArgumentParser() 
parser.add_argument("--data_dir", type=str, default="codes/data/", help="dataset folder, should end with /")
parser.add_argument("--fig_dir", type=str, default="codes/figs/", help="path to save figures, should end with /")
parser.add_argument("--weight_dir", type=str, default="codes/models/trained/", help="path to save trained models, end with /")
parser.add_argument("--result_dir", type=str, default="codes/data/results/", help="training logs dir, end with /")
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
parser.add_argument("--eval_type", type=str, default="xs", help="cross-subject (xs) or cross-view (xv) evaluation")
parser.add_argument("--return_pred", type=bool, default=False, help="return prediction results for the whole test set")
parser.add_argument("--n_process", type=int, default=None, help="number of process for multiprocessing, or None to run in serial")
parser.add_argument("--debug", action="store_true", help="debug mode, use a small fraction of datset")

def multiprocess_wrapper(args_and_eval_id):
    # args_and_eval_id is a tuple with elements (args, eval_id)
    args, eval_id = args_and_eval_id
    args.eval_id = eval_id
    train_model(args)

def train_model(args):
    
    if args.n_process is not None:
        print("subprocess {} is beginning to train a {} model".format(os.getpid(), args.model_type))
    else:
        print("beginning to train a {} model".format(args.model_type))
        
    trainset = TCGDataset(args.data_dir, args.label_type, args.eval_type, args.eval_id, training=True)
    testset = TCGDataset(args.data_dir, args.label_type, args.eval_type, args.eval_id, training=False)
    
    # choose a single frame model or a sequence version of monoloco  
    if args.model_type == "single":
        trainset, testset = TCGSingleFrameDataset(trainset), TCGSingleFrameDataset(testset)
        if args.debug:
            trainset, testset = Subset(trainset, indices=range(1024)), Subset(testset, indices=range(1024))
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        model = MonolocoModel(args.input_size, args.output_size, args.linear_size, args.dropout, args.n_stage).to(device)
    elif args.model_type == "sequence":
        if args.debug:
            trainset, testset = Subset(trainset, indices=range(10)), Subset(testset, indices=range(2))
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=tcg_collate_fn)
        # always run one sequence for testing, so no need for padding, and no artificial data padded 
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
            # for single frame model, the output shape is (N, C), for sequence model it's (N, T, C)
            loss = criterion(pred.reshape(-1, args.output_size), label.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item()) 
        
        train_loss = sum(batch_loss)/len(batch_loss)
        
        test_acc = compute_accuracy(model, testloader)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
        # scheduler.step(train_loss)
        if args.n_process == None:
            # only print this in single frame mode, or the output will be unordered 
            print("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}".format(epoch, train_loss, test_acc))
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
    

    # save training records  
    model_prefix = "TCGSingleFrame_" if args.model_type == "single" else "TCGSeqModel_"
    task_prefix = "{}_{}_".format(args.eval_type, args.eval_id)
    time_prefix = "{}".format(datetime.datetime.now()).replace(" ", "_").replace(":", ".")
    save_model_path = args.weight_dir + model_prefix + task_prefix + time_prefix + ".pth"
    save_info_path  = args.result_dir + model_prefix + task_prefix + time_prefix + ".txt"
    
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), save_model_path)
    
    with open(save_info_path, "w") as f:
        f.write("label_{}_lr_{}_hidden_{}_drop_{}_stage_{}\n".format(args.label_type,
                                args.lr, args.linear_size, args.dropout, args.n_stage))
        for epoch, (loss, acc) in enumerate(zip(train_loss_list, test_acc_list)):
            f.write("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}\n".format(epoch, loss, acc))
            
    if args.n_process is not None:
        print("subprocess {} has finished training a {} model with \
            best testing accuracy {}".format(os.getpid(), args.model_type, best_test_acc))
        
if __name__ == "__main__":
    args = parser.parse_args(["--model_type", "sequence", "--eval_type", "xv", "--debug", "--num_epoch", "2", "--n_process", "3"])
    # args = parser.parse_args()
    args.output_size = TCGDataset.get_output_size(args.label_type)
    num_splits = TCGDataset.get_num_split(args.eval_type)
    
    if args.n_process is None:
        for idx in range(num_splits):
            args.eval_id = idx
            train_model(args)
            
    elif isinstance(args.n_process, int):
        
        from torch.multiprocessing import Pool
        
        input_args = [(args, eval_id) for eval_id in range(num_splits)]
        with Pool(processes=args.n_process) as p:
            combined_results = p.map(multiprocess_wrapper, input_args)