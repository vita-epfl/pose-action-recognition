import os
import numpy as np
import copy 
import argparse
import datetime
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
torch.autograd.set_detect_anomaly(True)

from threading import Lock
from torch.multiprocessing import Pool
from models import MonolocoModel, TempMonolocoModel
from utils.tcg_metrics import compute_accuracy, get_all_predictions, get_eval_metrics
from utils.tcg_dataset import TCGDataset, TCGSingleFrameDataset, tcg_collate_fn

# define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# set value for some arguments 
parser = argparse.ArgumentParser() 
# # local paths
parser.add_argument("--data_dir", type=str, default="poseact/data/", help="dataset folder, should end with /")
parser.add_argument("--fig_dir", type=str, default="poseact/figs/", help="path to save figures, should end with /")
parser.add_argument("--weight_dir", type=str, default="poseact/models/trained/", help="path to save trained models, end with /")
parser.add_argument("--result_dir", type=str, default="poseact/out/results/", help="training logs dir, end with /")

# remote paths 
# parser.add_argument("--data_dir", type=str, default="./data/tcg_dataset/", help="dataset folder, should end with /")
# parser.add_argument("--fig_dir", type=str, default="./figs/", help="path to save figures, should end with /")
# parser.add_argument("--weight_dir", type=str, default="./models/trained/", help="path to save trained models, end with /")
# parser.add_argument("--result_dir", type=str, default="./out/results/", help="training logs dir, end with /")

parser.add_argument("--label_type", type=str, default="major", help="major for 4 classes, sub for 15 classes")
parser.add_argument("--batch_size", type=int, default=128, help="batch size, use a small value (1) for sequence model")
parser.add_argument("--num_epoch", type=int, default=50, help="number of training epochs")
parser.add_argument("--linear_size", type=int, default=256, help="size of hidden linear layer")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--n_stage", type=int, default=3, help="number of stages in a monoloco model")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate") 
parser.add_argument("--model_type", type=str, choices=["single", "sequence"], default="single", 
                    help="train single frame model or sequenced model")
parser.add_argument("--eval_type", type=str, default="xs", help="cross-subject (xs) or cross-view (xv) evaluation")
parser.add_argument("--return_pred", action="store_true", help="return prediction results for the whole test set")
parser.add_argument("--n_process", type=int, default=None, help="number of process for multiprocessing, or None to run in serial")
parser.add_argument("--debug", action="store_true", help="debug mode, use a small fraction of datset")
parser.add_argument("--save_res", action="store_true", help="store training log and trained network")
parser.add_argument("--verbose", action="store_true", help="being more verbose, like print average loss at each epoch")
parser.add_argument("--relative_kp", action="store_true", help="use relative coordinates")
parser.add_argument("--use_velocity", action="store_true", help="add velocity to key points")

def multiprocess_wrapper(args_and_eval_id):
    # args_and_eval_id is a tuple with elements (args, eval_id)
    args, eval_id = args_and_eval_id
    args.eval_id = eval_id
    return train_model(args)

def train_model(args):
    
    # to lock stdout and avoid overlapped printing 
    process_lock:Lock = Lock()
    if args.n_process is not None:
        process_lock.acquire()
        print("subprocess {} is beginning to train a {} model".format(os.getpid(), args.model_type))
        process_lock.release()
    else:
        process_lock.acquire()
        print("beginning to train a {} model".format(args.model_type))
        process_lock.release()
        
    trainset = TCGDataset(args.data_dir, args.label_type, args.eval_type, args.eval_id, training=True, 
                          relative_kp=args.relative_kp, use_velocity=args.use_velocity)
    testset = TCGDataset(args.data_dir, args.label_type, args.eval_type, args.eval_id, training=False, 
                         relative_kp=args.relative_kp, use_velocity=args.use_velocity)
    if args.debug:
        print("using a 2 epochs and first 2 sequences for debugging")
        args.num_epoch = 2
        trainset.seqs = trainset.seqs[:2]
        testset.seqs = testset.seqs[:2]
        if args.model_type == "sequence":
            args.batch_size = 2

    # choose a single frame model or a sequence version of monoloco  
    if args.model_type == "single":
        trainset, testset = TCGSingleFrameDataset(trainset), TCGSingleFrameDataset(testset)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        model = MonolocoModel(trainset.n_feature, args.output_size, args.linear_size, args.dropout, args.n_stage).to(device)
    elif args.model_type == "sequence":
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=tcg_collate_fn, drop_last=True)
        # always run one sequence for testing, so no need for padding, and no artificial data padded 
        testloader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=tcg_collate_fn)
        model = TempMonolocoModel(trainset.n_feature, args.output_size, args.linear_size, args.dropout, args.n_stage).to(device)
    

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
        if args.n_process == None and args.verbose:
            # only print this in single frame mode, or the output will be unordered 
            print("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}".format(epoch, train_loss, test_acc))
        elif args.verbose:
            print("Subprocess {} Epoch {} Avg Loss {:.4f} Test Acc {:.4f}".format(os.getpid(), epoch, train_loss, test_acc))
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
    

    # save training records  
    model_prefix = "TCGSingleFrame_" if args.model_type == "single" else "TCGSeqModel_"
    task_prefix = "{}_{}_".format(args.eval_type, args.eval_id)
    time_prefix = "{}".format(datetime.datetime.now()).replace(" ", "_").replace(":", ".")
    save_model_path = args.weight_dir + model_prefix + task_prefix + time_prefix + ".pth"
    save_info_path  = args.result_dir + model_prefix + task_prefix + time_prefix + ".txt"
    
    model.load_state_dict(best_weights)
    
    if args.save_res:
        torch.save(model.state_dict(), save_model_path)
        
        with open(save_info_path, "w") as f:
            f.write("label_{}_lr_{}_hidden_{}_drop_{}_stage_{}\n".format(args.label_type,
                                    args.lr, args.linear_size, args.dropout, args.n_stage))
            for epoch, (loss, acc) in enumerate(zip(train_loss_list, test_acc_list)):
                f.write("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}\n".format(epoch, loss, acc))
            
    if args.n_process is not None:
        process_lock.acquire() 
        print("subprocess {} has finished training a {} model with best testing accuracy {}".format(os.getpid(), args.model_type, best_test_acc))
        process_lock.release()
        
    if args.return_pred:
        results = get_all_predictions(model, testloader)
        return results
    
if __name__ == "__main__":
    # start REAL sub process as specified here https://pytorch.org/docs/stable/notes/multiprocessing.html
    mp.set_start_method('spawn')
    
    # args = parser.parse_args(["--model_type", "sequence", "--eval_type", "xv", "--debug", "--num_epoch", "2", "--return_pred", "--n_process", "3"])
    args = parser.parse_args(["--model_type", "single", "--eval_type", "xv", "--debug", "--num_epoch", "2", "--return_pred", "--relative_kp", "--use_velocity"])
    args.output_size = TCGDataset.get_output_size(args.label_type)
    num_splits = TCGDataset.get_num_split(args.eval_type)
    
    if args.n_process is None:
        print("Starting to run the experiment in serial")
        combined_results = []
        for idx in range(num_splits):
            print("This is the {} th split in the TCG dataset".format(idx))
            args.eval_id = idx
            result = train_model(args)
            combined_results.append(result)
            
    elif isinstance(args.n_process, int):
        print("Starting to run the experiment with {} subprocesses".format(args.n_process))
        input_args = [(args, eval_id) for eval_id in range(num_splits)]
        with Pool(processes=args.n_process) as p:
            combined_results = p.map(multiprocess_wrapper, input_args)
            
    print("Done training")
    
    print("Begin to calculate various metrics over all the train-test splits")
    acc, f1, jac, cfx = get_eval_metrics(combined_results)
    print("Average results over all the splits:")
    print("accuracy {:.4f} f1 score {:.4f} Jaccard score {:.4f}".format(acc, f1, jac))
    print("Confusion matrix (rows are true label, columns are predicted):")
    print(cfx)