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
from poseact.utils.losses import MultiHeadClfLoss
from poseact.models import MultiHeadMonoLoco, multihead_resnet
from poseact.utils.titan_metrics import compute_accuracy, get_all_predictions, get_eval_metrics, summarize_results
from poseact.utils.titan_dataset import TITANDataset, TITANSimpleDataset, Person, Vehicle, Sequence, Frame

# define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def manual_add_arguments(args):
    """
        manually specify the default folders
    """
    if getattr(args, "pifpaf_out", None) is None:
        args.pifpaf_out = "{}/out/pifpaf_results/".format(args.base_dir) # pifpaf output folder, end with /
    if getattr(args, "dataset_dir", None) is None:
        args.dataset_dir = "{}/data/TITAN/".format(args.base_dir) # original TITAN dataset folder, should end with / 
    if getattr(args, "pickle_dir", None) is None:
        args.pickle_dir = "{}/out/".format(args.base_dir) # saved pickle file of the poses, should end with /
    if getattr(args, "save_dir", None) is None:
        args.save_dir = "{}/out/recognition/".format(args.base_dir) # path to qualitative results, should end with /
    if getattr(args, "fig_dir", None) is None:
        args.fig_dir = "{}/figs/".format(args.base_dir) # path to save figures, should end with /
    if getattr(args, "weight_dir", None) is None:
        args.weight_dir = "{}/out/trained/".format(args.base_dir) # path to save trained models, end with /
    if getattr(args, "result_dir", None) is None:
        args.result_dir = "{}/out/results/".format(args.base_dir) # training logs dir, end with /
    return args

# set value for some arguments 
parser = argparse.ArgumentParser() 

# base path
parser.add_argument("--base_dir", type=str, default=".", help="root directory of the codes")
parser.add_argument("--pifpaf_out", type=str, default=None, help="pifpaf output folder")
parser.add_argument("--dataset_dir", type=str, default=None, help="original TITAN dataset folder")
parser.add_argument("--pickle_dir", type=str, default=None, help="saved pickle file of the poses")
parser.add_argument("--save_dir", type=str, default=None, help="path to qualitative results")
parser.add_argument("--fig_dir", type=str, default=None, help="path to save figures")
parser.add_argument("--weight_dir", type=str, default=None, help="path to save trained models")
parser.add_argument("--result_dir", type=str, default=None, help="training logs dir")

# model and training related arguments 
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--num_epoch", type=int, default=50, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate") 
parser.add_argument("--workers", type=int, default=0, help="number of workers for dataloader") 
# parser.add_argument("--input_size", type=int, default=34, help="input size, number of joints times feature dimension")
parser.add_argument("--linear_size", type=int, default=128, help="size of hidden linear layer")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--n_stage", type=int, default=3, help="number of stages in a monoloco model")
parser.add_argument("--torchhub_pretrain", action="store_true", help="use a pretrained resnet from torch hub")
parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file name usually a xxxx.pth file in args.weight_dir")

# dataset related arguments
parser.add_argument("--merge_cls", action="store_true", 
                    help="completely remove unlearnable classes, and merge the multiple action sets into one")
parser.add_argument("--inflate", type=float, default=None, 
                    help="inflate the minority classes to some proportion of the majority class")
parser.add_argument("--relative_kp", action="store_true", 
                    help="turn the absolute key point coordinates into center + relative")
parser.add_argument("--rm_center", action="store_true", help="remove the center point")
parser.add_argument("--normalize", action="store_true", help="divide the (x, y) of a point by (w, h) of the bbox")
parser.add_argument("--use_img", action="store_true", 
                    help="crop patches from the original image, don't use poses")
parser.add_argument("--drop_last", action="store_false", help="drop the last batch (only use in training), True if not set")

# loss related arguments 
# parser.add_argument("--n_tasks", type=int, default=5, help="number of tasks for multi-task loss, 5 for TITAN")
parser.add_argument("--imbalance", type=str, default="manual", choices=["manual", "focal", "both"], 
                    help="method to tackle imbalanced data")
parser.add_argument("--gamma", type=float, default=1.5, help="the gamma parameter for focal loss, should be a positive integer")
parser.add_argument("--anneal_factor", type=float, default=0.0, help="annealing factor for alpha balanced cross entropy")
parser.add_argument("--uncertainty", action="store_true", help="use task uncertainty")
parser.add_argument("--mask_cls", action="store_true", help="maskout some unlearnable classes")

# logging related arguments 
parser.add_argument("--task_name", type=str, default="Baseline", help="a name for this training task, used in save name")
parser.add_argument("--select_best", action="store_true", help="select the checkpoint with best validation accuracy")
parser.add_argument("--test_only", action="store_true", help="run a test on a pretrained model")
parser.add_argument("--debug", action="store_true", help="debug mode, use a small fraction of datset")
parser.add_argument("--save_model", action="store_true", help="store trained network")
parser.add_argument("--verbose", action="store_true", help="being more verbose, like print average loss at each epoch")

if __name__ == "__main__":
    
    setup_multiprocessing()
    
    # ["--debug","--base_dir", "poseact", "--imbalance", "focal", "--gamma", "2", "--save_model", "--merge_cls", "--use_img"]
    # ["--debug","--base_dir", "poseact", "--imbalance", "focal", "--gamma", "2", "--save_model", "--merge_cls", "--relative_kp", "--normalize", "--rm_center"]
    # ["--base_dir", "poseact", "--linear_size", "128", "--test_only", "--ckpt", "TITAN_Relative_KP803217.pth"]
    args = parser.parse_args(["--base_dir", "poseact", "--linear_size", "128",  "--relative_kp", "--merge_cls"])
    args = manual_add_arguments(args)
    
    # prepare train, validation and test splits, as well as the dataloaders 
    trainset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.pickle_dir, True, "train")
    valset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.pickle_dir, True, "val")
    testset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.pickle_dir, True, "test")
    
    if args.debug:
        print("using a 2 epochs and first 2 sequences for debugging")
        args.num_epoch = 2
        trainset.seqs = trainset.seqs[:2]
        valset.seqs = trainset.seqs[:2]
        testset.seqs = trainset.seqs[:2]
        for i in range(2):
            trainset.seqs[i].frames = trainset.seqs[i].frames[:5]
            valset.seqs[i].frames = valset.seqs[i].frames[:5]
            testset.seqs[i].frames = testset.seqs[i].frames[:5]
        
    trainset = TITANSimpleDataset(trainset, merge_cls=args.merge_cls, inflate=args.inflate, use_img=args.use_img,
                                  relative_kp=args.relative_kp, rm_center=args.rm_center, normalize=args.normalize)
    valset = TITANSimpleDataset(valset, merge_cls=args.merge_cls, inflate=args.inflate, use_img=args.use_img,
                                  relative_kp=args.relative_kp, rm_center=args.rm_center, normalize=args.normalize)
    testset = TITANSimpleDataset(testset, merge_cls=args.merge_cls, inflate=args.inflate, use_img=args.use_img,
                                  relative_kp=args.relative_kp, rm_center=args.rm_center, normalize=args.normalize)
        
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.workers, collate_fn=TITANSimpleDataset.collate, drop_last=args.drop_last)
    valloader = DataLoader(valset, batch_size=args.batch_size,shuffle=False,
                           num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    
    input_size, output_size = trainset.n_feature, trainset.n_cls
    
    if not args.use_img:
        model = MultiHeadMonoLoco(input_size, output_size, args.linear_size, args.dropout, args.n_stage).to(device)
        model_params = list(model.parameters())
    else:
        assert args.merge_cls, "use --merge_cls in commandline"
        ckpt = "{}/{}".format(args.weight_dir, args.ckpt) if args.ckpt else None
        model = multihead_resnet(output_size=output_size, ckpt_path=ckpt, pretrained=args.torchhub_pretrain)
        # if pretrain, then train the last layer only 
        if args.torchhub_pretrain or ckpt:
            model_params = list(model.fc.parameters())
        else:
            model_params = list(model.parameters())
    model.to(device)    
    
    # load a pretrained model if specified 
    if args.test_only and args.ckpt is not None:
        pretrained = "{}/{}".format(args.weight_dir, args.ckpt)
        try:
            model.load_state_dict(torch.load(pretrained))
            args.num_epoch = 0 # don't train
        except:
            print("failed to load pretrained, train from scratch instead")
        
    criterion = MultiHeadClfLoss(n_tasks=len(output_size), imbalance=args.imbalance, gamma=args.gamma, 
                                 anneal_factor=args.anneal_factor, uncertainty=args.uncertainty, 
                                 device=device, mask_cls=args.mask_cls)
    loss_params = list(criterion.parameters()) 
    
    # criterion.parameters will be an empty list if uncertainty is false 
    params = model_params + loss_params
    optimizer = optim.Adam(params=params, lr=args.lr)
    
    # training loop 
    best_test_acc = -1
    train_loss_list, test_acc_list = [], [] 
    for epoch in range(args.num_epoch):
        model.train()
        batch_loss = [] 
        for pose, label in trainloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            # for single frame model, the output shape is (N, C), for sequence model it's (N, T, C)
            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item()) 
        
        train_loss = sum(batch_loss)/len(batch_loss)
        
        test_acc = compute_accuracy(model, valloader)
        if test_acc > best_test_acc and args.select_best:
            best_test_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
        # scheduler.step(train_loss)
        print("Epoch {} Avg Loss {:.4f} test Acc {:.4f}".format(epoch, train_loss, test_acc))

        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
    
    if args.save_model:
        task_name = args.task_name
        slurm_job_id = os.environ.get("SLURM_JOBID", None)
        if slurm_job_id is not None:
            task_name = task_name + str(slurm_job_id)
        time_suffix = "{}".format(datetime.datetime.now()).replace(" ", "_").replace(":", ".")
        filename = "{}/TITAN_{}_{}.pth".format(args.weight_dir, task_name, time_suffix)
        torch.save(model.state_dict(), filename)
        print("model saved to {}".format(filename))
    
    result_list, label_list, score_list = get_all_predictions(model, testloader)
    acc, f1, jac, cfx, ap = get_eval_metrics(result_list, label_list, score_list)
    summarize_results(acc, f1, jac, cfx, ap, args.merge_cls)
        
    
    