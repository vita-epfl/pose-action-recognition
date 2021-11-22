import os
import numpy as np
import copy 
import argparse
import datetime
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
torch.autograd.set_detect_anomaly(True)

from models import MultiHeadMonoLoco
from utils.titan_metrics import compute_accuracy, get_all_predictions, get_eval_metrics, per_class_acc
from utils.losses import MultiHeadClfLoss
from titan_dataset import TITANDataset, TITANSimpleDataset, Person, Vehicle, Sequence, Frame
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix, accuracy_score

# define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def manual_add_arguments(args):
    """
        manually specify the folder directory
    """
    args.output_size = [4, 7, 9, 13, 4]
    args.pifpaf_out = "{}/out/pifpaf_results/".format(args.base_dir) # pifpaf output folder, end with /
    args.dataset_dir = "{}/data/TITAN/".format(args.base_dir) # original TITAN dataset folder, should end with / 
    args.save_dir = "{}/out/".format(args.base_dir) # saved pickle file of the poses, should end with /
    args.fig_dir = "{}/figs/".format(args.base_dir) # path to save figures, should end with /
    args.weight_dir = "{}/out/trained/".format(args.base_dir) # path to save trained models, end with /
    args.result_dir = "{}/out/results/".format(args.base_dir) # training logs dir, end with /
    return args

# set value for some arguments 
parser = argparse.ArgumentParser() 

# base path
parser.add_argument("--base_dir", type=str, default=".", help="root directory of the codes")

parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--num_epoch", type=int, default=50, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate") 
parser.add_argument("--workers", type=int, default=0, help="number of workers for dataloader") 
parser.add_argument("--input_size", type=int, default=34, help="input size, number of joints times feature dimension")
parser.add_argument("--linear_size", type=int, default=256, help="size of hidden linear layer")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--n_stage", type=int, default=3, help="number of stages in a monoloco model")

# loss related arguments 
parser.add_argument("--n_tasks", type=int, default=5, help="number of tasks for multi-task loss, 5 for TITAN")
parser.add_argument("--imbalance", type=str, default="manual", choices=["manual", "focal", "both"], 
                    help="method to tackle imbalanced data")
parser.add_argument("--gamma", type=float, default=1.5, help="the gamma parameter for focal loss, should be a positive integer")
parser.add_argument("--anneal_factor", type=float, default=0.0, help="annealing factor for alpha balanced cross entropy")
parser.add_argument("--uncertainty", action="store_true", help="use task uncertainty")
parser.add_argument("--mask_cls", action="store_true", help="maskout some unlearnable classes")

parser.add_argument("--task_name", type=str, default="Baseline", help="a name for this training task, used in save name")
parser.add_argument("--select_best", action="store_true", help="select the checkpoint with best validation accuracy")
parser.add_argument("--test_only", action="store_true", help="run a test on a pretrained model")
parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file name usually a xxxx.pth file in args.weight_dir")
parser.add_argument("--debug", action="store_true", help="debug mode, use a small fraction of datset")
parser.add_argument("--save_model", action="store_true", help="store trained network")
parser.add_argument("--verbose", action="store_true", help="being more verbose, like print average loss at each epoch")

if __name__ == "__main__":

    args = parser.parse_args()
    args = manual_add_arguments(args)
    
    # prepare train, validation and test splits, as well as the dataloaders 
    trainset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.save_dir, True, "train")
    valset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.save_dir, True, "val")
    testset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.save_dir, True, "test")
    
    trainset = TITANSimpleDataset(trainset)
    valset = TITANSimpleDataset(valset)
    testset = TITANSimpleDataset(testset)
    
    if args.debug:
        print("using a 2 epochs and 1000 samples for debugging")
        args.num_epoch = 2
        trainset = Subset(trainset, indices=range(1000))
        valset = Subset(trainset, indices=range(1000))
        testset = Subset(trainset, indices=range(1000))
        
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    valloader = DataLoader(valset, batch_size=args.batch_size,shuffle=False,
                           num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    
    model = MultiHeadMonoLoco(args.input_size, args.output_size, args.linear_size, args.dropout, args.n_stage).to(device)
    
    # load a pretrained model if specified 
    if args.test_only and args.ckpt is not None:
        pretrained = "{}/{}".format(args.weight_dir, args.ckpt)
        try:
            model.load_state_dict(torch.load(pretrained))
            args.num_epoch = 0 # don't train
        except:
            print("failed to load pretrained, train from scratch instead")
        
    criterion = MultiHeadClfLoss(n_tasks=args.n_tasks, imbalance=args.imbalance, gamma=args.gamma, 
                                 anneal_factor=args.anneal_factor, uncertainty=args.uncertainty, device=device)
    # criterion.parameters will be an empty list if uncertainty is false 
    params = list(model.parameters()) + list(criterion.parameters()) 
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
        time_suffix = "{}".format(datetime.datetime.now()).replace(" ", "_").replace(":", ".")
        filename = "{}/TITAN_{}_{}.pth".format(args.weight_dir, args.task_name, time_suffix)
        torch.save(model.state_dict(), filename)
        print("model saved to {}".format(filename))
    
    result_list, label_list, score_list = get_all_predictions(model, testloader)
    acc, f1, jac, cfx, ap = get_eval_metrics(result_list, label_list, score_list)
    
    print("In general, overall accuracy {:.4f} avg Jaccard {:.4f} avg F1 {:.4f}".format(
                                np.mean(acc), np.mean(jac), np.mean(f1)))
    
    action_hierarchy = ["communicative", "complex_context", "atomic", "simple_context", "transporting"]
    for idx, layer in enumerate(action_hierarchy):
        # some classes have 0 instances (maybe) and recalls will be 0, resulting in a nan
        print("")
        print("For {} actions accuracy {:.4f} Jaccard score {:.4f} f1 score {:.4f} mAP {:.4f}".format(
            layer, acc[idx], jac[idx], f1[idx], np.mean(ap[idx])))
        print("Accuracy for each class: {}".format(per_class_acc(cfx[idx])))
        print("Average Precision for each class is {}".format(np.round(ap[idx], decimals=4).tolist()))
        print("Confusion matrix (elements in a row share the same true label, those in the same columns share predicted):")
        print("The corresponding classes are {}".format(Person.get_attr_dict(type=layer)))
        print(cfx[idx])
        print("")
        
    
    