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
# from utils import compute_accuracy, get_all_predictions, get_eval_metrics
from utils.losses import MultiHeadClfLoss
from titan_dataset import TITANDataset, TITANSimpleDataset, Person, Vehicle, Sequence, Frame
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix, accuracy_score

# define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# set value for some arguments 
parser = argparse.ArgumentParser() 

# # local paths
# parser.add_argument("--pifpaf_out", type=str, default="codes/out/pifpaf_results/", help="pifpaf output folder, end with /")
# parser.add_argument("--dataset_dir", type=str, default="codes/data/", help="original TITAN dataset folder, should end with /")
# parser.add_argument("--save_dir", type=str, default="codes/out/", help="saved pickle file of the poses, should end with /")
# parser.add_argument("--fig_dir", type=str, default="codes/figs/", help="path to save figures, should end with /")
# parser.add_argument("--weight_dir", type=str, default="codes/models/trained/", help="path to save trained models, end with /")
# parser.add_argument("--result_dir", type=str, default="codes/out/results/", help="training logs dir, end with /")

# # remote paths 
parser.add_argument("--pifpaf_out", type=str, default="./out/pifpaf_results/", help="pifpaf output folder, end with /")
parser.add_argument("--dataset_dir", type=str, default="./data/TITAN/", help="original TITAN dataset folder, should end with /")
parser.add_argument("--save_dir", type=str, default="./out/", help="saved pickle file of the poses, should end with /")
parser.add_argument("--fig_dir", type=str, default="./figs/", help="path to save figures, should end with /")
parser.add_argument("--weight_dir", type=str, default="./models/trained/", help="path to save trained models, end with /")
parser.add_argument("--result_dir", type=str, default="./out/results/", help="training logs dir, end with /")

parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--num_epoch", type=int, default=50, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate") 
parser.add_argument("--workers", type=int, default=0, help="number of workers for dataloader") 
parser.add_argument("--input_size", type=int, default=51, help="input size, number of joints times feature dimension")
parser.add_argument("--linear_size", type=int, default=256, help="size of hidden linear layer")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--n_stage", type=int, default=3, help="number of stages in a monoloco model")

# parser.add_argument("--return_pred", action="store_true", help="return prediction results for the whole test set")
parser.add_argument("--debug", action="store_true", help="debug mode, use a small fraction of datset")
# parser.add_argument("--save_res", action="store_true", help="store training log and trained network")
# parser.add_argument("--verbose", action="store_true", help="being more verbose, like print average loss at each epoch")

def compute_accuracy(model:nn.Module, testloader:DataLoader):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            for one_pred, one_label in zip(pred, label.permute(1, 0)):
                _, predicted = torch.max(one_pred.data, -1)
                correct += (predicted == one_label).sum().item()
            total += label.numel()
            
    return correct / total

def get_all_predictions(model:nn.Module, testloader:DataLoader):
    """ all prediction results of a model on a test set as well as the true label 
        return two lists of numpy array, will be good for sklearn metrics, like sklearn.metrics.f1_score
    """
    
    device = next(model.parameters()).device
    model.eval()
    label_list, result_list = [[] for _ in range(5)], [[] for _ in range(5)]
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            for idx, (one_pred, one_label) in enumerate(zip(pred, label.permute(1, 0))):
                _, pred_class = torch.max(one_pred.data, -1)

                result_list[idx].append(pred_class)
                label_list[idx].append(one_label)
            
    # result list is a list for the 5 layers of actions in TITAN hierarchy 
    # communicative, complex_context, atomic, simple_context, transporting
    result_list = [torch.cat(one_list, dim=0).cpu().detach().numpy() for one_list in result_list]
    label_list = [torch.cat(one_list, dim=0).cpu().detach().numpy() for one_list in label_list]
    
    return result_list, label_list

def get_eval_metrics(result_list, label_list, n_classes=[4, 7, 9, 13, 4]):
    """ 
    """
    acc_list, f1_list, jac_list, cfx_list = [[] for _ in range(4)]
    for idx, (pred, label) in enumerate(zip(result_list, label_list)):
        acc = accuracy_score(y_true=label, y_pred=pred)
        f1 = f1_score(y_true=label, y_pred=pred, average='macro')
        jac = jaccard_score(y_true=label, y_pred=pred, average='macro')
        cfx = confusion_matrix(y_true=label, y_pred=pred, labels=range(n_classes[idx]))
        acc_list.append(acc)
        f1_list.append(f1)
        jac_list.append(jac)
        cfx_list.append(cfx)
    
    return  acc_list, f1_list, jac_list, cfx_list


if __name__ == "__main__":
    
    args = parser.parse_args()
    args.output_size = [4, 7, 9, 13, 4]
    # prepare train, validation and test splits, as well as the dataloaders 
    trainset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.save_dir, True, "train")
    valset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.save_dir, True, "val")
    testset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.save_dir, True, "test")
    
    trainset = TITANSimpleDataset(trainset)
    valset = TITANSimpleDataset(valset)
    testset = TITANSimpleDataset(testset)
    
    if args.debug:
        trainset = Subset(trainset, indices=range(5000))
        valset = Subset(trainset, indices=range(5000))
        testset = Subset(trainset, indices=range(5000))
        
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    valloader = DataLoader(valset, batch_size=args.batch_size,shuffle=False,
                           num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    
    model = MultiHeadMonoLoco(args.input_size, args.output_size, args.linear_size, args.dropout, args.n_stage).to(device)
    
    criterion = MultiHeadClfLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    
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
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
        # scheduler.step(train_loss)
        print("Epoch {} Avg Loss {:.4f} test Acc {:.4f}".format(epoch, train_loss, test_acc))

        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
    
    result_list, label_list = get_all_predictions(model, testloader)
    acc, f1, jac, cfx = get_eval_metrics(result_list, label_list)
    
    action_hierarchy = ["communicative", "complex_context", "atomic", "simple_context", "transporting"]
    for idx, layer in enumerate(action_hierarchy):
        print("For {} actions accuracy {:.4f} f1 score {:.4f} Jaccard score {:.4f}".format(layer, acc[idx], f1[idx], jac[idx]))
        print("Confusion matrix (elements in a row share the same true label, those in the same columns share predicted):")
        print(cfx[idx])
        
    
    