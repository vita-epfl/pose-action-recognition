import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    f1_score, 
    jaccard_score, 
    confusion_matrix, 
    accuracy_score,
    average_precision_score
    )

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
    if isinstance(testloader.dataset, Subset):
        n_tasks = len(testloader.dataset.dataset.n_cls)
    else:
        n_tasks = len(testloader.dataset.n_cls)
        
    label_list, result_list, score_list = [[[] for _ in range(n_tasks)] for _ in range(3)]
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            for idx, (one_pred, one_label) in enumerate(zip(pred, label.permute(1, 0))):
                _, pred_class = torch.max(one_pred.data, -1)

                result_list[idx].append(pred_class)
                label_list[idx].append(one_label)
                score_list[idx].append(one_pred)
            
    # result list is a list for the 5 layers of actions in TITAN hierarchy 
    # communicative, complex_context, atomic, simple_context, transporting
    result_list = [torch.cat(one_list, dim=0).cpu().detach().numpy() for one_list in result_list]
    label_list = [torch.cat(one_list, dim=0).cpu().detach().numpy() for one_list in label_list]
    score_list = [torch.cat(one_list, dim=0).cpu().detach().numpy() for one_list in score_list]
    
    return result_list, label_list, score_list

def to_one_hot(n_cls, label):
    """ 
        n_cls (int): number of classes
        label (tensor): true label of the data, tensor format 
    """
    one_hot_label = np.eye(n_cls)[label]
    return one_hot_label

def softmax(score:np.ndarray):
    return F.softmax(torch.tensor(score), dim=-1).detach().numpy() 

def per_class_acc(cfx_mtx):
    correct = np.diagonal(cfx_mtx)
    total = np.sum(cfx_mtx, axis=1)
    return np.nan_to_num(correct/total, nan=0)

def get_eval_metrics(result_list, label_list, score_list):
    """ 
    """
    n_classes = [int(score.shape[1]) for score in score_list]
    acc_list, f1_list, jac_list, cfx_list, ap_list = [[] for _ in range(5)]
    for idx, (pred, label, score) in enumerate(zip(result_list, label_list, score_list)):
        acc = accuracy_score(y_true=label, y_pred=pred)
        f1 = f1_score(y_true=label, y_pred=pred, average='macro')
        jac = jaccard_score(y_true=label, y_pred=pred, average='macro')
        cfx = confusion_matrix(y_true=label, y_pred=pred, labels=range(n_classes[idx]))
        # the network outputs logits, so use softmax to convert them to prediction scores 
        ap = average_precision_score(y_true=to_one_hot(n_classes[idx], label), 
                                     y_score=softmax(score), average=None) 
        ap = np.nan_to_num(ap, nan=0)
        acc_list.append(acc)
        f1_list.append(f1)
        jac_list.append(jac)
        cfx_list.append(cfx)
        ap_list.append(ap)
    
    return  acc_list, f1_list, jac_list, cfx_list, ap_list