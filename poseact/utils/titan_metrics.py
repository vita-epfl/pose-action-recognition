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
    """ all prediction results, true label of a model on a test set as well as prediction score
        return three lists of numpy array, will be good for sklearn metrics, like sklearn.metrics.f1_score
        the length of each list is the number of action sets, in titan it would be 5
        
        shape each element in the lists:
            result_list: (n_samples, 1), predicted class for an action set for each sample in the test loader
            label_list: (n_samples, 1) true label
            score_list: (n_samples, n_classes) n_classses is the number of classes in this action set 
    """
    
    device = next(model.parameters()).device
    model.eval()
    n_tasks = len(model.output_size)
        
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

def per_class_recall(cfx_mtx):
    correct = np.diagonal(cfx_mtx)
    total = np.sum(cfx_mtx, axis=1)
    return np.nan_to_num(correct/total, nan=0)

def per_class_precision(cfx_mtx):
    correct = np.diagonal(cfx_mtx)
    total = np.sum(cfx_mtx, axis=0)
    return np.nan_to_num(correct/total, nan=0)

def per_class_f1(cfx_mtx):
    precision = per_class_precision(cfx_mtx)
    recall = per_class_recall(cfx_mtx)
    return np.nan_to_num(2*precision*recall/(precision+recall), nan=0)

def get_eval_metrics(result_list, label_list, score_list):
    """ see `get_all_predictions` for input shape 
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