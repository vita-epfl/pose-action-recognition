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
from poseact.utils.casr_dataset import Person
from poseact.utils.losses import IGNORE_INDEX 

def compute_accuracy(model:nn.Module, testloader:DataLoader, is_sequence=False):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            if is_sequence:
                N, T, C = pred.shape 
                pred = pred.view(N*T, C)
                label = label.view(N*T)
            _, predicted = torch.max(pred.data, -1)
            valid_idx = torch.not_equal(label, IGNORE_INDEX)
            label = label[valid_idx]
            predicted = predicted[valid_idx]
            correct += (predicted == label).sum().item()
            total += label.numel()
            
    return correct / total

def get_all_predictions(model:nn.Module, testloader:DataLoader, is_sequence=False):
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
        
    label_list, result_list, score_list = [[] for _ in range(3)]
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            if is_sequence:
                N, T, C = pred.shape 
                pred = pred.view(N*T, C)
                label = label.view(N*T)
            _, pred_class = torch.max(pred.data, -1)
            result_list.append(pred_class)
            label_list.append(label)
            score_list.append(pred)
    
    results = torch.cat(result_list, dim=0).cpu().detach().numpy()
    labels = torch.cat(label_list, dim=0).cpu().detach().numpy()
    scores = torch.cat(score_list, dim=0).cpu().detach().numpy()
    
    valid_idx = np.not_equal(labels, IGNORE_INDEX)
    results = results[valid_idx]
    labels = labels[valid_idx] 
    scores = scores[valid_idx]
    
    return results, labels, scores

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

def get_eval_metrics(combined_results):
    """ see `get_all_predictions` for input shape 
    """
    n_classes = len(np.unique(list(Person.action_dict.values())))
    acc_list, f1_list, jac_list, cfx_list = [[] for _ in range(4)]
    for pred, label, _ in combined_results:
        acc = accuracy_score(y_true=label, y_pred=pred)
        f1 = f1_score(y_true=label, y_pred=pred, average='macro')
        jac = jaccard_score(y_true=label, y_pred=pred, average='macro')
        cfx = confusion_matrix(y_true=label, y_pred=pred, labels=range(n_classes))
        acc_list.append(acc)
        f1_list.append(f1)
        jac_list.append(jac)
        cfx_list.append(cfx)
    
    return  acc_list, f1_list, jac_list, cfx_list

def summarize_results(acc, f1, jac, cfx):
    
    print("In general, overall accuracy {:.4f} avg Jaccard {:.4f} avg F1 {:.4f}".format(np.mean(acc), np.mean(jac), np.mean(f1)))
    print("Best accuracy {:.4f} best Jaccard {:.4f} best F1 {:.4f}".format(np.max(acc), np.max(jac), np.max(f1)))
    print("Worst accuracy {:.4f} Worst Jaccard {:.4f} Worst F1 {:.4f}".format(np.min(acc), np.min(jac), np.min(f1)))
    print("Confusion matrix (elements in a row share the same true label, those in the same columns share predicted):")
    print("The corresponding classes are {}".format(Person.action_dict))
    avg_cfx = np.mean(cfx, axis=0)
    print(avg_cfx)
    class_prec, class_rec,class_f1 = per_class_precision(avg_cfx), per_class_recall(avg_cfx),per_class_f1(avg_cfx)
    print("Precision for each class: {}".format(class_prec))
    print("Recall for each class: {}".format(class_rec))
    print("F1 score for each class: {}".format(class_f1))