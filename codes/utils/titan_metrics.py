import torch 
import numpy as np 
import torch.nn as nn 
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, jaccard_score, confusion_matrix, accuracy_score

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