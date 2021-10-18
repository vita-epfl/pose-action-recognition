import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader

# This function computes the accuracy on the test dataset
def compute_accuracy(model:nn.Module, testloader:DataLoader):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            _, predicted = torch.max(pred.data, -1)
            total += label.numel()
            correct += (predicted == label).sum().item()
    return correct / total

def get_all_predictions(model:nn.Module, testloader:DataLoader):
    """ all prediction results of a model on a test set as well as the true label 
        return a numpy array, will be good for sklearn metrics, like sklearn.metrics.f1_score
    """
    
    device = next(model.parameters()).device
    model.eval()
    label_list, result_list = [], []
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose) 
            _, pred_class = torch.max(pred.data, -1)

            result_list.append(pred_class)
            label_list.append(label)
            
    if len(result_list[0].shape) == 2: # sequence model 
        assert testloader.batch_size == 1, "please use batch size 1 for testing, to avoide padding artificial data"
        result_list = [result.reshape(-1) for result in result_list]
        label_list = [label.reshape(-1) for label in label_list]
        
    elif len(result_list[0].shape) == 1: # single frame model 
        N = result_list[0].shape
    
    all_result_tensor = torch.cat(result_list, dim=0)
    all_label_tensor = torch.cat(label_list, dim=0)
    
    return all_result_tensor.cpu().detach().numpy(), all_label_tensor.cpu().detach().numpy()