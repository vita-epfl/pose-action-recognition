""" 
    we first trained the model on all the categories of actions in TITAN, but the dataset itself 
    contains too many "none of the above", which we think is not a helpful annotation, and induces 
    too much negative samples. Therefore we picked out several learnable classes out of the original
    action hierarchy. Nevertheless, we want to see how the previously trained model behaves on the 
    picked classes. 
"""
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from poseact.models import MultiHeadMonoLoco
from poseact.utils.titan_metrics import get_eval_metrics, summarize_results
from poseact.utils.titan_dataset import TITANDataset, TITANSimpleDataset, Person, Sequence, Frame
from poseact.titan_train import manual_add_arguments 

from sklearn.metrics import (
    f1_score, 
    jaccard_score, 
    confusion_matrix, 
    accuracy_score,
    average_precision_score
    )

# define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# set value for some arguments 
parser = argparse.ArgumentParser() 
parser.add_argument("--base_dir", type=str, default=".", help="root directory of the project")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--num_epoch", type=int, default=50, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate") 
parser.add_argument("--workers", type=int, default=0, help="number of workers for dataloader") 
parser.add_argument("--input_size", type=int, default=34, help="input size, number of joints times feature dimension")
parser.add_argument("--linear_size", type=int, default=256, help="size of hidden linear layer")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--n_stage", type=int, default=3, help="number of stages in a monoloco model")
parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file name usually a xxxx.pth file in args.weight_dir")
parser.add_argument("--merge_cls", action="store_true", help="completely remove unlearnable classes, and merge the multiple action sets into one")

np.set_printoptions(precision=4, suppress=True)

if __name__ == "__main__":

    # ["--base_dir", "poseact", "--linear_size", "128", "--merge_cls", "--ckpt", "TITAN_Baseline_2021-11-04_12.01.49.069328.pth"]
    args = parser.parse_args(["--base_dir", "poseact", "--linear_size", "128", "--merge_cls", "--ckpt", "TITAN_Baseline_2021-11-04_12.01.49.069328.pth"])
    args = manual_add_arguments(args)
    
    testset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.pickle_dir, True, "test")
    testset = TITANSimpleDataset(testset, merge_cls=args.merge_cls)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    
    model = MultiHeadMonoLoco(args.input_size, [4, 7, 9, 13, 4], args.linear_size, args.dropout, args.n_stage).to(device)
    pretrained = "{}/{}".format(args.weight_dir, args.ckpt)
    model.load_state_dict(torch.load(pretrained))
    
    device = next(model.parameters()).device
    model.eval()
    n_tasks = len(model.output_size)
        
    result_list, score_list = [[[] for _ in range(n_tasks)] for _ in range(2)]
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            for idx, one_pred in enumerate(pred):
                _, pred_class = torch.max(one_pred.data, -1)

                result_list[idx].append(pred_class)
                score_list[idx].append(one_pred)
    
    label_list= []
    with torch.no_grad():
        for _, label in testloader:
            label_list.append(label)

    # result list is a list for the 5 layers of actions in TITAN hierarchy 
    # communicative, complex_context, atomic, simple_context, transporting
    result_list = [torch.cat(one_list, dim=0).cpu().detach().numpy() for one_list in result_list]
    score_list = [torch.cat(one_list, dim=0).cpu().detach().numpy() for one_list in score_list]
    label_list = [torch.cat(label_list, dim=0).cpu().detach().numpy().flatten()]
    
    final_result_list = [np.zeros(label_list[0].shape).flatten()]
    final_score_list = [np.zeros((label_list[0].shape[0], 5))]
    
    walking = (result_list[2] == 7)
    standing = (result_list[2] == 6)
    sitting = (result_list[2] == 4)
    bending = (result_list[2] == 0)
    biking = (result_list[3] == 0)
    motorcycling = (result_list[3] == 7)
    
    final_result_list[0][walking] = 0
    final_result_list[0][standing] = 1
    final_result_list[0][sitting] = 2
    final_result_list[0][bending] = 3
    final_result_list[0][biking] = 4
    final_result_list[0][motorcycling] = 4
    
    result_list = final_result_list
    
    final_score_list[0][:, 0] = score_list[2][:, 7]
    final_score_list[0][:, 1] = score_list[2][:, 6]
    final_score_list[0][:, 2] = score_list[2][:, 4]
    final_score_list[0][:, 3] = score_list[2][:, 0]
    final_score_list[0][:, 4] = np.max(np.concatenate((score_list[3][:, 0].reshape(-1,1), 
                                                       score_list[3][:, 7].reshape(-1,1)), axis=1), axis=1)
    
    score_list = final_score_list
    
    acc, f1, jac, cfx, ap = get_eval_metrics(result_list, label_list, score_list)
    summarize_results(acc, f1, jac, cfx, ap, merge_cls=True)
        
    
    