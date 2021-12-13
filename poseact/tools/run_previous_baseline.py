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
from poseact.utils.titan_metrics import per_class_precision, per_class_recall, per_class_f1
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
parser.add_argument("--base_dir", type=str, default="..", help="root directory of the project")
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

if __name__ == "__main__":

    # ["--base_dir", "poseact", "--linear_size", "128", "--merge_cls", "--ckpt", "TITAN_Baseline_2021-11-04_12.01.49.069328.pth"]
    args = parser.parse_args()
    args = manual_add_arguments(args)
    
    testset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.save_dir, True, "test")
    testset = TITANSimpleDataset(testset, merge_cls=args.merge_cls)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
    
    model = MultiHeadMonoLoco(args.input_size, [4, 7, 9, 13, 4], args.linear_size, args.dropout, args.n_stage).to(device)
    pretrained = "{}/{}".format(args.weight_dir, args.ckpt)
    model.load_state_dict(torch.load(pretrained))
    
    device = next(model.parameters()).device
    model.eval()
    n_tasks = len(model.output_size)
        
    result_list = [[] for _ in range(n_tasks)] 
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            pred = model(pose)
            for idx, one_pred in enumerate(pred):
                _, pred_class = torch.max(one_pred.data, -1)

                result_list[idx].append(pred_class)
    
    label_list= []
    with torch.no_grad():
        for _, label in testloader:
            label_list.append(label)

    # result list is a list for the 5 layers of actions in TITAN hierarchy 
    # communicative, complex_context, atomic, simple_context, transporting
    result_list = [torch.cat(one_list, dim=0).cpu().detach().numpy() for one_list in result_list]
    label_list = [torch.cat(label_list, dim=0).cpu().detach().numpy()]
    
    final_result_list = [torch.zeros(label_list[0].shape)]
    
    squatting = (result_list[2] == 5)
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
    
    n_classes = testset.n_cls
    acc_list, f1_list, jac_list, cfx_list, ap_list = [[] for _ in range(5)]
    for idx, (pred, label) in enumerate(zip(result_list, label_list)):
        acc = accuracy_score(y_true=label, y_pred=pred)
        f1 = f1_score(y_true=label, y_pred=pred, average='macro')
        jac = jaccard_score(y_true=label, y_pred=pred, average='macro')
        cfx = confusion_matrix(y_true=label, y_pred=pred, labels=range(n_classes[idx]))
        acc_list.append(acc)
        f1_list.append(f1)
        jac_list.append(jac)
        cfx_list.append(cfx)
    
    print("In general, overall accuracy {:.4f} avg Jaccard {:.4f} avg F1 {:.4f}".format(
                                np.mean(acc_list), np.mean(jac_list), np.mean(f1_list)))
    if args.merge_cls:
        action_hierarchy = ["valid_action"]
    else:
        action_hierarchy = ["communicative", "complex_context", "atomic", "simple_context", "transporting"]
    
    for idx, layer in enumerate(action_hierarchy):
        # some classes have 0 instances (maybe) and recalls will be 0, resulting in a nan
        prec, rec, f1 = per_class_precision(cfx_list[idx]), per_class_recall(cfx_list[idx]),per_class_f1(cfx_list[idx])
        print("")
        print("For {} actions accuracy {:.4f} Jaccard score {:.4f} f1 score {:.4f}".format(
            layer, acc_list[idx], jac_list[idx], f1_list[idx]))
        print("Precision for each class: {}".format(prec))
        print("Recall for each class: {}".format(rec))
        print("F1 score for each class: {}".format(f1))
        print("Confusion matrix (elements in a row share the same true label, those in the same columns share predicted):")
        print("The corresponding classes are {}".format(Person.get_attr_dict(layer)))
        print(cfx_list[idx])
        print("")
        
    
    