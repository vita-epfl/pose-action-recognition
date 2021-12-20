import os
import sys 
import numpy as np
import PIL
import glob
import torch
import openpifpaf
import argparse
import subprocess 

import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from torch.utils.data import DataLoader, Subset

import multiprocessing as mp 
from itertools import product 
from multiprocessing import Pool
from models import MultiHeadMonoLoco
from poseact.titan_train import manual_add_arguments
from poseact.utils import setup_multiprocessing, make_save_dir
from poseact.utils.titan_dataset import TITANDataset, TITANSimpleDataset, Person, Sequence, Frame, get_all_clip_names
from poseact.utils.titan_metrics import get_all_predictions, get_eval_metrics, summarize_results

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
parser.add_argument("--ckpt", type=str, default="TITAN_Relative_KP803217.pth", help="default checkpoint file name")

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
args = parser.parse_args(["--base_dir", "poseact", "--linear_size", "128", "--test_only", "--ckpt", "TITAN_Relative_KP803217.pth", "--relative_kp", "--merge_cls"])
args = manual_add_arguments(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.pickle_dir, True, "test")

simple_testset = TITANSimpleDataset(testset, merge_cls=args.merge_cls, inflate=args.inflate, use_img=args.use_img,
                                  relative_kp=args.relative_kp, rm_center=args.rm_center, normalize=args.normalize)

input_size, output_size = simple_testset.n_feature, simple_testset.n_cls
model = MultiHeadMonoLoco(input_size, output_size, args.linear_size, args.dropout, args.n_stage).to(device)
pretrained = "{}/{}".format(args.weight_dir, args.ckpt)
model.load_state_dict(torch.load(pretrained))
model.eval()
model.to(device)    

testloader = DataLoader(simple_testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=TITANSimpleDataset.collate)
result_list, label_list, score_list = get_all_predictions(model, testloader)
acc, f1, jac, cfx, ap = get_eval_metrics(result_list, label_list, score_list)
summarize_results(acc, f1, jac, cfx, ap)
        
another_result_list = []
another_all_poses = []
another_all_labels = []
another_score_list = []
for seq in testset.seqs:
    print("processing {}".format(seq.seq_name))
    for frame in seq.frames:
        pose_array, box_array, label_array = frame.collect_objects(True)
        if pose_array.size == 0:
            continue
        pose_array = TITANSimpleDataset.to_relative_coord(pose_array)
        another_all_poses.append(pose_array)
        another_all_labels.append(label_array)
        pose_tensor = torch.tensor(pose_array, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(pose_tensor)
        another_score_list.append(pred[0].detach().cpu().numpy())
        _, pred_class = torch.max(pred[0].data, -1)
        another_result_list.append(pred_class.detach().cpu().numpy())

another_all_poses = np.concatenate(another_all_poses, axis=0).flatten()
another_all_labels = np.concatenate(another_all_labels, axis=0).flatten()
another_score_list = np.concatenate(another_score_list, axis=0)
another_result_list = np.concatenate(another_result_list, axis=0).flatten()

acc, f1, jac, cfx, ap = get_eval_metrics([another_result_list], [another_all_labels], [another_score_list])
summarize_results(acc, f1, jac, cfx, ap)
    
assert np.allclose(simple_testset.all_labels, another_all_labels)
assert np.allclose(simple_testset.all_poses, another_all_poses)
assert np.allclose(result_list[0], another_result_list)

print()