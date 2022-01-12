import os 
import PIL
import torch
import argparse
import openpifpaf

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

from poseact.utils import to_relative_coord
from poseact.models import MonolocoModel
from poseact.utils.casr_dataset import CASRDataset, CASRSimpleDataset, Person, Sequence

# set value for some arguments 
parser = argparse.ArgumentParser() 

parser.add_argument("--base_dir", type=str, default=".", help="root directory of the codes")
parser.add_argument("--linear_size", type=int, default=128, help="size of hidden linear layer")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--n_stage", type=int, default=3, help="number of stages in a monoloco model")
parser.add_argument("--num_runs", type=int, default=12, help="number of different train-val-test setup")
parser.add_argument("--ckpt", type=str, default="CASR_Baseline832369_8_2022-01-08_15.15.24.108630.pth", help="default checkpoint")

colors = {"left":"r", "right":"g", "stop":"b", "none":"m"}
inverse_action_dict = {0:"left", 1:"right", 2:"stop", 3:"none"}

def manual_add_args(args):
    base_dir = args.base_dir
    args.pickle_dir = "{}/out/casrdata".format(base_dir)
    args.save_dir = "{}/out/".format(base_dir)
    args.weight_dir = "{}/out/trained/".format(base_dir)
    args.viz_dir = "{}/out/casr_viz".format(base_dir)
    args.data_dir = "{}/data/CASR_Dataset/images".format(base_dir)
    return args  
    
def draw_and_save(img, box, action, label=None, save_path=None, verbose=False):
    """ draw actions, bounding boxes on an image and save to save_path

    Args:
        img (array): numpy array, converted from PIL image
        boxes (array): bounding boxes of size (N, 4), format x, y, w, h
        actions (array): an array of integers indicationg the type of actions
        labels (array, optional): ground truth labels, available if using pickle file. Defaults to None.
        save_path (str, optional): save path. Defaults to None.
    """
    with openpifpaf.show.image_canvas(img) as ax:
        box_color = colors[action]
        gt_color = colors[label]

        x, y, w, h = box
        rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor=box_color, facecolor="none",alpha=0.85)
        ax.add_patch(rect)
        ax.text(x=x, y=y+h, s=action, fontsize=4, color="w", alpha=0.8,
            bbox=dict(facecolor=box_color, edgecolor='none', pad=0, alpha=0.5))
        ax.text(x=x, y=y+h+20, s="GT: "+label, fontsize=4, color="w", alpha=0.8,
            bbox=dict(facecolor=gt_color, edgecolor='none', pad=0, alpha=0.5))
                
        if save_path is not None:
            plt.savefig(save_path, dpi=350)
            if verbose:
                print("file saved to {}".format(save_path))
        else:
            if verbose:
                print("save path doesn't exit, don't save image") 

if __name__ == "__main__":
    # ["--base_dir", "poseact", "--model_type", "sequence", "--batch_size", "32"]
    args = parser.parse_args() # ["--base_dir", "poseact", "--debug"]
    args = manual_add_args(args)
    raw_seqs = CASRDataset.load_prepared_seqs(args.save_dir)
    ytset = [seq for seq in raw_seqs if seq.is_yt==True]
    input_size, output_size = 36, 4
    model = MonolocoModel(input_size, output_size, args.linear_size, args.dropout, args.n_stage)
    model.load_state_dict(torch.load("{}/{}".format(args.weight_dir, args.ckpt)))
    model.eval()
    img_path_fn = lambda seq, frame: "{}/{}/{}".format(args.data_dir, seq, frame)
    save_path_fn = lambda seq, frame: "{}/{}/{}".format(args.viz_dir, seq, frame)
    for seq in ytset:
        
        seq_save_folder = "{}/{}".format(args.viz_dir, seq.seq_name)
        if not os.path.exists(seq_save_folder):
            os.makedirs(seq_save_folder, exist_ok=False)
            
        for person in seq.persons:
            if np.allclose(person.key_points, 0):
                continue
            pose = to_relative_coord(torch.tensor(person.key_points, dtype=torch.float32).unsqueeze(0))
            pred = model(pose)
            _, pred_class = torch.max(pred.data, -1)
            action = inverse_action_dict[pred_class.item()]
            label = inverse_action_dict[person.action]
            bbox = person.pred_box
            image = PIL.Image.open(img_path_fn(seq.seq_name, person.frame)).convert('RGB')
            draw_and_save(image, bbox, action, label, save_path_fn(seq.seq_name, person.frame), verbose=True)
            
    
    