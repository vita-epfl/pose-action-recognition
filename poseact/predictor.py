"""
predict on an image: python predictor.py --function image --image_path example.png

"""

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

import multiprocessing as mp 
from itertools import product 
from multiprocessing import Pool
from models import MultiHeadMonoLoco
from poseact.titan_train import manual_add_arguments
from poseact.utils import setup_multiprocessing, make_save_dir
from poseact.utils.titan_dataset import TITANDataset, TITANSimpleDataset, Person, Sequence, Frame, get_all_clip_names


# print('OpenPifPaf version', openpifpaf.__version__)
# print('PyTorch version', torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# is this the correct way to add force complete pose???
from openpifpaf import decoder, network, visualizer, show, logger

def configure_pifpaf():
    # use the same checkpoint as we extract the poses 
    parser = argparse.ArgumentParser()
    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)
    # give an empty list to this parser, so it won't use the real command line input
    # we just need the default parameters of pifpaf 
    pifpaf_args = parser.parse_args([]) 
    pifpaf_args.force_complete_pose = True
    decoder.configure(pifpaf_args)

def process_one_seq(input_args):
    """ call a sub process to run a single sequence 
    """
    args, seq_idx = input_args
    command = ["python", "{}/predictor.py".format(args.base_dir),
            "--function", "titan_single",
            "--seq_idx", "{}".format(seq_idx),
            "--save_dir", "{}".format(args.save_dir),
            "--split", "{}".format(args.split)]
    shell_command = " ".join(command) # if shell=True, the first arguments can not be a list 
    print("subprocess {} is running the command: {}".format(os.getpid(), shell_command))
    sys.stdout.flush()
    process_result = subprocess.run(shell_command, shell=True, stdout=subprocess.DEVNULL)
    
class Predictor():
    
    def __init__(self, args) -> None:
        self.relative_kp = not args.no_relative_kp 
        self.merge_cls = not args.no_merge_cls
        self.threshold = args.threshold 
        self.alpha = args.alpha
        self.dpi = args.dpi
        ckpt_path = "{}/out/trained/{}".format(args.base_dir, args.ckpt)
        self.model, self.predictor = self.get_models(ckpt_dir=ckpt_path, json=True)
        self.colors = {"walking":"r", "standing":"g", "sitting":"b", "bending":"m", "biking":"c"}
        
    def predict_action(self, pifpaf_pred, json=True):
        """predict the type of actions with an action recognition model 
        and pifpaf output (json format, converted to dict in python)
        if json is false, then pifpaf_pred should be a pose array of size (N, V, C)
        
        Returns:
            result_list : a list of strings 
        """
        if json: # json file, direct output of pifpaf
            kp_list = [np.array(pred['keypoints']).reshape(-1, 3)[:, :2] for pred in pifpaf_pred]
            kp_list = np.array(kp_list)
        else:
            kp_list = pifpaf_pred
            
        if self.relative_kp:
            kp_list = TITANSimpleDataset.to_relative_coord(kp_list)
            
        kp_tensor = torch.tensor(kp_list, dtype=torch.float).to(device)
        with torch.no_grad():
            pred = self.model(kp_tensor)
        result_list = []
        for idx, one_pred in enumerate(pred):
            _, pred_class = torch.max(one_pred.data, -1)
            result_list.append(pred_class.detach().cpu().numpy())
        result_list = np.array(result_list).T
        # result_list shape (n_persons, n_actions), n_actions will be 1 if args.merge_cls is True
        return result_list

    def get_img_save_path(self, img_path, res_folder):
        
        if res_folder is not None:
            frame_name = img_path.replace("\\", "/").split("/")[-1]
            save_path = "{}/{}.annotated.jpg".format(res_folder, frame_name)
        else:
            save_path = img_path + ".annotated.jpg"
        
        return save_path
    
    def annotate_img(self, img_path, res_folder=None):
        
        if not os.path.exists(img_path):
            print("image file doesn't exists")
            return
        
        save_path = self.get_img_save_path(img_path, res_folder)
        
        pil_im = PIL.Image.open(img_path).convert('RGB')
        predictions, gt_anns, image_meta = self.predictor.pil_image(pil_im)
        predictions = list(filter(lambda item: item['score']>self.threshold, predictions))
        if len(predictions)>0:
            actions = self.predict_action(pifpaf_pred=predictions)
            boxes = [pred["bbox"] for pred in predictions]
        else:
            actions, boxes = [], []
        actions = [Person.pred_list_to_str(action) for action in actions]
        
        im = np.asarray(pil_im)
        
        # annotation_painter = openpifpaf.show.AnnotationPainter()
        
        self.draw_and_save(im, boxes, actions, None, save_path)
        
    def draw_and_save(self, img, boxes, actions, labels=None, save_path=None, verbose=False):
        """ draw actions, bounding boxes on an image and save to save_path

        Args:
            img (array): numpy array, converted from PIL image
            boxes (array): bounding boxes of size (N, 4), format x, y, w, h
            actions (array): an array of integers indicationg the type of actions
            labels (array, optional): ground truth labels, available if using pickle file. Defaults to None.
            save_path (str, optional): save path. Defaults to None.
        """
        with openpifpaf.show.image_canvas(img) as ax:
            ax.text(100, 100, "pifpaf detects {} person".format(len(boxes)))
            for idx, (action, box) in enumerate(zip(actions, boxes)):
                
                if self.merge_cls:
                    box_color = self.colors[action[0]]
                    action_to_show = action
                    if labels is not None:
                        label_to_show = labels[idx] if labels is not None else []
                        gt_color = self.colors[label_to_show[0]]
                else:
                    # color according to atomic action
                    box_color = self.colors[action[2]]
                    # atomic and simple context 
                    action_to_show = action[2:4]
                    if labels is not None:
                        label_to_show = labels[2:4] if labels is not None else []
                        gt_color = self.colors[label_to_show[2]]
                    
                x, y, w, h = box
                rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor=box_color, facecolor="none",alpha=0.85)
                ax.add_patch(rect)
                for cnt, s in enumerate(action_to_show):
                    ax.text(x=x, y=y+h+cnt*20, s=s, fontsize=4, color="w", alpha=0.8,
                        bbox=dict(facecolor=box_color, edgecolor='none', pad=0, alpha=0.5))
                    
                if labels is not None:
                    for new_cnt, s in enumerate(label_to_show):
                        ax.text(x=x, y=y+h+new_cnt+(cnt+1)*20, s="GT: "+s, fontsize=4, color="w", alpha=0.8,
                            bbox=dict(facecolor=gt_color, edgecolor='none', pad=0, alpha=0.5))
                    
            if save_path is not None:
                plt.savefig(save_path, dpi=self.dpi)
                if verbose:
                    print("file saved to {}".format(save_path))
            else:
                if verbose:
                    print("save path doesn't exit, don't save image") 
                
    def get_models(self, ckpt_dir=None, json=True):
        model = MultiHeadMonoLoco(36, [5], 128, 0.2, 3).to(device)
        if os.path.exists(ckpt_dir):
            model.load_state_dict(torch.load(ckpt_dir)) # action predictor 
        else:
            print("can not load state dict, use initial model instead")
        predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30', json_data=True) # pifpaf predictor 
        model.eval()
        return model, predictor
    
    def run_img(self, img_path, res_folder):
        
        if isinstance(res_folder, str):
            if not os.path.exists(res_folder):
                print("create folder at {}".format(res_folder))
                os.makedirs(res_folder, exist_ok=False)
        else:
            res_folder = None
            
        self.annotate_img(img_path, res_folder) 

    def run_seq(self, seq_folder, save_folder):
        src_folder = "{}".format(seq_folder)
        all_img_files = glob.glob("{}/*.*".format(src_folder))
        for img_path in all_img_files:
            self.run_img(img_path, save_folder)

    def run_multiple_seq(self, base_dir, save_dir, seq_nums):
        for seq in seq_nums:
            seq_folder = "{}/data/TITAN/images_anonymized/clip_{}/images/".format(base_dir, seq)
            save_folder = "{}/clip_{}/".format(save_dir, seq)
            self.run_seq(seq_folder, save_folder)
            
    def prepare_dataset(self, args):
        self.save_dir = args.save_dir
        self.base_dir = args.base_dir
        self.all_clips = get_all_clip_names(args.dataset_dir)
        self.dataset = TITANDataset(args.pifpaf_out, args.dataset_dir, args.pickle_dir, True, args.split)
        
    def predict_one_sequence(self, idx):
        sys.stdout.flush()
        seq = self.dataset.seqs[idx]
        save_dir = make_save_dir(self.save_dir, seq.seq_name)
        for frame in seq.frames:
            pose_array, box_array, label_array = frame.collect_objects(self.merge_cls)
            if pose_array.size == 0: # skip if pifpaf doesn't detect anybody 
                actions, labels = [], []
            else:
                actions = self.predict_action(pifpaf_pred=pose_array, json=False)
                actions = [Person.pred_list_to_str(action) for action in actions]
                labels = [Person.pred_list_to_str(label) for label in label_array]
                
            img, img_path = frame.read_img(self.base_dir)
            save_path = self.get_img_save_path(img_path, save_dir)
            self.draw_and_save(img, box_array, actions, labels, save_path)
        
    def run(self, args):
        
        function_name = args.function
        
        if function_name == "image":
            base_dir, image_path = args.base_dir, args.image_path
            img_path = "{}/{}".format(base_dir, image_path)
            self.run_img(img_path, res_folder=None)
            
        elif function_name == "seq":
            base_dir, image_path = args.base_dir, args.image_path
            src_folder = "{}/{}".format(base_dir, image_path)
            all_img_files = glob.glob("{}/*.*".format(src_folder))
            for img_path in all_img_files:
                self.run_img(img_path, res_folder=args.save_dir)
                
        elif function_name == "all":
            base_dir, save_dir = args.base_dir, args.save_dir
            
            all_clips = get_all_clip_names(args.dataset_dir)
            clip_nums = [int(clip.split("_")[-1]) for clip in all_clips]
            # clip_nums = [1, 2, 16, 26, 319, 731] # for debugging locally 
            self.run_multiple_seq(base_dir, save_dir, clip_nums)
            
        elif function_name == "titanseqs": 
            # load the pre-extracted pickle file and run prediction frame by frame
            if args.n_process >= 2:
                self.prepare_dataset(args)
                all_seq_idx = range(len(self.dataset.seqs))
                # all_seq_idx = [0, 1] # for debugging locally 
                input_args = list(product([args], all_seq_idx))
                with Pool(processes=args.n_process) as p:
                    p.map(process_one_seq, input_args)
            else:
                self.prepare_dataset(args)
                for idx in range(len(self.dataset.seqs)):
                    self.predict_one_sequence(idx)
        
        # run on a single sequence in the prepared titan dataset          
        elif function_name == "titan_single":
            self.prepare_dataset(args)
            seq_idx = args.seq_idx
            clip_name = self.dataset.seqs[seq_idx].seq_name
            pid = os.getpid()
            print("Process {} is running predictions on {} of the {} split".format(pid, clip_name, self.dataset.split))
            self.predict_one_sequence(seq_idx)
            print("Process {} has completed predictions on {} of the {} split".format(pid, clip_name, self.dataset.split))
            

if __name__ == "__main__":
    
    setup_multiprocessing()
    configure_pifpaf()
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--function", type=str, default="image", help="which function to call")
    parser.add_argument("--image_path", type=str, default=None, help="path to an image")
    parser.add_argument("--base_dir", type=str, default="./", help="default root working directory")
    parser.add_argument("--save_dir", type=str, default="./out/recognition/", help="to save annotated pictures")
    parser.add_argument("--ckpt", type=str, default="TITAN_Relative_KP803217.pth", help="default checkpoint file name")
    parser.add_argument("--no_relative_kp", action="store_true",help="use absolute key point corrdinates")
    parser.add_argument("--no_merge_cls", action="store_true",  help="keep the original action hierarchy in titan")
    parser.add_argument("--n_process", type=int, default=0, help="number of process for multiprocessing, or 0 to run in serial")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"], help="split of dataset")
    parser.add_argument("--seq_idx", type=int, default=0, help="index of sequence")
    parser.add_argument("--threshold", type=float, default=0.3, help="confidence threshold for instances")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--dpi", type=int, default=350)
    # ["--base_dir", "poseact/", "--save_dir", "poseact/out/recognition/" ,"--function", "titan_single", "--seq_idx", "0"]
    args = parser.parse_args()
    # print(args)
    args = manual_add_arguments(args)
    predictor = Predictor(args)
    predictor.run(args)


