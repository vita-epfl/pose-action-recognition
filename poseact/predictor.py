import os
import numpy as np
import PIL
import glob
import torch
import openpifpaf
import argparse

import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from models import MultiHeadMonoLoco
from titan_dataset import Person, get_all_clip_names, TITANSimpleDataset

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


class Predictor():
    def __init__(self, args) -> None:
        self.relative_kp = not args.no_relative_kp 
        self.merge_cls = not args.no_merge_cls
        self.threshold = args.threshold 
        self.alpha = args.alpha
        self.dpi = args.dpi
        ckpt_path = "{}/out/trained/{}".format(args.base_dir, args.ckpt)
        self.model, self.predictor = self.get_models(ckpt_dir=ckpt_path, json=True)
    
    def predict_action(self, pifpaf_pred):
        """predict the type of actions with an action recognition model 
        and pifpaf output (should be json format, converted to dict in python)

        Returns:
            result_list : a list of strings 
        """
        kp_list = []
        for pred in pifpaf_pred:
            kp_list.append(np.array(pred['keypoints']).reshape(-1, 3)[:, :2])
        if self.relative_kp:
            kp_list = np.array(kp_list)
            kp_list = TITANSimpleDataset.convert_to_relative_coord(kp_list)
        kp_tensor = torch.tensor(kp_list, dtype=torch.float).to(device)
        pred = self.model(kp_tensor)
        result_list = []
        for idx, one_pred in enumerate(pred):
            _, pred_class = torch.max(one_pred.data, -1)
            result_list.append(pred_class.detach().cpu().numpy())
        result_list = np.array(result_list).T
        # result_list[0] will be the actions of the first person, in number
        return result_list

    def annotate_img(self, img_path, res_folder=None):
        
        colors = {"walking":"r", "standing":"g", "sitting":"b", "bending":"m", "biking":"c"}
        
        if res_folder is not None:
            frame_name = img_path.replace("\\", "/").split("/")[-1]
            save_path = "{}/{}.annotated.jpg".format(res_folder, frame_name)
        else:
            save_path = img_path + ".annotated.jpg"
            
        if not os.path.exists(img_path):
            print("image file doesn't exists")
            return
        
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
        annotation_painter = openpifpaf.show.AnnotationPainter()
        
        with openpifpaf.show.image_canvas(im) as ax:
            # annotation_painter.annotations(ax, predictions, alpha=alpha)
            # current_fig = plt.gcf()
            ax.text(100, 100, "pifpaf detects {} person".format(len(predictions)))
            for action, box in zip(actions, boxes):
                
                if self.merge_cls:
                    box_color = colors[action[0]]
                    action_to_show = action
                else:
                    # color according to atomic action
                    box_color = colors[action[2]]
                    # atomic and simple context 
                    action_to_show = action[2:4]
                    
                x, y, w, h = box
                rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor=box_color, facecolor="none",alpha=0.85)
                ax.add_patch(rect)
                for cnt, s in enumerate(action_to_show):
                    ax.text(x=x, y=y+h+cnt*20, s=s, fontsize=4, color="w", alpha=0.8,
                        bbox=dict(facecolor=box_color, edgecolor='none', pad=0, alpha=0.5))
                    
            # plt.show()
            plt.savefig(save_path, dpi=self.dpi)
            print("file saved to {}".format(save_path))
            
    def get_models(self, ckpt_dir=None, json=True):
        model = MultiHeadMonoLoco(36, [5], 128, 0.2, 3).to(device)
        if os.path.exists(ckpt_dir):
            model.load_state_dict(torch.load(ckpt_dir)) # action predictor 
        else:
            print("can not load state dict, use initial model instead")
        predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30', json_data=True) # pifpaf predictor 
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
                self.run_img(img_path, save_folder=args.save_dir)
        elif function_name == "all":
            base_dir, save_dir = args.base_dir, args.save_dir
            pifpaf_out = "{}/out/pifpaf_results/".format(base_dir)
            
            all_clips = get_all_clip_names(pifpaf_out)
            clip_nums = [int(clip.split("_")[-1]) for clip in all_clips]
            clip_nums = [1, 2, 16, 26, 319, 731] # for debugging locally 
            
            self.run_multiple_seq(base_dir, save_dir, clip_nums)
        elif function_name == "prepared":
            # load the pre-extracted pickle file and run prediction frame by frame
            pass 
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--function", type=str, default="image", help="which function to call")
    parser.add_argument("--image_path", type=str, default=None, help="path to an image")
    parser.add_argument("--base_dir", type=str, default="./", help="default root working directory")
    parser.add_argument("--save_dir", type=str, default="./out/recognition/", help="to save annotated pictures")
    parser.add_argument("--ckpt", type=str, default="TITAN_Relative_KP_803217.pth", help="default checkpoint file name")
    parser.add_argument("--no_relative_kp", action="store_true",help="use absolute key point corrdinates")
    parser.add_argument("--no_merge_cls", action="store_true",  help="keep the original action hierarchy in titan")
    parser.add_argument("--threshold", type=float, default=0.3, help="confidence threshold for instances")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--dpi", type=int, default=350)
    # ["--base_dir", "codes/", "--save_dir", "codes/out/recognition/" ,"--function", "all"]
    args = parser.parse_args()
    # print(args)
    
    configure_pifpaf()
    predictor = Predictor(args)
    predictor.run(args)


