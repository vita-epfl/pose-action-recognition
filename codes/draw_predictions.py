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
from titan_dataset import Person, get_all_clip_names

# print('OpenPifPaf version', openpifpaf.__version__)
# print('PyTorch version', torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# is this the correct way to add force complete pose???
from openpifpaf import decoder
from utils.monoloco_run import cli
args =  cli()
args.force_complete_pose = True
decoder.configure(args)

def predict_action(model, pifpaf_pred):
    """predict the type of actions with an action recognition model 
    and pifpaf output (should be json format, converted to dict in python)

    Returns:
        result_list : a list of strings 
    """
    kp_list = []
    for pred in pifpaf_pred:
        kp_list.append(np.array(pred['keypoints']).reshape(-1, 3)[:, :2])
    # kp_list = np.concatenate(kp_list, axis=0)
    kp_list = torch.tensor(kp_list, dtype=torch.float).to(device)
    pred = model(kp_list)
    result_list = []
    for idx, one_pred in enumerate(pred):
        _, pred_class = torch.max(one_pred.data, -1)
        result_list.append(pred_class.detach().cpu().numpy())
    result_list = np.array(result_list).T
    # result_list[0] will be the actions of the first person, in number
    return result_list

def annotate_img(model, predictor, img_path, res_folder=None, alpha=0.2, dpi=250):
    
    if res_folder is not None:
        frame_name = img_path.replace("\\", "/").split("/")[-1]
        save_path = "{}/{}.annotated.jpg".format(res_folder, frame_name)
    else:
        save_path = img_path + ".annotated.jpg"
        
    if not os.path.exists(img_path):
        print("image file doesn't exists")
        return 
    
    pil_im = PIL.Image.open(img_path).convert('RGB')
    predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
    
    if len(predictions)>0:
        actions = predict_action(model, pifpaf_pred=predictions)
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
            # atomic and simple context 
            x, y, w, h = box
            rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor="r", facecolor="none",alpha=0.85)
            ax.add_patch(rect)
            for cnt, s in enumerate(action[2:4]):
                ax.text(x=x, y=y+h+cnt*20, s=s, fontsize=4, color="w", alpha=0.8,
                    bbox=dict(facecolor='red', edgecolor='none', pad=0, alpha=0.5))
        # plt.show()
        plt.savefig(save_path, dpi=dpi)
        print("file saved to {}".format(save_path))

def run_img(model, predictor, img_path, res_folder, **kwargs):
    
    if isinstance(res_folder, str):
        if not os.path.exists(res_folder):
            print("create folder at {}".format(res_folder))
            os.makedirs(res_folder, exist_ok=False)
    else:
        res_folder = None
        
    annotate_img(model, predictor, img_path, res_folder, **kwargs) 

def run_seq(model, predictor, seq_folder, save_folder, **kwargs):
    src_folder = "{}/images/".format(seq_folder)
    all_img_files = glob.glob("{}/*.png".format(src_folder))
    for img_path in all_img_files:
        run_img(model, predictor, img_path, save_folder, **kwargs)

def run_multiple_seq(model, predictor, base_dir, save_dir, seq_nums, **kwargs):
    for seq in seq_nums:
        seq_folder = "{}/data/TITAN/images_anonymized/clip_{}/".format(base_dir, seq)
        save_folder = "{}/clip_{}/".format(save_dir, seq)
        run_seq(model, predictor, seq_folder, save_folder, **kwargs)
     
def get_models(ckpt_dir=None, json=True):
    model = MultiHeadMonoLoco(34, [4, 7, 9, 13, 4], 128, 0.2, 3).to(device)
    if os.path.exists(ckpt_dir):
        model.load_state_dict(torch.load(ckpt_dir)) # action predictor 
    else:
        print("can not load state dict, use initial model instead")
    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30', json_data=True) # pifpaf predictor 
    return model, predictor

def test_run_image(args):
    base_dir = args.base_dir
    ckpt_path = "{}/out/trained/{}".format(base_dir, args.ckpt)
    model, predictor = get_models(ckpt_path)

    img_path = "{}/out/titan_clip/example.png".format(base_dir)
    run_img(model, predictor, img_path, res_folder=None, alpha=args.alpha, dpi=args.dpi)

def run_all_titan_seqs(args):
    
    base_dir, save_dir = args.base_dir, args.save_dir
    ckpt_path = "{}/out/trained/{}".format(base_dir, args.ckpt)
    pifpaf_out = "{}/out/pifpaf_results/".format(base_dir)
    
    all_clips = get_all_clip_names(pifpaf_out)
    clip_nums = [int(clip.split("_")[-1]) for clip in all_clips]
    # clip_nums = [16, 26, 319, 731]
    
    model, predictor = get_models(ckpt_path)
    run_multiple_seq(model, predictor, base_dir, save_dir, clip_nums, alpha=args.alpha, dpi=args.dpi)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--base_dir", type=str, default="./", help="default root working directory")
    parser.add_argument("--save_dir", type=str, default="./out/recognition/", help="to save annotated pictures")
    parser.add_argument("--ckpt", type=str, default="TITAN_Baseline_2021-11-04_12.01.38.803868.pth", 
                        help="default checkpoint file name")
    parser.add_argument("--function", type=str, default="None", help="which function to call")
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--dpi", type=int, default=350)
    args = parser.parse_args()

    function_dict = {"image": test_run_image, 
                     "all": run_all_titan_seqs}
    
    function = function_dict.get(args.function, None)
    if function:
        function(args)


