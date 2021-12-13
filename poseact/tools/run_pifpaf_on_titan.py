""" run pifpaf prediction on TITAN dataset
"""
import os 
import sys 
import glob 
import shutil
import ctypes
import subprocess
import argparse
import multiprocessing as mp 
from multiprocessing import Pool
from openpifpaf import Predictor
from poseact.utils import setup_multiprocessing 

parser = argparse.ArgumentParser() 
parser.add_argument("--base_dir", type=str, default="./")
parser.add_argument("--long_edge", type=int, default=1920)
parser.add_argument("--n_process", type=int, default=0)
parser.add_argument("--mode", type=str, default="single", choices=["single", "track"])
args = parser.parse_args() # ["--base_dir", "poseact", "--mode", "track"]

base_dir = args.base_dir
if args.mode == "single":
    output_dir = "{}/out/pifpaf_results_{}/".format(base_dir, args.long_edge)
elif args.mode == "track":
    output_dir = "{}/out/pifpaf_track_results_{}/".format(base_dir, args.long_edge)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_clip_names = glob.glob("{}/data/TITAN/images_anonymized/*".format(base_dir), recursive=True)
clips = [name.replace("\\", "/").split(sep="/")[-1] for name in all_clip_names]
clips = sorted(clips, key=lambda item: int(item.split(sep="_")[-1]))
        
def process_one_seq(seq_idx):
    clip = clips[seq_idx]
    print("Running pifpaf on {}".format(clip))
    
    clip_save_path = output_dir + clip + "/"
    if not os.path.exists(clip_save_path):
        os.mkdir(clip_save_path)    
    # the --glob command should be quoted with "", 
    # otherwise the shell will convert the wildcard to all matching files
    command = ["python", "-m", "openpifpaf.predict", 
               "{}/out/titan_clip/example.png".format(base_dir), 
               "--glob", "\"{}/data/TITAN/images_anonymized/{}/images/*.png\"".format(base_dir, clip), 
               "--checkpoint=shufflenetv2k30",  
               "--long-edge",  "{}".format(args.long_edge),
               "--force-complete-pose",
               "--json-output", clip_save_path]
    shell_command = " ".join(command) # if shell=True, the first arguments can not be a list 
    process_result = subprocess.run(shell_command, shell=True, stdout=subprocess.DEVNULL)
    if process_result.returncode == 0:
        print("Completed prediction on {}".format(clip))
    else:
        print("Failed to run on {}".format(clip))

def track_one_seq(seq_idx):
    
    clip = clips[seq_idx]
    pid = os.getpid()
    print("Process {} is running pifpaf on {}".format(pid, clip))
    sys.stdout.flush()
    
    original_images = glob.glob("{}/data/TITAN/images_anonymized/{}/images/*.png".format(base_dir, clip), recursive=True)
    tmp_folder = "{}/scratch/TITAN/{}".format(base_dir, clip)
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    for img_path in original_images:
        frame_name = img_path.replace("\\", "/").split("/")[-1].replace(".png", "")
        frame_number = int(frame_name)
        tmp_number = int(frame_number/6)
        new_frame_name = "{:06d}.png".format(tmp_number)
        new_frame_path = "{}/{}".format(tmp_folder, new_frame_name)
        shutil.copy(img_path, new_frame_path)

    # otherwise the shell will convert the wildcard to all matching files
    command = ["python", "-m", "openpifpaf.video", 
               "--long-edge={}".format(args.long_edge),
               "--checkpoint=tshufflenetv2k16",
               "--decoder=trackingpose:0",
               "--source", "{}/%06d.png".format(tmp_folder, clip), 
               "--force-complete-pose",
               "--json-output"]
    shell_command = " ".join(command) # if shell=True, the first arguments can not be a list 
    process_result = subprocess.run(shell_command, shell=True, stdout=subprocess.DEVNULL)
    if process_result.returncode == 0:
        print("Completed pose tracking on {}".format(clip))
    else:
        print("Failed to run on {}".format(clip))
    sys.stdout.flush()
    
    pifpaf_out_file = "{}/%06d.png.openpifpaf.json".format(tmp_folder)
    json_save_dir = "{}/TITAN_{}_track.json".format(output_dir, clip)
    shutil.copy(pifpaf_out_file, json_save_dir)
    
if __name__ == "__main__":
    
    if args.mode == "single":
        process_function = process_one_seq
    elif args.mode == "track":
        process_function = track_one_seq
            
    if args.n_process > 2:
        setup_multiprocessing()
        with Pool(processes=args.n_process) as p:
            p.map(process_function, range(len(clips)))
    else:
        for idx in range(len(clips)):
            process_function(idx)

    if args.mode == "single":
        # delete the example file 
        for clip in clips:
            clip_save_path = output_dir + clip + "/"
            example_file = clip_save_path + "example.png.predictions.json"
            if os.path.exists(example_file):
                os.remove(example_file)
