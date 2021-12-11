""" run pifpaf prediction on TITAN dataset
"""
import os 
import sys 
import glob 
import ctypes
import subprocess
import argparse
import multiprocessing as mp 
from multiprocessing import Pool
from openpifpaf import Predictor

parser = argparse.ArgumentParser() 
parser.add_argument("--base_dir", type=str, default="./")
parser.add_argument("--long_edge", type=int, default=1920)
parser.add_argument("--n_process", type=int, default=0)
parser.add_argument("--mode", type=str, default="single", choices=["single", "track"])
args = parser.parse_args() # ["--base_dir", "codes", "--long_edge", "3333"]

base_dir = args.base_dir
output_dir = "{}/out/pifpaf_results_{}/".format(base_dir, args.long_edge)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_clip_names = glob.glob("{}/data/TITAN/images_anonymized/*".format(base_dir), recursive=True)
clips = [name.split(sep="/")[-1] for name in all_clip_names]
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
        print("Completed pose tracking on {}".format(clip))
    else:
        print("Failed to run on {}".format(clip))

if __name__ == "__main__":
    if args.n_process > 2:
        mp.set_start_method('spawn')
        if sys.platform.startswith("linux"):
            try:
                libgcc_s = ctypes.CDLL("/usr/lib64/libgcc_s.so.1")
                print("loaded /usr/lib64/libgcc_s.so.1")
            except:
                pass 
        with Pool(processes=args.n_process) as p:
            p.map(process_one_seq, range(len(clips)))
    else:
        for idx in range(len(clips)):
            process_one_seq(idx)
    
    # delete the example file 
    for clip in clips:
        clip_save_path = output_dir + clip + "/"
        example_file = clip_save_path + "example.png.predictions.json"
        if os.path.exists(example_file):
            os.remove(example_file)



    