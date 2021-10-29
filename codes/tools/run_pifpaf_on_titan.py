""" run pifpaf prediction on TITAN dataset
"""
import os 
import glob 
import subprocess
import multiprocessing as mp 
from multiprocessing import Pool
from openpifpaf import Predictor

LOCAL_RUN = False
base_dir = "codes" if LOCAL_RUN else "."
output_dir = "{}/out/pifpaf_results/".format(base_dir)

all_clip_names = glob.glob("{}/data/TITAN/images_anonymized/*".format(base_dir), recursive=True)
clips = [name.split(sep="/")[-1] for name in all_clip_names]
clips = sorted(clips, key=lambda item: int(item.split(sep="_")[-1]))

# for clip in clips:
#     print("Running pifpaf on {}".format(clip))
    
#     clip_save_path = output_dir + clip + "/"
#     if not os.path.exists(clip_save_path):
#         os.mkdir(clip_save_path)
#     command = ["python", "-m", "openpifpaf.predict", 
#                "{}/data/titan_clip/example.png".format(base_dir), 
#                "--glob", "\"{}/data/TITAN/images_anonymized/{}/images/*.png\"".format(base_dir, clip), 
#                "--checkpoint=shufflenetv2k30",  
#                "--long-edge",  "1920",
#                "--force-complete-pose",
#                "--json-output", clip_save_path]
#     shell_command = " ".join(command) # if shell=True, the first arguments can not be a list 
#     process_result = subprocess.run(shell_command, shell=True)
#     if process_result.returncode == 0:
#         print("Completed prediction on {}".format(clip))
#     else:
#         print("Failed to run on {}".format(clip))
        
def process_one_seq(seq_idx):
    clip = clips[seq_idx]
    print("Running pifpaf on {}".format(clip))
    
    clip_save_path = output_dir + clip + "/"
    if not os.path.exists(clip_save_path):
        os.mkdir(clip_save_path)
    command = ["python", "-m", "openpifpaf.predict", 
               "{}/out/titan_clip/example.png".format(base_dir), 
               "--glob", "\"{}/data/TITAN/images_anonymized/{}/images/*.png\"".format(base_dir, clip), 
               "--checkpoint=shufflenetv2k30",  
               "--long-edge",  "1920",
               "--force-complete-pose",
               "--json-output", clip_save_path]
    shell_command = " ".join(command) # if shell=True, the first arguments can not be a list 
    process_result = subprocess.run(shell_command, shell=True, stdout=subprocess.DEVNULL)
    if process_result.returncode == 0:
        print("Completed prediction on {}".format(clip))
    else:
        print("Failed to run on {}".format(clip))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    with Pool(processes=4) as p:
        p.map(process_one_seq, range(len(clips)))
    
    # delete the example file 
    for clip in clips:
        clip_save_path = output_dir + clip + "/"
        example_file = clip_save_path + "example.png.predictions.json"
        if os.path.exists(example_file):
            os.remove(example_file)



    