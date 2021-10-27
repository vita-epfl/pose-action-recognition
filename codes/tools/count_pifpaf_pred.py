""" count how many person pifpaf can detect when setting different `--long-edge`
"""
# first run this
# sbatch python_wrapper.sh -m openpifpaf.predict data/titan_clip/example.png --glob "./data/TITAN/images_anonymized/clip*15/images/*.png" --checkpoint=shufflenetv2k30  --long-edge 1920 --force-complete-pose
# and then look at the output log  
import re 
import numpy as np 
job_numbers = [720, 719, 699, 700, 701, 702, 703, 704, 705]
# job_numbers = [644]
pattern_fn = "slurm-773{}_4294967294.log".format
file_dir = "codes/out/"

avg_det_list, long_edge_list = [], []
for num in job_numbers:
    file_name = pattern_fn(num)
    # print(file_name)
    full_path = file_dir + file_name
    
    with open(full_path, "r") as f:
        file_content = f.readlines()
    first_line = file_content[0]
    long_edge = int(re.findall(r"long-edge \d+", first_line)[0].split()[-1])
    num_detection = []
    for line in file_content[3:]:
        phrase = re.findall(r"cifcaf:\d+ annotations", line)
        if len(phrase) == 1:
            num = int(phrase[0].split()[0].split(":")[-1])
            num_detection.append(num)
            
    avg_det = sum(num_detection)/len(num_detection)
    print("Pifpaf detects {:.2f} persons on average with long edge {}".format(avg_det, long_edge))
    
    avg_det_list.append(avg_det)
    long_edge_list.append(long_edge)
    
p = np.polyfit(long_edge_list, avg_det_list, 1)
print("With a linear regression: n_person = {:.2f}*long_edge/100 + {:.2f}".format(100*p[0], p[1]))