""" count the number of instances for the original titan dataset, 
    as well as the selected action set 
    usage: python tools/count_gt_annos.py 
"""

import glob
import pandas as pd
from collections import Counter
from typing import Dict

def simplify_key(key:str):
    simplify_dict = {'getting in 4 wheel vehicle': 'getting in 4 wv',
                     'getting off 2 wheel vehicle': "getting off 2 wv",
                     "getting on 2 wheel vehicle":'getting on 2 wv',
                     'getting out of 4 wheel vehicle':'getting out of 4 wv',
                     "crossing a street at pedestrian crossing":"crossing legally",
                     "jaywalking (illegally crossing NOT at pedestrian crossing)":"crossing illegally",
                     "waiting to cross street":"waiting to cross",
                     "walking along the side of the road": 'walking on the side',
                     'carrying with both hands':"carrying",
                     }
    if key in simplify_dict.keys():
        return simplify_dict[key]
    else:
        return key

def print_dict_in_percentage(dict_record:Dict[str, int]):
    total_count = sum(list(dict_record.values()))
    print()
    for key, value in dict_record.items():
        print("\'{}\':{:.2f}%".format(simplify_key(key), value/total_count*100))
    print()
    
base_dir = "./"
pifpaf_out = "{}/out/pifpaf_results/".format(base_dir)
dataset_dir = "{}/data/TITAN/".format(base_dir)
save_dir = "{}/out/".format(base_dir)

all_files = glob.glob(dataset_dir+"titan_0_4/*.csv")

all_df = []

communicative, complex_context, atomic, simple_context, transporting = [Counter() for _ in range(5)]

for filename in all_files:
    print("procesing {}".format(filename))
    df = pd.read_csv(filename)
    for row_id in range(len(df)):
        row_data = df.iloc[row_id,:]
        if row_data["label"] != "person":
            continue
        communicative.update({row_data["attributes.Communicative"]:1})
        complex_context.update({row_data["attributes.Complex Contextual"]:1})
        atomic.update({row_data["attributes.Atomic Actions"]:1})
        simple_context.update({row_data["attributes.Simple Context"]:1})
        transporting.update({row_data["attributes.Transporting"]:1})

for category in [communicative, complex_context, atomic, simple_context, transporting]:
    
    print_dict_in_percentage(category)

df = pd.read_csv("{}/splits/test_set.txt".format(dataset_dir), header=None)
seqs = df[0].to_numpy().tolist()
count_dict = {"walking":0, "standing":0, "sitting":0, "bending":0, "biking":0} 
for seq in seqs:
    file_path = "{}/titan_0_4/{}.csv".format(dataset_dir, seq)
    df = pd.read_csv(file_path)
    for row_id in range(len(df)):
        row_data = df.iloc[row_id,:]
        if row_data["label"] != "person":
            continue
        simple_action = row_data["attributes.Simple Context"]
        atomic_action = row_data["attributes.Atomic Actions"]
        if simple_action in ["biking", "motorcycling"]:
            count_dict["biking"] += 1
        elif atomic_action in ["walking", "standing", "sitting", "bending"]:
            count_dict[atomic_action] += 1
            
print(count_dict)
print(sum(count_dict.values()))