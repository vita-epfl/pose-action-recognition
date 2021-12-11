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
    
base_dir = "poseact/"
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