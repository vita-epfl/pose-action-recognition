""" heuristics from https://github.com/charlesbvll/monoloco/blob/main/monoloco/activity.py
"""
import numpy as np 
from poseact.utils.losses import IGNORE_INDEX
from poseact.utils.casr_dataset import Person, Sequence
from torch.utils.data import DataLoader
from poseact.utils.casr_dataset import CASRDataset, CASRSimpleDataset
from poseact.utils.casr_metrics import summarize_results, get_eval_metrics

def is_turning(kp):
    """
    Returns flag if a cyclist is turning
    """
    x=0
    y=1

    nose = 0
    l_ear = 3
    r_ear = 4
    l_shoulder = 5
    l_elbow = 7
    l_hand = 9
    r_shoulder = 6
    r_elbow = 8
    r_hand = 10

    head_width = kp[x][l_ear]- kp[x][r_ear]
    head_top = (kp[y][nose] - head_width)

    l_forearm = [kp[x][l_hand] - kp[x][l_elbow], kp[y][l_hand] - kp[y][l_elbow]]
    l_arm = [kp[x][l_shoulder] - kp[x][l_elbow], kp[y][l_shoulder] - kp[y][l_elbow]]

    r_forearm = [kp[x][r_hand] - kp[x][r_elbow], kp[y][r_hand] - kp[y][r_elbow]]
    r_arm = [kp[x][r_shoulder] - kp[x][r_elbow], kp[y][r_shoulder] - kp[y][r_elbow]]

    l_angle = (90/np.pi) * np.arccos(np.dot(l_forearm/np.linalg.norm(l_forearm), l_arm/np.linalg.norm(l_arm)))
    r_angle = (90/np.pi) * np.arccos(np.dot(r_forearm/np.linalg.norm(r_forearm), r_arm/np.linalg.norm(r_arm)))

    if kp[x][l_shoulder] > kp[x][r_shoulder]:
        is_left = kp[x][l_hand] > kp[x][l_shoulder] + np.linalg.norm(l_arm)
        is_right = kp[x][r_hand] < kp[x][r_shoulder] - np.linalg.norm(r_arm)
        l_too_close = kp[x][l_hand] > kp[x][l_shoulder] and kp[y][l_hand]>=head_top
        r_too_close = kp[x][r_hand] < kp[x][r_shoulder] and kp[y][r_hand]>=head_top
    else:
        is_left = kp[x][l_hand] < kp[x][l_shoulder] - np.linalg.norm(l_arm)
        is_right = kp[x][r_hand] > kp[x][r_shoulder] + np.linalg.norm(r_arm)
        l_too_close = kp[x][l_hand] <= kp[x][l_shoulder] and kp[y][l_hand]>=head_top
        r_too_close = kp[x][r_hand] >= kp[x][r_shoulder] and kp[y][r_hand]>=head_top


    is_l_up = kp[y][l_hand] < kp[y][l_shoulder]
    is_r_up = kp[y][r_hand] < kp[y][r_shoulder]

    is_left_risen = is_l_up and l_angle >= 30 and not l_too_close
    is_right_risen = is_r_up and r_angle >= 30 and not r_too_close

    is_left_down = is_l_up and l_angle >= 30 and not l_too_close
    is_right_down = is_r_up and r_angle >= 30 and not r_too_close

    if is_left and l_angle >= 40 and not(is_left_risen or is_right_risen):
        return 'left'

    if is_right and r_angle >= 40 or (is_left_risen or is_right_risen):
        return 'right'

    if is_left_down or is_right_down:
        return 'stop'

    return "none"

def get_predictions(dataset:CASRSimpleDataset):
    label_list, result_list, score_list = [[] for _ in range(3)]

    for pose, label in zip(dataset.all_poses, dataset.all_labels):
        pred = Person.action_dict[is_turning(pose.T.numpy())]
        label_list.append(label.data)
        result_list.append(pred)
        score_list.append(1)
    
    results = np.array(result_list)
    labels = np.array(label_list)
    scores = np.array(score_list)
    
    valid_idx = np.not_equal(labels, IGNORE_INDEX)
    results = results[valid_idx]
    labels = labels[valid_idx] 
    scores = scores[valid_idx]
    
    return results, labels, scores

if __name__ == "__main__":
    """ run heuristics on CASR
    """
    save_dir = "./out/"
    all_test_results, all_yt_results = [], []
    for run_id in range(12):
        testset = CASRDataset(save_dir, run_id=run_id, split="test", relative_kp=False)
        ytset = CASRDataset(save_dir, run_id=run_id, split="yt", relative_kp=False)
        testset, ytset = CASRSimpleDataset(testset), CASRSimpleDataset(ytset)
        all_test_results.append(get_predictions(testset))
        all_yt_results.append(get_predictions(ytset))
    
    acc, f1, jac, cfx = get_eval_metrics(all_test_results)
    summarize_results(acc, f1, jac, cfx)
    acc, f1, jac, cfx = get_eval_metrics(all_yt_results)
    summarize_results(acc, f1, jac, cfx) 
        
    
