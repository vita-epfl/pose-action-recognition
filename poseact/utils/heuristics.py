""" heuristics from https://github.com/charlesbvll/monoloco/blob/main/monoloco/activity.py
"""
import numpy as np 

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


def is_phoning(kp):
    """
    Returns flag of alert if someone is using their phone (talking on the phone)
    """
    x=0
    y=1

    nose = 0
    l_ear = 3
    l_shoulder = 5
    l_elbow = 7
    l_hand = 9
    r_ear = 4
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

    is_l_up = kp[y][l_hand] < kp[y][l_shoulder]
    is_r_up = kp[y][r_hand] < kp[y][r_shoulder]

    l_too_close = kp[x][l_hand] <= kp[x][l_shoulder] and kp[y][l_hand]>=head_top
    r_too_close = kp[x][r_hand] >= kp[x][r_shoulder] and kp[y][r_hand]>=head_top

    is_left_phone = is_l_up and l_angle <= 30 and l_too_close
    is_right_phone = is_r_up and r_angle <= 30 and r_too_close

    print("Top of head y is :", head_top)
    print("Nose height :", kp[y][nose])
    print("Right elbow x: {} and y: {}".format(kp[x][r_elbow], kp[y][r_elbow]))
    print("Left elbow x: {} and y: {}".format(kp[x][l_elbow], kp[y][l_elbow]))

    print("Right shoulder height :", kp[y][r_shoulder])
    print("Left shoulder height :", kp[y][l_shoulder])

    print("Left hand x = ", kp[x][l_hand])
    print("Left hand y = ", kp[y][l_hand])

    print("Is left hand up : ", is_l_up)

    print("Right hand x = ", kp[x][r_hand])
    print("Right hand y = ", kp[y][r_hand])

    print("Is right hand up : ", is_r_up)

    print("Left arm angle :", l_angle)
    print("Right arm angle :", r_angle)

    print("Is left hand close to head :", l_too_close)
    print("Is right hand close to head:", r_too_close)

    if is_left_phone or is_right_phone:
        return True

    return False