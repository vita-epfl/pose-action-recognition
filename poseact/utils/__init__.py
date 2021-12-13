import os 
import sys 
import ctypes
import multiprocessing as mp 
from .losses import MultiHeadClfLoss

def setup_multiprocessing():
    mp.set_start_method('spawn')
    if sys.platform.startswith("linux"):
        try:
            libgcc_s = ctypes.CDLL("/usr/lib64/libgcc_s.so.1")
        except:
            pass 

def make_save_dir(base_dir, subdir_name, return_folder=False):
    assert os.path.isdir(base_dir), "base dir does not exits"
    subdir = base_dir + "/" + subdir_name
    if not os.path.exists(subdir):
        print("making folder at {}".format(subdir))
        os.mkdir(subdir)
    return subdir