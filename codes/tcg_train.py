import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from models import MonolocoModel
from utils import compute_accuracy
from tcg_dataset import TCGDataset, TCGSingleFrameDataset, tcg_train_test_split, tcg_collate_fn

# define device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# set value for some arguments 
data_dir = "codes/data"
fig_dir = "codes/figs"
weight_dir = "codes/models/trained/"
label_type = "major"
batch_size = 512
num_epoch = 100
input_size = 17*3
output_size = 15 if label_type == "sub" else 4
linear_size = 256
dropout = 0.2
n_stage = 3 
lr = 0.002

tcg_dataset = TCGDataset(data_dir, label_type)
tcg_trainset, tcg_testset = tcg_train_test_split(tcg_dataset)
# split sequences first and then take frames from two sets of sequences, to avoid data correlation 
# previously when I didn't notice this, the testing accuracy was about 85% 
trainset, testset = TCGSingleFrameDataset(tcg_trainset), TCGSingleFrameDataset(tcg_testset)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

model = MonolocoModel(input_size, output_size, linear_size, dropout, n_stage).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

train_loss_list, test_acc_list = [], [] 
for epoch in range(num_epoch):
    model.train()
    batch_loss = [] 
    for pose, label in trainloader:
        pose, label = pose.flatten(-2,-1).to(device), label.to(device)
        pred = model(pose)
        loss = criterion(pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item()) 
    
    train_loss = sum(batch_loss)/len(batch_loss)
    # scheduler.step(train_loss)
    test_acc = compute_accuracy(model, testloader)
    print("Epoch {} Avg Loss {:.4f} Test Acc {:.4f}".format(epoch, train_loss, test_acc))
    train_loss_list.append(train_loss) 
    test_acc_list.append(test_acc)