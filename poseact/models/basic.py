import torch
import torch.nn as nn 
import torch.nn.functional as F 

class MonolocoModel(nn.Module):
    """
    Ported from https://github.com/vita-epfl/monoloco/blob/main/monoloco/network/architectures.py
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        N, V, C = x.shape 
        x = x.view(N, V*C)
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        # need to verify this 
        # return F.softmax(y, dim=-1)
        return y


class MyLinear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):

        y = self.w1(x)
        if x.shape[0]>1:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        if x.shape[0]>1:
            y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class TempMonolocoModel(nn.Module):
    """
    Ported from https://github.com/vita-epfl/monoloco/blob/main/monoloco/network/architectures.py
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        
        self.lstm = nn.LSTM(input_size=linear_size, hidden_size=linear_size, num_layers=1, batch_first=True)

    def forward(self, x:torch.Tensor):
        # batch size, sequence length, number of body joints, feature dimension (3 for 3D space)
        N, T, V, C = x.shape 
        device = x.device
        x = x.view(N*T, V*C)
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        
        y = y.view(N, T, -1)
        h0 = torch.zeros((1, N, self.linear_size)).to(device)
        c0 = torch.zeros((1, N, self.linear_size)).to(device)
        # getting Cuda Error : RuntimeError: CUDNN_STATUS_EXECUTION_FAILED possibly out of memory
        # https://discuss.pytorch.org/t/cuda-error-runtimeerror-cudnn-status-execution-failed/17625/13
        # didn't see it when using batch size 1, but RNN is just toooooooooo slow
        # TODO try to replace it with the attention block in transformer 
        y, (hc, cn) = self.lstm(y, (h0, c0))
        
        y = self.w2(y)
        # need to verify this 
        # return F.softmax(y, dim=-1)
        return y
    
    
class MultiHeadMonoLoco(nn.Module):

    def __init__(self, input_size, output_size=[4, 7, 9, 13, 4], linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.output_heads = []
        for size in self.output_size:
            self.output_heads.append(nn.Linear(self.linear_size, size))
        self.output_heads = nn.ModuleList(self.output_heads)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        N, V, C = x.shape 
        x = x.view(N, V*C)
        # pre-processing
        y = self.w1(x)
        if x.shape[0]>1:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
            
        # predictions = [F.softmax(head(y), dim=-1) for head in self.output_heads]
        predictions = [head(y) for head in self.output_heads]
        
        return predictions