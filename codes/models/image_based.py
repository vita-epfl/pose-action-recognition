import torch 
import torch.nn as nn 

from typing import List
from torchvision.models import resnet50

class MultiHeadLinear(nn.Module):
    
    def __init__(self, input_size:List[int], output_size:List[int]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.heads = []
        for in_size, out_size in zip(self.input_size,self.output_size):
            self.heads.append(nn.Linear(in_features=in_size, out_features=out_size))
        self.heads = nn.ModuleList(self.heads)
        
    def forward(self, x):
        
        predictions = [head(x) for head in self.heads]
        
        return predictions

def multihead_resnet(output_size:List[int], ckpt_path=None, pretrained=False):
    model = resnet50(pretrained=pretrained)
    if pretrained==False and ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    in_features = [model.fc.in_features for _ in output_size]
    model.fc = MultiHeadLinear(input_size=in_features, output_size=output_size)
    model.output_size = output_size
    
    return model