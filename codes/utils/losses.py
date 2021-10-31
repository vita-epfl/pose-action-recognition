import torch 
import torch.nn as nn 

class MultiHeadClfLoss(nn.Module):
    
    """ for TITAN dataset, separately predict the actions in the hierarchy,
        then separately evaluate the losses, and sum them up 
    """
    
    def __init__(self, weights = [1, 1, 1, 1, 1]):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.weights = [w/sum(weights) for w in weights]
    def forward(self, pred, target):
        """ pred: a list of prediction results from multiple prediction heads of a network
                  the ith element has size (N, C_i)
            target: label tensor of size (N, M), where M is the number of heads in that network 
        """
        losses = []
        for idx, (pred_i, target_i) in enumerate(zip(pred, target.permute(1, 0))):
            losses.append(self.weights[idx]*self.cross_entropy(pred_i, target_i))
        
        return sum(losses)
             
    

        