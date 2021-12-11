import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """ proposed in paper https://arxiv.org/pdf/1708.02002.pdf
        implementation from https://github.com/clcarwin/focal_loss_pytorch
        changed `Variable()` into `.requires_grad_()`, removed `long`
        original code runs on python2 and cpu, modified to run on python3 and GPU
    """

    def __init__(self, gamma=0, alpha=None, device="cpu", ignore_index:int=None):
        super(FocalLoss, self).__init__()
        # registered gamma and alpha as nn.Parameters, so we can use self.to(device) to move them easily
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.alpha = alpha
        self.ignore_index = ignore_index
        if isinstance(alpha, (float, int)):  # binary classification
            self.alpha = nn.Parameter(torch.tensor([alpha, 1-alpha]), requires_grad=False)
        if isinstance(alpha, list):  # multi-class classification
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.to(device)

    def forward(self, pred, target):
        
        if pred.dim() == 3: # N, T, C 
            N, T, C = pred.shape
            pred = pred.contiguous().view(N*T, C)
            
        if pred.dim() == 4:
            N, C, H, W = pred.shape
            pred = pred.view(N, C, H*W)  # N,C,H,W => N,C,H*W
            pred = pred.permute(0, 2, 1)    # N,C,H*W => N,H*W,C
            pred = pred.contiguous().view(N*H*W, C)   # N,H*W,C => N*H*W,C
            
        target = target.view(-1, 1)
        if self.ignore_index is not None:
            keep_flag = torch.logical_not(torch.eq(target, self.ignore_index))# keep those not equal to ignore index
            target = target[keep_flag.flatten()]
            pred = pred[keep_flag.flatten()]

        logpt = F.log_softmax(pred, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp().requires_grad_()

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at.requires_grad_()

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()


class MultiHeadClfLoss(nn.Module):

    """ for TITAN dataset, separately predict the actions in the hierarchy,
        then separately evaluate the losses, and sum them up 
        BE SURE to add the parameters of this loss into the optimizer, if you 
        use uncertainty-based multitask loss 
    """
    # data proportion in percentage, statistics from pifpaf detection,
    # all data included, not just train split
    EPS = 1e-2
    data_prop = [
        [5.53, 10.41, 2.22, 81.84],  # 4 communicative
        [0.17, 0.06, 0.03, 0.07, 0.36, 0.6, 98.71],  # 7 complex context
        [1.36, 0.01, 0.00, 0.42, 4.19, 0.12, 13.58, 80.10, 0.23],  # 9 atomic
        [3.24, 0.24, 0.15, 4.79, 0.70, 0.71, 7.11, 0.23, 0.24, 0.92, 37.70, 29.32, 14.67],  # 13 simple context
        [8.01, 0.81, 2.28, 88.90],  # 4 transporting
    ]
    ignored_classes = [
        None, 
        [0, 1, 2, 3],
        None,
        [3, 4, 5, 6],
        None
    ]
    def __init__(self, n_tasks=5, imbalance="manual", gamma=0, anneal_factor=0, uncertainty=False, device="cpu", mask_cls=False):
        """ 
            a multitask loss for action recognition task on TITAN dataset https://usa.honda-ri.com/titan

        Args:

            n_task (int, optional): number of tasks, TITAN has 5 categories of actions so . Defaults to 5.

            imbalance (str, optional): how to deal with imbalanced data. Use focal loss if set to "focal", weight the classes with `MultiHeadClfLoss.data_prop` if set to "manual", use alpha-weighted focal loss if "both". Defaults to "manual".

            annel_factor (int, optional): anneal factor for the manually set weights, -1 for inverse weight to the data percentage, 0 for equal weights, any value between will be valid. Defaults to 0 (don't weight).

            uncertainty (bool, optional): use task uncertainty (aleatoric homoscedastic loss, see https://arxiv.org/abs/1705.07115). Defaults to False.

        """
        super().__init__()
        self.device = device
        self.anneal_factor = anneal_factor
        self.use_uncertainty = uncertainty
        self.mask_cls = mask_cls # whether to mask out unlearnable classes 
        self.ignore_index = -100 # replace some class labels with this, to ignore them during training
        
        if n_tasks == 5:
            self.cls_weights = self.compute_class_weights()
        else:
            self.cls_weights = [None for _ in range(n_tasks)]
            
        if imbalance == "manual":
            self.base_loss = [nn.CrossEntropyLoss(weight=self.cls_weights[idx], ignore_index=self.ignore_index) 
                                        for idx in range(n_tasks)]
        elif imbalance == "focal":
            self.base_loss = [FocalLoss(gamma=gamma, device=self.device, ignore_index=self.ignore_index) 
                                        for idx in range(n_tasks)]
        elif imbalance == "both":
            self.base_loss = [FocalLoss(gamma=gamma, alpha=self.cls_weights[idx], 
                                        device=self.device, ignore_index=self.ignore_index) 
                                        for idx in range(n_tasks)]
        # basically, in multitask loss, total_loss = sum(one_loss/var + log_var)
        # for numerical stability 1/var = exp(-log_var)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks)) if self.use_uncertainty else None
        self.to(device)

    def compute_class_weights(self):
        cls_weights = []
        for prop in self.data_prop:
            annealed_rate = (torch.tensor(prop)+self.EPS)**self.anneal_factor  # raise to some power
            weight = annealed_rate/(torch.sum(annealed_rate))  # normalize sum to 1
            cls_weights.append(nn.Parameter(weight, requires_grad=False))
        # registered as nn.Parameters, so we can use to(device) to move them easily
        return nn.ParameterList(cls_weights)

    def mask_out_classes(self, target:torch.Tensor):
        """ 
            replace the classes in cls.ignored_class with -100 (same as nn.CrossEntropyLoss)
            so they will be ignored during training 
            
            target: label tensor of size (N, M), where M is the number of heads in that network 
            
            pytorch will modify `target` in-place, so no need to return some new tensor 
        """
        for ignored_i, target_i in zip(self.ignored_classes, target.permute(1, 0)):
            if ignored_i is None:
                continue
            target_copy = target_i.clone().detach().cpu().numpy()
            kept_flag = np.isin(target_copy, ignored_i)
            target_i[kept_flag] = self.ignore_index
            
    def forward(self, pred, target):
        """ pred: a list of prediction results from multiple prediction heads of a network
                  the ith element has size (N, C_i)
            target: label tensor of size (N, M), where M is the number of heads in that network 
        """
        if self.mask_cls:
            self.mask_out_classes(target)
        loss_list = []
        for idx, (pred_i, target_i) in enumerate(zip(pred, target.permute(1, 0))):
            if self.use_uncertainty:
                loss = torch.exp(-self.log_vars[idx])*self.base_loss[idx](pred_i, target_i) + self.log_vars[idx]
            else:
                loss = self.base_loss[idx](pred_i, target_i)
            loss_list.append(loss)

        return sum(loss_list)


def test_gpu_forward():
    # add some debug tests here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = [torch.randn(100, n_cls).to(device) for n_cls in [4, 7, 9, 13, 4]]
    label = torch.cat([torch.randint(0, n_cls, (100,)).reshape(-1, 1).to(device)
                       for n_cls in [4, 7, 9, 13, 4]], dim=1)

    param_set = [('manual', -0.75, True), ('focal', 0, True), ('both', -0.75, True)]

    for imbalance, anneal_factor, uncertainty in param_set:
        print("Testing {} mode with anneal factor {} Uncertainty {}".format(
            imbalance, anneal_factor, uncertainty))
        criterion = MultiHeadClfLoss(n_tasks=5, imbalance=imbalance, gamma=2, anneal_factor=anneal_factor,
                                     uncertainty=uncertainty, device=device, mask_cls=True)
        # it looks like the parameters will change by at most lr in one step
        optimizer = optim.Adam(criterion.parameters(), lr=0.5)

        for _ in range(10):
            loss = criterion(pred, label)
            # print("original parameters: ", list(criterion.parameters()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("updated parameters: ", list(criterion.parameters()))

def test_mask_out_label():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = [torch.randn(100, n_cls).to(device) for n_cls in [4, 7, 9, 13, 4]]
    label = torch.cat([torch.randint(0, n_cls, (100,)).reshape(-1, 1).to(device)
                       for n_cls in [4, 7, 9, 13, 4]], dim=1)
    criterion = MultiHeadClfLoss(n_tasks=5, imbalance="both", gamma=2, anneal_factor=-0.75,
                                     uncertainty=True, device=device, mask_cls=True)
    loss = criterion(pred, label)

def test_ignore_index():
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    data  = torch.randn((12, 10))
    model = nn.Linear(10, 3)
    pred = model(data)
    label = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])  
    
    ignore_flag = np.isin(label, torch.tensor([0, 1]))
    label[ignore_flag] = -100
    
    loss1 = criterion(pred, label)
    loss2 = criterion(pred[8:], label[8:])
    assert torch.allclose(loss1, loss2)
    
    fl = FocalLoss(gamma=2, alpha=None, ignore_index=-100)
    loss3 = fl.forward(pred, label)
    loss4 = fl.forward(pred[8:], label[8:])
    assert torch.allclose(loss3, loss4)
    
if __name__ == "__main__":
    test_mask_out_label()
    test_ignore_index()
    test_gpu_forward()
