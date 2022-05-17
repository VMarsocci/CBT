import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import utils


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    
    def __init__(self):
        super().__init__()

        # -EWC:
        self.ewc_lambda = 0     #-> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.fisher_n = None    #-> number of batches for estimating FI-matrix (if "None", full pass over dataset)
        self.b_s = 1     #-> batch size to compute FI-matrix
        self.EWC_task_count = 0 #-> keeps track of number of quadratic loss terms (for "offline EWC")
        self.est_fisher_info = {}
        self.prev_task_info = {}

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass

    #----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, dataset, collate_fn=None):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        for n, p in self.named_parameters():
            if p.requires_grad:
                # n = n.replace('.', '__')
                self.est_fisher_info[n] = p.detach().clone().zero_()
                self.prev_task_info[n] = p.detach().clone().cpu().numpy()


        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        data_loader = utils.get_data_loader(dataset, batch_size=self.b_s, cuda=self._is_on_cuda(), collate_fn=collate_fn)

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index,((y1, y2), _) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            y1 = y1.to(self._device())
            y2 = y2.to(self._device())
            
            output = self(y1, y2)
            loss = output[0] 
            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            loss.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # n = n.replace('.', '__')
                    if p.grad is not None:
                        self.est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        self.est_fisher_info = {n: (p/index).cpu().numpy() for n, p in self.est_fisher_info.items()}

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        # self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)

        return self.est_fisher_info, self.prev_task_info


    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count>0:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                    mean = torch.from_numpy(self.prev_task[n]).float().to(self._device())
                    fisher_ = torch.from_numpy(self.fisher[n]).float().to(self._device())
                    # Calculate EWC-loss
                    losses.append((fisher_ * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())
            
    def get_info(self, diz, ewc_lambda):
        self.fisher = diz[()]['fisher']
        self.prev_task = diz[()]['prev_task']
        self.ewc_lambda = ewc_lambda
