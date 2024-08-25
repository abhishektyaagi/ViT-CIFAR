""" import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import numpy as np
import pdb
import math
from maskGenerator1Diag import get_mask_pseudo_diagonal_numpy, get_mask_pseudo_diagonal_torch
from torch_sparse_soft_topk_google.isotonic_dykstra import isotonic_dykstra_mask
from torch_sparse_soft_topk_google.topk import sparse_soft_topk_mask_dykstra
from torch_sparse_soft_topk_google.isotonic_pav import sparse_soft_topk_mask_pav

#isotonic_dykstra = types.SimpleNamespace()
#isotonic_dykstra.isotonic_dykstra_mask = isotonic_dykstra_mask

seed = 5
torch.manual_seed(seed)

class CustomFullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, K=1, diagPos=[], alphaLR=0.01):
        super(CustomFullyConnectedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.topkLR = alphaLR
        print("Learning rate for alpha is: ", self.topkLR)
        self.num_permutations = in_features  # In the case of a square matrix
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the vectors V as learnable parameters with default Linear layer initialization
        seed = 5
        torch.manual_seed(seed)
        #self.layer = nn.Linear(out_features, out_features)
        self.V = nn.Parameter(torch.empty(out_features, out_features, device=self.device, dtype=torch.float32, requires_grad=True))
        #Initialize the diagonal of V with values using init uniform
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        # Assign the weights of self.layer to self.V
        #with torch.no_grad():  # Ensure no gradients are tracked during this operation
        #    self.V.copy_(self.layer.weight.data)
        
        #Initialize alphas as learnable parameters 
        self.alpha = nn.Parameter(torch.empty(self.num_permutations, device=self.device, requires_grad=True))
        #Set the value of all alphas to be 1/num_permutations
        nn.init.constant_(self.alpha, 1/self.num_permutations)

        #nn.init.uniform_(self.alpha, 0.0, 1.0)  # Initialize alpha with small positive values

        #Add assertion to check if all alphas are non-negative
        assert torch.all(self.alpha >= 0)

        # Initialize the alpha vector with 30 random values set to 1 and the rest to 0
        #self.alpha = torch.zeros(self.num_permutations, device=self.device)
        #Set specific indices to 1 by going through the values in diagPos
        #pdb.set_trace()
        #self.alpha[diagPos] = 1

    def compute_weights(self):
        #pdb.set_trace()
        #print("Learning rate for alpha is: ", self.topkLR)
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        #self.alpha_topk = sparse_soft_topk_mask_pav(self.alpha, self.K, 0.01, 4/3, 50).to(self.device)
        #print(torch.count_nonzero(self.alpha_topk))
        #pdb.set_trace()

        #Make alpha_tpok zero where the value is less than 0.5. NOTE: THIS IS NOT DIFFERENTIABLE
        #self.alpha_topk[self.alpha_topk < 0.5] = 0

        #Print the number of non-zero values in alpha_topk. Print it after every 100seconds
        #print(torch.sum(self.alpha_topk != 0))

        # Find non-zero alpha indices
        #non_zero_alpha_indices = torch.nonzero(self.alpha, as_tuple=False).squeeze()
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()
        #print(non_zero_alpha_indices)
        #pdb.set_trace()
        #Store matrices in a list
        #matrix_list = []

        #Initialize the weight matrix with zeros of size
        WSum = torch.zeros((self.in_features,self.out_features), device=self.device)
        #For each indice, get the corresponding mask
        for i in non_zero_alpha_indices:
            #mask1_np = get_mask_pseudo_diagonal_numpy((self.in_features,self.out_features), sparsity=0.99967, experimentType="randDiagOneLayer", layerNum=1, numDiag=1
            #                                    , diag_pos=i, currLayer=1, debug=0)
            mask1 = get_mask_pseudo_diagonal_torch((self.in_features,self.out_features), sparsity=0.99967, experimentType="randDiagOneLayer", diag_pos=i)
            
            #Multiply mask with diagonalized matrix having V[i] as diagonal
            #mask1 = torch.as_tensor(mask1_np, dtype=torch.float32).to(self.device).detach()
            #result = mask1*torch.diag(self.V[i]).to(self.device)
            result = self.alpha_topk[i]*torch.matmul(mask1, torch.diag(self.V[i]).to(self.device))
            #result = mask1
            #matrix_list.append(result)
            WSum += result 

        #Sum all the matrices in the list
        #W = torch.sum(matrix_list, dim=0)
        W = WSum
        return W

    @property
    def weights(self):
        return self.compute_weights()

    def forward(self, x):
        x = x.to(self.device)
        W = self.weights    
        out = F.linear(x, W)
        return out

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr
        print("New learning rate for alpha is: ", self.topkLR)

 """

import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import numpy as np
import math
import pdb
#from maskGenerator1Diag import get_mask_pseudo_diagonal_numpy, get_mask_pseudo_diagonal_torch
from maskGenerator1DiagRect import get_mask_pseudo_diagonal_numpy, get_mask_pseudo_diagonal_torch
from torch_sparse_soft_topk_google.isotonic_dykstra import isotonic_dykstra_mask
from torch_sparse_soft_topk_google.topk import sparse_soft_topk_mask_dykstra
from torch_sparse_soft_topk_google.isotonic_pav import sparse_soft_topk_mask_pav

seed = 5
torch.manual_seed(seed)

class CustomFullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, sparsity = 0.1, diagPos=[], alphaLR=0.01):
        super(CustomFullyConnectedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.total_permutations = max(in_features, out_features)
        self.diag_length = min(in_features, out_features)

        num_params = in_features * out_features
        req_params = int((1-sparsity) * num_params)
        K = math.ceil(req_params/min(in_features, out_features))

        self.K = K
        self.topkLR = alphaLR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.V = nn.Parameter(torch.empty(self.total_permutations, self.diag_length, device=self.device, dtype=torch.float32, requires_grad=True))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        self.alpha = nn.Parameter(torch.empty(self.total_permutations, device=self.device, requires_grad=True))
        nn.init.constant_(self.alpha, 1/self.in_features)
        #pdb.set_trace()
        assert torch.all(self.alpha >= 0)

    def compute_weights(self):
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()
        #print(non_zero_alpha_indices)
        #pdb.set_trace()
        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0) 
        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)
        for i in non_zero_alpha_indices:
            mask1 = get_mask_pseudo_diagonal_torch((self.out_features, self.in_features), sparsity=0.99967, experimentType="randDiagOneLayer", diag_pos=i)
            
            if self.out_features > self.in_features:
                result = self.alpha_topk[i] * torch.matmul(mask1, torch.diag(self.V[i]).to(self.device))
            else:
                mask1 = mask1.T
                result = self.alpha_topk[i] * (torch.matmul(mask1, torch.diag(self.V[i])).T.to(self.device))
            WSum += result
        return WSum

    @property
    def weights(self):
        return self.compute_weights()

    """ def forward(self, x):
        pdb.set_trace()
        if x.dim() == 3:
            batch_size, seq_len, feature_dim = x.shape
            x = x.view(batch_size * seq_len, feature_dim)
        W = self.weights    
        out = F.linear(x, W)
        if x.dim() == 3:
            out = out.view(batch_size, seq_len, -1)
        return out """
    
    def forward(self, x):
        x = x.to(self.device)
        W = self.weights
        #pdb.set_trace()    
        out = F.linear(x, W)
        return out


    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr
        #print("New learning rate for alpha is: ", self.topkLR)
