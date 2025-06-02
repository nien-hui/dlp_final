"""
Contains the complete implementation to reproduce the results of the "Mettack"
---
The implementation contains all the related benchmarks:
    - GCNGuard
    - RGCN
    - GCN-Jaccard
    - Our proposed Noisy GCN.

To use the benchmarks (GCNGuard, RGCN ...), please adapt the argument "defense"
in the "test" function. We provided an example of their use in the main section
of this file.
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense.noisy_gcn import Noisy_GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import *
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.defense import *
from deeprobust.graph.data import Dataset
import argparse
from scipy.sparse import csr_matrix
import pickle
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
import scipy
import numpy as np

from sklearn.preprocessing import normalize

from deeprobust.graph.defense.noisy_gcn_with_prune import Noisy_PGCN

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, 
                    help='Disables CUDA training.') 
parser.add_argument('--seed', type=int, default=4, help='Random seed.') 
parser.add_argument('--epochs', type=int, default=200, 
                    help='Number of epochs to train.') 
parser.add_argument('--lr', type=float, default=0.01, 
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, 
                    help='Weight decay (L2 loss on parameters).') 
parser.add_argument('--hidden', type=int, default=16, 
                    help='Number of hidden units.') 
parser.add_argument('--dropout', type=float,     default=0.5, 
                    help='Dropout rate (1 - keep probability).') 
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 
                'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset') 
parser.add_argument('--ptb_rate', type=float, default=0.05, 
                                                        help='pertubation rate') 
parser.add_argument('--model', type=str, default='Meta-Self', 
                    choices=['A-Meta-Self', 'Meta-Self'], help='model variant') 

parser.add_argument('--modelname', type=str, default='GCN',
                                            choices=['GCN', 'GAT','GIN', 'JK']) 
parser.add_argument('--defensemodel', type=str, default='GCNJaccard',
                                    choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=False,
                                                        choices=[True, False]) 
parser.add_argument('--beta_max', type=float, default=0.20) 
parser.add_argument('--beta_min', type=float, default=0.01) 


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# Load the Dataset
data = Dataset(root='/tmp/', name=args.dataset) 
adj, features, labels = data.adj, data.features, data.labels 

# Extract the Train/Val/Test idx
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)

# Transforming the perturbation rate into edges
perturbations = int(args.ptb_rate * (adj.sum()//2)) 

# Preprocessing and sparsifying the adjacency and the feature matrix
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
adj, features = csr_matrix(adj), csr_matrix(features)

# Transform to undirected adjacency (spacially useful for OGB Data)
adj = adj + adj.T
adj[adj>1] = 1


# Setup GCN as the Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, 
            dropout=0.5, with_relu=False, with_bias=False, weight_decay=5e-4, 
                                                            device=device) 

surrogate = surrogate.to(device) 
surrogate.fit(features, adj, labels, idx_train, train_iters=100) 

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

# Initialize the Attack
# if 'A' in args.model:
#     model = MetaApprox(model=surrogate, nnodes=adj.shape[0],
#                         feature_shape=features.shape, attack_structure=True,
#                         attack_features=False, device=device, lambda_=lambda_)

# else:
#     model = Metattack(model=surrogate, nnodes=adj.shape[0],
#                         feature_shape=features.shape,  attack_structure=True,
#                         attack_features=False, device=device, lambda_=lambda_)

# model = model.to(device)


def test_noisy(new_adj):
    """
    Main function to test our proposed NoisyGCN
    ---
    Inputs:
        new_adj: the clean/perturbed adjacency to be tested

    Output:
        acc_test: The resulting accuracy test
    """


    best_acc_val = 0
    # We test the best noise value based on the validation nodes as specified
    # in the main paper
    for beta in np.arange(0, args.beta_max, args.beta_min):
        classifier = Noisy_GCN(nfeat=features.shape[1], nhid=16,
                                nclass=labels.max().item() + 1, dropout=0.5,
                                    device=device, noise_ratio_1=beta)

        classifier = classifier.to(device)

        classifier.fit(features, new_adj, labels, idx_train, train_iters=200,
                       idx_val=idx_val,
                       idx_test=idx_test,
                       verbose=False, attention=False)
        classifier.eval() 

        # Validation Acc
        acc_val, _ = classifier.test(idx_val)

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            acc_test, _ = classifier.test(idx_test) 
        print(acc_val)

    return acc_test.item()

def test_prune_noisy(new_adj):
    # Pre-processing the input adjacency if need
    best_acc_val = 0
    # We test the best noise value based on the validation nodes as specified 
    # in the main paper 
    for beta in np.arange(0, args.beta_max, args.beta_min): 
        classifier = Noisy_PGCN(nfeat=features.shape[1], nhid=20,lr=0.01, 
                                nclass=labels.max().item() + 1, dropout=0., 
                                    device=device, noise_ratio_1=beta) 

        classifier = classifier.to(device)

        classifier.fit(features, new_adj, labels, idx_train, train_iters=200,
                       idx_val=idx_val,
                       idx_test=idx_test,
                       verbose=False, attention=False) 
        classifier.eval() 

        # Validation Acc 
        acc_val, _ = classifier.test(idx_val) 
        print(acc_val) 

        if acc_val > best_acc_val: 
            best_acc_val = acc_val 
            acc_test, _ = classifier.test(idx_test) 

    return acc_test.item() 


def test(adj, defense="GCN"):
    """
    Main function to test the considered benchmarks
    ---
    Inputs:
        adj: the clean/perturbed adjacency to be tested
        defense (str,): The considered defense method (Guard, Jaccard ..)

    Output:
        acc_test: The resulting accuracy test
    """

    if defense == "GCN":
        classifier = globals()[args.modelname](nfeat=features.shape[1],
            nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    elif defense == "Guard":
        classifier = globals()[args.modelname](nfeat=features.shape[1],
            nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = True

    else:
        classifier = globals()[defense](nnodes=adj.shape[0], nhid=16,
                        nfeat=features.shape[1], nclass=labels.max().item() + 1,
                                                    dropout=0.5, device=device)
        attention = False

    classifier = classifier.to(device)

    classifier.fit(features, adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val, idx_test=idx_test, verbose=False,
                                                attention=attention)
    classifier.eval()

    acc_test, _ = classifier.test(idx_test)
    return acc_test.item()


import pandas as pd

if __name__ == '__main__':
    """
    Main function containing the Mettack implementation, please note that you
    need to uncomment the last part to use the other benchamarks
    """
    df = pd.read_csv('METTACK_citeseer.csv', index_col=0) 

    model = Metattack(model=surrogate, nnodes=adj.shape[0], 
                            feature_shape=features.shape,  attack_structure=True, 
                            attack_features=False, device=device, lambda_=lambda_) 

    model = model.to(device)
    # Apply the Attack and get the resulting adjacency 
    perturbations = int(0.05 * (adj.sum()//2)) 
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, 
                                                            ll_constraint=False) 
    modified_adj = model.modified_adj
    modified_adj_sparse = csr_matrix(modified_adj.cpu().numpy()) 


    model = Metattack(model=surrogate, nnodes=adj.shape[0], 
                            feature_shape=features.shape,  attack_structure=True, 
                            attack_features=False, device=device, lambda_=lambda_) 

    model = model.to(device)
    perturbations = int(0.1 * (adj.sum()//2))
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, 
                                                            ll_constraint=False) 
    modified_adj2 = model.modified_adj 
    modified_adj_sparse2 = csr_matrix(modified_adj2.cpu().numpy()) 


    # print('=== testing NoisyGCN ===')
    # attention=False
    # acc_noise_clean=test_noisy(adj)
    # acc_noise_attacked=test_noisy(modified_adj_sparse) 
    # # acc_noise_attacked2=test_noisy(modified_adj_sparse2) 
    # print('---------------')
    # print("NoisyGCN Non Attacked Acc - {}" .format(acc_noise_clean)) 
    # print("NoisyGCN Attacked 5% Acc - {}" .format(acc_noise_attacked)) 
    # # print("NoisyGCN Attacked 10% Acc - {}" .format(acc_noise_attacked2)) 
    # print('---------------') 


    print('=== testing NoisyPGCN ===') 
    pacc_noise_clean=test_prune_noisy(adj) 
    pacc_noise_attacked=test_prune_noisy(modified_adj_sparse) 
    pacc_noise_attacked2=test_prune_noisy(modified_adj_sparse2) 
    print('---------------')
    print("NoisyPGCN Non Attacked Acc - {}" .format(pacc_noise_clean)) 
    print("NoisyPGCN Attacked 5% Acc - {}" .format(pacc_noise_attacked)) 
    print("NoisyPGCN Attacked 10% Acc - {}" .format(pacc_noise_attacked2)) 
    print('---------------') 


    new_row_df = pd.DataFrame({'noisypgcn_0': [pacc_noise_clean], 
                               'noisypgcn_5': [pacc_noise_attacked], 
                               'noisypgcn_10': [pacc_noise_attacked2], 
                               }) 

    # 用 concat 拼接，ignore_index=True 会让最终的索引重新编号 
    df = pd.concat([df, new_row_df], ignore_index=True) 

    df.to_csv('METTACK_citeseer.csv') 


    # To run another defense: 
    # --- Normal GCN --- #
    # print('=== testing Normal GCN ===')
    # acc_gcn_non_attacked = test(adj)
    # acc_gcn_attacked = test(modified_adj_sparse)

    # l_acc_gcn_non.append(acc_gcn_non_attacked)
    # l_acc_gcn_att.append(acc_gcn_attacked)


    # --- RGCN --- #
    # # print('=== testing RGCN ===')
    # attention = False
    # acc_rgcn_non_attacked = test(adj, defense = "RGCN")
    # acc_rgcn_attacked = test(modified_adj_sparse, defense = "RGCN")

    # --- GNNGuard --- #
    # print('=== testing GNNGuard ===')
    # attention = True
    # acc_non_attacked = test(adj, defense="Guard")
    # acc_attacked = test(modified_adj_sparse, defense="Guard")


    # print('---------------')
    # print("NoisyGCN Non Attacked Acc - {}" .format(acc_noise_clean))
    # print("NoisyGCN Attacked Acc - {}" .format(acc_noise_attacked))
    # print('---------------')
