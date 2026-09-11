""" 
Contains the complete implementation to reproduce the results of the "DICE"
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

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN 
from deeprobust.graph.global_attack import DICE 
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCNJaccard, GCNSVD, RGCN
from scipy.sparse import csr_matrix
# from deeprobust.graph.defense.noisy_gcn import Noisy_GCN
from deeprobust.graph.defense.noisy_gcn_with_prune import Noisy_PGCN
from deeprobust.graph.defense.noisy_gcn import Noisy_GCN

import argparse

import warnings
import wandb 

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=6, help='Random seed.') 
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 
                'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset') 
parser.add_argument('--ptb_rate', type=float, default=0.05,  
                                                    help='pertubation rate') 
parser.add_argument('--beta_max', type=float, default=1,   
                                                    help='Noise upper-range') 
parser.add_argument('--beta_min', type=float, default=0.01, 
                                                    help='Noise lower-range') 


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

np.random.seed(args.seed) 
torch.manual_seed(args.seed) 
if args.cuda:
    torch.cuda.manual_seed(args.seed) 

# Load the Dataset
data = Dataset(root='/tmp/', name=args.dataset) 

def test(new_adj, defense = "GCN"): 
    """
    Main function to test the considered benchmarks
    ---
    Inputs:
        adj: the clean/perturbed adjacency to be tested
        defense (str,): The considered defense method (Guard, Jaccard ..)

    Output:
        acc_test: The resulting accuracy test
    """

    # Pre-processing the input adjacency if need
    if not new_adj.is_sparse:
        new_adj = csr_matrix(new_adj.cpu())

    if not features.is_sparse:
        features_local = csr_matrix(features)

    if defense == "GCN":
        classifier = GCN(nfeat=features_local.shape[1], nhid=16,
                    nclass=labels.max().item() + 1, dropout=0.5, device=device) 
        attention = False

    elif defense == "Guard":
        classifier = GCN(nfeat=features_local.shape[1], nhid=16,
                    nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = True

    else:
        classifier = globals()[defense](nnodes=new_adj.shape[0],
                    nfeat=features_local.shape[1], nhid=16,
                    nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    classifier = classifier.to(device)

    classifier.fit(features_local, new_adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=False, attention=attention)

    classifier.eval()
    output = classifier.predict().cpu()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item())) 

    return acc_test.item()


def test_noisy(new_adj, perturb):
    """
    Main function to test our proposed NoisyGCN
    ---
    Inputs:
        new_adj: the clean/perturbed adjacency to be tested

    Output:
        acc_test: The resulting accuracy test
    """

    # Pre-processing the input adjacency if need
    if not new_adj.is_sparse:
        new_adj = csr_matrix(new_adj.cpu())

    if not features.is_sparse:
        features_local = csr_matrix(features)

    elif features.is_sparse:
        features_local = features

    best_acc_val = 0 
    acc_list = []
    
    # We test the best noise value based on the validation nodes as specified
    # in the main paper
    for beta in np.arange(0, args.beta_max, args.beta_min): 
        classifier = Noisy_GCN(nfeat=features.shape[1], nhid=16, 
            nclass=labels.max().item() + 1, dropout=0.5, device=device, 
                                                    noise_ratio_1=beta) 

        classifier = classifier.to(device) 

        classifier.fit(features_local, new_adj, labels, idx_train, 
                        train_iters=200, idx_val=idx_val, idx_test=idx_test, 
                        verbose=False, attention=False) 
        classifier.eval() 

        # Validation Acc
        acc_val, _ = classifier.test(idx_val)

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            acc_test, _ = classifier.test(idx_test) 

        acc_list.append(acc_test.item()) 

    df[f'noisy_gcn_{perturb}'] = acc_list 
        
    return acc_test.item() 


def test_prune_noisy(new_adj, perturb):
    # Pre-processing the input adjacency if need
    if not new_adj.is_sparse:
        new_adj = csr_matrix(new_adj.cpu())

    if not features.is_sparse:
        features_local = csr_matrix(features)

    elif features.is_sparse:
        features_local = features

    best_acc_val = 0
    acc_list = []
    # We test the best noise value based on the validation nodes as specified
    # in the main paper
    for beta in np.arange(0, args.beta_max, args.beta_min): 
        classifier = Noisy_PGCN(nfeat=features.shape[1], nhid=20, 
            nclass=labels.max().item() + 1, dropout=0., device=device, 
                                                    noise_ratio_1=beta) 

        classifier = classifier.to(device) 

        classifier.fit(features_local, new_adj, labels, idx_train, 
                        train_iters=200, idx_val=idx_val, idx_test=idx_test, 
                        verbose=False, attention=False) 
        classifier.eval() 

        # Validation Acc
        acc_val, _ = classifier.test(idx_val) 

        # if acc_val > best_acc_val:
        #     best_acc_val = acc_val
        #     acc_test, _ = classifier.test(idx_test) 
        
        best_acc_val = acc_val
        acc_test, _ = classifier.test(idx_test) 

        acc_list.append(acc_test.item()) 


    df[f'noisy_pgcn_{perturb}'] = acc_list 

    return acc_test.item() 


import pandas as pd

if __name__ == '__main__': 
    """
    Main function containing the Mettack implementation, please note that you
    need to uncomment the last part to use the other benchamarks
    """ 

    # df = pd.read_csv('beta_test.csv', index_col=0) 
    df = pd.DataFrame() 

    adj, features, labels = data.adj, data.features, data.labels 
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test 


    # Setup Attack Model
    model = DICE() 
    n_perturbations = int(0.05 * (adj.sum()//2)) 
    model.attack(adj, labels, n_perturbations) 


    ##model2
    model2 = DICE() 
    n_perturbations = int(0.1 * (adj.sum()//2)) 
    model2.attack(adj, labels, n_perturbations) 


    # Preprocessing the data
    adj, features, labels = preprocess(adj, features, labels,
                                                        preprocess_adj=False) 
    
    modified_adj = model.modified_adj
    modified_adj = torch.FloatTensor(modified_adj.todense()) 

    modified_adj2 = model2.modified_adj
    modified_adj2 = torch.FloatTensor(modified_adj2.todense()) 

    
    print('=== testing NoisyGCN ===') 
    attention=False 
    acc_noise_clean=test_noisy(adj, 0) 
    acc_noise_attacked=test_noisy(modified_adj, 5) 
    acc_noise_attacked2=test_noisy(modified_adj2, 10) 
    print('---------------') 
    print("NoisyGCN Non Attacked Acc - {}" .format(acc_noise_clean)) 
    print("NoisyGCN Attacked 5% Acc - {}" .format(acc_noise_attacked)) 
    print("NoisyGCN Attacked 10% Acc - {}" .format(acc_noise_attacked2)) 
    print('---------------') 


    print('=== testing pruning NoisyGCN ===') 
    attention=False 
    pacc_noise_clean=test_prune_noisy(adj, 0) 
    pacc_noise_attacked=test_prune_noisy(modified_adj, 5) 
    pacc_noise_attacked2=test_prune_noisy(modified_adj2, 10) 
    print('---------------') 
    print("NoisyPGCN Non Attacked Acc - {}" .format(pacc_noise_clean)) 
    print("NoisyPGCN Attacked 5% Acc - {}" .format(pacc_noise_attacked)) 
    print("NoisyPGCN Attacked 10% Acc - {}" .format(pacc_noise_attacked2)) 
    print('---------------') 



    df.to_csv('beta_test.csv') 

    # --- Normal GCN --- #
    # print('=== testing Normal GCN ===')
    #
    # modified_adj = model.modified_adj
    # modified_adj = torch.FloatTensor(modified_adj.todense())
    #
    # acc_gcn_non_attacked = test(adj)
    # acc_gcn_attacked = test(modified_adj)


    # Test other defense methods:
    # --- GNNGuard --- #

    # print('=== testing GNNGuard ===')
    # #attention = True
    # acc_non_attacked = test(adj, defense="Guard")
    # acc_attacked = test(modified_adj, defense="Guard")


    # --- GCNJaccard --- #
    # print('=== testing GCNJaccard ===')
    # attention = False
    # acc_jaccard_non_attacked = test(adj, defense = "GCNJaccard")
    # acc_jaccard_attacked = test(modified_adj, defense = "GCNJaccard")

    # print('---------------')
    # print("NoisyGCN Non Attacked Acc - {}" .format(acc_noise_clean))
    # print("NoisyGCN Attacked Acc - {}" .format(acc_noise_attacked))
    # print('---------------')