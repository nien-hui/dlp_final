"""
    Implementation of the proposed NoisyGCN. This implementation is based on the
    original GCN code (available in the file gcn.py in the same folder) that was
    extracted from the GNNGuard implementation which is also based on the DeepRobust
    package.
    ---
    The main addition different is based on the function "inject_noise" that takes
    the embedding vector and produce a noise to be injected.
"""

import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
import scipy
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
from deeprobust.graph.utils import *
#from torch_geometric.nn import GINConv, GATConv, GCNConv, JumpingKnowledge

from torch_geometric.nn import GINConv, GATConv, GCNConv, JumpingKnowledge

from deeprobust.graph.defense.torch_conv_guard import GCNConv

from torch.nn import Sequential, Linear, ReLU
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix 

from .apply_pruning import pruning


def inject_noise(x, scale_noise):

    """
    Main function to sample noise
    ---
    Inputs:
        x: the embedding vector
        scale_noise (float,): The noise scale -- denoted as \beta in the paper

    Output:
        noise: The sampled produced noise to be added to the embedding vector
    """

    # Initiate a centred gaussian
    loc = torch.zeros(x.shape, dtype=torch.float32)
    scale = torch.ones(x.shape, dtype=torch.float32)
    noise = torch.distributions.Normal(loc, scale).sample()

    # Rescale the gaussian based on the noise ratio
    noise = scale_noise * noise

    return noise.to(x.device)


class Noisy_PGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, drop=False, weight_decay=5e-4, n_edge=1,with_relu=True,
                 with_bias=True, device=None, noise_ratio_1=0.1):

        super(Noisy_PGCN, self).__init__()


        assert device is not None, "Please specify 'device'!"

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr

        self.attention = False

        weight_decay = 0

        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay

        self.with_relu = with_relu
        self.with_bias = with_bias
        self.n_edge = n_edge
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None


        self.gate = Parameter(torch.rand(1).detach())
        self.test_value = Parameter(torch.rand(1).detach()) 

        self.drop_learn_1 = Linear(2, 1)
        self.drop_learn_2 = Linear(2, 1)
        self.drop = drop
        self.bn1 = torch.nn.BatchNorm1d(nhid) 
        self.bn2 = torch.nn.BatchNorm1d(nhid) 
        nclass = int(nclass) 


        self.noise_ratio_1 = noise_ratio_1 

        """GCN from geometric"""
        """network from torch-geometric, """

        self.gc1 = GCNConv(nfeat, nhid, bias=True,)
        self.gc2 = GCNConv(nhid, nclass, bias=True,)

        self.reg_weight = 1e-5


    def forward(self, x, adj, l1_reg=False): 
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""
        x = x.to_dense()

        """GCN and GAT"""
        if self.attention:
            adj = self.att_coef(x, adj, i=0).to(self.device)


        edge_index = adj._indices() 

        x = self.gc1(x, edge_index, edge_weight=adj._values()) 

        if self.training: 
            noise = inject_noise(x, self.noise_ratio_1) 
            x = x + noise 


        x = F.relu(x) 
        # x = self.bn1(x)
        if self.attention:  # if attention=True, use attention mechanism
            adj_2 = self.att_coef(x, adj, i=1).to(self.device)
            adj_memory = adj_2.to_dense()  # without memory

        ### l1 loss
        # l1_act = 0
        # if l1_reg:
        #     l1_act = x.abs().sum() 
        
        x = F.relu(x) 

        l1_act = 0
        if l1_reg:
            l1_act = x.abs().sum() 

        # x = self.bn1(x) 
        if self.attention:  # if attention=True, use attention mechanism
            adj_2 = self.att_coef(x, adj, i=1).to(self.device)
            adj_memory = adj_2.to_dense()  # without memory 

            # adj_memory = self.gate * adj.to_dense() + (1 - self.gate) * adj_2.to_dense()
            row, col = adj_memory.nonzero()[:,0], adj_memory.nonzero()[:,1]
            edge_index = torch.stack((row, col), dim=0)
            adj_values = adj_memory[row, col]
        else:
            edge_index = adj._indices()
            adj_values = adj._values()


        x = F.dropout(x, self.dropout, training=self.training) 

        # x = F.dropout(x, self.dropout, training=self.training) 
        x = self.gc2(x, edge_index, edge_weight=adj_values) 

        if l1_act: 
            return F.log_softmax(x, dim=1), self.reg_weight * l1_act 
        else: 
            return F.log_softmax(x, dim=1) 


    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.drop_learn_1.reset_parameters()
        self.drop_learn_2.reset_parameters()
        try:
            self.gate.reset_parameters()
            self.fc2.reset_parameters()
        except:
            pass

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim<0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                     att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1) # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)#.cuda()
        att_adj = torch.tensor(att_adj, dtype=torch.int64)#.cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj

    def add_loop_sparse(self, adj, fill_value=1):
        # make identify sparse tensor
        row = torch.range(0, int(adj.shape[0]-1), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n.to(self.device)

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=81, att_0=None,
            attention=False, model_name=None, initialize=True, verbose=False, normalize=False, patience=510, ):


        '''
            train the gcn model, when idx_val is not None, pick the best model 
            according to the validation loss 
        '''
        self.sim = None

        self.idx_test = idx_test
        self.attention = attention
        # if self.attention:
        #     att_0 = self.att_coef_1(features, adj)
        #     adj = att_0 # update adj
        #     self.sim = att_0 # update att_0

        # self.device = self.gc1.weight.device

        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            self.idx_test = idx_test 
            self.attention = attention 

        if initialize:
            self.initialize() 

        if type(adj) is not torch.Tensor: 
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device) 
        else: 
            features = features.to(self.device) 
            adj = adj.to(self.device) 
            labels = labels.to(self.device) 


        # normalize = False # we don't need normalize here, the norm is conducted in the GCN (self.gcn1) model
        # if normalize:
        #     if utils.is_sparse_tensor(adj):
        #         adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        #     else:
        #         adj_norm = utils.normalize_adj_tensor(adj)
        # else:
        #     adj_norm = adj
        # add self loop
        adj = self.add_loop_sparse(adj)


        """The normalization gonna be done in the GCNConv"""
        self.adj_norm = adj
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

            self._train_without_val(labels, idx_train, train_iters, verbose) 



    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=None)   # this weight is the weight of each training nodes
            loss_train.backward()
            optimizer.step()
            if verbose and i % 20 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0


        for i in range(50): 
            # print('epoch', i)
            self.train()
            optimizer.zero_grad()
            output, reg_loss = self.forward(self.features, self.adj_norm, l1_reg=True) 
            loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + reg_loss 
            loss_train.backward() 
            optimizer.step() 
            self.eval() 

            loss_val = F.nll_loss(output[idx_val], labels[idx_val]) 
            acc_val = utils.accuracy(output[idx_val], labels[idx_val]) 
            # print(acc_val)
            # acc_test = utils.accuracy(output[self.idx_test], labels[self.idx_test])

            # if verbose and i % 5 == 0:
            #     print('Epoch {}, training loss: {}, val acc: {}, '.format(i, loss_train.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===') 
            if best_loss_val > loss_val: 
                best_loss_val = loss_val 
                self.output = output
                weights = deepcopy(self.state_dict()) 
            
            if acc_val > best_acc_val: 
                best_acc_val = acc_val 
                self.output = output 
                weights = deepcopy(self.state_dict()) 

            # if i == 50:

        ##### prune now 
        # print("do prune")
        example_input = [self.features, self.adj_norm] 
        pruning(self, example_input) 
        ##### 

        weights = deepcopy(self.state_dict())

        for i in range(train_iters-50): 
            # print('epoch', i)
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm) 
            loss_train = F.nll_loss(output[idx_train], labels[idx_train]) 
            loss_train.backward() 
            optimizer.step() 
            self.eval() 

            loss_val = F.nll_loss(output[idx_val], labels[idx_val]) 
            acc_val = utils.accuracy(output[idx_val], labels[idx_val]) 

            # print(acc_val)
            # acc_test = utils.accuracy(output[self.idx_test], labels[self.idx_test])

            # if verbose and i % 5 == 0:
            #     print('Epoch {}, training loss: {}, val acc: {}, '.format(i, loss_train.item(), acc_val))

            if best_loss_val > loss_val: 
                best_loss_val = loss_val 
                self.output = output
                weights = deepcopy(self.state_dict()) 

            if acc_val > best_acc_val: 
                best_acc_val = acc_val 
                self.output = output 
                weights = deepcopy(self.state_dict()) 


        if verbose:
            print('=== picking the best model according to the performance on validation ===') 


        self.load_state_dict(weights) 
        # """my test"""
        # output_ = self.forward(self.features, self.adj_norm)
        # acc_test_ = utils.accuracy(output_[self.idx_test], labels[self.idx_test])
        # print('With best weights, test acc:', acc_test_)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose): 
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))


            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()  # here use the self.features and self.adj_norm in training stage
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, output

    def _set_parameters(self):
        # TODO
        pass

    def predict(self, features=None, adj=None):
        '''By default, inputs are unnormalized data'''
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)
