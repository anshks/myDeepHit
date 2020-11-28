from utils import *
# from modules import *
# from network import *
from class_DeepHit import *
import math
import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

#==================add regularisation===========================

ground_truth_G = createGraph()
dummy = pd.read_csv("synthetic_final.csv")
# dummy = pd.read_csv("metabric_deepSurv.csv")
label = np.asarray(dummy['label']); label = label.reshape((len(label),1))
time = np.asarray(dummy['time']).astype(int); time = time.reshape((len(time),1))
# print(time)
X = np.asarray(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]])
# X = f_get_Normalization(X, 'standard')

data_variable_size = 9
x_dims = 1

num_Event = int(len(np.unique(label)) - 1)
num_Category = int(np.max(time) * 1.2)
print(num_Category)

mask1 = create_mask1(time,label,num_Event, num_Category)
mask2 = create_mask2(time,label,num_Event, num_Category)

X_train,X_test,Y_train,Y_test, E_train,E_test, mask1_train, mask1_test, mask2_train, mask2_test = train_test_split(X, time, label, mask1, mask2, test_size=0.2, random_state=1234)

X_train,X_val,Y_train,Y_val, E_train,E_val, mask1_train, mask1_val, mask2_tain, mask2_val = train_test_split(X_train, Y_train, E_train, mask1_train, mask2_train, test_size=0.2, random_state=1234)

def f_get_minibatch(mb_size, x, label, time, mask1, mask2):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)
    x_mb = torch.from_numpy(x[idx, :]).float()
    k_mb = torch.from_numpy(label[idx, :]) # censoring(0)/event(1,2,..) label
    t_mb = torch.from_numpy(time[idx, :])
    m1_mb = mask1[idx, :, :].float() #fc_mask
    m2_mb = mask2[idx, :].float() #fc_mask
    return x_mb, k_mb, t_mb, m1_mb, m2_mb

#===================================
# training:
#===================================

def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer, tau_A, graph_threshold, mask1_train, mask2_train, lr, data_variable_size, x_dims, num_Event, num_Category, alpha, beta, batch_size, X_train, Y_train, E_train):

    nll1_train = []
    kl_train = []
    mse_train = []
    shd_trian = []
    nll2_train = []

    survival.train()

    flat, events, time, temp_mask1, temp_mask2 = f_get_minibatch(batch_size, X_train, E_train, Y_train, mask1_train, mask2_train)

    optimizer.zero_grad()

    d_out = survival(flat)

    loss_nll2 = deephit_nll(d_out, events.float(), time.float(), temp_mask1)
    loss_rank = rank_loss(num_Event, num_Category, d_out, events.float(), time.float(), temp_mask2)

    loss = (alpha*loss_nll2) + (beta*loss_rank)

    loss.backward()
    loss = optimizer.step()

    nll2_train.append((alpha*loss_nll2.item())+(beta*loss_rank.item()))
    return np.mean(nll2_train), d_out

#Define Random Hyperparameterss
def get_random_hyperparmeters():
    SET_LAYERS        = [1,2,3] #number of layers
    SET_NODES         = [50, 100, 150, 200] #number of nodes

    SET_ACTIVATION_FN = ['ELU', 'LeakyReLU', 'ReLU'] #non-linear activation functions

    SET_ALPHA         = [0.1, 0.5, 1.0, 3.0, 5.0] #alpha values -> log-likelihood loss
    SET_BETA          = [0.1, 0.5, 1.0, 3.0, 5.0] #beta values -> ranking loss
    SET_GAMMA         = [0.1, 0.5, 1.0, 3.0, 5.0] #gamma values -> calibration loss

    # params = { 'EPOCH' : 30000,
    #         'keep_prob' : 0.6,
    #         'lr' : 1e-4,
    #         'h_dim_shared' : SET_NODES[np.random.randint(len(SET_NODES))],
    #         'h_dim_CS' : SET_NODES[np.random.randint(len(SET_NODES))],
    #         'num_layers_shared':SET_LAYERS[np.random.randint(len(SET_LAYERS))],
    #         'num_layers_CS':SET_LAYERS[np.random.randint(len(SET_LAYERS))],
    #         'active_fn': SET_ACTIVATION_FN[np.random.randint(len(SET_ACTIVATION_FN))],
    #
    #         'alpha':1.0, #default (set alpha = 1.0 and change beta and gamma)
    #         # 'beta':SET_BETA[np.random.randint(len(SET_BETA))],
    #         'beta':0,
    #         'gamma':0,   #default (no calibration loss)
    #         # 'alpha':SET_ALPHA[np.random.randint(len(SET_ALPHA))],
    #         # 'beta':SET_BETA[np.random.randint(len(SET_BETA))],
    #         # 'gamma':SET_GAMMA[np.random.randint(len(SET_GAMMA))]
    # }

    params = { 'EPOCH' : 50000,
                'keep_prob' : 0.4,
                'lr' : 1e-4,
                'h_dim_shared' : SET_NODES[np.random.randint(len(SET_NODES))],
                'h_dim_CS' : SET_NODES[np.random.randint(len(SET_NODES))],
                'num_layers_shared': SET_LAYERS[np.random.randint(len(SET_LAYERS))],
                'num_layers_CS': SET_LAYERS[np.random.randint(len(SET_LAYERS))],
                'active_fn': SET_ACTIVATION_FN[np.random.randint(len(SET_ACTIVATION_FN))],
                'alpha':SET_ALPHA[np.random.randint(len(SET_ALPHA))], #default (set alpha = 1.0 and change beta and gamma)
                'beta':0,
                'gamma':0,   #default (no calibration loss)
                # 'alpha':SET_ALPHA[np.random.randint(len(SET_ALPHA))],
                # 'beta':SET_BETA[np.random.randint(len(SET_BETA))],
                # 'gamma':SET_GAMMA[np.random.randint(len(SET_GAMMA))]
        }
    return params
    
max_c_INDEX = []
for i in range(10):
    #DAG-GNN Parameters
    print("Iter "+str(i))
    z_dims = 1
    encoder_hidden = 64
    decoder_hidden = 64
    batch_size = 64
    encoder_dropout = 0
    decoder_dropout = 0
    factor = True
    lr_decay = 200
    gamma= 1
    c_A = 1
    tau_A = 0.0
    lambda_A = 0.
    graph_threshold = 0.3

    batch_size = 64
    #DeepHit Parameters
    input_dims={}
    input_dims['in_dim'] = data_variable_size
    input_dims['num_Event'] = num_Event
    input_dims['num_Category'] = num_Category

    params = get_random_hyperparmeters()
    EPOCH = params['EPOCH']
    lr = params['lr']

    network_settings={}
    network_settings['h_dim_shared'] = params['h_dim_shared']
    network_settings['h_dim_CS'] = params['h_dim_CS']
    network_settings['num_layers_shared'] = params['num_layers_shared']
    network_settings['num_layers_CS'] = params['num_layers_CS']
    network_settings['active_fn'] = params['active_fn']
    network_settings['keep_prob'] = params['keep_prob']
    network_settings['initial_W'] = 'xavier_normal'
    alpha = params['alpha']
    beta = params['beta']

    print(params)

    survival = DeepHit(input_dims, network_settings)

    #=======================================
    #set up training parameters
    #=======================================

    optimizer = optim.Adam(list(survival.parameters()), lr = lr)

    #===================================
    # main
    #===================================

    best_ELBO_loss = np.inf
    best_NLL1_loss = np.inf
    best_MSE_loss = np.inf
    best_NLL2_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL1_graph = []
    best_MSE_graph = []

    h_A_new = torch.tensor(1.)
    h_tol = 1e-8
    k_max_iter = 1
    h_A_old = np.inf

    c_scores = []
    epoch_index = []
    for step_k in range(k_max_iter):
        for epoch in range(EPOCH):
            # print("iter: "+str(step_k)+" Epoch: "+str(epoch))
            NLL2_loss, surv_out = train(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer, tau_A, graph_threshold, mask1_train, mask2_train, lr, data_variable_size, x_dims, num_Event, num_Category, alpha, beta, batch_size, X_train, Y_train, E_train)

            if NLL2_loss < best_NLL2_loss:
                best_NLL2_loss = NLL2_loss
                # print(NLL2_loss)

            if(epoch%1001==0):
                survival.train(mode = False)
                X_te = torch.Tensor(X_val).float()
                hr_pred2 = survival(X_te)
                hr_pred2 = hr_pred2.detach().numpy()

                EVAL_TIMES = [80, 160, 240]
                FINAL1 = np.zeros([num_Event, len(EVAL_TIMES), 1])
                result1 = np.zeros([num_Event, len(EVAL_TIMES)])
                for t, t_time in enumerate(EVAL_TIMES):
                    eval_horizon = int(t_time)

                    if eval_horizon >= num_Category:
                        print( 'ERROR: evaluation horizon is out of range')
                        result1[:, t] = -1
                    else:
                        # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
                        risk = np.sum(hr_pred2[:,:,:(eval_horizon+1)], axis=2) #risk score until EVAL_TIMES
                        for k in range(num_Event):
                            # result1[k, t] = c_index(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                            # result2[k, t] = brier_score(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                            result1[k, t] = weighted_c_index(Y_train, (E_train[:,0] == k+1).astype(int), risk[:,k], Y_val, (E_val[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

                tmp_valid = np.mean(result1)
                c_scores.append(tmp_valid)
                epoch_index.append(epoch)

                print("C-Index :", (result1, epoch))

    # print("Best Epoch: {:04d}".format(best_epoch))
    # m = max(c_scores)
    # print("Max C-index :" , m)
    # max_c_INDEX.append((params, m))
    # test()
    # #Metric - Concordance index

    X_te = torch.Tensor(X_test).float()
    hr_pred2 = survival(X_te)
    hr_pred2 = hr_pred2.detach().numpy()

    EVAL_TIMES = [80, 160, 240]
    FINAL1 = np.zeros([num_Event, len(EVAL_TIMES), 1])
    result1 = np.zeros([num_Event, len(EVAL_TIMES)])
    for t, t_time in enumerate(EVAL_TIMES):
        eval_horizon = int(t_time)

        if eval_horizon >= num_Category:
            print( 'ERROR: evaluation horizon is out of range')
            result1[:, t] = -1
        else:
            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(hr_pred2[:,:,:(eval_horizon+1)], axis=2) #risk score until EVAL_TIMES
            for k in range(num_Event):
                # result1[k, t] = c_index(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                # result2[k, t] = brier_score(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                result1[k, t] = weighted_c_index(Y_train, (E_train[:,0] == k+1).astype(int), risk[:,k], Y_test, (E_test[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

    FINAL1[:, :, 0] = result1

    ### SAVE RESULTS
    row_header = []
    for t in range(num_Event):
        row_header.append('Event_' + str(t+1))

    col_header1 = []
    for t in EVAL_TIMES:
        col_header1.append(str(t) + 'yr c_index')

    # c-index result
    df1 = pd.DataFrame(result1, index = row_header, columns=col_header1)

    print('--------------------------------------------------------')
    print('- C-INDEX: ')
    print(df1)
    print('--------------------------------------------------------')

    #End of Survival metric

    plt.plot(epoch_index, c_scores)
    plt.savefig('Iter_'+str(i+1)+'.png')
    plt.close()

    print("Best NLL2 loss: ",best_NLL2_loss)

print(max_c_INDEX)
