from utils import *
from modules import *
from network import *
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


#==================add regularisation===========================


ground_truth_G = createGraph()
df = pd.read_csv("data.csv")
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, label, time = [], [], [], [], [], [], [], [], [], [], [], []
c = 5498
for ind in df.index:

    if df['label'][ind] == 4:
        x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, time_, label_ = df['x1'][ind], df['x2'][ind], df['x3'][ind], df['x4'][ind], df['x5'][ind],  df['x6'][ind], df['x7'][ind], df['x8'][ind], df['x9'][ind], df['x10'][ind], df['time'][ind], 1
        x1.append(x_1); x2.append(x_2); x3.append(x_3); x4.append(x_4); x5.append(x_5)
        x6.append(x_6); x7.append(x_7); x8.append(x_8); x9.append(x_9); x10.append(x_10);
        time.append(time_)
        label.append(label_)

    elif c >= 0 and df['label'][ind] == 0:
        x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, time_, label_ = df['x1'][ind], df['x2'][ind], df['x3'][ind], df['x4'][ind], df['x5'][ind],  df['x6'][ind], df['x7'][ind], df['x8'][ind], df['x9'][ind], df['x10'][ind], df['time'][ind], 0
        c -= 1
        x1.append(x_1); x2.append(x_2); x3.append(x_3); x4.append(x_4); x5.append(x_5)
        x6.append(x_6); x7.append(x_7); x8.append(x_8); x9.append(x_9); x10.append(x_10)
        time.append(time_)
        label.append(label_)

data = {"x1":x1, "x2":x2, "x3":x3, "x4":x4, "x5":x5, "x6":x6, "x7":x7, "x8":x8, "x9":x9, "x10":x10, "time":time, "label":label}
dummy = pd.DataFrame(data, columns = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "time", "label"])

label = np.asarray(label).reshape((len(label),1))
time = np.asarray(time).reshape((len(time),1))
X = np.array(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]])

X = f_get_Normalization(X, 'standard')

X_train,X_val,Y_train,Y_val=train_test_split(X,time,test_size=0.25, random_state=0)
X_train,X_val,E_train,E_val=train_test_split(X,label,test_size=0.25, random_state=0)

train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(E_train))
trainloader = DataLoader(train, batch_size = 64, shuffle = False)

data_variable_size = 10
x_dims = 1

num_Event = 1
num_Category = 300

mask1 = create_mask1(time,label,num_Event, num_Category)
mask2 = create_mask2(time,label,num_Event, num_Category)

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr

#===================================
# training:
#===================================

def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer, tau_A, graph_threshold, mask1, mask2, lr, data_variable_size, x_dims, num_Event, num_Category, alpha, beta):

    nll1_train = []
    kl_train = []
    mse_train = []
    shd_trian = []
    nll2_train = []

    encoder.train()
    decoder.train()
    survival.train()
    scheduler.step()

    # optimizer, lr = update_optimizer(optimizer, lr, c_A)
    print("c_A",c_A)
    #64x10 1x64
    for i, data in enumerate(trainloader):
        inputs ,relations, time, events = data
        inputs, relations, time, events = Variable(inputs).double(), Variable(relations).double(), Variable(time), Variable(events)

        # reshape data
        # relations = relations.resize_((list(relations.size())[0],10,1))
        inputs = inputs.unsqueeze(2)
        # inputs = inputs.float()
        optimizer.zero_grad()
        # print(inputs.type())
        # print("hey")
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(inputs, rel_rec, rel_send)
        edges = logits

        flat = edges.view(-1,data_variable_size)
        flat = flat.float()
        d_out = survival(flat)

        dec_x, output, adj_A_tilt_decoder = decoder(inputs, edges, data_variable_size * x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')

        target = inputs
        preds = output
        variance = 0

        #reconstruction accuracy loss
        loss_nll1 = nll_gaussian(preds, target, variance)

        loss_kl = kl_gaussian_sem(logits)

        temp_mask1 = torch.Tensor(list(time.size())[0], num_Event, num_Category)
        temp_mask2 = torch.Tensor(list(time.size())[0], num_Category)

        for i in range(len(time)):
            temp_mask1[i] = mask1[int(time[i])]
            temp_mask2[i] = mask2[int(time[i])]

        loss_nll2 = deephit_nll(d_out, events.float(), time.float(), temp_mask1)

        loss_rank = rank_loss(num_Event, num_Category, d_out, events.float(), time.float(), temp_mask2)

        loss = loss_kl + loss_nll1 + (alpha*loss_nll2 + (beta*loss_rank))

        one_adj_A = origin_A
        sparse_loss = tau_A*torch.sum(torch.abs(one_adj_A))

        h_A = _h_A(origin_A, data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss

        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, tau_A*lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metric
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(graph))

        mse_train.append(F.mse_loss(preds, target).item())
        nll1_train.append(loss_nll1.item())
        nll2_train.append(loss_nll2.item()+(0.1*loss_rank.item()))
        kl_train.append(loss_kl.item())
        shd_trian.append(shd)

    return np.mean(np.mean(kl_train)  + np.mean(nll1_train)), np.mean(nll1_train), np.mean(mse_train), graph, origin_A, np.mean(nll2_train), d_out

#Define Random Hyperparameterss
def get_random_hyperparmeters():
    SET_LAYERS        = [1,2,3,5] #number of layers
    SET_NODES         = [50, 100, 200, 300] #number of nodes

    SET_ACTIVATION_FN = ['ELU','GELU','LeakyReLU'] #non-linear activation functions

    SET_ALPHA         = [0.1, 0.5, 1.0, 3.0, 5.0] #alpha values -> log-likelihood loss
    SET_BETA          = [0.1, 0.5, 1.0, 3.0, 5.0] #beta values -> ranking loss
    SET_GAMMA         = [0.1, 0.5, 1.0, 3.0, 5.0] #gamma values -> calibration loss

    # params = { 'EPOCH' : 5,
    #         'keep_prob' : 0.6,
    #         'lr' : 1e-4,
    #         'h_dim_shared' : SET_NODES[np.random.randint(len(SET_NODES))],
    #         'h_dim_CS' : SET_NODES[np.random.randint(len(SET_NODES))],
    #         'num_layers_shared':SET_LAYERS[np.random.randint(len(SET_LAYERS))],
    #         'num_layers_CS':SET_LAYERS[np.random.randint(len(SET_LAYERS))],
    #         'active_fn': SET_ACTIVATION_FN[np.random.randint(len(SET_ACTIVATION_FN))],
    #
    #         'alpha':1.0, #default (set alpha = 1.0 and change beta and gamma)
    #         'beta':SET_BETA[np.random.randint(len(SET_BETA))],
    #         'gamma':0,   #default (no calibration loss)
    #         # 'alpha':SET_ALPHA[np.random.randint(len(SET_ALPHA))],
    #         # 'beta':SET_BETA[np.random.randint(len(SET_BETA))],
    #         # 'gamma':SET_GAMMA[np.random.randint(len(SET_GAMMA))]
    # }
    params = { 'EPOCH' : 400,
                'keep_prob' : 0.6,
                'lr' : 1e-4,
                'h_dim_shared' : 100,
                'h_dim_CS' : 200,
                'num_layers_shared': 2,
                'num_layers_CS': 1,
                'active_fn': 'ELU',

                'alpha':1.0, #default (set alpha = 1.0 and change beta and gamma)
                'beta':0.1,
                'gamma':0,   #default (no calibration loss)
                # 'alpha':SET_ALPHA[np.random.randint(len(SET_ALPHA))],
                # 'beta':SET_BETA[np.random.randint(len(SET_BETA))],
                # 'gamma':SET_GAMMA[np.random.randint(len(SET_GAMMA))]
    }
    return params

for i in range(2):
    #DAG-GNN Parameters
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
    # Generate off-diagonal interaction graph
    off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    # add adjacency matrix A
    adj_A = np.zeros((data_variable_size, data_variable_size))

    encoder = MLPEncoder(data_variable_size * x_dims, x_dims, encoder_hidden,
                         z_dims, adj_A,
                         batch_size = batch_size,
                         do_prob = encoder_dropout, factor = factor).double()

    decoder = MLPDecoder(data_variable_size * x_dims,
                         z_dims, x_dims, encoder,
                         data_variable_size = data_variable_size,
                         batch_size = batch_size,
                         n_hid = decoder_hidden,
                         do_prob = decoder_dropout).double()

    survival = DeepHit(input_dims, network_settings)

    #=======================================
    #set up training parameters
    #=======================================

    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters())+list(survival.parameters()), lr = lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay,gamma=gamma)

    triu_indices = get_triu_offdiag_indices(data_variable_size)
    tril_indices = get_tril_offdiag_indices(data_variable_size)

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

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
        while c_A < 1e+20:
            for epoch in range(EPOCH):
                print("iter: "+str(step_k)+" Epoch: "+str(epoch))
                ELBO_loss, NLL1_loss, MSE_loss, graph, origin_A, NLL2_loss, surv_out = train(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer, tau_A, graph_threshold, mask1, mask2, lr, data_variable_size, x_dims, num_Event, num_Category, alpha, beta)
                if ELBO_loss < best_ELBO_loss:
                    best_ELBO_loss = ELBO_loss
                    best_epoch = epoch
                    best_ELBO_graph = graph
                    print("ELBO",ELBO_loss)

                if NLL1_loss < best_NLL1_loss:
                    best_NLL1_loss = NLL1_loss
                    best_epoch = epoch
                    best_NLL1_graph = graph
                    print("NLL1",NLL1_loss)

                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch
                    best_MSE_graph = graph
                    print("MSE",MSE_loss)

                if NLL2_loss < best_NLL2_loss:
                    best_NLL2_loss = NLL2_loss
                    print(NLL2_loss)

                # test()
                # #Metric - Concordance index
                X_tr = ((torch.Tensor(X_train)).unsqueeze(2)).double()
                enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(X_tr, rel_rec, rel_send)
                edges = logits

                flat = edges.view(-1,data_variable_size).float()
                hr_pred = survival(flat)
                hr_pred = (torch.exp(hr_pred)).detach().numpy()

                X_te = ((torch.Tensor(X_val)).unsqueeze(2)).double()
                enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(X_te, rel_rec, rel_send)
                edges = logits

                flat = edges.view(-1,data_variable_size).float()
                hr_pred2 = survival(flat)
                hr_pred2 = (torch.exp(hr_pred2)).detach().numpy()

                EVAL_TIMES = [12,24,36]
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

                FINAL1[:, :, 0] = result1
                c_scores.append(result1[0][0])
                epoch_index.append(epoch)
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
                # print('Concordance Index for training dataset:', ci_train)
                # print('Concordance Index for test dataset:', ci_test)
                #End of Survival metric

            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
            if ELBO_loss > 2 * best_ELBO_loss:
                break

            # update parameters
            A_new = origin_A.data.clone()
            h_A_new = _h_A(A_new, data_variable_size)
            if h_A_new.item() > 0.25 * h_A_old:
                c_A*=10
            else:
                break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break

    print("Best Epoch: {:04d}".format(best_epoch))

    # test()
    # #Metric - Concordance index
    X_tr = ((torch.Tensor(X_train)).unsqueeze(2)).double()
    enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(X_tr, rel_rec, rel_send)
    edges = logits

    flat = edges.view(-1,data_variable_size).float()
    hr_pred = survival(flat)
    hr_pred = (torch.exp(hr_pred)).detach().numpy()

    X_te = ((torch.Tensor(X_val)).unsqueeze(2)).double()
    enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(X_te, rel_rec, rel_send)
    edges = logits

    flat = edges.view(-1,data_variable_size).float()
    hr_pred2 = survival(flat)
    hr_pred2 = (torch.exp(hr_pred2)).detach().numpy()

    EVAL_TIMES = [12,24,36]
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
    # print('Concordance Index for training dataset:', ci_train)
    # print('Concordance Index for test dataset:', ci_test)
    #End of Survival metric

    plt.plot(epoch_index, c_scores)
    plt.savefig('Iter_'+str(i+1)+'.png')
    plt.close()

    print("Best NLL2 loss: ",best_NLL2_loss)

    print (best_ELBO_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(best_ELBO_graph))
    print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_NLL1_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(best_NLL1_graph))
    print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print (best_MSE_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(best_MSE_graph))
    print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph = origin_A.data.clone().numpy()
    graph[np.abs(graph) < 0.1] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(graph))
    print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.2] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(graph))
    print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.3] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(graph))
    print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
