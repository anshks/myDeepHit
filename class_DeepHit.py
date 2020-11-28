import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import network

class DeepHit(nn.Module):
    def __init__(self, input_dims, network_settings):
        super().__init__()

        #INPUT DIMENSIONS
        self.in_dim = input_dims['in_dim']
        self.num_Event = input_dims['num_Event']
        self.num_Category = input_dims['num_Category']

        #NETWORK HYPER-PARAMETERS
        self.h_dim_shared = network_settings['h_dim_shared']
        self.h_dim_CS = network_settings['h_dim_CS']
        self.num_layers_shared = network_settings['num_layers_shared']
        self.num_layers_CS = network_settings['num_layers_CS']

        self.active_fn = network_settings['active_fn']
        self.initial_W = network_settings['initial_W']

        self.keep_prob =  network_settings['keep_prob']

        self.net = network.CreateLayers(self.initial_W)
        self.shared_layers = self.net.create_FCNet(self.in_dim, self.num_layers_shared, self.h_dim_shared, self.active_fn, self.h_dim_shared, self.active_fn, self.keep_prob)

        self.CS_layers = []
        for i in range(self.num_Event):
            self.CS_layers.append(self.net.create_FCNet(self.in_dim+self.h_dim_shared, self.num_layers_CS, self.h_dim_CS, self.active_fn, self.h_dim_CS, self.active_fn, self.keep_prob))

        self.dr = nn.Dropout(self.keep_prob)
        # self.final_layer = nn.Linear(self.num_Event*self.h_dim_CS, self.num_Event*self.num_Category)
        self.final_layer = self.net.create_FCNet(self.num_Event*self.h_dim_CS, 1, 0, None, self.num_Event*self.num_Category, None, self.keep_prob)

    def forward(self, x):
        # print(x)
        shared_out = self.shared_layers.forward(x)
        # print(shared_out)
        last_x = x
        h = torch.cat((last_x, shared_out),1)
        # print(h)
        out = torch.Tensor(self.num_Event,list(x.shape)[0],self.h_dim_CS)
        for i in range(self.num_Event):
            out[i] = self.CS_layers[i].forward(h)
        # print(out)
        out = torch.stack(tuple(out), axis=1)
        # print(out)
        out = out.reshape((-1, self.num_Event*self.h_dim_CS))
        # print(out)
        out = self.dr(out)
        # print(out)
        out = self.final_layer.forward(out)
        # print(out)
        out = F.softmax(out,-1)
        # print(out)
        out = out.reshape((-1, self.num_Event, self.num_Category))
        # print(out)
        return out
