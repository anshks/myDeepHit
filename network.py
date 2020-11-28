import torch
import torch.nn as nn
from torch.nn import init

class graphConv(nn.Module):

	def __init__(self, input_features, output_features, in_dims, num_k, graph_shift):
		super().__init__()
		self.input_features = input_features
		self.output_features = output_features
		self.num_k = num_k
		self.graph_shift = graph_shift
		# self.weight = Parameter(torch.Tensor(num_k, input_features, output_features))
		self.weight = Parameter(torch.Tensor(num_k, in_dims, in_dims)) #Think about this if you wish to change the input and output features
		init.xavier_uniform(self.weight)

	def forward(self, nodes):
		"""
		nodes -- N*D
		graph_shift -- N*N
		"""
		out = torch.Tensor(list(nodes.size())[0], list(nodes.size())[1], self.output_features)
		for ind in range(list(nodes.size())[0]):
			S = torch.eye(list(nodes.size())[1])
			nodes_embed = torch.zeros(list(nodes.size())[1], self.output_features)
			for k in range(self.num_k):
				# nodes_embed += torch.mm(torch.mm(S, nodes[ind]), self.weight[k])
				nodes_embed += torch.mm(self.weight[k], torch.mm(S, nodes[ind]))
				S = torch.mm(S, self.graph_shift)
			out[ind] = nodes_embed
		return out

class CreateLayers(nn.Module):
	def __init__(self,w_init):
		super().__init__()

		self.w_init = w_init

		self.initializations = {
						'uniform': init.uniform_,
						'normal': init.normal_,
						'dirac': init.dirac_,
						'xavier_uniform': init.xavier_uniform_,
						'xavier_normal': init.xavier_normal_,
						'kaiming_uniform': init.kaiming_uniform_,
						'kaiming_normal': init.kaiming_normal_,
						'orthogonal': init.orthogonal_,
						'ones' : init.ones_
		}
		self.activations = nn.ModuleDict([
				['ELU', nn.ELU()],
				['ReLU', nn.ReLU()],
				['Tanh', nn.Tanh()],
				['LogSigmoid', nn.LogSigmoid()],
				['LeakyReLU', nn.LeakyReLU()],
				['SELU', nn.SELU()],
				['CELU', nn.CELU()],
				['GELU', nn.GELU()],
				['Sigmoid', nn.Sigmoid()],
				['Softmax', nn.Softmax()],
				['LogSoftmax', nn.LogSoftmax()]
		])

	def init_weights(self,m):

		if type(m) == nn.Linear:
			self.initializations[self.w_init](m.weight)

	def create_FCNet(self, in_dim, num_layers, h_dim, h_fn, o_dim, o_fn, keep_prob=1.0):
		'''
			GOAL             : Create FC network with different specifications
			in_dims          : number of input units
			num_layers       : number of layers in FCNet
			h_dim  (int)     : number of hidden units
			h_fn             : activation function for hidden layers (default: tf.nn.relu)
			o_dim  (int)     : number of output units
			o_fn             : activation function for output layers (defalut: None)
			w_init           : initialization for weight matrix (defalut: Xavier)
			keep_prob        : keep probabilty [0, 1]  (if None, dropout is not employed)
		'''

		# default active functions (hidden: relu, out: None)
		if h_fn is None:
			h_fn = 'ReLU'
		if o_fn is None:
			o_fn = None

		layers = []
		for layer in range(num_layers):
			if num_layers == 1:
				layers.append(nn.Linear(in_dim,o_dim))  #Discusss
				if o_fn != None:
					layers.append(self.activations[o_fn])
			else:
				if layer == 0:
					layers.append(nn.Linear(in_dim,h_dim))
					layers.append(self.activations[h_fn])
					if not keep_prob is None:
						layers.append(nn.Dropout(keep_prob))
				elif layer > 0 and layer != (num_layers-1): # layer > 0:
					layers.append(nn.Linear(h_dim,h_dim)) #probably wrong
					layers.append(self.activations[h_fn])
					if not keep_prob is None:
						layers.append(nn.Dropout(keep_prob))
				else: # layer == num_layers-1 (the last layer)
					layers.append(nn.Linear(h_dim,o_dim))
					if o_fn != None:
						layers.append(self.activations[o_fn])

		out = nn.Sequential(*layers)

		if self.w_init != None:
			out.apply(self.init_weights)
		return out

	def create_GCNet(self, num_gcn_layers, graph_shift, num_k, gcn_input_features, gcn_hidden_features, gcn_output_features):
		gcn_layers = []
		for layer in range(num_gcn_layers):
			if num_gcn_layers == 1:
				gcn_layers.append(graphConv(gcn_input_features, gcn_output_features, in_dims, num_k, graph_shift))
				gcn_layers.append(self.activations[h_fn]) #Can change to o_fn
			else:
				if layer == 0:
					gcn_layers.append(graphConv(gcn_input_features, gcn_hidden_features, in_dims, num_k, graph_shift))
					gcn_layers.append(self.activations[h_fn])
				elif layer > 0 and layer != (num_gcn_layers-1):
					gcn_layers.append(graphConv(gcn_hidden_features, gcn_hidden_features, in_dims, num_k, graph_shift))
					gcn_layers.append(self.activations[h_fn])
				else:
					gcn_layers.append(graphConv(gcn_hidden_features, gcn_output_features, in_dims, num_K, graph_shift))
					gcn_layers.append(self.activations[h_fn]) #Can change to o_fn

		out = nn.Sequential(*gcn_layers)
		return out
