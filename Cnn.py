import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.nodule):
	def __init__(sel):
		super(TheModelClass, self).__init__()

		self.cnn = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
										  nn.Linear(16 * 5 * 5, 120),
										  nn.Linear(120, 84),
										  #one neuron,
										 )

	def forward(self, x):	
		x = self.cnn(x)
		return x

model = CNN()
iterations = 10
criterion = nn.MSELoss()
lr = 0.001
weight_decay=0.0001
optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)
autoencoder_state = torch.load("/home/debopriyo/ng_sural/data/BOXREC/AVA/models/scae_model_epoch_"+\
											   str(epoch+1)+"_loss_"+str(round(runningLoss/(trainset_len/BatchSize),6))+".pth")
model.load_state_dict(autoencoder_state['state_dict']) 
model.load_state_dict(autoencoder_state['optimizer'])