
import torch
from PIL import Image
import matplotlib.pyplot as pyplot
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable as V
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
from AVA_Dataset_All import AVA_Dataset_All
#from google.colab import files
#%matplotlib inline
#%%
path = "/home/ksnehalreddy/Desktop/Fashion Reco/Autoencoder_fashion/"
#normalise the image ****************************************************************
#transform.normalise by finding mean and std*****************************************
# with open('mean_std.txt','r') as f:
  # text = f.readlines()
_mean = [0.485,0.456,0.406]
_std  = [0.229,0.224,0.225]

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(_mean,_std)])
BatchSize = 128

trainset_autoencoder = AVA_Dataset_All(root_dir=path+"resized_images_320_320_padded_white",transform=transform)
trainset_len_autoencoder = trainset.__len__()
print(trainset_autoencoder[0][0].shape)
trainloader_autoencoder = torch.utils.data.DataLoader(trainset,batch_size=BatchSize,shuffle=True,num_workers=10)

#testset = tv.datasets.CIFAR10(root='./CIFAR10',train=False,download=True,transform=transform)
#testloader = torch.utils.data.DataLoader(testset,batch_size=BatchSize,shuffle=False,num_workers=4)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%%
use_gpu = torch.cuda.is_available()

if use_gpu :
    print('GPU is available...')
#%%

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp =   
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv_encoder = nn.Sequential(
                          nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
                          nn.LeakyReLU(0.1),
                          # nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                          # nn.LeakyReLU(0.1),
                          # nn.MaxPool2d(kernel_size = 3, stride=2, padding=1, dilation=1)
                          nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                          nn.LeakyReLU(0.1),
                          # nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                          # nn.LeakyReLU(0.1)
                        )


        self.conv_decoder = nn.Sequential(
      #                    nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
       #                   nn.LeakyReLU(0.1),
        #                  nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                          # nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                          # nn.LeakyReLU(0.1),
                          #nn.Upsample(scale_factor = 2, mode='bilinear'),
                          nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                          nn.LeakyReLU(0.1),
                          #nn.Upsample(scale_factor = 2, mode = 'bilinear'),
			  Interpolate(scale_factor = 2, mode = 'bilinear'),
        #nn.MaxUnpool2d(kernel_size = 3, stride=2)
                          nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size=3, stride = 1, padding = 1),
                          nn.ReLU()
                        )

    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.conv_decoder(x)
        return x

net = autoencoder()

if use_gpu:
    net = net.cuda()
    # net = net.to(torch.device('cuda'))

print(net)

# init_weights = copy.deepcopy(net.conv_encoder[0].weight.data)
#%%
iterations = 10
criterion = nn.MSELoss()
lr = 1e-3
optimizer = optim.Adam(net.parameters(), lr = lr)

#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

trainLoss = []
# net.train()
for epoch in range(iterations):
    epochStart = time.time()
    runningLoss = 0.0
    print("Epoch:",epoch+1,"started...")
    for i, data in enumerate(trainloader, 0):
        # if i % 10 == 0:
        #     print("batch :",i)
    #get the inputs
        inputs = data
        #wrap them in variable
        if use_gpu:
          inputs = V(inputs).cuda()
        else:
          inputs = V(inputs)

        optimizer.zero_grad() # zeroes the gradient buffers of all parameters

        outputs = net(inputs) # forward

        loss = criterion(outputs, inputs) # calculate loss

        loss.backward() # backpropagate the loss

        optimizer.step()

        runningLoss += loss.data.item()
        # if i % 100 == 99:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, runningLoss / 100 * (i + 1) ))


    trainLoss.append(runningLoss/(trainset_len/BatchSize))
    # if i % 100 == 99:    # print every 2000 mini-batches
    #     print('[%d, %5d] loss: %.6f' %
    #           (epoch + 1, i + 1, trainLoss[i]))
    if epoch % 5 == 0:
        lr = lr * 0.1
        optimizer = optim.Adam(net.parameters(), lr = lr)

    epochEnd = time.time() - epochStart

    print("Iteration: {:.0f} / {:.0f} ; Training Loss: {:.6f} ; Time Consumed: {:.0f}m {:.0f}s"
                        .format(epoch + 1, iterations, trainLoss[i], epochEnd//60, epochEnd%60))

    model_path = "/home/debopriyo/ng_sural/data/BOXREC/AVA/models/scae_model_epoch_"+\
                    str(epoch+1)+"_loss_"+str(round(runningLoss/(trainset_len/BatchSize),6))+".pth"
    torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': runningLoss/(trainset_len/BatchSize)}, model_path)

print("Finished Training...")

class CNN(nn.Module):
  def __init__(sel):
    super(TheModelClass, self).__init__()

    self.conv1_conv2 = net.conv_encoder

    self.cnn = nn.Sequential(
                  nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                  nn.LeakyReLU(0.1),
                  nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
                  nn.LeakyReLU(0.1),
                  )
    self.fcn = nn.Sequential(
                  nn.Linear(128 * 80 * 80, 1024),
                  nn.Dropout(0.5),
                  nn.Linear(1024, 1024),
                  nn.Dropout(0.5),
                  nn.Linear(1024,1)
                  nn.Sigmoid()
                )

  def forward(self, x): 
    x = self.conv1_conv2(x)
    x = self.cnn(x)
    x = self.view(-1,128 * 80 * 80)
    x = self.fcn(x)
    
    return x

final_net = CNN()

iterations = 10
criterion = nn.BCELoss()
lr = 0.001
weight_decay=0.0001
optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)

for epoch in range(iterations):
    epochStart = time.time()
    runningLoss = 0.0
    print("Epoch:",epoch+1,"started...")
    for i, data in enumerate(trainloader, 0):
        # if i % 10 == 0:
        #     print("batch :",i)
    #get the inputs
        inputs = data
        #wrap them in variable
        if use_gpu:
          inputs = V(inputs).cuda()
        else:
          inputs = V(inputs)

        optimizer.zero_grad() # zeroes the gradient buffers of all parameters

        outputs = net(inputs) # forward

        loss = criterion(outputs, inputs) # calculate loss

        loss.backward() # backpropagate the loss

        optimizer.step()

        runningLoss += loss.data.item()
        # if i % 100 == 99:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, runningLoss / 100 * (i + 1) ))


    trainLoss.append(runningLoss/(trainset_len/BatchSize))
    # if i % 100 == 99:    # print every 2000 mini-batches
    #     print('[%d, %5d] loss: %.6f' %
    #           (epoch + 1, i + 1, trainLoss[i]))
    if epoch % 5 == 0:
        lr = lr * 0.1
        optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)

    epochEnd = time.time() - epochStart

    print("Iteration: {:.0f} / {:.0f} ; Training Loss: {:.6f} ; Time Consumed: {:.0f}m {:.0f}s"
                        .format(epoch + 1, iterations, trainLoss[i], epochEnd//60, epochEnd%60))
