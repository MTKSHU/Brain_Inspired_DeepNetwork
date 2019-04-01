import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(128 * 160 * 160, 1024),
                                 nn.Linear(1024, 1024),
                                 nn.Linear(1024, 1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        x = self.cnn(x)
        return x


model = CNN()
iterations = 10
criterion = nn.BCELoss()
lr = 0.001
weight_decay = 0.0001
epoch_num = 100
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
for epoch in range(epoch_num):
    # train code here
    pass

# autoencoder_state = torch.load(
#         "/home/debopriyo/ng_sural/data/BOXREC/AVA/models/scae_model_epoch_" + str(epoch + 1) + "_loss_" + str(
#             round(runningLoss / (trainset_len / BatchSize), 6)) + ".pth")

# model.load_state_dict(autoencoder_state['state_dict'])
# model.load_state_dict(autoencoder_state['optimizer'])
