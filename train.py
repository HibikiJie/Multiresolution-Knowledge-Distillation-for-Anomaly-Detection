from models.network import SourceNetwork, ClonerNetwork
from utils.loss_functions import LossFunction
from utils.dataset import ADDataset
from torch.utils.data import DataLoader
# from torch import nn
import torch

'''set parameter'''
learning_rate = 0.001
num_epoch = 100000
weight_root = 'weight'
batch_size = 32
num_workers = 4
device = 'cuda:0'
lamda = 0.5

'''create network、 optimizer、 and loss function'''
device = torch.device(device if torch.cuda.is_available() else 'cpu')
source_network = SourceNetwork().to(device)
cloner_network = ClonerNetwork().to(device)
loss_function = LossFunction(lamda=lamda)
optimizer = torch.optim.Adam(cloner_network.parameters(), lr=learning_rate)

'''create data loader'''
data_loader = DataLoader(ADDataset(root='data/leather'), batch_size, shuffle=True, num_workers=num_workers)

'''train'''
for epoch in range(num_epoch):
    loss_sum = 0
    for i, image in enumerate(data_loader):
        image = image.to(device)

        predict = cloner_network(image)
        real = source_network(image)
        loss = loss_function(predict, real)

        '''study'''
        optimizer.zero_grad()  # Clear the previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Adjust weights

        loss_sum += loss.item()
    print(f'epoch: {epoch};  loss: {str(loss_sum / len(data_loader))[0:10]}')  # print logs

    '''save weights'''
    torch.save(cloner_network.state_dict(), f'{weight_root}/cloner_network.pt')
