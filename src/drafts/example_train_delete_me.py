import numpy as np
import torch
from torch import nn
from torch.optim import optimizer
from torch.utils.data import DataLoader

from src.data.base_dataset import BasicDataset
from src.models.fc_model import BaseModel

data_folder = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/data/raw/free-field'

dataset = BasicDataset(data_folder, preload=True)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
inputs, targets = next(iter(dataloader))
input_size = np.prod(inputs.shape[2:])
output_size = np.prod(targets.shape[2:])
model = BaseModel(input_size, output_size, 3, np.linspace(input_size, output_size, 5)[1:-1].astype(int))
criteria = nn.MSELoss()
lr = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 5
print_every = 1
dtype = torch.float32
device = torch.device('cpu')

model.type(dtype)
model.train()

for epoch in range(epochs):  # loop over the dataset multiple times
    print('epoch %d' % (epoch+1))

    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs, targets = inputs.to(device=device, dtype=dtype), targets.to(device=device, dtype=dtype)
        inputs = inputs.view([-1, *inputs.shape[-3:]])
        targets = targets.view([-1, *targets.shape[-3:]])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(inputs)
        loss = criteria(output, targets.view([targets.shape[0], -1]))
        loss.backward()
        optimizer.step()

        # print statistics
        if i % print_every == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))

print('Finished Training')
