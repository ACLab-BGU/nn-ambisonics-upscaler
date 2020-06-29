import os
from torch.utils.data import DataLoader
from src.data.base_dataset import BasicDataset


path = os.path.join(os.path.dirname(__file__), 'raw', 'free-field')

dataset = BasicDataset(path, preload=True)

dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)

for idx, (data, label) in enumerate(dataloader):
    print(idx)
    print(data.shape)
    print(label.shape)
