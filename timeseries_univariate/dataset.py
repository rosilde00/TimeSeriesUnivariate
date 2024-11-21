import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class TemperatureDataset(Dataset):
    def __init__(self, temperatures, window):
        self.temperatures = temperatures
        self.window = window
        self.len = len(self.temperatures)-self.window-1
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.temperatures.iloc[index:index+self.window].values).to(torch.float32)
        y = torch.from_numpy(self.temperatures.iloc[index+self.window].values).to(torch.float32).squeeze()
        return x,y

def get_dataset(temperatures, batch_size, window):
    dataset = TemperatureDataset(temperatures, window)
    splitted_dataset = random_split(dataset, [0.7, 0.3])

    train = DataLoader(splitted_dataset[0], batch_size)
    val = DataLoader(splitted_dataset[1], batch_size)
    
    return train, val

def get_ordered_dataset(temperatures, window, batch_size):
    dataset = TemperatureDataset(temperatures, window)
    dataloader = DataLoader(dataset, batch_size)
    return dataloader