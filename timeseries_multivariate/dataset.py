import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np

class PowerDataset(Dataset):
    def __init__(self, weather, window):
        self.weather = weather
        self.window = window
        self.len = len(self.weather)-self.window
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.weather.iloc[index:index+self.window].values).to(torch.float32)
        y = torch.from_numpy(np.array(self.weather.iloc[index+self.window]['Global_active_power'])).to(torch.float32)
        return x,y

def get_dataset(weather, batch_size, window):
    dataset = PowerDataset(weather, window)
    splitted_dataset = random_split(dataset, [0.7, 0.3])

    train = DataLoader(splitted_dataset[0], batch_size)
    val = DataLoader(splitted_dataset[1], batch_size)
    
    return train, val

def get_ordered_dataset(weather, window, batch_size):
    dataset = PowerDataset(weather, window)
    dataloader = DataLoader(dataset, batch_size)
    return dataloader

def dataset_preproc(data):
    data = data.drop(columns=['Date', 'Time'])
    data = data.astype(np.float32)
    
    data = data.fillna(data.mean())
    
    for c in data.columns:
        if c != 'Global_active_power':
            val = data[c].values
            normalized_data = (val - val.mean())/(val.std() + 1e-8)
            data[c] = normalized_data

    correlazioni = data.corr()['Global_active_power']
    for i in range(0,len(correlazioni.index)):
        if -0.1<correlazioni.iloc[i]<0.1:
            data = data.drop(columns=[correlazioni.index[i]])
            
    return data