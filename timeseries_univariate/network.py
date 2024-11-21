import torch
from torch import nn
from torch.optim.adamw import AdamW
import numpy as np

class TemperatureForecasting (nn.Module):
    
    def __init__(self):
        super().__init__()  
        self.lstm = nn.LSTM(1, 150, batch_first=True)
        self.linear = nn.Linear(150, 1)
        
    def forward (self, temp):
        lstm_out, _ = self.lstm(temp)
        res = self.linear(lstm_out[:,-1,:])
        return res
    
def create_model():
    return TemperatureForecasting()
  
def train_loop(model, dataloader, batch_size):
    size = len(dataloader.dataset) 
    model.train() 
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    for batch, (x, y) in enumerate(dataloader): 
        pred = model(x).squeeze()
        loss = loss_fn(pred, y.float())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 30 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"MSE: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
def validation_loop(model, dataloader):
    model.eval()
    num_batches = len(dataloader)
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    avg_mse, avg_mae = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            avg_mse += mse(pred, y.float()).item()
            avg_mae += mae(pred, y.float()).item()

    avg_mse /= num_batches
    avg_mae /= num_batches
    
    print(f"MSE: {avg_mse:>5f} - MAE: {avg_mae:>5f}")
    return avg_mse, avg_mae

def test_loop(model, dataloader):
    model.eval()
    predictions = np.array([])
    
    with torch.no_grad():
        for x, _ in dataloader:
            pred = model(x)
            predictions = np.append(predictions, pred)
            
    return predictions

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.count = 0
        self.early_stop = False

    def __call__(self, validation_loss):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.count = 0
        return self.early_stop