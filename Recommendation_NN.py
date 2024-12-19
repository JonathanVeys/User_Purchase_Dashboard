import torch 
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class simpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = self.data.iloc[:,-1].values
        self.features = self.data.iloc[:, :-1].values

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return features, labels

class predictiveNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        pass


data = pd.read_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset = simpleDataset(data)

train_len = int(len(data) * 0.7)
valid_len = int(len(data) * 0.2)
test_len = int(len(data) * 0.1) + 1

train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_len, valid_len, test_len])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


