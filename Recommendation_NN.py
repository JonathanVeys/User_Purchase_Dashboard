import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class Churn_Dataset(Dataset):
    def __init__(self, data):
        self.features = data.drop(columns=['Churn']).values
        self.labels = data['Churn'].values

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        features = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return features, label
    
class predictiveNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super(predictiveNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim[2], output_dim)
        )
    
    def forward(self, x):
        return(self.model(x))
    
data = pd.read_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/Processesed-Churn-Data.csv')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = predictiveNN(input_dim=int(data.shape[1]-1), output_dim=2, hidden_dim=[16, 32, 16], dropout=0.3)
model.to(device)

train_len = int(len(data) * 0.8)
valid_len = int(len(data) - train_len)

Churn_dataset = Churn_Dataset(data)
train_dataset, valid_dataset = random_split(Churn_dataset, [train_len, valid_len])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

class_counts = data['Churn'].value_counts().to_dict()
class_weights = [1 / class_counts[0], 1 / class_counts[1]]
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.SGD(model.parameters(), lr=0.003)
max_epochs = 50
train_loss = []
valid_loss = []
valid_accuracy = []

for epoch in range(max_epochs):
    model.train()
    epoch_train_loss = 0
    for features, labels in tqdm(train_dataloader, desc='Training'):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model(features)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
    train_loss.append(epoch_train_loss/len(train_dataloader))

    with torch.no_grad():
        model.eval()
        epoch_valid_loss = 0
        total = 0
        correct = 0
        for features, labels in tqdm(valid_dataloader, desc='Validation'):
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(output, labels)
            epoch_valid_loss += loss.item()
        valid_accuracy.append(correct/total)
        valid_loss.append(epoch_valid_loss/len(valid_dataloader))
    print(f'Epoch: {epoch+1}/{max_epochs} | Training Loss: {train_loss[-1]} | Validation Loss: {valid_loss[-1]} | Validation Accuracy: {valid_accuracy[-1]}')
torch.save(model, '/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/churn_prediction_model.pth')
plt.plot([n for n in range(len(train_loss))], train_loss, color='red', linestyle='-', label='Training Loss')
plt.plot([n for n in range(len(valid_loss))], valid_loss, color='blue', linestyle='-', label='Validation Loss')
plt.plot([n for n in range(len(valid_accuracy))], valid_accuracy, color='blue', linestyle='-.', label='Validation Accuracy')
plt.grid()
plt.legend()
plt.show()



