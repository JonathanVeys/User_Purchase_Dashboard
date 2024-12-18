import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class purchaseDataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = data

        self.features = data.iloc[:, :-1].values
        self.labels = data.iloc[:, -1].values
        pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return features, labels
    
data = pd.read_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/purchase_dataset.csv')
data = data.iloc[:, 3:]
purchase_dataset = purchaseDataset(data)

train_len = int(0.7 * len(purchase_dataset))
valid_len = int(0.2 * len(purchase_dataset))
test_len = int(0.1 * len(purchase_dataset))

train_dataset, valid_dataset, test_dataset = random_split(purchase_dataset, [train_len, valid_len, test_len])
features, label = train_dataset[0]

print("Train Dataset length: ", len(train_dataset))
print("Valid Dataset length: ", len(valid_dataset))
print("Test Dataset length: ", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class RecommendationNN(nn.Module):
    def __init__(self, input_dim=6, output_dim=2, hidden_dims=[32, 64, 32]):
        super(RecommendationNN, self).__init__()
        
        # Sequential model with Dropout and BatchNorm
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),  # Normalize activations
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dims[2], output_dim)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x  # Raw logits; apply activation in the loss function
        
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

y_train = [label for _,label in purchase_dataset]
unique, counts = np.unique(y_train, return_counts=True)
class_weights = torch.tensor([len(y_train) / count for count in counts], dtype=torch.float32)
class_weights = class_weights.to(device)

model = RecommendationNN()
model.to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

max_epochs = 50


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=30, device='cpu'):
    model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    train_track_loss = []
    valid_track_loss = []
    valid_track_accuracy = []

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        total_train_samples = 0

        # Training Loop
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            train_loss += loss.item() * inputs.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs, 1)  # Get class predictions
            train_correct += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)
        # Average training loss and accuracy
        train_loss /= total_train_samples
        train_acc = train_correct / total_train_samples

        train_track_loss.append(train_loss)

        # Validation Loop
        model.eval()
        val_loss, val_correct = 0.0, 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)
        
        val_loss /= total_val_samples
        val_acc = val_correct / total_val_samples

        valid_track_loss.append(val_loss)
        valid_track_accuracy.append(val_acc)

        # Print epoch metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    print("Training complete.")
    if best_model_state:
        print("Saving the best model...")
        torch.save(best_model_state, "best_recommendation_model.pth")

    return(train_track_loss, valid_track_loss, valid_track_accuracy)

train_loss, valid_loss, valid_accuracy = train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=15, device=device)

plt.plot([n for n in range(len(train_loss))], train_loss)
plt.plot([n for n in range(len(valid_loss))], valid_loss)
plt.plot([n for n in range(len(valid_accuracy))], valid_loss)
plt.show()
