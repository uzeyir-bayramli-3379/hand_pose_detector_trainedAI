import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary

device='cuda' if torch.cuda.is_available() else 'cpu'
class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values
        self.labels = self.data[:, -1]  # Assume the last column is the label
        self.features = self.data[:, :-1]
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
    
dataframe = pd.read_csv('vulcan_gesture_data.csv')
dataset = PandasDataset(dataframe)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
for features, labels in train_dataloader:
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")


class TestModel(nn.Module):
    def __init__(self):

        super(TestModel, self).__init__()

        self.input_layer = nn.Linear(63,64)
        self.relu=nn.ReLU()
        self.linear = nn.Linear(64, 3)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.linear(x)
        return x
model=TestModel()
features, labels = next(iter(train_dataloader))
output=model(features)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
for epoch in range(30):
    for features, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(str(epoch) + " " + str(loss.item()))
model.eval()
with torch.no_grad():
    all_preds = []
    all_labels = []
    for features, labels in test_dataloader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")
torch.save(model.state_dict(), 'vulcan_gesture_model.pth')