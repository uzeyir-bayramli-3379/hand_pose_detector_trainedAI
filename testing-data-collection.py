import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
for features, labels in dataloader:
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

scaler = StandardScaler()
dataframe.iloc[:, :-1] = scaler.fit_transform(dataframe.iloc[:, :-1])
dataframe.fillna(dataframe.mean(), inplace=True)