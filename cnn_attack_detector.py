import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.models as models

class adv_obs_dataset(Dataset):
    def __init__(self, csv_benign, csv_adversarial, transform=None, dtype=torch.float32):
        """
        Args:
            csv_benign (str): Path to CSV file for class 0.
            csv_adversarial (str): Path to CSV file for class 1.
            transform (callable, optional): Optional transform to be applied on a sample.
            dtype (torch.dtype): Desired dtype for the features.
        """
        # Load both datasets
        self.benign = pd.read_csv(csv_benign).values.astype(np.float32)
        self.adv = pd.read_csv(csv_adversarial).values.astype(np.float32)

        # Create labels
        labels_benign = np.zeros(len(self.benign ), dtype=np.int64)
        labels_adv = np.ones(len( self.adv), dtype=np.int64) 

        # Stack data 
        print("Number of Benign Samples: ", len(self.benign))
        print("Number of Adversarial Samples: ", len(self.adv))
        self.data = np.vstack((self.benign, self.adv)).astype(np.float32)
        print("Total Number of Samples in Dataset: ", len(self.data))

        # Stack labels
        self.labels = np.concatenate((labels_benign, labels_adv))
        self.labels = self.labels.reshape(-1,1)  # Ensure labels are 1D

        print(f"x shape: {self.data.shape}, y shape: {self.labels.shape}")

        # Shuffle
        indices = np.random.permutation(len(self.labels))
        self.data = self.data[indices]
        self.labels = self.labels[indices]

        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(y, dtype=self.dtype)

        # Pad features to 36 elements, then reshape to 6x6
        if x.numel() < 36:
            x = torch.nn.functional.pad(x, (0, 36 - x.numel()))
        x = x.view(6, 6) 

        return x, y

class cnn_detector(nn.Module):
    def __init__(self):
        super(cnn_detector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(32, 32) 
        self.output = nn.Linear(32,1)       
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)  # Pooling layer
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.pool2(x)  # Pooling layer
        x = self.flat(x)  # Flatten the tensor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.sigmoid(x)

        return x
    

def train_adv_cnn_classifier(epochs=10, batch_size=64, learning_rate=0.001):

    transform = transforms.ToTensor()
     
    full_dataset = adv_obs_dataset("signal_processing/multi_fgsm015_adversarial_obs_data.csv", "signal_processing/multi_fgsm015_benign_obs_data.csv")

    # Split into train and val
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # Model, loss, optimizer
    model = cnn_detector()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader):
        
            optimizer.zero_grad() # zero the gradients

            outputs = model(batch_x.unsqueeze(1))  # Add channel dimension
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Validation loop
    model.eval()
    correct = 0
    total = 0   
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader):
            outputs = model(batch_x.unsqueeze(1))#.squeeze(1)
            predictions = (outputs > 0.5).long()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    val_acc = correct / total
    print(f"Validation Accuracy: {100 * val_acc:.2f}%")

    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
}, "best_model.pt")

def debugging():
    batch_size=64

    full_dataset = adv_obs_dataset("signal_processing/multi_fgsm015_adversarial_obs_data.csv", "signal_processing/multi_fgsm015_benign_obs_data.csv")

    # Split into train and val
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    example_x, example_y = next(iter(train_loader))
    print("x:", example_x[0].view(example_x[0].size(0), -1).shape)

if __name__ == "__main__":
    train_adv_cnn_classifier()
    # debugging()