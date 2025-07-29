

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score
import os

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
        data_benign = pd.read_csv(csv_benign, index_col=0).values
        data_adv = pd.read_csv(csv_adversarial, index_col=0).values

        # Create labels
        labels_benign = np.zeros(len(data_benign), dtype=np.int64)
        labels_adv = np.ones(len(data_adv), dtype=np.int64)

        # Stack data and labels
        print("Number of Benign Samples: ", len(data_benign))
        print("Number of Adversarial Samples: ", len(data_adv))
        self.data = np.vstack((data_benign, data_adv)).astype(np.float32)
        print("Total Number of Samples in Dataset: ", len(self.data))
        self.labels = np.concatenate((labels_benign, labels_adv))


        # Shuffle
        indices = np.random.permutation(len(self.labels))
        self.data = self.data[indices]
        self.labels = self.labels[indices]

        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        sample = torch.tensor(sample, dtype=self.dtype)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label


class adv_detector_nn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        super(adv_detector_nn, self).__init__()
        # Layer 1        
        self.input_bn = nn.BatchNorm1d(input_dim)

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        # Layer 2
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        # Layer 3
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        # Layer 4
        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)

        # Output layer
        self.output = nn.Linear(128, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        #x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        #x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)
        #x = self.dropout(x)

        x = self.fc4(x)
        x = F.relu(x)
        x = self.bn4(x)


        x = self.output(x)
        x = torch.sigmoid(x)  # Or remove this if using BCEWithLogitsLoss
        return x



class ResNet1D(nn.Module):
    def __init__(self, base_model, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim

        # Replace conv1 with a 1D-compatible layer
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Keep rest of the resnet body
        self.resnet = base_model
        self.resnet.conv1 = self.conv1
        self.resnet.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Adjust the avgpool to 1D
        self.resnet.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Change final fully connected layer
        self.resnet.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        # Assume x shape is [B, input_dim]
        x = x.unsqueeze(1)  # [B, 1, input_dim]
        return self.resnet(x)
        
def train_adv_classifier(epochs=60, batch_size=64, lr=1e-3):
    path = os.path.join(os.getcwd(), 'mujoco_ant_obs_dataset')
    full_dataset = adv_obs_dataset(f"{path}/adversarial_obs_data.csv", f"{path}/benign_obs_data.csv")
    input_dim = full_dataset.data.shape[1]

    # Split into train and val
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # Model, loss, optimizer
    model = adv_detector_nn(input_dim)

    ############################################################
    ##### Use this code for pre-trained resnet classifier ######
    # Load the pretrained ResNet-34 model
    # base_model = models.resnet34(pretrained=True)
    # model = ResNet1D(base_model, input_dim=input_dim, output_dim=1)
    ############################################################

    criterion = nn.BCELoss()  # Since output is sigmoid
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze(1)  # shape: (batch_size,)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
        
    # Validation loop
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader):
            outputs = model(batch_x).squeeze(1)
            predictions = (outputs > 0.5).long()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    val_acc = correct / total
    print(f"Validation Accuracy: {100 * val_acc:.2f}%")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall: {recall_score(all_labels, all_preds):.4f}")
    print(f"F1 Score: {f1_score(all_labels, all_preds):.4f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, "best_model.pt")

def load_test_model():
    # Load data 
    path = os.path.join(os.getcwd(), 'mujoco_ant_obs_dataset')
    full_dataset = adv_obs_dataset(f"{path}/adversarial_obs_data.csv", f"{path}/benign_obs_data.csv")
    input_dim = full_dataset.data.shape[1]

    # Split into train and val, set up test loader
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=12)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=64)

    # Initialize Model 
    checkpoint = torch.load("best_model.pt")
    model = adv_detector_nn(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load states 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Validation loop
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader):
            outputs = model(batch_x).squeeze(1)
            predictions = (outputs > 0.5).long()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    val_acc = correct / total
    print(f"Validation Accuracy: {100 * val_acc:.2f}%")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall: {recall_score(all_labels, all_preds):.4f}")
    print(f"F1 Score: {f1_score(all_labels, all_preds):.4f}")


if __name__ == '__main__':
    # train_adv_classifier()
    load_test_model()