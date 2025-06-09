

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




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
        data_benign = pd.read_csv(csv_benign).values
        data_adv = pd.read_csv(csv_adversarial).values

        # Create labels
        labels_benign = np.zeros(len(data_benign), dtype=np.int64)
        labels_adv = np.ones(len(data_adv), dtype=np.int64)

        # Stack data and labels
        self.data = np.vstack((data_benign, data_adv)).astype(np.float32)
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



def train_adv_classifier():

    dataset = CSVDualClassDataset("class0.csv", "class1.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_data, batch_labels in dataloader:
        print(batch_data.shape, batch_labels.shape)
        break


if __name__ == '__main__':
    train_adv_classifier()