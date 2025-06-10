
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE_adv_obs_dataset(Dataset):
    def __init__(self, benign_csv, adv_csv, transforms=None, dtype=torch.float32):
        """
        Args:
            benign_csv (str): CSV path for data used as labels (and inputs).
            adv_csv (str): CSV path for data used as inputs.
        """
        self.benign = pd.read_csv(benign_csv).values.astype(np.float32)
        self.adv = pd.read_csv(adv_csv).values.astype(np.float32)

        assert self.benign.shape == self.adv.shape, "Both files must have the same shape"

        self.dtype = dtype
        self.N = len(self.benign)
        self.transforms = transforms

        # Create self-pairs and cross-pairs
        # Total dataset: 2N entries
        self.data = np.concatenate([self.benign, self.adv], axis=0)
        self.labels = np.concatenate([self.benign, self.benign], axis=0)  # file1 is always the label

        #For tensor data       
        #self.data_min = self.data.min(dim=0).values  # per-feature min
        #self.data_max = self.data.max(dim=0).values  # per-feature max
        #For np data
        self.data_min = torch.Tensor(np.min(self.data, axis=0)).to(device)
        self.data_max = torch.Tensor(np.max(self.data, axis=0)).to(device)


    def normalize(self, x):
        return (x - self.data_min) / (self.data_max - self.data_min + 1e-8)

    def denormalize(self, x_norm):
        return x_norm * (self.data_max - self.data_min + 1e-8) + self.data_min

    def __len__(self):
        return 2 * self.N

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transforms:
            x = self.transforms(x)

        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(y, dtype=self.dtype)
        return x, y

"""
A Convolutional Variational Autoencoder, made to match that in the Defence-VAE paper: https://github.com/lxuniverse/defense-vae/blob/master/black_box/vae_models.py
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=4096, zDim=128):
        super(VAE, self).__init__()
        self.encoding_dim = zDim
        # Vae model made to match that in the Defence-VAE code base
        # Encoder
        self.encConv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias= False)
        self.encConv1_bn = nn.BatchNorm1d(64)
        self.encConv2 = nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=3, bias= False)
        self.encConv2_bn = nn.BatchNorm1d(64)
        self.encConv3 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.encConv3_bn = nn.BatchNorm1d(128)
        self.encConv4 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias= False)
        self.encConv4_bn = nn.BatchNorm1d(256)
        # Latent space
        self.mu_layer = nn.Linear(1024, self.encoding_dim)
        self.logvar_layer = nn.Linear(1024, self.encoding_dim)

        # Decoder
        self.fc3 = nn.Linear(128, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv1_bn = nn.BatchNorm1d(128)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv2_bn = nn.BatchNorm1d(64)
        self.deconv3 = nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=2, bias= False)
        self.deconv3_bn = nn.BatchNorm1d(64)
        self.deconv4 = nn.ConvTranspose1d(64, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        #print(x.shape)
        out = self.relu(self.encConv1_bn(self.encConv1(x)))
        #print(out.shape)
        out = self.relu(self.encConv2_bn(self.encConv2(out)))
        #print(out.shape)
        out = self.relu(self.encConv3_bn(self.encConv3(out)))
        #print(out.shape)
        out = self.relu(self.encConv4_bn(self.encConv4(out)))
        #print(out.shape)
        h1 = out.view(out.size(0), -1)

        # mu and logVar respectively
        mu = self.mu_layer(h1)

        logvar = self.logvar_layer(h1)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into transpose convolutional layers
        # The generated output is the same size of the original input
        #print('DECODER INPUT: ', z.shape)
        h3 = self.relu(self.fc3(z)).view(-1, 256, 4)#.unsqueeze(2)
        #print(h3.shape)
        #out = h3.view(h3.size(0), 1028)
        out = self.relu(self.deconv1_bn(self.deconv1(h3)))
        #print(out.shape)
        out = self.relu(self.deconv2_bn(self.deconv2(out)))
        #print(out.shape)
        out = self.relu(self.deconv3_bn(self.deconv3(out)))
        #print(out.shape)
        out = self.sigmoid(self.deconv4(out))
        #print(out.shape)
        return out

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        decoded_z = self.decoder(z)

        return decoded_z, mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

def train_vae_mnist(EPOCHS=10):
    """
    Determine if any GPUs are available
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    """
    Initialize Hyperparameters
    """
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = EPOCHS


    """
    Create dataloaders to feed data into the neural network
    Default MNIST dataset is used and standard train/test split is performed
    """
    '''
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                        transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=1)'''
    # Split into train and validation
    dataset = VAE_adv_obs_dataset('benign_obs_data.csv','adversarial_obs_data.csv')#,transforms=transforms.ToTensor())
    val_fraction = 0.2
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=32)


    """
    Initialize the network and the Adam optimizer
    """
    net = VAE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    KL_losses = []
    Recon_losses = []
    for epoch in range(num_epochs):
        avg_batch_KL = []
        avg_batch_recon = []
        for idx, data in tqdm(enumerate(train_loader, 0)):
            imgs, benign_imgs = data
            imgs = imgs.unsqueeze(1).to(device)
            benign_imgs = benign_imgs.unsqueeze(1).to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = net(imgs)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            #kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
            #loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            #print('Min/max recon_x:', out.min(), out.max())
            #print('Min/max x:', benign_imgs.min(), benign_imgs.max())
            loss, bce, kld = net.loss(out,dataset.normalize(benign_imgs),mu,logVar)

            #Record batch loss  
            avg_batch_recon.append(torch.mean(bce).item())
            avg_batch_KL.append(torch.mean(kld).item())

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Record average losses for epoch
        KL_losses.append(sum(avg_batch_KL)/len(avg_batch_KL))
        Recon_losses.append(sum(avg_batch_recon)/len(avg_batch_recon))

        print('Epoch {}: Loss {}'.format(epoch, loss))


    """
    The following part takes a random image from test loader to feed into the VAE.
    Both the original image and generated image from the distribution are shown.
    """
    '''
    sample_count = 10
    image_count = 1
    net.eval()
    with torch.no_grad():
        for data in random.sample(list(test_loader), sample_count):
            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            out, mu, logVAR = net(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.savefig('../../results/vae_examples/vae_examples'+str(image_count)+'.png')
            plt.clf()
            image_count += 1
    '''

    # Save model parameters
    torch.save(net.state_dict(), './defence_vae.pth')


if __name__ == '__main__':
    train_vae_mnist()