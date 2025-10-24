from typing import List

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from ddpm.unet import UNet
from ddpm.diffusion import DenoiseDiffusion



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class sensor_diffusion_dataset(Dataset):
    def __init__(self, csv_benign, csv_adversarial, transform=None, dtype=torch.float32):
        """
        Args:
            csv_benign (str): Path to CSV file for benign data (ground truth).
            csv_adversarial (str): Path to CSV file for adversarial data (input).
            transform (callable, optional): Optional transform to be applied on a sample.
            dtype (torch.dtype): Desired dtype for the features.
        """
        # Load both datasets
        data_benign = pd.read_csv(csv_benign, index_col=0).values
        data_adversarial = pd.read_csv(csv_adversarial, index_col=0).values

        # For diffusion model: adversarial data is input, benign data is target
        print("Number of Benign Samples: ", len(data_benign))
        print("Number of Adversarial Samples: ", len(data_adversarial))
        
        # Use adversarial data as input and benign data as target
        # We need to match the number of samples
        min_samples = min(len(data_benign), len(data_adversarial))
        self.data = data_adversarial[:min_samples].astype(np.float32)  # Input (adversarial)
        self.targets = data_benign[:min_samples].astype(np.float32)   # Target (benign)
        
        print("Total Number of Samples in Dataset: ", len(self.data))

        # Shuffle
        indices = np.random.permutation(len(self.data))
        self.data = self.data[indices]
        self.targets = self.targets[indices]

        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]      # Adversarial input
        target = self.targets[idx]  # Benign target

        if self.transform:
            sample = self.transform(sample)
            target = self.transform(target)

        sample = torch.tensor(sample, dtype=self.dtype)
        target = torch.tensor(target, dtype=self.dtype)
        
        # Reshape 1D sensor data to 2D for UNet: [sensor_dim] -> [1, 1, 1, sensor_dim]
        # This creates a "1D image" that UNet can process
        sample = sample.unsqueeze(0).unsqueeze(0)#.unsqueeze(0)  # [1, 1, 1, sensor_dim]
        target = target.unsqueeze(0).unsqueeze(0)#.unsqueeze(0)  # [1, 1, 1, sensor_dim]
        print(sample.shape)
        print(target.shape)

        return sample, target


class Diffusion_model():
    def __init__(self,experiment_name,debug=False):
        self.debug = debug
        self.experiment_name = experiment_name
        self.device = device
        self.experiment_path = './Experiments/'+experiment_name+'/'
        self.checkpoint_path = self.experiment_path+experiment_name+'_diffusion_checkpoint.zip'
        self.training_loss_plot_path = self.experiment_path+experiment_name+'_training_loss.png'
        self.loss_vals_path = self.experiment_path+self.experiment_name+'_latest_losses.csv'
        self.origin_sampling_path = self.experiment_path+'origin_sampling/'

        self.adv_run_path = './Experiments/ADVERSARIAL_'+experiment_name+'/'
        self.adv_checkpoint_path = self.adv_run_path+experiment_name+'_diffusion_checkpoint.zip'
        self.adv_training_loss_plot_path = self.adv_run_path+experiment_name+'_training_loss.png'
        self.adv_loss_vals_path = self.adv_run_path+self.experiment_name+'_latest_losses.csv'
        self.adv_origin_sampling_path = self.adv_run_path+'origin_sampling/'


        print("###########")
        print("Initiating Diffusion model with device: ", self.device)
        print("###########")
        # For sensor data: 1D data reshaped to 2D for UNet
        self.image_channels = 1  # Single channel for sensor data
        # Sensor data dimensions (will be determined from CSV)
        self.sensor_dim = None  # Will be set when loading data
        # Number of channels in the initial feature map
        self.n_channels = 32
        # The list of channel numbers at each resolution.
        # The number of channels is `channel_multipliers[i] * n_channels`
        # Reduced for 1D sensor data
        self.channel_multipliers = [1, 2, 2]
        # The list of booleans that indicate whether to use attention at each resolution
        self.is_attention = [False, False, True]

        # Number of time steps $T$
        self.n_steps = 3
        # Batch size
        self.batch_size = 8
        # Number of samples to generate
        self.n_samples = 16
        # Learning rate
        self.learning_rate = 1e-4

        # Number of training epochs
        self.epochs = 40

        self.normalize = None#transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         #                  std=[0.229, 0.224, 0.225])

        #self.pre_proc = transforms.Compose([transforms.Resize(self.image_size),
                                            #transforms.ToTensor()])
                                            #self.normalize])
        self.pre_proc = None

        self.data_folder = './mujoco_ant_obs_dataset/'  # folder with MuJoCo sensor data
        self.benign_csv = './mujoco_ant_obs_dataset/benign_obs_data.csv'
        self.adversarial_csv = './mujoco_ant_obs_dataset/adversarial_obs_data.csv'

        # No transforms needed for sensor data
        self.data_transforms = None

        self.training_data_loader = None
        self.adv_training_data_loader = None


        # Model will be created in run() method after we know sensor dimensions
        self.eps_model = None
        self.diffusion = None
        self.optimizer = None




    def sample(self,epoch_num):
        #Sample Images
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            # Remove noise for $T$ steps
            for t_ in tqdm(range(self.n_steps)):#monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            #tracker.save('sample', x)
            torchvision.utils.save_image(x, self.experiment_path+'sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')


    def sample_with_prior(self):
        #Sample Images
        dataset = CaptionDataset(self.data_folder, self.data_name, 'VAL', transform=self.pre_proc)
        # Dataloader
        val_loader = torch.utils.data.DataLoader(dataset,batch_size=self.n_samples, shuffle=True, num_workers=0, pin_memory=False)
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            #x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
            #                device=self.device)
            imgs, _,_,_ = next(iter(val_loader))
            imgs = imgs.to(device)
            #x = x + imgs
            t_sequence = torch.full(size=(self.n_samples,),fill_value=self.n_steps-1, device=imgs.device, dtype=torch.long)
            print(t_sequence.size())
            print(imgs.size())
            x = self.diffusion.q_sample(imgs,t_sequence)
            # Remove noise for $T$ steps
            for t_ in tqdm(range(self.n_steps)):#monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x_new = x.new_full((self.n_samples,), t, dtype=torch.long).to(device)
                x = self.diffusion.p_sample(x, x_new)

            # Log samples
            #tracker.save('sample', x)
            if not os.path.exists(self.origin_sampling_path):
                os.makedirs(self.origin_sampling_path)
            torchvision.utils.save_image(x, self.origin_sampling_path+self.experiment_name+'.png')
    

    def train(self):
        """
        ### Train
        """
        losses = []

        # Iterate through the dataset
        for i, (adversarial_data, benign_data) in enumerate(tqdm(self.training_data_loader)):
            # Move data to device
            adversarial_data = adversarial_data.to(self.device)#.permute(1,0,2)
            benign_data = benign_data.to(self.device)#.permute(1,0,2)
            
            # For diffusion model, we train on adversarial data to reconstruct benign data
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss - using adversarial data as input, benign as target
            print(f"Adversarial data shape: {adversarial_data.shape}")
            print(f"Benign data shape: {benign_data.shape}")
            
            # Call adv_denoiseing_loss with benign data as target
            x0_reconst, loss = self.diffusion.adv_denoiseing_loss(adversarial_data, benign_x=benign_data)
            
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            losses.append(loss.item())
            #Test break
            if self.debug is True:
                if(i == 0):
                    break

        #return avg loss for epoch
        return sum(losses) / len(losses)
            


    def run(self):
        
        # Dataset - use sensor data
        dataset = sensor_diffusion_dataset(self.benign_csv, self.adversarial_csv, transform=self.data_transforms)
        
        # Set sensor dimension from loaded data and create model
        if len(dataset) > 0:
            sample_data, _ = dataset[0]
            self.sensor_dim = sample_data.shape[0]
            print(f"Sensor data dimension: {self.sensor_dim}")
            
            # Create model now that we know the sensor dimension
            

            self.eps_model = UNet(
                image_channels=self.image_channels,
                n_channels=self.n_channels,
                ch_mults=self.channel_multipliers,
                is_attn=self.is_attention,
                ).to(self.device)

            # Create [DDPM class](index.html)
            self.diffusion = DenoiseDiffusion(
                eps_model=self.eps_model,
                n_steps=self.n_steps,
                device=self.device,
            )

            # Adam optimizer
            self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)
        
        # select smaller subset for training
        subset_factor = 8 #ie. use one nth of the total dataset
        subset_size = range(int(len(dataset)/subset_factor))
        subset = torch.utils.data.Subset(dataset, subset_size) # will select the first subset of dataset
        # Dataloader
        self.training_data_loader = torch.utils.data.DataLoader(subset,batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False)

        prev_epoch_count = 0
        losses = []
        losses_path = self.loss_vals_path

        if os.path.exists(losses_path):
            print("Loading previous losses...")
            with open(losses_path, newline='') as file:
                reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
                losses = list(reader)
                #strip the nested list
                losses = losses[0]
            prev_epoch_count = len(losses)
            print(losses)
            print("Done!")
        else:
            print("No losses to load, fresh training...")

        print("Beginning Training starting at epoch: ", prev_epoch_count)

        #Training loop
        for i in range(self.epochs):
            print("Epoch ", prev_epoch_count + i, ":")
            # Train the model
            epoch_loss = self.train()
            print('Loss: ',epoch_loss)
            # Sample some images
            #if(i % 2 == 0):
            print("Saving Samples to: "+self.experiment_path)
            self.origin_sampling(prev_epoch_count + i)
            print("Done!")
            # Save the model
            print("Saving Checkpoint to "+self.checkpoint_path)
            self.save_params()
            print("Done!")
            
            losses.append(epoch_loss)

            #Update loss plot at each epoch
            #print(list(range(prev_epoch_count + i + 1)))
            plt.plot(list(range(prev_epoch_count + i + 1)), losses, label='Training Loss')
            
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            
            plt.legend(loc='best')
            plt.savefig(self.training_loss_plot_path)
            plt.clf()

            #Save losses
            with open(losses_path, 'w') as file:
                wr = csv.writer(file,quoting=csv.QUOTE_NONNUMERIC)
                wr.writerow(losses)

    def origin_sampling(self,epoch_num,starting_t=None,eval=False,subset_factor=1,samples=16):
        if(starting_t is None):
            starting_t = self.n_steps - 1
        #Sample sensor data
        
        # Dataset - use sensor data
        dataset = sensor_diffusion_dataset(self.benign_csv, self.adversarial_csv, transform=self.data_transforms)
        # select smaller subset for sampling
        subset_size = range(int(len(dataset)/subset_factor))
        subset = torch.utils.data.Subset(dataset, subset_size)
        # Dataloader
        val_loader = torch.utils.data.DataLoader(subset,batch_size=samples, shuffle=True, num_workers=0, pin_memory=False)
        
        with torch.no_grad():
            # Load sensor data
            adversarial_data, benign_data = next(iter(val_loader))
            adversarial_data = adversarial_data.to(device)
            benign_data = benign_data.to(device)
            
            # Use adversarial data as input for diffusion process
            t = torch.full(size=(samples,),fill_value=self.n_steps-1, device=adversarial_data.device, dtype=torch.long)
            xT = self.diffusion.origin_q_sample(adversarial_data,t,base_t=starting_t)
            xt = xT
            xt_next = None
            
            # Remove noise for $T$ steps
            for t_ in tqdm(range(self.n_steps)):
                S_theta = self.eps_model(xt, t)
                ### Simplified denoising step
                xt_next = xt - (1/self.n_steps)*(xT - S_theta)

                #Advance to the next time step
                t_val = self.n_steps - (t_ % self.n_steps) - 1
                t = torch.full(size=(samples,), fill_value=t_val, device=adversarial_data.device, dtype=torch.long)
                xt = xt_next

            # Save reconstructed sensor data as numpy arrays instead of images
            if not os.path.exists(self.origin_sampling_path):
                os.makedirs(self.origin_sampling_path)
            
            # Reshape back to 1D for saving: [batch, 1, 1, sensor_dim] -> [batch, sensor_dim]
            xt_1d = xt#.squeeze(1).squeeze(1)  # Remove the extra dimensions
            adversarial_1d = adversarial_data#.squeeze(1).squeeze(1)
            benign_1d = benign_data#.squeeze(1).squeeze(1)
            xT_1d = xT#.squeeze(1).squeeze(1)
            
            if(eval == True):
                # Save reconstructed data
                np.save(self.origin_sampling_path+'reconstructed_'+self.experiment_name+'_epoch'+str(epoch_num)+'.npy', xt_1d.cpu().numpy())
                np.save(self.origin_sampling_path+'original_adversarial_'+self.experiment_name+'_epoch'+str(epoch_num)+'.npy', adversarial_1d.cpu().numpy())
                np.save(self.origin_sampling_path+'target_benign_'+self.experiment_name+'_epoch'+str(epoch_num)+'.npy', benign_1d.cpu().numpy())
                np.save(self.origin_sampling_path+'noisy_'+self.experiment_name+'_epoch'+str(epoch_num)+'.npy', xT_1d.cpu().numpy())
            else:
                # Save reconstructed data
                np.save(self.experiment_path+'reconstructed_'+self.experiment_name+'_epoch'+str(epoch_num)+'.npy', xt_1d.cpu().numpy())
                np.save(self.experiment_path+'original_adversarial_'+self.experiment_name+'_epoch'+str(epoch_num)+'.npy', adversarial_1d.cpu().numpy())
                np.save(self.experiment_path+'target_benign_'+self.experiment_name+'_epoch'+str(epoch_num)+'.npy', benign_1d.cpu().numpy())
                np.save(self.experiment_path+'noisy_'+self.experiment_name+'_epoch'+str(epoch_num)+'.npy', xT_1d.cpu().numpy())

    def adv_training(self,adv_loss=None,adv_target=None,victim=None,whitebox=False,base_t=None,targeted_latent=False,target_features=None):
        """
        ### Standard Denoising Training (formerly adversarial training)
        """
        losses = []
        
        #CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)
        #CrossEntropyLoss = torch.nn.NLLLoss().to(device)
        # Iterate through the dataset
        for i, (imgs, caps) in enumerate(tqdm(self.training_data_loader)):#monit.iterate('Train', self.data_loader):
            # Increment global step
            #tracker.add_global_step()
            # Move data to device
            #Un-comment below to display images
            '''
            print(imgs[0].permute(1,2,0).size())
            plt.imshow(imgs[0].permute(1,2,0))
            plt.show()
            '''
            imgs = imgs.to(self.device)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss - using standard denoising loss only
            loss = self.diffusion.denoiseing_loss(imgs)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            #tracker.save('loss', loss)
            losses.append(loss.item())
            #Test break
            if self.debug is True:
                if(i == 0):
                    break

        #return avg loss for epoch
        avg_loss = sum(losses) / len(losses)
        print('Denoising Loss: ', avg_loss)

        return avg_loss


    def adv_run(self, subset_factor=8, noise_scale=None):
        """
        Standard denoising training run (formerly adversarial training)
        """
        # Dataset
        dataset = CaptionDataset(self.data_transforms, self.data_folder)
        # select smaller subset of coco to train
        subset_size = range(int(len(dataset)/subset_factor))
        subset = torch.utils.data.Subset(dataset, subset_size)
        # Dataloader
        self.training_data_loader = torch.utils.data.DataLoader(subset,batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        
        prev_epoch_count = 0
        losses = []
        losses_path = self.loss_vals_path

        base_t = None
        if(noise_scale):
            base_t = int(self.n_steps*noise_scale)

        if os.path.exists(losses_path):
            print("Loading previous losses...")
            with open(losses_path, newline='') as file:
                reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
                losses = list(reader)
                #strip the nested list
                losses = losses[0]
            prev_epoch_count = len(losses)
            print(losses)
            print("Done!")
        else:
            print("No losses to load, fresh training...")

        print("Beginning Training starting at epoch: ", prev_epoch_count)

        #Training loop
        for i in range(self.epochs):
            print("Epoch ", prev_epoch_count + i, ":")
            # Train the model
            epoch_loss = self.adv_training(None, None, None, base_t=base_t)
            print('Loss: ',epoch_loss)
            # Sample some images
            print("Saving Samples to: "+self.adv_run_path)
            self.origin_sampling(prev_epoch_count + i,adv=True)
            print("Done!")
            # Save the model
            print("Saving Checkpoint to "+self.adv_checkpoint_path)
            self.save_params(adv=True)
            print("Done!")
            
            losses.append(epoch_loss)

            #Update loss plot at each epoch
            plt.plot(list(range(prev_epoch_count + i + 1)), losses, label='Training Loss')
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            
            plt.legend(loc='best')
            plt.savefig(self.adv_training_loss_plot_path)
            plt.clf()

            #Save losses
            with open(losses_path, 'w') as file:
                wr = csv.writer(file,quoting=csv.QUOTE_NONNUMERIC)
                wr.writerow(losses)

    def generate_attack_images(self, img_count):
        pass #TODO generate and save a number of adverarial images

    def eval_attack(self,victim,epoch_num,starting_t=None,subset_factor=1,samples=1):
        if(starting_t is None):
            starting_t = self.n_steps - 1
        
        # Dataset
        dataset = CaptionDataset(self.data_transforms, self.data_folder, split_type='train')
        # select smaller subset of coco to train
        #subset_factor = 8 #ie. use one nth of the total dataset
        subset_size = range(int(len(dataset)/subset_factor))
        subset = torch.utils.data.Subset(dataset, subset_size) # will select the first 10 of dataset_train
        # Dataloader #TODO shuffle has been set to False
        val_loader = torch.utils.data.DataLoader(subset,batch_size=samples, shuffle=True, num_workers=0, pin_memory=False)
        
        with torch.no_grad():
            imgs, _ = next(iter(val_loader))
            imgs = imgs.to(device)

            t = torch.full(size=(samples,),fill_value=self.n_steps-1, device=imgs.device, dtype=torch.long)
            xT = self.diffusion.origin_q_sample(imgs,t,base_t=starting_t)
            xt = xT
            xt_next = None
            # Remove noise for $T$ steps
            for t_ in tqdm(range(self.n_steps)):
                
                S_theta = self.eps_model(xt, t)
                ### The below can be simplified
                '''
                xt_tilda = (1 - alpha_t)*S_theta + (alpha_t*xT)
                xt_next_tilda = (1 - alpha_t_next)*S_theta + alpha_t_next*xT
                xt_next = xt - xt_tilda + xt_next_tilda
                '''
                ### to the following
                xt_next = xt - (1/self.n_steps)*(xT - S_theta)

                #Advance to the next time step
                t_val = self.n_steps - (t_ % self.n_steps) - 1
                t = torch.full(size=(samples,), fill_value=t_val, device=imgs.device, dtype=torch.long)
                xt = xt_next

            # Log samples
            #tracker.save('sample', x)
            eval_path = self.adv_run_path + 'Eval/Epoch_'+str(epoch_num)+'/'
            eval_path_original = eval_path + 'Original/'
            eval_path_adv = eval_path + 'Adversarial/'
            eval_path_noisy = eval_path + 'Noisy/'

            if not os.path.exists(eval_path):
                os.makedirs(eval_path)
            if not os.path.exists(eval_path_original):
                os.makedirs(eval_path_original)
            if not os.path.exists(eval_path_adv):
                os.makedirs(eval_path_adv)
            if not os.path.exists(eval_path_noisy):
                os.makedirs(eval_path_noisy)

            torchvision.utils.save_image(xt, eval_path_adv+'adv_'+str(epoch_num)+'.png')
            torchvision.utils.save_image(imgs, eval_path_original+'original_'+str(epoch_num)+'.png')
            torchvision.utils.save_image(xT, eval_path_noisy+'noisy_'+str(epoch_num)+'.png')

            victim.gen_caption(eval_path_adv+'adv_'+str(epoch_num)+'.png',eval_path+'generated_caption.png')
            

    def save_params(self,adv=False):
        if(adv == True):
            torch.save(self.eps_model.state_dict(),self.adv_checkpoint_path)
        else:
            torch.save(self.eps_model.state_dict(),self.checkpoint_path)


    def load_params(self,adv=False):
        if(adv == True):
            self.eps_model.load_state_dict(torch.load(self.adv_checkpoint_path))
        else:
            self.eps_model.load_state_dict(torch.load(self.checkpoint_path))
