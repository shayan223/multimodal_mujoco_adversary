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

from ddpm.unet import UNet
from ddpm.diffusion import DenoiseDiffusion
from show_attend_tell_V2.dataset import ImageCaptionDataset as CaptionDataset
from show_attend_tell_V2.GLOBAL_VALS import IMAGE_SIZE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Number of channels in the image. $3$ for RGB.
        self.image_channels = 1
        # Image size (resize coco images to image_size x image_size)
        self.image_size = IMAGE_SIZE
        # Number of channels in the initial feature map
        self.n_channels = 32
        # The list of channel numbers at each resolution.
        # The number of channels is `channel_multipliers[i] * n_channels`
        self.channel_multipliers = [1, 2, 2, 4]
        # The list of booleans that indicate whether to use attention at each resolution
        self.is_attention = [False, False, False, True]

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

        self.data_folder = './mujoco_ant_obs_dataset/'  # folder with data files saved by create_input_files.py
        self.data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

        self.data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         #std=[0.229, 0.224, 0.225])
        ])

        self.training_data_loader = None
        self.adv_training_data_loader = None


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
            # Calculate loss
            #loss = self.diffusion.loss(imgs)
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
        return sum(losses) / len(losses)
            


    def run(self):
        
        # Dataset
        dataset = CaptionDataset(self.data_transforms, self.coco_path)
        # select smaller subset of coco to train
        subset_factor = 8 #ie. use one nth of the total dataset
        subset_size = range(int(len(dataset)/subset_factor))
        subset = torch.utils.data.Subset(dataset, subset_size) # will select the first 10 of dataset_train
        # Dataloader #TODO shuffle has been set to False
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

    def origin_sampling(self,epoch_num,starting_t=None,eval=False,adv=False,subset_factor=1,samples=16):
        if(starting_t is None):
            starting_t = self.n_steps - 1
        #Sample Images
        #TODO 'VAL' was changed to 'TRAIN' and shuffle from true to false
        #dataset = CaptionDataset(self.data_transforms, self.data_folder)#CaptionDataset(self.data_folder, self.data_name, 'VAL', transform=self.pre_proc)
        # Dataloader
        #val_loader = torch.utils.data.DataLoader(dataset,batch_size=self.n_samples, shuffle=True, num_workers=0, pin_memory=False)
            
        
        # Dataset
        dataset = CaptionDataset(self.data_transforms, self.data_folder, split_type='train')
        # select smaller subset of coco to train
        #subset_factor = 8 #ie. use one nth of the total dataset
        subset_size = range(int(len(dataset)/subset_factor))
        subset = torch.utils.data.Subset(dataset, subset_size) # will select the first 10 of dataset_train
        # Dataloader #TODO shuffle has been set to False
        val_loader = torch.utils.data.DataLoader(subset,batch_size=samples, shuffle=True, num_workers=0, pin_memory=False)
        
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            #x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
            #                device=self.device)
            imgs, _ = next(iter(val_loader))
            imgs = imgs.to(device)
            #x = x + imgs
            
            '''
            if(starting_t is not None):
                T_max = torch.full(size=(self.n_samples,),fill_value=self.n_steps - 1, device=imgs.device, dtype=torch.long)
                t = torch.full(size=(self.n_samples,),fill_value=starting_t, device=imgs.device, dtype=torch.long)
                xT = self.diffusion.q_sample(imgs,T_max)
                xt = self.diffusion.q_sample(imgs,t)
            else:
            '''
            t = torch.full(size=(samples,),fill_value=self.n_steps-1, device=imgs.device, dtype=torch.long)
            xT = self.diffusion.origin_q_sample(imgs,t,base_t=starting_t)
            xt = xT
            xt_next = None
            # Remove noise for $T$ steps
            for t_ in tqdm(range(self.n_steps)):#monit.iterate('Sample', self.n_steps):
                #alpha_t = t / self.n_steps
                #alpha_t = alpha_t.view(-1,1,1,1)
                #print(alpha_t.size())
                #alpha_t_next = (t - 1)/ self.n_steps
                #alpha_t_next = alpha_t_next.view(-1,1,1,1)
                #print(alpha_t_next.size())
                
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
            if not os.path.exists(self.origin_sampling_path):
                os.makedirs(self.origin_sampling_path)
            if not os.path.exists(self.adv_origin_sampling_path):
                os.makedirs(self.adv_origin_sampling_path)
            if(eval == True):
                torchvision.utils.save_image(xt, self.origin_sampling_path+'sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')
                torchvision.utils.save_image(imgs, self.origin_sampling_path+'PRE_sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')
                torchvision.utils.save_image(xT, self.origin_sampling_path+'NOISY_sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')
            elif(adv == True):
                torchvision.utils.save_image(xt, self.adv_origin_sampling_path+'sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')
                torchvision.utils.save_image(imgs, self.adv_origin_sampling_path+'PRE_sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')
                torchvision.utils.save_image(xT, self.adv_origin_sampling_path+'NOISY_sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')
            else:
                torchvision.utils.save_image(xt, self.experiment_path+'sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')
                torchvision.utils.save_image(imgs, self.experiment_path+'PRE_sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')
                torchvision.utils.save_image(xT, self.experiment_path+'NOISY_sample_'+self.experiment_name+'_epoch'+str(epoch_num)+'.png')

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
