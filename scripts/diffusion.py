from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import time

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
from torch.utils.data import Dataset, Subset

from ddpm.unet_mlp_1d import UNetMLP1D
from ddpm.diffusion import DenoiseDiffusion



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class sensor_diffusion_dataset(Dataset):
    def __init__(
        self,
        csv_benign,
        csv_adversarial,
        transform=None,
        dtype=torch.float32,
        normalization_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            csv_benign (str): Path to CSV file for benign data (ground truth).
            csv_adversarial (str): Path to CSV file for adversarial data (input).
            transform (callable, optional): Optional transform to be applied on a sample.
            dtype (torch.dtype): Desired dtype for the features.
        """
        # Load both datasets
        data_benign = pd.read_csv(csv_benign, index_col=0).values.astype(np.float32)
        data_adversarial = pd.read_csv(csv_adversarial, index_col=0).values.astype(np.float32)

        # For diffusion model: adversarial data is input, benign data is target
        print("Number of Benign Samples: ", len(data_benign))
        print("Number of Adversarial Samples: ", len(data_adversarial))
        
        # Match the number of samples - use n = number of benign samples
        n = len(data_benign)
        # Use n adversarial samples (matched with benign samples)
        n_adversarial = min(n, len(data_adversarial))
        
        # Create dataset with 2n samples:
        # First n samples: benign data as input, benign data as target (self-labeled)
        # Next n samples: adversarial data as input, benign data as target (matched with benign counterpart)
        benign_inputs = data_benign[:n]
        benign_targets = data_benign[:n]  # Benign samples labeled with themselves
        
        adversarial_inputs = data_adversarial[:n_adversarial]
        adversarial_targets = data_benign[:n_adversarial]  # Adversarial samples labeled with benign counterpart
        
        # If we have fewer adversarial samples than benign, we'll only use what we have
        # This ensures we have at least n samples (all benign), and up to 2n if we have enough adversarial
        self.data = np.vstack([benign_inputs, adversarial_inputs]).astype(np.float32)
        self.targets = np.vstack([benign_targets, adversarial_targets]).astype(np.float32)
        self.sample_types = np.concatenate([
            np.zeros(len(benign_inputs), dtype=np.int64),
            np.ones(len(adversarial_inputs), dtype=np.int64),
        ])
        
        print("Total Number of Samples in Dataset: ", len(self.data))
        print("  - Benign samples (self-labeled): ", len(benign_inputs))
        print("  - Adversarial samples (labeled with benign counterpart): ", len(adversarial_inputs))

        # Shuffle
        indices = np.random.permutation(len(self.data))
        self.data = self.data[indices]
        self.targets = self.targets[indices]
        self.sample_types = self.sample_types[indices]

        if normalization_stats is None:
            mean = self.data.mean(axis=0)
            std = self.data.std(axis=0)
            std[std < 1e-6] = 1.0
            self.normalization_stats = {
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            }
        else:
            self.normalization_stats = {
                "mean": normalization_stats["mean"].astype(np.float32),
                "std": normalization_stats["std"].astype(np.float32),
            }

        self.data = self.normalize_array(self.data)
        self.targets = self.normalize_array(self.targets)

        self.transform = transform
        self.dtype = dtype

    def normalize_array(self, value: np.ndarray) -> np.ndarray:
        return (value - self.normalization_stats["mean"]) / self.normalization_stats["std"]

    def denormalize_array(self, value: np.ndarray) -> np.ndarray:
        return (value * self.normalization_stats["std"]) + self.normalization_stats["mean"]

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
        
        # Pad from 29 to 32 dimensions by adding zeros at the end
        #if sample.shape[0] == 29:
        #    sample = torch.cat([sample, torch.zeros(3, dtype=self.dtype)], dim=0)
        #if target.shape[0] == 29:
        #    target = torch.cat([target, torch.zeros(3, dtype=self.dtype)], dim=0)
        
        # Reshape 1D sensor data to 2D for UNet: [sensor_dim] -> [1, 1, 1, sensor_dim]
        # This creates a "1D image" that UNet can process
        sample = sample.unsqueeze(0)#.unsqueeze(0)#.unsqueeze(0)  # [1, 1, 1, sensor_dim]
        target = target.unsqueeze(0)#.unsqueeze(0)#.unsqueeze(0)  # [1, 1, 1, sensor_dim]
        #print(sample.shape)
        #print(target.shape)

        metadata = {
            "sample_type": int(self.sample_types[idx]),
        }

        return sample, target, metadata


class Diffusion_model():
    def __init__(
        self,
        experiment_name,
        debug=False,
        inference_mode: str = "stochastic_light",
        inference_steps: Optional[int] = None,
        renoise_strength: float = 1.0,
        benign_csv: Optional[str] = None,
        adversarial_csv: Optional[str] = None,
    ):
        self.debug = debug
        self.experiment_name = experiment_name
        self.repo_root = Path(__file__).resolve().parents[1]
        self.experiment_root = self.repo_root / 'Experiments'
        self.device = device
        self.experiment_path = self.experiment_root / experiment_name
        self.checkpoint_path = self.experiment_path / f'{experiment_name}_diffusion_checkpoint.zip'
        self.training_loss_plot_path = self.experiment_path / f'{experiment_name}_training_loss.png'
        self.loss_vals_path = self.experiment_path / f'{self.experiment_name}_latest_losses.csv'
        self.origin_sampling_path = self.experiment_path / 'origin_sampling'
        self.normalization_stats_path = self.experiment_path / f'{experiment_name}_normalization_stats.json'
        self.split_metadata_path = self.experiment_path / f'{experiment_name}_dataset_splits.json'

        self.adv_run_path = self.experiment_root / f'ADVERSARIAL_{experiment_name}'
        self.adv_checkpoint_path = self.adv_run_path / f'{experiment_name}_diffusion_checkpoint.zip'
        self.adv_training_loss_plot_path = self.adv_run_path / f'{experiment_name}_training_loss.png'
        self.adv_loss_vals_path = self.adv_run_path / f'{self.experiment_name}_latest_losses.csv'
        self.adv_origin_sampling_path = self.adv_run_path / 'origin_sampling'


        print("###########")
        print("Initiating Diffusion model with device: ", self.device)
        print("###########")
        # For sensor data: 1D data reshaped to 2D for UNet
        self.image_channels = 1  # Single channel for sensor data
        # Sensor data dimensions (will be determined from CSV)
        self.sensor_dim = None  # Will be set when loading data
        # Original sensor dimension before padding
        self.original_sensor_dim = 29  # Original dimension before padding to 32
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
        self.batch_size = 64
        # Number of samples to generate
        self.n_samples = 16
        # Learning rate
        self.learning_rate = 1e-4

        # Number of training epochs
        self.epochs = 100
        self.train_ratio = 0.8
        self.val_ratio = 0.1

        self.normalize = None#transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         #                  std=[0.229, 0.224, 0.225])

        #self.pre_proc = transforms.Compose([transforms.Resize(self.image_size),
                                            #transforms.ToTensor()])
                                            #self.normalize])
        self.pre_proc = None

        self.data_folder = str(self.repo_root)
        self.benign_csv = str(Path(benign_csv) if benign_csv is not None else self.repo_root / 'benign_obs_data.csv')
        self.adversarial_csv = str(Path(adversarial_csv) if adversarial_csv is not None else self.repo_root / 'adversarial_obs_data.csv')

        # No transforms needed for sensor data
        self.data_transforms = None

        self.training_data_loader = None
        self.validation_data_loader = None
        self.test_data_loader = None
        self.adv_training_data_loader = None
        self.normalization_stats = None
        self.dataset_split_metadata: Dict[str, object] = {}
        self.inference_mode = inference_mode
        self.inference_steps = inference_steps if inference_steps is not None else self.n_steps
        self.renoise_strength = renoise_strength

        # Load dataset to determine input shape
        dataset = sensor_diffusion_dataset(self.benign_csv, self.adversarial_csv, transform=self.data_transforms)
        self.normalization_stats = dataset.normalization_stats
        if len(dataset) > 0:
            sample_data, _, _ = dataset[0]
            self.input_shape = sample_data.shape  # Shape after processing (e.g., [1, sensor_dim])
            self.sensor_dim = sample_data.shape[-1]  # Last dimension is the sensor dimension
            print(f"Input shape determined: {self.input_shape}")
            print(f"Sensor dimension: {self.sensor_dim}")
        else:
            raise ValueError("Dataset is empty, cannot determine input shape")
        
        # Initialize model
        self.eps_model = UNetMLP1D(
            input_channels=1,      # Input vector dimension
            n_channels=32,          # Base number of channels
            ch_mults=(1, 2, 2, 4),  # Channel multipliers for each resolution
            is_attn=(False, False, True, True),  # Attention at higher resolutions
            n_blocks=2              # Number of blocks per resolution
        ).to(self.device)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)
        self.set_inference_config(self.inference_mode, self.inference_steps, self.renoise_strength)

    def set_inference_config(self, mode: str = "stochastic_light", steps: Optional[int] = None,
                             renoise_strength: Optional[float] = None):
        self.inference_mode = mode
        if steps is not None:
            self.inference_steps = max(1, int(steps))
        if renoise_strength is not None:
            self.renoise_strength = float(renoise_strength)

        if self.inference_mode == "deterministic":
            self.use_stochastic_renoise = False
            self.eps_model.eval()
        elif self.inference_mode == "stochastic_heavy":
            self.use_stochastic_renoise = True
            self.eps_model.train()
        else:
            self.use_stochastic_renoise = True
            self.eps_model.eval()

    def _normalization_stats_to_json(self) -> Dict[str, List[float]]:
        return {
            "mean": self.normalization_stats["mean"].tolist(),
            "std": self.normalization_stats["std"].tolist(),
        }

    def _save_normalization_stats(self):
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        with open(self.normalization_stats_path, 'w', encoding='utf-8') as stats_file:
            json.dump(self._normalization_stats_to_json(), stats_file, indent=2)

    def _load_normalization_stats(self):
        if self.normalization_stats_path.exists():
            with open(self.normalization_stats_path, 'r', encoding='utf-8') as stats_file:
                loaded = json.load(stats_file)
            self.normalization_stats = {
                "mean": np.asarray(loaded["mean"], dtype=np.float32),
                "std": np.asarray(loaded["std"], dtype=np.float32),
            }

    def normalize_tensor(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalization_stats["mean"], dtype=torch.float32, device=value.device)
        std = torch.tensor(self.normalization_stats["std"], dtype=torch.float32, device=value.device)
        while mean.dim() < value.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (value - mean) / std

    def denormalize_tensor(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalization_stats["mean"], dtype=torch.float32, device=value.device)
        std = torch.tensor(self.normalization_stats["std"], dtype=torch.float32, device=value.device)
        while mean.dim() < value.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (value * std) + mean

    def _build_split_loaders(self):
        dataset = sensor_diffusion_dataset(
            self.benign_csv,
            self.adversarial_csv,
            transform=self.data_transforms,
            normalization_stats=self.normalization_stats,
        )
        total_size = len(dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, test_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
        self.training_data_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False
        )
        self.validation_data_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False
        )
        self.test_data_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False
        )
        self.dataset_split_metadata = {
            "total_size": total_size,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
        }
        with open(self.split_metadata_path, 'w', encoding='utf-8') as split_file:
            json.dump(self.dataset_split_metadata, split_file, indent=2)

    def evaluate_loader(self, loader) -> Dict[str, float]:
        if loader is None:
            return {"overall": float("nan"), "clean": float("nan"), "adversarial": float("nan")}

        self.eps_model.eval()
        overall_losses: List[float] = []
        clean_losses: List[float] = []
        adv_losses: List[float] = []

        with torch.no_grad():
            for adversarial_data, benign_data, metadata in loader:
                adversarial_data = adversarial_data.to(self.device)
                benign_data = benign_data.to(self.device)
                x0_reconst, loss = self.diffusion.adv_denoiseing_loss(adversarial_data, benign_x=benign_data)
                per_sample = F.mse_loss(x0_reconst, benign_data, reduction='none').mean(dim=(1, 2))
                sample_types = metadata["sample_type"]
                overall_losses.extend(per_sample.detach().cpu().tolist())
                for sample_loss, sample_type in zip(per_sample.detach().cpu().tolist(), sample_types.tolist()):
                    if int(sample_type) == 0:
                        clean_losses.append(sample_loss)
                    else:
                        adv_losses.append(sample_loss)

        return {
            "overall": float(np.mean(overall_losses)) if overall_losses else float("nan"),
            "clean": float(np.mean(clean_losses)) if clean_losses else float("nan"),
            "adversarial": float(np.mean(adv_losses)) if adv_losses else float("nan"),
        }




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
        self.eps_model.train()
        for i, (adversarial_data, benign_data, _metadata) in enumerate(tqdm(self.training_data_loader)):
            # Move data to device
            adversarial_data = adversarial_data.to(self.device)#.permute(1,0,2)
            benign_data = benign_data.to(self.device)#.permute(1,0,2)
            
            # For diffusion model, we train on adversarial data to reconstruct benign data
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss - using adversarial data as input, benign as target
            #print(f"Adversarial data shape: {adversarial_data.shape}")
            #print(f"Benign data shape: {benign_data.shape}")
            
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
        
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        self._save_normalization_stats()
        self._build_split_loaders()

        prev_epoch_count = 0
        losses = []
        losses_path = self.loss_vals_path

        if losses_path.exists():
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
            val_metrics = self.evaluate_loader(self.validation_data_loader)
            print('Loss: ',epoch_loss)
            print('Validation metrics: ', val_metrics)
            # Sample some images
            #if(i % 2 == 0):
            print("Saving Samples to: "+str(self.experiment_path))
            self.origin_sampling(prev_epoch_count + i)
            print("Done!")
            # Save the model
            print("Saving Checkpoint to "+str(self.checkpoint_path))
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
        dataset = sensor_diffusion_dataset(
            self.benign_csv,
            self.adversarial_csv,
            transform=self.data_transforms,
            normalization_stats=self.normalization_stats,
        )
        # select smaller subset for sampling
        subset_size = range(int(len(dataset)/subset_factor))
        subset = torch.utils.data.Subset(dataset, subset_size)
        # Dataloader
        val_loader = torch.utils.data.DataLoader(subset,batch_size=samples, shuffle=True, num_workers=0, pin_memory=False)
        
        with torch.no_grad():
            # Load sensor data
            adversarial_data, benign_data, _metadata = next(iter(val_loader))
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
            if not self.origin_sampling_path.exists():
                self.origin_sampling_path.mkdir(parents=True, exist_ok=True)
            
            # Reshape back to 1D for saving: [batch, 1, 1, sensor_dim] -> [batch, sensor_dim]
            xt_1d = self.denormalize_tensor(xt.squeeze(1))
            adversarial_1d = self.denormalize_tensor(adversarial_data.squeeze(1))
            benign_1d = self.denormalize_tensor(benign_data.squeeze(1))
            xT_1d = self.denormalize_tensor(xT.squeeze(1))
            
            if(eval == True):
                # Save reconstructed data
                np.save(self.origin_sampling_path / f'reconstructed_{self.experiment_name}_epoch{epoch_num}.npy', xt_1d.cpu().numpy())
                np.save(self.origin_sampling_path / f'original_adversarial_{self.experiment_name}_epoch{epoch_num}.npy', adversarial_1d.cpu().numpy())
                np.save(self.origin_sampling_path / f'target_benign_{self.experiment_name}_epoch{epoch_num}.npy', benign_1d.cpu().numpy())
                np.save(self.origin_sampling_path / f'noisy_{self.experiment_name}_epoch{epoch_num}.npy', xT_1d.cpu().numpy())
            else:
                # Save reconstructed data
                np.save(self.experiment_path / f'reconstructed_{self.experiment_name}_epoch{epoch_num}.npy', xt_1d.cpu().numpy())
                np.save(self.experiment_path / f'original_adversarial_{self.experiment_name}_epoch{epoch_num}.npy', adversarial_1d.cpu().numpy())
                np.save(self.experiment_path / f'target_benign_{self.experiment_name}_epoch{epoch_num}.npy', benign_1d.cpu().numpy())
                np.save(self.experiment_path / f'noisy_{self.experiment_name}_epoch{epoch_num}.npy', xT_1d.cpu().numpy())

    def benchmark_inference_speed(self, batch_size: int = 1, warmup: int = 5, iterations: int = 50) -> Dict[str, float]:
        batch_size = max(1, int(batch_size))
        warmup = max(0, int(warmup))
        iterations = max(1, int(iterations))

        sample = torch.zeros((batch_size, self.sensor_dim), dtype=torch.float32, device=self.device)

        def sync_device():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        for _ in range(warmup):
            _ = self.inference(sample)
        sync_device()

        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = self.inference(sample)
        sync_device()
        elapsed_seconds = time.perf_counter() - start_time

        total_samples = batch_size * iterations
        seconds_per_batch = elapsed_seconds / iterations
        seconds_per_sample = elapsed_seconds / total_samples
        return {
            "uses_origin_q_sample": self.use_stochastic_renoise,
            "starting_t": self.n_steps - 1,
            "inference_steps": max(1, int(self.inference_steps)),
            "batch_size": batch_size,
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "elapsed_seconds": elapsed_seconds,
            "ms_per_batch": seconds_per_batch * 1000.0,
            "ms_per_sample": seconds_per_sample * 1000.0,
            "samples_per_second": total_samples / elapsed_seconds if elapsed_seconds > 0 else float("inf"),
        }

    def inference(self, input_vector, starting_t=None, return_numpy=False):
        """
        Inference function that takes a vector input, runs the diffusion loop, and returns the denoised output.
        
        Args:
            input_vector: Input vector (can be numpy array or torch tensor). 
                         Can be 1D [sensor_dim] or 2D [batch_size, sensor_dim] or match input_shape.
            starting_t: Starting timestep for diffusion (default: n_steps - 1)
            return_numpy: If True, return numpy array; if False, return torch tensor (default: False)
        
        Returns:
            Denoised output vector with same shape as input (or numpy array if return_numpy=True)
        """
        if starting_t is None:
            starting_t = self.n_steps - 1
        
        # Convert to tensor if needed
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.tensor(input_vector, dtype=torch.float32)
        
        # Ensure input is on the correct device
        input_vector = input_vector.to(self.device)
        input_vector = self.normalize_tensor(input_vector)
        
        # Handle different input shapes
        original_shape = input_vector.shape
        if len(original_shape) == 1:
            # Single vector: [sensor_dim] -> [1, 1, sensor_dim] to match input_shape
            input_vector = input_vector.unsqueeze(0)  # [1, sensor_dim]
            batch_size = 1
            single_sample = True
        elif len(original_shape) == 2:
            # Batch of vectors: [batch_size, sensor_dim]
            batch_size = original_shape[0]
            single_sample = False
        else:
            # Already in the right shape (e.g., [batch, 1, sensor_dim])
            batch_size = original_shape[0]
            single_sample = False
            # Flatten to [batch, sensor_dim] if needed
            if len(original_shape) > 2:
                input_vector = input_vector.view(batch_size, -1)
        
        # Ensure the input matches the expected shape
        # The model expects [batch, 1, sensor_dim] based on input_shape
        if len(input_vector.shape) == 2:
            input_vector = input_vector.unsqueeze(1)  # [batch, 1, sensor_dim]
        
        with torch.no_grad():
            # Use adversarial data as input for diffusion process
            effective_steps = max(1, int(self.inference_steps))
            t = torch.full(size=(batch_size,), fill_value=min(starting_t, self.n_steps - 1),
                          device=input_vector.device, dtype=torch.long)
            if self.use_stochastic_renoise:
                base_t = max(0.0, min(float(starting_t) * self.renoise_strength, float(self.n_steps - 1)))
                xT = self.diffusion.origin_q_sample(input_vector, t, base_t=base_t)
            else:
                xT = input_vector.clone()
            xt = xT
            xt_next = None
            
            # Remove noise for $T$ steps
            for t_ in range(effective_steps):
                S_theta = self.eps_model(xt, t)
                # Simplified denoising step
                xt_next = xt - (1/effective_steps)*(xT - S_theta)
                
                # Advance to the next time step
                t_val = max(0, effective_steps - t_ - 1)
                t = torch.full(size=(batch_size,), fill_value=t_val, 
                              device=input_vector.device, dtype=torch.long)
                xt = xt_next
            
            # Extract output (remove extra dimensions if needed)
            output = xt
            if len(output.shape) > 2:
                # Reshape from [batch, 1, sensor_dim] to [batch, sensor_dim]
                output = output.squeeze(1)
            output = self.denormalize_tensor(output)
            
            # If single sample, remove batch dimension
            if single_sample:
                output = output.squeeze(0)
            
            # Convert to numpy if requested
            if return_numpy:
                output = output.cpu().numpy()
            
            return output

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
        target_path = self.adv_checkpoint_path if adv else self.checkpoint_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if(adv == True):
            torch.save(self.eps_model.state_dict(),self.adv_checkpoint_path)
        else:
            torch.save(self.eps_model.state_dict(),self.checkpoint_path)
            self._save_normalization_stats()


    def load_params(self,adv=False):
        if not adv:
            self._load_normalization_stats()
        if(adv == True):
            self.eps_model.load_state_dict(torch.load(self.adv_checkpoint_path, map_location=self.device))
        else:
            self.eps_model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.eps_model.to(self.device)
        self.set_inference_config(self.inference_mode, self.inference_steps, self.renoise_strength)
