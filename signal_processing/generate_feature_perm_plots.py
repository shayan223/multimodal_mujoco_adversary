import torch
from captum.attr import FeaturePermutation, KernelShap
from attack_detector import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import signal_processing.generate_shap_plots as generate_shap_plots

# ----------------- Initialize Model -----------------
checkpoint = torch.load("trained_models/nn_angular_015.pt")
model = adv_detector_nn(29)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load states 
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Put model in eval mode
model.eval()

# Set Seeds
torch.manual_seed(13)
np.random.seed(13)
 

# ----------------- Load data -----------------
# Load 007 Data
path = os.path.join(os.getcwd(), 'mujoco_ant_obs_dataset/fgsm015_data/fgsm015_angular/angular_fgsm015_')
full_dataset = adv_obs_dataset(f"{path}adversarial_obs_data.csv", f"{path}benign_obs_data.csv")
input_dim = full_dataset.data.shape[1]

# Split into train and val, set up val loader
indices = list(range(len(full_dataset)))
batch_size = 500
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=12)
val_subset = torch.utils.data.Subset(full_dataset, val_indices)
val_loader = DataLoader(val_subset, batch_size=batch_size)

# Select a batch
batch_x, batch_y = next(iter(val_loader))

# ----------------- Define method, calculate attribution-----------------
feature_perm = FeaturePermutation(model)
attr = feature_perm.attribute(batch_x, target=0)

# ----------------- Visualize mean attributions -----------------

mean_attributions = attr.mean(dim=0).squeeze().cpu().numpy()
plt.figure(figsize=(10, 8))

# Define a color-blind friendly palette
# Using a palette with distinct hues for positive and negative values
# Example: Blue for positive, Orange for negative
# These colors are generally safe for common forms of color blindness.
colors = ['#1f77b4' if val >= 0 else '#ff7f0e' for val in mean_attributions]
plt.bar(np.arange(len(mean_attributions)), mean_attributions, color=colors)

# Add a legend
# We create two dummy artists for the legend
# Create new patches for the updated legend
positive_patch = mpatches.Patch(color='#1f77b4', label='Positive Contribution (Feature increases likelihood of being adversarial)')
negative_patch = mpatches.Patch(color='#ff7f0e', label='Negative Contribution (Feature decreases likelihood of being adversarial)')

# Update the legend
plt.legend(handles=[positive_patch, negative_patch], loc='best', title='Contribution to Attack Detection')

# plt.title(f'FCN Velocity Detector (0.015)\nAverage Feature Importance Across Batch (Feature Permutation)\nBatch Size = {batch_size}')
# plt.title(f'FCN Angular Attack Detector (0.007)\nAverage Feature SHAP Value\nBatch Size = {batch_size}')
plt.xlabel('Feature Index')
plt.ylabel('Average Attribution Score')
plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()
plt.savefig("/Users/DaniJusto/Desktop/Feature Importance/nn_angular_015_fp.png")