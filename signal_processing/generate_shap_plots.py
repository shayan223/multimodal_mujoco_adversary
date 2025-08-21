import torch
from attack_detector import *
import matplotlib.pyplot as plt
import numpy as np
import shap
import matplotlib.patches as mpatches

# --- Load Model ---
# Load Model and set it to evaluation mode
checkpoint = torch.load("trained_models/nn_velocity_015.pt")
model = adv_detector_nn(29)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Wrapper Function ---
# This wrapper now takes the model as an argument and handles data types correctly
def model_predict_wrapper(x):
    # SHAP passes a NumPy array, so convert it to a PyTorch tensor
    tensor_x = torch.from_numpy(x)
    
    # Get the model's raw outputs (logits), not the final predictions
    with torch.no_grad():
        outputs = model(tensor_x).squeeze(1)
        
    # Return the raw outputs as a NumPy array
    return outputs.numpy()

def visualize():
    path = os.path.join(os.getcwd(), 'mujoco_ant_obs_dataset/fgsm015_data/fgsm015_velocity/velocity_fgsm015_')
    full_dataset = adv_obs_dataset(f"{path}adversarial_obs_data.csv", f"{path}benign_obs_data.csv")

    indices = list(range(len(full_dataset)))
    batch_size = 5
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=12)
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # Select Batches and convert them to numpy arrays for SHAP
    train_batch_x, _ = next(iter(train_loader))
    test_batch_x, _ = next(iter(val_loader))

    explainer_background = train_batch_x.numpy()
    to_explain = test_batch_x.numpy()

    # --- Initialize and run the Explainer ---
    # The wrapper is now a function reference, not a call, and the background data is a NumPy array
    explainer = shap.KernelExplainer(model_predict_wrapper, explainer_background)

    # Calculate SHAP values for the data you want to explain
    shap_values = explainer.shap_values(to_explain)

    plt.figure(figsize=(10, 8)) 
    
    # Summary plot
    shap.summary_plot(shap_values, to_explain, plot_type="summary", show=False)

    # Get the current axes and set title
    ax = plt.gca()
    plt.show()

    # Force plot for a single data point
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0, :], to_explain[0, :])

if __name__ == '__main__':
    visualize()