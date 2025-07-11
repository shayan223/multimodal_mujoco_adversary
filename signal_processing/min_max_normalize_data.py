import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


filename = "multi_benign_obs_data"
file_path = os.path.join(os.getcwd(), "signal_processing", "data", "mujoco_ant_obs_dataset/only_success_500k", f'{filename}.csv')
df = pd.read_csv(file_path)

# Split into features and label
X = df.drop(columns = 'adversarial', axis=1)
y = df['adversarial']

# MinMax Normalize the features
X = pd.DataFrame(MinMaxScaler().fit_transform(X)) 

# Add label
X['adversarial'] = y

# Save to csv
X.to_csv(os.path.join(os.getcwd(), "signal_processing","data", f'{filename}_normalized.csv'))


