import pandas as pd
import os 

#Read in Adversarial Dataset
adv_file = os.path.join(os.getcwd(), "mujoco_ant_obs_dataset", "velocity_500k", "veladversarial_obs_data.csv")
adversarial_df = pd.read_csv(adv_file, header=0)

#Read in Benign Dataset
ben_file = os.path.join(os.getcwd(), "mujoco_ant_obs_dataset", "velocity_500k", "velbenign_obs_data.csv" )
benign_df = pd.read_csv(ben_file, header=0)

# Drop the 'Unamed 0' Column because pandas automatically assigns indices
adversarial_df = adversarial_df.drop(columns = 'Unnamed: 0')
benign_df = benign_df.drop(columns = 'Unnamed: 0')

# Add Labels
adversarial_df['adversarial'] = 1
benign_df['adversarial'] = 0


# Sample Observations and Create Joined Dataframe
adversarial_sampled = adversarial_df.sample(n=5000, random_state=4321)
benign_sampled = benign_df.sample(n=5000, random_state=2341)
combined_df = pd.concat([adversarial_sampled,benign_sampled])

# # Create Joined Dataframe
# combined_df = pd.concat([adversarial_sampled,benign_sampled])

# Save to file in data folder
#combined_df.to_csv(os.path.join(os.getcwd(), "signal_processing","data", 'combined_labeled_obs_data.csv'))
combined_df.to_csv(os.path.join(os.getcwd(), "signal_processing","dataset_samples", "combined_velocity_data_5000.csv"))