import pandas as pd
import os 

#Read in Adversarial Dataset
adv_file = os.path.join(os.getcwd(), "signal_processing", "data", "adversarial_obs_data.csv")
adversarial_df = pd.read_csv(adv_file, header=0)

#Read in Benign Dataset
ben_file = os.path.join(os.getcwd(), "signal_processing","data", "benign_obs_data.csv")
benign_df = pd.read_csv(ben_file, header=0)

# Drop the 'Unamed 0' Column
adversarial_df.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)
benign_df.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)

# Add Labels
adversarial_df['adversarial'] = 1
benign_df['adversarial'] = 0

# Create Joined Dataframe
combined_df = pd.concat([adversarial_df,benign_df])

# Save to file in data folder
combined_df.to_csv(os.path.join(os.getcwd(), "signal_processing","data", 'combined_labeled_obs_data.csv'))