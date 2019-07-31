import pandas as pd
import numpy as np
import os

# Set user data dir
datadir = 'prepared_user_data/'

# Save dir
savedir = 'combined_samples/'

# Get files in dir
files = sorted(os.listdir(datadir))

# Create empty dataframe to add new samples
combined_data = pd.DataFrame()

# For each user extract each sample and add to combined_data.
for user in files:
    df = pd.read_csv(datadir + user)
    
    # Checks if the labels exist.
    if 'STRESSED' not in df.columns:
        print(user, 'has no label data.')
        continue
        
    # Extract indexes of labeled rows.
    label_indexes = list(df[df.STRESSED.notnull()].index)

    # This loop checks the difference between two label indexes.
    # If the difference is lower than treshold (12, two hours),
    # It continues to next index until pass the treshold.
    # Therefore, every instance's lentgh become more than treshold.
    start = 0
    new_index_ranges = []
    for i in label_indexes:
        if i - start >= 12:
            new_index_ranges.append([start, i+1])
            start = i+1
        else:
            continue

    # Extract each sample from the user data and append to combined_data.
    for ranges in new_index_ranges:
        sample = df.iloc[ranges[0]: ranges[1], :]
        combined_data = combined_data.append(sample, ignore_index=True, sort=False)
    print(user, 'is completed.')

combined_data.to_csv(savedir + 'combined_data.csv')
print('ALL COMPLETED.')