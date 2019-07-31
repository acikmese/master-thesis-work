import pandas as pd
import numpy as np
import os

def resample_aggregations(columns):
    # Function to assign aggregation method during resampling.
    agg_dict = {}
    for i in columns:
        if 'level' in i:
            agg_dict[i] = np.mean
        elif 'total' in i:
            agg_dict[i] = np.max
        else:
            agg_dict[i] = np.sum
    return agg_dict


def resample_data(df, labels, res_range='10min'):
    # Resamples data according to res_range and it chooses which calculation will be done
    # for each features based on resample_aggregations fuction.
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    res_aggs = resample_aggregations(list(df.columns))
    df = df.resample(res_range).agg(res_aggs)
    df = pd.merge_asof(df, labels, left_index=True, right_index=True, tolerance=pd.Timedelta(res_range))
    return df


# Set user data dir
datadir = 'prepared_user_data_seconds/'

# Save dir
savedir = 'combined_samples/'

# Get files in dir
files = sorted(os.listdir(datadir))

# Create empty dataframe to add new samples
combined_data = pd.DataFrame()

for res_range in ['10min', '15min', '20min', '30min', '45min', '60min']:
    # For each user extract each sample and add to combined_data.
    for user in files:
        df = pd.read_csv(datadir + user)

        # Checks if the labels exist.
        if 'STRESSED' not in df.columns:
            print(user, 'has no label data.')
            continue

        labels = df.loc[df.STRESSED.notnull(), ['timestamp', 'STRESSED']]
        labels = labels.set_index('timestamp')
        labels.index = pd.to_datetime(labels.index)

        df = df.drop(columns=['hour_of_day', 'STRESSED'])

        df = resample_data(df, labels, res_range=res_range)

        df['hour_of_day'] = df.index.hour

        df = df.reset_index()

        # Extract indexes of labeled rows.
        label_indexes = list(df[df.STRESSED.notnull()].index)

        # This loop checks the difference between two label indexes.
        # If the difference is lower than treshold (12, two hours),
        # It continues to next index until pass the treshold.
        # Therefore, every instance's length become more than treshold.
        
        # Create different tresholds for each resample range to have minimum 2 hour period.
        tresholds = {'10min': 12, '15min': 8, '20min': 6, '30min': 4, '45min': 2, '60min': 2}
        
        start = 0
        new_index_ranges = []
        for i in label_indexes:
            if i - start >= tresholds[res_range]:
                new_index_ranges.append([start, i+1])
                start = i+1
            else:
                continue

        # Extract each sample from the user data and append to combined_data.
        for ranges in new_index_ranges:
            sample = df.iloc[ranges[0]: ranges[1], :]
            combined_data = combined_data.append(sample, ignore_index=True, sort=False)
        print(user, 'is completed.')

    combined_data.to_csv(savedir + 'combined_data_all_' + res_range + '.csv')
    print('All Completed for', res_range)
    combined_data = pd.DataFrame()
print('ALL COMPLETED.')