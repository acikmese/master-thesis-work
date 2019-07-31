import pandas as pd
import numpy as np
import os
import scipy.stats as stats


def get_user_list(loc):
    # Collecting all student codes from activity folder (which represents all students)
    user_codes = []
    for x in sorted(os.listdir(loc + 'sensing/activity/')):
        # Chooses the string before "." and after "_"
        user_codes.append(x.split('.')[0].split('_')[1])
    return user_codes


### DATA READ FUNCTIONS BEGIN ###
## Sensing ##

# All functions starting with "get" takes the data according to function and 
# prepare for dataset creation process.
def get_activity(user, loc):
    # Reads the data.
    activity = pd.read_csv(loc + 'sensing/activity/activity_' + user + '.csv')
    # Change column names.
    activity.columns = ['timestamp', 'activity_inference']
    # make timestamp unique and take the mode for different values of activity inference.
    # mode is taken because it eliminates multiple rows for a timestamp. therefore, we will
    # have one value for each timestamp.
    activity = activity.groupby("timestamp")['activity_inference'].apply(lambda x: x.mode()[0]).reset_index()
    # convert timestamp time with seconds.
    activity.timestamp = pd.to_datetime(activity.timestamp, unit='s')
    # make timestamp index of rows.
    activity = activity.set_index('timestamp')
    # fill empty values with backward filling.
    activity = activity.asfreq('s', method='bfill')
    return activity

def get_audio(user, loc):
    audio = pd.read_csv(loc + 'sensing/audio/audio_' + user + '.csv')
    audio.columns = ['timestamp', 'audio_inference']
    # make timestamp unique and take the mode for different values of audio inference
    audio = audio.groupby("timestamp")['audio_inference'].apply(lambda x: x.mode()[0]).reset_index()
    audio.timestamp = pd.to_datetime(audio.timestamp, unit='s')
    audio = audio.set_index('timestamp')
    audio = audio.asfreq('s', method='bfill')
    return audio

def get_conversation(user, loc):
    conversation = pd.read_csv(loc + 'sensing/conversation/conversation_' + user + '.csv')
    conversation.columns = ['start_timestamp', 'end_timestamp']
    conversation.start_timestamp = pd.to_datetime(conversation.start_timestamp, unit='s')
    conversation.end_timestamp = pd.to_datetime(conversation.end_timestamp, unit='s')
    return conversation

def get_bluetooth(user, loc):
    bluetooth = pd.read_csv(loc + 'sensing/bluetooth/bt_' + user + '.csv', index_col=False)
    bluetooth.time = pd.to_datetime(bluetooth.time, unit='s')
    return bluetooth

def get_wifi(user, loc):
    wifi = pd.read_csv(loc + 'sensing/wifi/wifi_' + user + '.csv', index_col=False)
    wifi.time = pd.to_datetime(wifi.time, unit='s')
    return wifi

def get_dark(user, loc):
    dark = pd.read_csv(loc + 'sensing/dark/dark_' + user + '.csv', index_col=False)
    dark.start = pd.to_datetime(dark.start, unit='s')
    dark.end = pd.to_datetime(dark.end, unit='s')
    return dark

def get_phone_charge(user, loc):
    phonecharge = pd.read_csv(loc + 'sensing/phonecharge/phonecharge_' + user + '.csv', index_col=False)
    phonecharge.start = pd.to_datetime(phonecharge.start, unit='s')
    phonecharge.end = pd.to_datetime(phonecharge.end, unit='s')
    return phonecharge

def get_phone_lock(user, loc):
    phonelock = pd.read_csv(loc + 'sensing/phonelock/phonelock_' + user + '.csv', index_col=False)
    phonelock.start = pd.to_datetime(phonelock.start, unit='s')
    phonelock.end = pd.to_datetime(phonelock.end, unit='s')
    return phonelock

## Other than sensing ##

def get_sms(user, loc):
    sms = pd.read_csv(loc + 'sms/sms_' + user + '.csv', index_col=False)
    sms = sms[['timestamp']]
    sms = sms.groupby('timestamp').count().reset_index()
    sms['timestamp'] = pd.to_datetime(sms.timestamp, unit='s')
    sms['sms'] = 1
    sms = sms.set_index('timestamp')
    return sms

def get_call_log(user, loc):
    call_log = pd.read_csv(loc + 'call_log/call_log_' + user + '.csv', index_col=False)
    if 'CALLS_date' not in call_log.columns:
        call_log = call_log[['timestamp']]
        call_log = call_log.groupby('timestamp').count().reset_index()
        call_log['timestamp'] = pd.to_datetime(call_log.timestamp, unit='s')
        call_log['call_log'] = 1
        call_log = call_log.set_index('timestamp')
    else:
        call_log['timestamp'] = pd.to_datetime(call_log.timestamp, unit='s')
        call_log['CALLS_date'] = pd.to_datetime(call_log.CALLS_date, unit='ms')
        call_log = call_log[['timestamp', 'CALLS_date', 'CALLS_duration']]
        call_log = call_log.groupby(['timestamp', 'CALLS_date']).sum().reset_index()
        call_log['call_log'] = 1
    return call_log

# One time call this because it includes all of the users' deadlines.
def get_deadlines(loc):
    deadlines = pd.read_csv(loc + 'education/deadlines.csv', index_col=False)
    deadlines = deadlines.T
    deadlines.columns = deadlines.iloc[0, :]
    deadlines = deadlines.drop(deadlines.index[0])
    deadlines.index = pd.to_datetime(deadlines.index)
    return deadlines

def get_app_usage(user, loc):
    app = pd.read_csv(loc + 'app_usage/running_app_' + user + '.csv', index_col=False)
    app = app[['timestamp', 'RUNNING_TASKS_numRunning']]
    app = app.groupby('timestamp').sum().reset_index()
    app['timestamp'] = pd.to_datetime(app.timestamp, unit='s')
    app.columns = ['timestamp', 'running_apps']
    app = app.set_index('timestamp')
    return app

### DATA READ FUNCTIONS END ###


### DATA MERGER FUNCTIONS BEGIN ###
## Sensing ##

# All functions with "merge" adds each features to dataset.
# All merge operation is done based on timestamp.
# Activity is the main dataframe, all other data are merged on it.
# In this case, df = activity.
def merge_audio(df, audio):
    df = pd.merge(df, audio, left_index=True, right_index=True, how='outer')
    return df.reset_index()

def merge_conversation(df, conversation):
    # add conversation
    df['conversation'] = np.nan
    for i in range(conversation.shape[0]):
        start = conversation.iloc[i, 0]
        end = conversation.iloc[i, 1]
        df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end), 'conversation'] = 1
    return df

def merge_bluetooth(df, bluetooth):
    # add bluetooth
    bluetooth_new = pd.DataFrame()
    for time in bluetooth.time.unique():
        data = {'timestamp': time}
        item = bluetooth[bluetooth.time == time]
        data['total_devices_around'] = item.shape[0]
        data['total_nearer'] = item[(item.level >= -65) & (item.level <= 0)].shape[0]
        data['total_near'] = item[(item.level >= -80) & (item.level < -65)].shape[0]
        data['total_far'] = item[(item.level >= -90) & (item.level < -80)].shape[0]
        data['total_farther'] = item[(item.level >= -125) & (item.level < -90)].shape[0] # Normally -100 is max but for one anomaly.
        data['level_avg'] = round(item.level.mean())
        data['level_std'] = item.level.std()
        bluetooth_new = bluetooth_new.append(data, ignore_index=True)
    bluetooth_new.columns = ['bt_' + i for i in bluetooth_new.columns]
    df = pd.merge(df, bluetooth_new, left_on='timestamp', right_on='bt_timestamp', how='outer')
    df.drop(columns=['bt_timestamp'], inplace=True)
    return df

def merge_wifi(df, wifi):
    # add wifi
    wifi_new = pd.DataFrame()
    for time in wifi.time.unique():
        data = {'timestamp': time}
        item = wifi[wifi.time == time]
        data['total_devices_around'] = item.shape[0]
        data['total_nearer'] = item[item.level >= -60].shape[0]
        data['total_near'] = item[(item.level >= -80) & (item.level < -60)].shape[0]
        data['total_far'] = item[(item.level >= -100) & (item.level < -80)].shape[0]
        data['level_avg'] = round(item.level.mean())
        data['level_std'] = item.level.std()
        wifi_new = wifi_new.append(data, ignore_index=True)
    wifi_new.columns = ['wifi_' + i for i in wifi_new.columns]
    df = pd.merge(df, wifi_new, left_on='timestamp', right_on='wifi_timestamp', how='left')
    df.drop(columns=['wifi_timestamp'], inplace=True)
    return df

def merge_dark(df, dark):
    # add dark
    df['phone_in_dark'] = np.nan
    for i in range(dark.shape[0]):
        start = dark.iloc[i, 0]
        end = dark.iloc[i, 1]
        df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end), 'phone_in_dark'] = 1
    return df

def merge_phone_charge(df, phone_charge):
    # phone charge
    df['phone_charging'] = np.nan
    for i in range(phone_charge.shape[0]):
        start = phone_charge.iloc[i, 0]
        end = phone_charge.iloc[i, 1]
        df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end), 'phone_charging'] = 1
    return df

def merge_phone_lock(df, phone_lock):
    # phone locked
    df['phone_locked'] = np.nan
    for i in range(phone_lock.shape[0]):
        start = phone_lock.iloc[i, 0]
        end = phone_lock.iloc[i, 1]
        df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end), 'phone_locked'] = 1
    return df

## Other than sensing ##

def merge_sms(df, sms):
    df = pd.merge(df, sms, left_on='timestamp', right_index=True, how='left')
    return df

def merge_call_log(df, call_log):
    if 'CALLS_date' in call_log.columns:
        only_time = call_log[['timestamp', 'call_log']]
        only_time = only_time.set_index('timestamp')
        df = pd.merge(df, only_time, left_on='timestamp', right_index=True, how='left')
        call_dur = call_log[['CALLS_date', 'CALLS_duration']]
        call_dur.columns = ['CALLS_date', 'call_duration']
        call_dur = call_dur.set_index('CALLS_date')
        df = pd.merge(df, call_dur, left_on='timestamp', right_index=True, how='left')
        df.loc[df.call_duration.notnull(), 'call_log'] = 1
    else:
        df = pd.merge(df, call_log, left_on='timestamp', right_index=True, how='left')
    return df

def merge_deadlines(df, deadlines, user):
    df = df.set_index('timestamp')
    if user in deadlines.columns:
        user_deadlines = deadlines[user]
        for i in range(user_deadlines.shape[0]):
            df.loc[(df.index.month==user_deadlines.index[i].month) &
                   (df.index.day==user_deadlines.index[i].day), 
                   'deadlines'] = deadlines[user][i]
    else:
        df['deadlines'] = 0
    return df.reset_index()

def merge_app_usage(df, app_usage): # This is same with merge_sms but for understanding.
    df = pd.merge(df, app_usage, left_on='timestamp', right_index=True, how='left')
    return df


## Merge combiner ##
def merge_all(user_name, activity, audio, conversation, bluetooth, wifi, dark, 
              phone_charge, phone_lock, sms, call_log, deadlines, app_usage):
    # Firstly add hour of the day to the dataset.
    activity['hour_of_day'] = activity.index.hour
    df = merge_audio(activity, audio)
    print('audio merge completed.')
    df = merge_conversation(df, conversation)
    print('conversation merge completed.')
    df = merge_bluetooth(df, bluetooth)
    print('bluetooth merge completed.')
    df = merge_wifi(df, wifi)
    print('wifi merge completed.')
    df = merge_dark(df, dark)
    print('dark merge completed.')
    df = merge_phone_charge(df, phone_charge)
    print('phone_charge merge completed.')
    df = merge_phone_lock(df, phone_lock)
    print('phone_lock merge completed.')
    df = merge_sms(df, sms)
    print('sms merge completed.')
    df = merge_call_log(df, call_log)
    print('call_log merge completed')
    df = merge_deadlines(df, deadlines, user_name)
    print('deadlines merge completed')
    df = merge_app_usage(df, app_usage)
    print('app_usage merge completed')
    return df


### EMA Functions ###
# EMA functions to create labels.

def ema(user, typ, cols, loc):
    data = pd.read_json(loc + 'EMA/response/' + typ + '/' + typ + '_' + user + '.json')
    if 'null' in data.columns:
        data = data.drop(columns='null')
    if 'location' in data.columns:
        data = data.drop(columns='location')
    # Checks if the given columns exists in data, if exists drop them.
    dr = True 
    for j in cols:
        if j not in data.columns:
            dr = False
    if dr == True:
        data = data.dropna(subset=cols)
    return data

def label_generator(stress, mood2, resample_factor):
    stress['level'] = stress['level'].replace([1,2,3], 1)
    stress['level'] = stress['level'].replace([4,5], 0)
    stress.columns = ['STRESSED', 'resp_time']
    mood2 = mood2.replace([1, 3], 0)
    mood2 = mood2.replace([2], 1)
    mood2.columns = ['STRESSED', 'resp_time']
    labels = stress.append(mood2)
    labels = labels.sort_values(by='resp_time', ascending=True)
    labels = labels.set_index('resp_time').resample(resample_factor).max()
    labels = labels[labels.STRESSED.notnull()]
    return labels

def resample_aggregations(columns):
    # Function to assign aggregation method during resampling
    agg_dict = {}
    for i in columns:
        if ('conversation' in i) | ('phone' in i) | ('inference' in i):
            agg_dict[i] = np.sum
        elif 'level' in i:
            agg_dict[i] = np.mean
        elif 'total' in i:
            agg_dict[i] = np.max
    return agg_dict



def main():
    # Set dataset directory
    dir_loc = '../../student-life-study-data/dataset/'

    # Get user list
    user_codes = get_user_list(dir_loc)

    # Create empty dataset
    dataset = pd.DataFrame()
    
    deadlines = get_deadlines(dir_loc)
    
    for user in user_codes:
        # Sensing
        activity = get_activity(user, dir_loc)
        print('activity data read for', user, 'is completed.')
        audio = get_audio(user, dir_loc)
        print('audio data read for', user, 'is completed.')
        conversation = get_conversation(user, dir_loc)
        print('conversation data read for', user, 'is completed.')
        bluetooth = get_bluetooth(user, dir_loc)
        print('bluetooth data read for', user, 'is completed.')
        wifi = get_wifi(user, dir_loc)
        print('wifi data read for', user, 'is completed.')
        dark = get_dark(user, dir_loc)
        print('dark data read for', user, 'is completed.')
        phone_charge = get_phone_charge(user, dir_loc)
        print('phone_charge data read for', user, 'is completed.')
        phone_lock = get_phone_lock(user, dir_loc)
        print('phone_lock data read for', user, 'is completed.')

        # Not Sensing
        sms = get_sms(user, dir_loc)
        print('sms data read for', user, 'is completed.')
        call_log = get_call_log(user, dir_loc)
        print('call_log data read for', user, 'is completed.')
        app_usage = get_app_usage(user, dir_loc)
        print('app_usage data read for', user, 'is completed.')
        
        # EMA
        stress = ema(user, 'Stress', ['level'], dir_loc)
        print('stress data read for', user, 'is completed.')
        mood2 = ema(user, 'Mood 2', ['how'], dir_loc)
        print('mood2 data read for', user, 'is completed.')
        
        print('ALL DATA READ IS COMPLETED FOR', user)


        df = merge_all(user, activity, audio, conversation,
                        bluetooth, wifi, dark,
                        phone_charge, phone_lock,
                        sms, call_log, deadlines, app_usage)
        
        print('Shape of df after merge:', str(df.shape))
        print('DATA MERGE IS COMPLETED FOR', user)
        
        # Create labels
        if stress.shape[1] == 2:
            stress['level'] = stress['level'].replace([1,2,3], 1)
            stress['level'] = stress['level'].replace([4,5], 0)
            stress.columns = ['STRESSED', 'resp_time']

        if mood2.shape[1] == 2:
            mood2 = mood2.replace([1, 3], 0)
            mood2 = mood2.replace([2], 1)
            mood2.columns = ['STRESSED', 'resp_time']

        labels = stress.append(mood2)
        labels = labels.sort_values(by='resp_time', ascending=True)
        labels = labels.set_index('resp_time')
        
        # Choose only valid timestamps
        df = df[df.timestamp.notnull()]
        
        # Fill empty values in dataset.
        df.loc[:, ['activity_inference',
                   'audio_inference']] = df.loc[:, ['activity_inference',
                                                     'audio_inference']].fillna(value=3)
        
        df.loc[:, ['conversation',
            'phone_in_dark',
            'phone_charging',
            'phone_locked',
            'sms',
            'call_log']] = df.loc[:, ['conversation',
                                        'phone_in_dark',
                                        'phone_charging',
                                        'phone_locked',
                                        'sms',
                                        'call_log']].fillna(value=0)
        
        df.loc[:, ['activity_inference', 
           'audio_inference']] = df.loc[:, ['activity_inference', 
                                            'audio_inference']].astype(int)
        
        # One Hot Encode
        df['activity_inference'] = df['activity_inference'].astype('category')
        df['audio_inference'] = df['audio_inference'].astype('category')
        df = pd.get_dummies(df)
        
        # Resampling is not done because all resampling can be done afterwards.
#         # Resampling
#         df = df.set_index('timestamp')
#         df.index = pd.to_datetime(df.index, unit='s')
        
#         res_aggs = resample_aggregations(list(df.columns))
#         df = df.resample('10min').agg(res_aggs)

#         # Merge df and labels
#         df = pd.merge_asof(df, labels, left_index=True, right_index=True, tolerance=pd.Timedelta('10m'))
        
        df = pd.merge(df, labels, left_on='timestamp', right_index=True, how='left')
        
        df.to_csv('prepared_user_data_seconds/' + user + '_data.csv', index=False, header=True)
        
        print(user, 'IS COMPLETED.')
        print('Shape of df is:', str(df.shape))
        

main()
print("ALL COMPLETED.")