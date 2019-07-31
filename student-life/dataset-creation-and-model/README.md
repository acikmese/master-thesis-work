# GUIDE OF FILES

"SAVED_FINAL_MODELS" folder consists of the final models that we trained. We can read these saved models to generate same results for the dataset or we can run "LSTM-Keras-Tez.ipynb" to see the similar results.

## These are the files that were used for the preparation of the dataset and training the model

1. "1-dataset-preparation-seconds.py" prepares a dataset which includes both sensing and non-sensing data. (THIS IS THE DATA PREPARATION CODE FOR OUR FINAL MODEL). This creates features for each user and save them to "prepared_user_data_seconds" folder.
2. "2-user_samples_combiner-all.py" prepares different datasets to feed into the model for different resample values like 10 min, 15 min, etc. It saves the dataset into "combined_samples" folder. Therefore, we have multiple dataset with different resampling times to feed into our model. (THIS IS THE DATASET GENERATING CODE FOR OUR FINAL MODEL, WE READ THE DATA FROM THIS FOLDER FOR MODEL TRAINING).
3. "LSTM-Keras-Tez.ipynb" is the notebook of our final model. The final model was trained by using Keras Deep Learning library. We used the dataset with 30 min resampled data "combined_samples/combined_data_all_30min.csv".

---

### These files are similar with above. However, these are the first trials and have some shortcomings. THIS CAN BE SEEN AS BACKUP

1. "1-dataset-preparation-only-sensing.py" prepares a dataset which only includes sensing data from mobile phone for each user. (NOT USED FOR FINAL MODEL - THIS ONLY INCLUDES SENSING DATA FOR OUR FIRST TRIAL). This creates features for each user and save them to "prepared_user_data" folder.

2. "2-user_samples_combiner-from-prepared-data.py" prepares a single dataset to feed into the model. This functions does not resample time. It saves the dataset into "combined_samples" folder.

---

"LGBM.ipynb" is the notebook for LightGBM model. It is not used in our final work, but we tried a simple gradient boosting model to see how it performs.

---

#### There are some other folders that are not mentioned above is unnecessary but keeped for backup. You can ignore "ignore-to-remove" and "ignore-backup" folders.