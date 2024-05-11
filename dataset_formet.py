import tensorflow as tf
import pandas as pd

# load data from json file
with open('data.json', 'r') as f:
    json_data=json.load(f)

# Preprocess JSON data as needed
# For example, convert it into a list of dictionaries
json_records=[record for record in json_data]

#convert json data into pandas dataframe
df_json=pd.DataFrame(json_records)

#load data from csv file
df_csv=pd.read_csv('data.csv')##########################################################################

#convert json, csv data into tensorflow datasets
dataset_json=tf.data.Dataset.from_tensor_slices((df_json['features'].values, df_json['lables'].values))
dataset_csv=tf.data.Dataset.from_tensor_slices((df_csv['features'].values, df_csv['labels'].values))

batch_size=32
#shuffle and batch the datasets
dataset_json=dataset_json.shuffle(buffer_size=len(df_json)).batch(batch_size)
dataset_csv=dataset_csv.shuffle(buffer_size=len(dataset_csv)).batch(batch_size)