#!/usr/bin/env python
# coding: utf-8

# In[26]:


# In[27]:


import pickle
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True ,help='Year of parquet file')
parser.add_argument('--month', type=int, required=True ,help='Month of parquet file')

args = parser.parse_args()

year = args.year
month = args.month
# In[28]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[29]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[36]:

df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')


# In[37]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[38]:


print(sum(y_pred) / len(y_pred))


# In[33]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[34]:


df_result = df[['ride_id']]
df_result['predict'] = y_pred


# In[35]:


df_result.to_parquet(
    "./result.parquet",
    engine='pyarrow',
    compression=None,
    index=False
)

