#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd





with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)



def prepare_data(dataframe, columns):
    dataframe['duration'] = dataframe.dropOff_datetime - dataframe.pickup_datetime
    dataframe['duration'] = dataframe.duration.dt.total_seconds() / 60

    dataframe = dataframe[(dataframe.duration >= 1) & (dataframe.duration <= 60)].copy()

    dataframe[columns] = dataframe[columns].fillna(-1).astype('int').astype('str')

    return dataframe

def read_data(filename ,categorical):
    df = pd.read_parquet(filename)
    
    df = prepare_data(df, categorical)
    
    return df


def main(year, month):

    input_file = f'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'taxi_type=fhv_year={year:04d}_month={month:02d}.parquet'

    categorical = ['PUlocationID', 'DOlocationID']

    df = read_data(input_file, categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)


if __name__ == "__main__":

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)