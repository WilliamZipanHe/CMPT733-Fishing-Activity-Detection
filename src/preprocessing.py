import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# fill missing values
def preprocess_missing_value(df):
    df['precip'] = df['precip'].fillna(0)
    df['speed_rolling_mean'] = df['speed_rolling_mean'].bfill()
    df['course_rolling_mean'] = df['course_rolling_mean'].bfill()
    df['speed_rolling_mean_3h'] = df['speed_rolling_mean_3h'].bfill()
    df['course_rolling_mean_3h'] = df['course_rolling_mean_3h'].bfill()
    df['course'] = df.groupby(['mmsi'])['course'].ffill().bfill()
    df['speed'] = df.groupby(['mmsi'])['speed'].ffill().bfill()
    return df

# one hot encode string params
def preprocess_one_hot(df):
    df = pd.get_dummies(df, columns = ['source'])
    df = pd.get_dummies(df, columns = ['gear_type'])
    df = pd.get_dummies(df, columns = ['hemisphere_ns'])
    df = pd.get_dummies(df, columns = ['hemisphere_ew'])
    return df

# scale all params
def preprocess_scaling(df, scaler):
    df[df.columns] = scaler.fit_transform(df)
    return df

def main(input_path, version):
    df = pd.read_csv('../data/'+ input_path)
    scaler = MinMaxScaler()
    del df['adjust_time_date']
    df = preprocess_missing_value(df)
    df = preprocess_one_hot(df)
    df = preprocess_scaling(df, scaler)
    print(df)
    df.to_csv('../data/preprocess_dataset_' + version + '.csv', index=False)


if __name__ == '__main__':
    input_path = sys.argv[1]
    version = sys.argv[2]
    main(input_path, version)
