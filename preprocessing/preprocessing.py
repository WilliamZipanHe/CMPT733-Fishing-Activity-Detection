import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


# fill missing values
def preprocess_missing_value(df):
    df['precip'] = df['precip'].fillna(0)
    df['sst'] = df['sst'].fillna(0)
    df['speed_rolling_mean'] = df['speed_rolling_mean'].bfill()
    df['course_rolling_mean'] = df['course_rolling_mean'].bfill()
    df['speed_rolling_mean_3h'] = df['speed_rolling_mean_3h'].bfill()
    df['course_rolling_mean_3h'] = df['course_rolling_mean_3h'].bfill()
    df['course'] = df.groupby(['mmsi'])['course'].ffill().bfill()
    df['speed'] = df.groupby(['mmsi'])['speed'].ffill().bfill()
    return df

# one hot encode string params
def preprocess_one_hot(df):
    df = pd.get_dummies(df, columns = ['hemisphere_ns'])
    df = pd.get_dummies(df, columns = ['hemisphere_ew'])

    if 'hemisphere_ns_north' not in df:
        df['hemisphere_ns_north'] = 0
    if 'hemisphere_ns_south' not in df:
        df['hemisphere_ns_south'] = 0
    if 'hemisphere_ew_east' not in df:
        df['hemisphere_ew_east'] = 0
    if 'hemisphere_ew_west' not in df:
        df['hemisphere_ew_west'] = 0

    # comment out for real life data--------------
    df = pd.get_dummies(df, columns = ['source'])
    df = pd.get_dummies(df, columns = ['gear_type'])
    return df

# scale all params and save scaler
def preprocess_scaling(df, scaler):
    # comment out for real life data--------------
    data = df.iloc[:, 1:]
    df[data.columns] = scaler.fit_transform(data)
    joblib.dump(scaler, '../scraping/scaler.gz')
    # comment out for training data--------------
    # df = df[['lat_x','lon_x','course','speed','timestamp','distance_from_shore','distance_from_port','mmsi','doy','dow','season','year','month','day','hour',\
    #     'sst','precip','speed_expand_mean','course_expand_mean','speed_rolling_mean','course_rolling_mean','speed_rolling_mean_3h','course_rolling_mean_3h',\
    #     'speed_rolling_mean_12h','course_rolling_mean_12h','speed_rolling_mean_1d','course_rolling_mean_1d','speed_rolling_mean_3d',\
    #     'course_rolling_mean_3d','source_crowd_sourced','source_dalhousie_longliner','source_dalhousie_ps','source_dalhousie_trawl',\
    #     'source_false_positives','source_gfw','gear_type_drifting_longlines','gear_type_fixed_gear','gear_type_pole_and_line','gear_type_purse_seines',\
    #     'gear_type_trawlers','gear_type_trollers','hemisphere_ns_north','hemisphere_ns_south','hemisphere_ew_east','hemisphere_ew_west'
    # ]]
    # scaler = joblib.load('scaler.gz')
    # df[df.columns] = scaler.transform(df)
    #---------------------------------------------
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
