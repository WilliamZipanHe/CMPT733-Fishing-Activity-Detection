import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
import joblib
import boto3


# sample inactive key, replace with bucket active key to run
session = boto3.Session(
    aws_access_key_id='AKIAW7MBQ4XXXXXXXXXX',
    aws_secret_access_key='LSTUcAAXXXXXXXXXXXXXXXXXXXXXXXXX',
)

s3 = session.resource('s3')

# add external dataset to new data -------------------------------------------------------------------
# function to get specific distance to port and distance to shore base on coordinates
def d2s_function(x, d_2_shore):
    num = d_2_shore.loc[:,str(round(x['lon']*100 + 18000))][round(x['lat']*100 + 9000)]
    return num

def d2p_function(x, d_2_port):
    num = d_2_port.loc[:,str(round(x['lon']*100 + 18000))][round(x['lat']*100 + 9000)]
    return num

# init input df - sea surface temparature
def init_sst_df(sst_df):
    sst_df["time_bnds"] = pd.to_datetime(sst_df["time_bnds"]).dt.to_period('M')
    return sst_df

# init input df - precipitation
def init_precip_df(precip_df):
    precip_df["time"] = pd.to_datetime(precip_df["time"]).dt.to_period('M')
    return precip_df

# merge sst and precip to new data
def combine_df_lon_lat(fishing_df, sst_df, precip_df):
    fishing_df['adjust_lat'] = fishing_df['lat'].apply(lambda x: custom_round(x))
    fishing_df['adjust_lon'] = fishing_df['lon'].apply(lambda x: custom_round(x))

    sst_df = sst_df[(sst_df['time_bnds'].dt.year == 2015) & (sst_df['time_bnds'].dt.month == 4)]
    precip_df = precip_df[(precip_df['time'].dt.year == 2015) & (precip_df['time'].dt.month == 4)]
    df_all = pd.merge(fishing_df, sst_df,  how='left', \
                      left_on=['adjust_lat','adjust_lon'], \
                      right_on = ['lat','lon'])
    df_all = pd.merge(df_all, precip_df,  how='left', \
                      left_on=['adjust_lat','adjust_lon'], \
                      right_on = ['lat','lon'])
    df_all = df_all.drop(columns=['adjust_time', 'adjust_lon', 'adjust_lat', 'time', 'lat', 'lon', 'lat_y','lon_y', 'time_bnds'])
    return df_all

# preprocess new data: missing value and onehot encoding-------------------------------------------------------------------
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

    df['source_crowd_sourced'] = 0
    df['source_dalhousie_longliner'] = 0
    df['source_dalhousie_ps'] = 0
    df['source_dalhousie_trawl'] = 0
    df['source_false_positives'] = 0
    df['source_gfw'] = 0
    df['gear_type_drifting_longlines'] = 0
    df['gear_type_fixed_gear'] = 0
    df['gear_type_pole_and_line'] = 0
    df['gear_type_purse_seines'] = 0
    df['gear_type_trawlers'] = 1
    df['gear_type_trollers'] = 0
    return df

# preprocess new data: scaling-------------------------------------------------------------------
# reorganize and scale all params
def preprocess_scaling(df, scaler):
    df = df[['lat_x','lon_x','course','speed','timestamp','distance_from_shore','distance_from_port','mmsi','doy','dow','season','year','month','day','hour',\
        'sst','precip','speed_expand_mean','course_expand_mean','speed_rolling_mean','course_rolling_mean','speed_rolling_mean_3h','course_rolling_mean_3h',\
        'speed_rolling_mean_12h','course_rolling_mean_12h','speed_rolling_mean_1d','course_rolling_mean_1d','speed_rolling_mean_3d',\
        'course_rolling_mean_3d','source_crowd_sourced','source_dalhousie_longliner','source_dalhousie_ps','source_dalhousie_trawl',\
        'source_false_positives','source_gfw','gear_type_drifting_longlines','gear_type_fixed_gear','gear_type_pole_and_line','gear_type_purse_seines',\
        'gear_type_trawlers','gear_type_trollers','hemisphere_ns_north','hemisphere_ns_south','hemisphere_ew_east','hemisphere_ew_west'
    ]]
    df[df.columns] = scaler.transform(df)
    return df

# add in additional feature base on existing data-------------------------------------------------------------------
# custom rounding functions
def custom_season(x):
    return np.round(int(x)/3)

def custom_round(x):
    return 0.5 + np.floor(float(x))

# add time features
def time_feature(df):
    df["adjust_time_date"] = pd.to_datetime(df['timestamp'], unit='s')
    df["adjust_time"] = pd.to_datetime(df["adjust_time_date"]).dt.to_period('M')
    df["doy"] = df["adjust_time_date"].dt.dayofyear
    df["dow"] = df["adjust_time_date"].dt.dayofweek
    df["season"] = df["adjust_time_date"].dt.month.apply(lambda x: custom_season(x))
    df["year"] = df["adjust_time_date"].dt.year
    df["month"] = df["adjust_time_date"].dt.month
    df["day"] = df["adjust_time_date"].dt.day
    df["hour"] = df["adjust_time_date"].dt.hour
    return df

# add hemisphere features
def hemisphere_feature(df):
    df['hemisphere_ns'] = np.where(df['lat_x'] > 0, 'north', 'south')
    df['hemisphere_ew'] = np.where(df['lon_x'] > 0, 'east', 'west')
    return df

# add expanding mean feature
def expanding_mean(df, param):
    df_expand_mean = df.groupby('mmsi')[param].expanding().mean().reset_index(0).sort_index()
    del df_expand_mean['mmsi']
    new_column = param + '_expand_mean'
    df_expand_mean.columns = [new_column]
    df = df.join(df_expand_mean)
    return df

# add rolling mean features
def rolling_mean(df, param):
    df_rolling_mean = df.groupby('mmsi')[param].rolling(5).mean().reset_index(0).sort_index()
    del df_rolling_mean['mmsi']
    new_column = param + '_rolling_mean'
    df_rolling_mean.columns = [new_column]
    df = df.join(df_rolling_mean)
    return df

def rolling_mean_with_time(df, param, time_string):
    df_rolling_mean_time = df.set_index(['adjust_time_date']).groupby(by='mmsi', sort=False)[param].rolling(time_string).mean().reset_index().sort_index()
    del df_rolling_mean_time['mmsi']
    del df_rolling_mean_time['adjust_time_date']
    new_column = param + '_rolling_mean_' + time_string
    df_rolling_mean_time.columns = [new_column]
    df = df.join(df_rolling_mean_time)
    return df

# Master function to manipulate new data and reupload to s3-----------------------------
def main():
    print('load files from s3')
    s3.Bucket('fishing733').download_file('vessel_scrape.csv', 'vessel_scrape.csv')
    df = pd.read_csv('vessel_scrape.csv')
    s3.Bucket('fishing733').download_file('dfp_bc_section.csv', 'dfp_bc_section.csv')
    d_2_port = pd.read_csv('dfp_bc_section.csv', index_col=0)
    s3.Bucket('fishing733').download_file('dfs_bc_section.csv', 'dfs_bc_section.csv')
    d_2_shore = pd.read_csv('dfs_bc_section.csv', index_col=0)
    s3.Bucket('fishing733').download_file('sst_grid_sm.csv', 'sst_grid.csv')
    sst_df = pd.read_csv('sst_grid.csv')
    s3.Bucket('fishing733').download_file('precip_grid_sm.csv', 'precip_grid.csv')
    precip_df = pd.read_csv('precip_grid.csv')
    s3.Bucket('fishing733').download_file('scaler.gz', 'scaler.gz')
    scaler = joblib.load('scaler.gz')
    s3.Bucket('fishing733').download_file('lgb.pkl', 'lgb.pkl')
    lgb_model = joblib.load('lgb.pkl')

    print('feature engineering...')
    df['distance_from_shore'] = df.apply(lambda x: d2s_function(x, d_2_shore), axis=1)
    df['distance_from_port'] = df.apply(lambda x: d2p_function(x, d_2_port), axis=1)
    df['source_crowd_sourced'] = 0
    df['source_dalhousie_longliner'] = 0
    df['source_dalhousie_ps'] = 0
    df['source_dalhousie_trawl'] = 0
    df['source_false_positives'] = 0
    df['source_gfw'] = 0
    df['gear_type_drifting_longlines'] = 0
    df['gear_type_fixed_gear'] = 0
    df['gear_type_pole_and_line'] = 0
    df['gear_type_purse_seines'] = 0
    df['gear_type_trawlers'] = 1
    df['gear_type_trollers'] = 0
    sst_df = init_sst_df(sst_df)
    precip_df = init_precip_df(precip_df)
    df = time_feature(df)
    df = combine_df_lon_lat(df, sst_df, precip_df) # comment out if don't need
    df = hemisphere_feature(df)
    df = expanding_mean(df, 'speed')
    df = expanding_mean(df, 'course')
    df = rolling_mean(df, 'speed')
    df = rolling_mean(df, 'course')
    df = rolling_mean_with_time(df, 'speed', '3h')
    df = rolling_mean_with_time(df, 'course', '3h')
    df = rolling_mean_with_time(df, 'speed', '12h')
    df = rolling_mean_with_time(df, 'course', '12h')
    df = rolling_mean_with_time(df, 'speed', '1d')
    df = rolling_mean_with_time(df, 'course', '1d')
    df = rolling_mean_with_time(df, 'speed', '3d')
    df = rolling_mean_with_time(df, 'course', '3d')

    print('preprocessing...')
    del df['adjust_time_date']
    df = preprocess_missing_value(df)
    df = preprocess_one_hot(df)
    df = preprocess_scaling(df, scaler)

    print('predicting...')
    new_df = df.copy()
    new_df[df.columns] = scaler.inverse_transform(df)
    del df['mmsi']
    del df['timestamp']
    del df['source_gfw']
    del df['source_false_positives']
    del df['source_dalhousie_trawl']
    del df['source_dalhousie_ps']
    del df['source_dalhousie_longliner']
    del df['source_crowd_sourced']
    pred_test = lgb_model.predict(df)
    new_df['pred'] = pred_test

    # print final df and upload back to s3
    print(new_df)
    print('upload file back to s3 vessel_data.csv')
    new_df.to_csv('vessel_data.csv', index=False)
    s3.Bucket('fishing733').upload_file('vessel_data.csv', 'vessel_data.csv')


if __name__ == '__main__':
    main()
