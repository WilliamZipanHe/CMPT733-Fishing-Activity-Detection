import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

import pandas as pd
import numpy as np
from pathlib import Path


# init input df - fishing gear
def init_fishing_df(path):
    fishing_df = pd.read_csv('../data/' + path)
    fishing_df = fishing_df[fishing_df['is_fishing'] > -0.5]
    fishing_df['is_fishing'] = [0 if x < 0.3 else 1 for x in fishing_df['is_fishing']]
    fishing_df = fishing_df[['is_fishing', 'lat', 'lon', 'course', 'speed', 'timestamp', 'distance_from_shore', 'distance_from_port', 'mmsi', 'source']]
    fishing_df['gear_type'] = Path(path).stem
    return fishing_df


# ------------------------This section only needed when adding sst/precip data-----------------------------
# init input df - sea surface temparature
def init_sst_df(path_sst):
    sst_df = pd.read_csv('../data/' + path_sst, index_col=0)
    sst_df["time_bnds"] = pd.to_datetime(sst_df["time_bnds"]).dt.to_period('M')
    return sst_df

# init input df - precipitation
def init_precip_df(path_precip):
    precip_df = pd.read_csv('../data/' + path_precip, index_col=0)
    precip_df["time"] = pd.to_datetime(precip_df["time"]).dt.to_period('M')
    return precip_df


# ------------------------This section only needed when slicing lon/lat or time-----------------------------
# custom rounding functions
def custom_season(x):
    return np.round(int(x)/3)

def custom_round(x):
    return 0.5 + np.floor(float(x))


# ------------------------Functions to combine/add features and feature engineering-----------------------------
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
    df = df.drop(columns=['timestamp'])
    return df

def hemisphere_feature(df):
    df['hemisphere_ns'] = np.where(df['lat_x'] > 0, 'north', 'south')
    df['hemisphere_ew'] = np.where(df['lon_x'] > 0, 'east', 'west')
    return df

def combine_df_lon_lat(fishing_df, sst_df, precip_df):
    fishing_df['adjust_lat'] = fishing_df['lat'].apply(lambda x: custom_round(x))
    fishing_df['adjust_lon'] = fishing_df['lon'].apply(lambda x: custom_round(x))
    df_all = pd.merge(fishing_df, sst_df,  how='left', \
                      left_on=['adjust_lat','adjust_lon', 'adjust_time'], \
                      right_on = ['lat','lon', 'time_bnds'])
    df_all = pd.merge(df_all, precip_df,  how='left', \
                      left_on=['adjust_lat','adjust_lon', 'adjust_time'], \
                      right_on = ['lat','lon', 'time'])
    df_all = df_all.drop(columns=['adjust_time', 'adjust_lon', 'adjust_lat', 'time', 'lat', 'lon', 'lat_y','lon_y', 'time_bnds'])
    return df_all

def expanding_mean(df, param):
    df_expand_mean = df.groupby('mmsi')[param].expanding().mean().reset_index(0).sort_index()
    del df_expand_mean['mmsi']
    new_column = param + '_expand_mean'
    df_expand_mean.columns = [new_column]
    df = df.join(df_expand_mean)
    return df

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


# ------------------------Master function to combine features df (each gear type)-----------------------------
# with sst and precip
def main(path, path_sst, path_precip, version):
    fishing_df = init_fishing_df(path)
    sst_df = init_sst_df(path_sst)
    precip_df = init_precip_df(path_precip)

    fishing_df = time_feature(fishing_df)
    fishing_df = combine_df_lon_lat(fishing_df, sst_df, precip_df) # comment out if don't need (need time feature function)
    fishing_df = hemisphere_feature(fishing_df)
    fishing_df = expanding_mean(fishing_df, 'speed')
    fishing_df = expanding_mean(fishing_df, 'course')
    fishing_df = rolling_mean(fishing_df, 'speed')
    fishing_df = rolling_mean(fishing_df, 'course')
    fishing_df = rolling_mean_with_time(fishing_df, 'speed', '3h')
    fishing_df = rolling_mean_with_time(fishing_df, 'course', '3h')
    fishing_df = rolling_mean_with_time(fishing_df, 'speed', '12h')
    fishing_df = rolling_mean_with_time(fishing_df, 'course', '12h')
    fishing_df = rolling_mean_with_time(fishing_df, 'speed', '1d')
    fishing_df = rolling_mean_with_time(fishing_df, 'course', '1d')
    fishing_df = rolling_mean_with_time(fishing_df, 'speed', '3d')
    fishing_df = rolling_mean_with_time(fishing_df, 'course', '3d')
    print(fishing_df)
    fishing_df.to_csv('../data/' + Path(path).stem + '_' + version + '.csv')


if __name__ == '__main__':
    fishing_path = sys.argv[1]
    sst_path = sys.argv[2] # comment out if don't need
    precip_path = sys.argv[3] # comment out if don't need
    version = sys.argv[4]
    main(fishing_path, sst_path, precip_path, version)
