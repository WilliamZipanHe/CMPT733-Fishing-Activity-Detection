import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from PIL import Image
import pandas as pd
import numpy


# this code is moved to scraping/scape_preprocess.py.
# However, the slice data function should still be used to cut down the data size before uploading to s3

# generate image file and convert to df
Image.MAX_IMAGE_PIXELS = None

im = Image.open('../data/distance-from-shore.tif')
im_array = numpy.array(im)
d_2_shore = pd.DataFrame(im_array)

im2 = Image.open('../data/distance-from-port-v1.tiff')
im_array2 = numpy.array(im2)
d_2_port = pd.DataFrame(im_array2)

# map shore distance to lat/lon
def d2s_function(x):
    num = d_2_shore.loc[round(x['lat']*100 + 9000)][round(x['lon']*100 + 18000)]
    return num

# map port distance to lat/lon
def d2p_function(x):
    num = d_2_port.loc[round(x['lat']*100 + 9000)][round(x['lon']*100 + 18000)]
    return num

# slice shore and port distance to smaller set to upload to s3 for reallife data (used only once)
def slice_data(df1, df2):
    df1 = df1.loc[4750+9000:5000+9000, -12750+18000:-12270+18000]
    df1.to_csv('../data/dfs_bc_section.csv')
    df2 = df2.loc[4750+9000:5000+9000, -12750+18000:-12270+18000]
    df2.to_csv('../data/dfp_bc_section.csv')
    sst_cut = pd.read_csv('../data/sst_grid.csv', index_col=0)
    sst_cut["time_bnds"] = pd.to_datetime(sst_cut["time_bnds"]).dt.to_period('M')
    sst_cut = sst_cut[(sst_cut['lat'] > 45) & (sst_cut['lat'] < 50) & (sst_cut['lon'] > -130) & \
                      (sst_cut['lon'] < -125) & (sst_cut['time_bnds'].dt.year == 2015)]
    sst_cut.to_csv('../data/sst_grid_sm.csv', index = False)
    precip_cut = pd.read_csv('../data/precip_grid.csv', index_col=0)
    precip_cut["time"] = pd.to_datetime(precip_cut["time"]).dt.to_period('M')
    precip_cut = precip_cut[(precip_cut['lat'] > 45) & (precip_cut['lat'] < 50) & (precip_cut['lon'] > -130) & \
                      (precip_cut['lon'] < -125) & (precip_cut['time'].dt.year == 2015)]
    precip_cut.to_csv('../data/precip_grid_sm.csv', index = False)

# setting general
def main():
    slice_data(d_2_shore, d_2_port)
    # df = pd.read_csv('../data/' + input_path)

    # getting distance data into df
    # df['distance_from_shore'] = df.apply(lambda x: d2s_function(x), axis=1)
    # df['distance_from_port'] = df.apply(lambda x: d2p_function(x), axis=1)

    # manually set source and gear data type
    # df['source_crowd_sourced'] = 0
    # df['source_dalhousie_longliner'] = 0
    # df['source_dalhousie_ps'] = 0
    # df['source_dalhousie_trawl'] = 0
    # df['source_false_positives'] = 0
    # df['source_gfw'] = 0
    # df['gear_type_drifting_longlines'] = 0
    # df['gear_type_fixed_gear'] = 0
    # df['gear_type_pole_and_line'] = 0
    # df['gear_type_purse_seines'] = 0
    # df['gear_type_trawlers'] = 1
    # df['gear_type_trollers'] = 0

    # df.to_csv('../data/' + vessel_name + '.csv', index = False)


if __name__ == '__main__':
    # input_path = sys.argv[1]
    # vessel_name = sys.argv[2]
    main()
