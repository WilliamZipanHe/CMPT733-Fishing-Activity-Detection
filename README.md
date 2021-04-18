# CMPT733-Fish-Activity-Detection
This project aims to predict fishing activity using our trained model from several known parameters including the sailing pattern of fishing vessels, location of the vessel, and fishing boundaries. From our prediction, we will be able to isolate vessels which display fishing activity but are not registered as an active fishing vessel. Researchers can furtheruse real-time satellite images/ AIS data as input to our trained model to classify vessel typeand identify potential illegal fishing activity, which would help protect the endangeredspecies.

# EDA

### vessel AIS data EDA:
explore vessel initial AIS training data

- input: individual vessel data
https://globalfishingwatch.org/data-download/datasets/public-training-data-v1

- function: eda/eda_ais.ipynb

### vessel mmsi data EDA:
input: 

### vessel map data EDA:
explore vessel location and trajectory

- input: individual vessel data
https://globalfishingwatch.org/data-download/datasets/public-training-data-v1

- function: eda/eda_vessel_map.ipynb

# Data integration and feature engineering:

### Sea surface temperature integration:
reduce size of sst grid to rounded lat/lon and time range

- input: SST file (sst.wkmean.1990-present.nc)
https://psl.noaa.gov/repository/entry/show?entryid=4abf55c8-335b-4117-b595-d7ce3d242f4f
save as sst_noaa.nc

- function: preprocesssing/feature_sst.ipynb

- output: sst_grid.csv

### Precipitation integration
reduce size of precipitation grid to rounded lat/lon and time range

- input: Precipitation file (precip.2012.nc - precip.2016.nc)
https://psl.noaa.gov/repository/entry/show?entryid=5ca8e807-458f-4c69-9eb2-d99b865dcf97
save as sst_noaa.nc

- function: preprocesssing/feature_precip.ipynb

- output: precip_grid.csv

### distance to port and distance to shore integration:
reduce size of distance grid to rounded lat/lon

- input: Distance to shore and distance to port image data
https://globalfishingwatch.org/data-download/datasets/public-distance-from-port-v1
save as distance-from-port-v1.tiff
https://globalfishingwatch.org/data-download/datasets/public-distance-from-shore-v1
save as distance-from-shore.tif

- function: preprocesssing/feature_engineering_new_vessel.py

- output: dfp_bc_section.csv, dfs_bc_section.csv

### Join external data and add new feature:
join sst and precip to current vessel data, and expand on original features

- input: individual vessel data, sst grid file, precipitation grid file
https://globalfishingwatch.org/data-download/datasets/public-training-data-v1
sst_grid.csv
precip_grid.csv

- function: preprocessing/feature_engineering.py
(example) python feature_engineering.py trollers.csv sst_grid.py precip_grid.py v1

- output: (example) trollers_v1.csv

Note*
The feature engineering of our data set is done seperatly on each vessel type, as we plan to train individual and combination models to evaluate the performance


### Join individual vessel dataset:
combine all individual datasets into a combined dataset

- input: all vessel csv files
(example) trollers_v1.csv

- function: src/joining_df.py
(example) python joining_df.py trollers_v1.csv trawlers_v1.csv v1

- output: combine_vessel_v1.csv

### Preprocess data:
Fill missing value, one hot encode categoried features and apply scaler

- input: vessel csv files or combine vessel file after feature_engineering.py

- function: preprocessing/preprocessing.py
python preprocessing.py combine_vessel.csv v1

- output: preprocess_dataset_v1.csv, scaler.gz

# Examine datasets:

### External data joining test
join sst and precip to each vessel data

- input: individual vessel data, sst grid file, precipitation grid file
(example) trollers.csv
sst_grid.csv
precip_grid.csv

- function: join_features_df.ipynb

### Examine preprocessed data:
test training preprocessing data and using the trained model to predict real life data:

- input: preprocess_dataset_v1.csv

- function: test_reallife_data_prediction.ipynb

# Scraping:

### Scraping online data: 
our data is a combination of ais download from multiple sites plus scraping, the following is an example of 2 of the test scarper built

- input: none
(credential required)

- function: scraping/scraper.ipynb

- outout: vessel_scrape.csv

### Feature engineering for scape/download vessel data
provide feature engineering external vessel AIS data, this was built to run on a schedular with s3 files connection which the dashboard can pull from

- input: from s3 scraped or downloaded vessel data, scaler from training model, current model, and the 2 cropped external distance to port and distance to shore datasets
(credential required)
vessel_scrape.csv
lgb.pkl
scaler.gz
dfp_bc_section.csv
dfs_bc_section.csv

- function: scraping/scrape_preprocess.py
(example) 

- output: to s3
