# CMPT733-Fish-Activity-Detection
This project aims to predict fishing activity using our trained model from several known parameters including the sailing pattern of fishing vessels, location of the vessel, and fishing boundaries. From our prediction, we will be able to isolate vessels which display fishing activity but are not registered as an active fishing vessel. Researchers can furtheruse real-time satellite images/ AIS data as input to our trained model to classify vessel typeand identify potential illegal fishing activity, which would help protect the endangeredspecies.

## Fishing Activity Dashboard URL
https://fishing-activity-detection.herokuapp.com

## Run the Dashboard Locally
Please use the following commands from command line:
```
cd dashboard
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```


## EDA

### vessel AIS data EDA:
explore vessel initial AIS training data

- input: individual vessel data \
https://globalfishingwatch.org/data-download/datasets/public-training-data-v1 \
(example) ../data/troller.csv

- function: eda/eda_ais.ipynb

### vessel map data EDA:
explore vessel location and trajectory

- input: individual vessel data \
https://globalfishingwatch.org/data-download/datasets/public-training-data-v1 \
(example) ../data/troller.csv

- function: eda/eda_vessel_map.ipynb

## Data integration and feature engineering:

### Sea surface temperature integration:
reduce size of sst grid to rounded lat/lon and time range

- input: SST file (sst.wkmean.1990-present.nc) \
https://psl.noaa.gov/repository/entry/show?entryid=4abf55c8-335b-4117-b595-d7ce3d242f4f \
save as ../data/sst_noaa.nc \
optional input to test join: individual vessel data

- function: preprocesssing/feature_sst.ipynb

- output: ../data/sst_grid.csv \
optional output to test join: ../data/feature_1.csv

### Precipitation integration
reduce size of precipitation grid to rounded lat/lon and time range

- input: Precipitation file (precip.2012.nc - precip.2016.nc) \
https://psl.noaa.gov/repository/entry/show?entryid=5ca8e807-458f-4c69-9eb2-d99b865dcf97 \
save as ../data/sst_noaa.nc \
optional input to test join: ../data/feature_1.csv

- function: preprocesssing/feature_precip.ipynb

- output: ../data/precip_grid.csv \
optional output to test join: ../data/feature_2.csv

### distance to port and distance to shore integration and reduce size of grid:
integrate distance to port and distance to shore and reduce size of distance grid to rounded lat/lon for for sst, precipitation, distance to port/shore (parameters are hard coded to target specific range to accomodate our test download/scrape real life data)

- input: Distance to shore and distance to port image data \
https://globalfishingwatch.org/data-download/datasets/public-distance-from-port-v1 \
save as ../data/distance-from-port-v1.tiff \
https://globalfishingwatch.org/data-download/datasets/public-distance-from-shore-v1 \
save as ../data/distance-from-shore.tif \
sst_grid.csv (From Sea surface temperature integration) \
precip_grid.csv (From Precipitation temperature integration)

- function: preprocesssing/minimize_external_feature.py \
(example) python minimize_external_feature.py

- output: ../data/dfp_bc_section.csv, ../data/dfs_bc_section.csv, ../data/sst_grid_sm.csv, ../data/precip_grid_sm.csv

### Join external data and add new feature:
join sst and precip to current vessel data, and expand on original features

- input: individual vessel data, sst grid file, precipitation grid file \
https://globalfishingwatch.org/data-download/datasets/public-training-data-v1 \
../data/sst_grid.csv (From Sea surface temperature integration) \
../data/precip_grid.csv (From Precipitation temperature integration)

- function: preprocessing/feature_engineering.py \
(example) python feature_engineering.py trollers.csv sst_grid.csv precip_grid.csv v1

- output: (example) ../data/trollers_v1.csv

Note* \
The feature engineering of our data set is done seperatly on each vessel type, as we plan to train individual and combination models to evaluate the performance


### Join individual vessel dataset:
combine all individual datasets into a combined dataset (comment out not required gear type)

- input: all 6 vessel csv files \
(example) ../data/trollers_v1.csv (From Join external data and add new feature)

- function: src/joining_df.py \
(example) python joining_df.py trollers_v1.csv trawlers_v1.csv v1

- output: (example) ../data/combine_gear_v1.csv

### Preprocess data:
Fill missing value, one hot encode categoried features and apply scaler

- input: vessel csv files or combine vessel file after feature_engineering.py
(example) ../data/combine_gear_v1.csv (From Join individual vessel dataset)

- function: preprocessing/preprocessing.py \
(example) python preprocessing.py combine_gear_v1.csv v1

- output: preprocess_dataset_v1.csv, scaler.gz

## Examine datasets:

### External data joining test
join sst and precip to each vessel data with print dataframe(depricated! - moved to feature_engineering.py)

- input: individual vessel data, sst grid file, precipitation grid file \
(example) trollers.csv \
sst_grid.csv \
precip_grid.csv

- function: join_features_df.ipynb

### Examine preprocessed data:
test training preprocessing data and using the trained model to predict real life data:

- input: preprocessed data set to train and real life preprocessed data to visualize \
(example) preprocess_dataset_v1.csv \
(example) preprocess_dataset_rl_v1.csv (require download/scrap/api external dataset then transform)

- function: test_reallife_data_prediction.ipynb \

Note* \
current scaler and model is set up for all vessels and all features

## Training
To train the model with preprocessed data, run
```
python src/train.py --data ./data/preprocess_dataset_v2.csv --model lgb --method bayesian
```
We now offered lgb (LightGBM) and xgb (XGBoost) models to choose from. For the method argument, users can use `vanilla` or `bayesian`, where vanilla use all the hyperparameter as the default value. The results will be written to result/results.csv

## Predicting and Evaluating
To predict the model and evaluate the preprocessed data, run
```
python src/predict_model_use.py
```
We use LightGBM + Bayesian Optimization in this file to build our model, f1 score figure will be plotted out and evaluation result will be printed out in the end。

## Scraping:

### Scraping online data: 
our data is a combination of ais download from multiple sites plus scraping, the following is an example of 2 of the test scarper built

- input: none \
(credential required)

- function: scraping/scraper.ipynb

- outout: vessel_scrape.csv

### Feature engineering for scape/download vessel data
provide feature engineering external vessel AIS data, this was built to run on a schedular with s3 files connection which the dashboard can pull from

- input: from s3 scraped or downloaded vessel data, scaler from training model, current model, and the 2 cropped external distance to port and distance to shore datasets
(credential required) \
vessel_scrape.csv \
lgb.pkl \
scaler.gz \
sst_grid_sm.csv \
precip_grid_sm.csv \
dfp_bc_section.csv \
dfs_bc_section.csv

- function: scraping/scrape_preprocess.py
(example) python scrape_preprocess.py

- output: to s3

Note* \
current scaler and model is set up for all vessels and all features
