import os
import re
import csv
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, fbeta_score, average_precision_score, auc,roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

def evaluations(y_test, y_pred, proba=True):
    '''
    input:
        y_test: a list or numpy 1d array. The actual labels.
        y_pred: a list or numpy 1d array. The predicted results.
        proba: True or False. If True, y_pred are a collection of probabilities. If False, y_pred are a collection of 1/0.
    output:
        a dictionary containing results of multiple evaluation metrics. The keys are the names of metrics and the values are numeric values.
    '''
    result_dict = {}
    result_dict['roc_auc_score'] = roc_auc_score(y_test, y_pred)
    result_dict['auc_pr'] = average_precision_score(y_test, y_pred)
    if proba is True:
        y_pred = [1 if i >0.5 else 0 for i in y_pred]
    result_dict['f05'] = fbeta_score(y_test, y_pred, beta=0.5)
    result_dict['f2'] = fbeta_score(y_test, y_pred, beta=2)
    result_dict['f1'] = f1_score(y_test, y_pred)
    result_dict['precision'] = precision_score(y_test, y_pred)
    result_dict['recall'] = recall_score(y_test, y_pred)
    result_dict['accuracy'] = accuracy_score(y_test, y_pred)
    return result_dict

def lgb_vanilla(train, target):
    data = lgb.Dataset(train, target)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': -1
    }
    model = lgb.train(params, data)
    return model

def xgb_vanilla(train, target):
    params = {
        'objective': 'binary:logistic',
    }
    data = xgb.DMatrix(train, target)
    model = xgb.train(params, data, num_boost_round=200)
    return model

def baeysian_opt_lgbm(train, target, init_iter=5, n_iters=20, random_state=42, seed=101):
    def lgb_f1_score(preds, dtrain):
        labels = dtrain.get_label()
        preds = preds.round(0)
        return 'f1', f1_score(labels, preds), True
    def hyp_lgbm(num_leaves, max_depth, bagging_fraction, colsample_bytree, learning_rate):
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1
        }
        params['num_leaves'] = int(round(num_leaves))
        params['max_depth'] = int(round(max_depth))
        params['bagging_fraction'] = bagging_fraction
        params['colsample_bytree'] = colsample_bytree
        params['learning_rate'] = learning_rate

        cv_results = lgb.cv(params, dtrain, nfold=5, num_boost_round=100, seed=seed, feval=lgb_f1_score)
        return np.max(cv_results['f1-mean'])

    dtrain = lgb.Dataset(train, target)

    pds = {
        'num_leaves': (100, 300),
        'max_depth': (5, 10),
        'bagging_fraction': (0.7, 1),
        'colsample_bytree': (0.7, 1),
        'learning_rate': (0.3, 0.5)
    }
    optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)
    return optimizer

def bo_lgb_train(opt, train, target):
    data = lgb.Dataset(train, target)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_jobs': 2
    }
    params.update(opt.max['params'])
    params['num_leaves'] = int(round(params['num_leaves']))
    params['max_depth'] = int(round(params['max_depth']))
    model = lgb.train(params, data, num_boost_round=100)
    return model

def baeysian_opt_xgbm(train, target, init_iter=5, n_iters=20, random_state=42, seed=101):
    def xgb_f1_score(preds, dtrain):
        labels = dtrain.get_label()
        preds = preds > 0.5
        return 'f1', f1_score(labels, preds)
    def hyp_xgbm(max_depth, subsample, colsample_bytree, colsample_bylevel):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
        }
        params['max_depth'] = int(round(max_depth))
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        params['colsample_bylevel'] = colsample_bylevel

        cv_results = xgb.cv(params, dtrain, nfold=5, seed=seed, feval=xgb_f1_score)
        return cv_results['test-f1-mean'].iloc[-1]

    dtrain = xgb.DMatrix(train, target)

    pds = {
        'max_depth': (3, 5),
        'subsample': (0.7, 1),
        'colsample_bytree': (0.7, 1),
        'colsample_bylevel': (0.7, 1)
    }
    optimizer = BayesianOptimization(hyp_xgbm, pds, random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)
    return optimizer

def bo_xgb_train(opt, train, target):
    data = xgb.DMatrix(train, target)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',

    }
    params.update(opt.max['params'])
    params['max_depth'] = int(round(params['max_depth']))
    model = xgb.train(params, data, num_boost_round=100)
    return model

def save_evaluations(result, trial_name, result_file='results.csv'):
    result['trial_name'] = trial_name
    csv_columns = list(result.keys())
    result = result,
    if os.path.isfile(result_file):
        with open(result_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for data in result:
                writer.writerow(data)
    else:
    	with open(result_file, 'w') as csvfile:
    		writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    		writer.writeheader()
    		for data in result:
    			writer.writerow(data)

def main(args):
    df = pd.read_csv(args.data)
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    x_train, x_test, y_train, y_test = train_test_split(df.drop(["is_fishing"], axis=1), df["is_fishing"], test_size=0.2, random_state=100)
    if args.model == 'lgb':
        if args.method == 'vanilla':
            model = lgb_vanilla(x_train, y_train)
        if args.method == 'bayesian':
            opt = baeysian_opt_lgbm(x_train, y_train, n_iters=10)
            model = bo_lgb_train(opt, x_train, y_train)
    if args.model == 'xgb':
        x_test = xgb.DMatrix(x_test)
        if args.method == 'vanilla':
            model = xgb_vanilla(x_train, y_train)
        if args.method == 'bayesian':
            opt = baeysian_opt_xgbm(x_train, y_train, n_iters=10)
            model = bo_xgb_train(opt, x_train, y_train)
    y_pred = model.predict(x_test)
    eval_result = evaluations(y_test, y_pred)
    save_evaluations(eval_result, trial_name= args.model + "+" + args.method)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser("train the binary classification model")
    parser.add_argument("--data", required=True, help="train data in .csv format")
    parser.add_argument("--model", choices=['lgb', 'xgb'], required=True, help="choose between lgb or xgb")
    parser.add_argument("--method", choices=['vanilla', 'bayesian'], required=True, help="vanilla or bayesian")
    args = parser.parse_args()
    main(args)
