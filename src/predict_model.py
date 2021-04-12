import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from evaluations import evaluations

def generate_mmsi_samples(df, target, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    mmsi_list = df['mmsi'].unique()
    test_num = int(len(mmsi_list) * test_size)
    mmsi_test = np.random.choice(mmsi_list, size=test_num, replace=False)
    df_test = df[df['mmsi'].isin(mmsi_test)].drop('mmsi', axis=1)
    df_train = df[~df['mmsi'].isin(mmsi_test)].drop('mmsi', axis=1)
    return df_train.drop(target, axis=1), df_test.drop(target, axis=1), df_train[target], df_test[target]

def baeysian_opt_lgbm(train, target, init_iter=5, n_iters=20, random_state=42, seed=101):
    def lgb_f1_score(preds, dtrain):
        labels = dtrain.get_label()
        preds = preds.round(0)
        return 'f1', f1_score(labels, preds), True
    def hyp_lgbm(num_leaves, max_depth, bagging_fraction, colsample_bytree, learning_rate):
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_boosting_round': 300,
            'verbose': -1
        }
        params['num_leaves'] = int(round(num_leaves))
        params['max_depth'] = int(round(max_depth))
        params['bagging_fraction'] = bagging_fraction
        params['colsample_bytree'] = colsample_bytree
        params['learning_rate'] = learning_rate

        cv_results = lgb.cv(params, dtrain, nfold=5, seed=seed, feval=lgb_f1_score)
        return np.max(cv_results['f1-mean'])

    dtrain = lgb.Dataset(train, target)

    pds = {
        'num_leaves': (100, 300),
        'max_depth': (3, 7),
        'bagging_fraction': (0.7, 1),
        'colsample_bytree': (0.7, 1),
        'learning_rate': (0.01, 0.1)
    }
    optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)
    return optimizer

def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess

def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'error', np.mean(labels != (preds > 0.5)), False

def accuracy(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'accuracy', np.mean(labels == (preds > 0.5)), True

def bo_lgb_train(opt, x_train, y_train, x_test, y_test):
    num_train, num_feature = x_train.shape
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test)
    evals_result = {}
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_boosting_round': 300,
        'n_jobs': 2
    }
    params.update(opt.max['params'])
    params['num_leaves'] = int(round(params['num_leaves']))
    params['max_depth'] = int(round(params['max_depth']))

    feature_name=['f' + str(i + 1) for i in range(num_feature)]

    print('Start training...')

    model = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets=[lgb_train,lgb_eval],
                    feature_name = feature_name,
                    evals_result=evals_result,
                    fobj=loglikelihood,
                    feval=[binary_error,accuracy],
                    verbose_eval=10)

    model.save_model('model.txt', num_iteration = model.best_iteration)
    print('Plot metrics recorded during training...')
    ax = lgb.plot_metric(evals_result, metric='accuracy')
    plt.show()
    return model

if __name__=="__main__":
    df = pd.read_csv(os.path.join('..', 'data', 'preprocess_dataset_v1.csv'))
    x_train, x_test, y_train, y_test = generate_mmsi_samples(df, target='is_fishing', test_size=0.2, random_seed=42)
    opt = baeysian_opt_lgbm(x_train, y_train, n_iters=1)
    model = bo_lgb_train(opt, x_train, y_train, x_test, y_test)
    y_pred = model.predict(x_test, num_iteration = model.best_iteration)

    print("----------------------Evaluation results are as follows-------------",
                    evaluations(y_test, y_pred, proba=True))
