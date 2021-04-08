from sklearn.metrics import f1_score,recall_score,precision_score,\
                             fbeta_score,average_precision_score,auc,roc_curve


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
    if proba is True:
        y_pred_proba = y_pred
        y_pred = [1 if i >0.5 else 0 for i in y_pred]
        fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba, pos_label=1)
        result_dict['auc'] = auc(fpr, tpr)
        result_dict['auc_pr'] = average_precision_score(y_test,y_pred_proba)
    result_dict['f05'] = fbeta_score(y_test, y_pred, beta=0.5)
    result_dict['f2'] = fbeta_score(y_test, y_pred, beta=2)
    result_dict['f1'] = f1_score(y_test, y_pred)
    result_dict['precision'] = precision_score(y_test, y_pred)
    result_dict['recall'] = recall_score(y_test, y_pred)
    return result_dict


if __name__=="__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    evaluations(y_test, y_pred)
