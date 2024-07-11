"""
description: read the silhouette csv files and train the randomforest
"""

import random
from datetime import datetime
from time import time
import numpy as np
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV

random.seed(123)


def read_data(train_file, val_file, test_file):
    # Load features from CSV
    train_features = pd.read_csv(train_file, delimiter=',').drop(columns=['dataset', 'graphId', 'num_nodes', 'num_edges'])
    val_features = pd.read_csv(val_file, delimiter=',').drop(columns=['dataset', 'graphId', 'num_nodes', 'num_edges'])
    test_features = pd.read_csv(test_file, delimiter=',').drop(columns=['dataset', 'graphId', 'num_nodes', 'num_edges'])

    # Calculate sum of filtrTime
    silhouetteTime = round(train_features["silhouetteTime"].sum() + val_features["silhouetteTime"].sum() + test_features[
        "silhouetteTime"].sum(), 3)
    print(f"Silhouette time: {silhouetteTime}")

    y_train = train_features['graphLabel']
    y_val = val_features['graphLabel']
    y_test = test_features['graphLabel']

    train_features_ = train_features.drop(columns=['graphLabel', 'silhouetteTime'])
    val_features_ = val_features.drop(columns=['graphLabel', 'silhouetteTime'])
    test_features_ = test_features.drop(columns=['graphLabel', 'silhouetteTime'])

    x_train = train_features_["silhouetteList"].str.strip('[]').str.split(",", n=-1, expand=True)
    x_val = val_features_["silhouetteList"].str.strip('[]').str.split(",", n=-1, expand=True)
    x_test = test_features_["silhouetteList"].str.strip('[]').str.split(",", n=-1, expand=True)

    return x_train, y_train, x_val, y_val, x_test, y_test, silhouetteTime


def tuning_hyperparameter():
    n_estimators = [int(a) for a in np.linspace(start=2, stop=5, num=5)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=6)]
    num_cv = 10
    bootstrap = [True, False]
    gridlength = len(n_estimators) * len(max_depth) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

    return param_grid, num_cv


def random_forest(param_grid, x_train, x_val, y_train, y_val, x_test, y_test, num_cv, silhouetteTime, filename='molhiv'):
    start = time()

    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=num_cv, n_jobs=10)
    grid.fit(x_train, y_train)
    param_choose = grid.best_params_

    rfc_pred = RandomForestClassifier(**param_choose, verbose=1).fit(x_train, y_train)

    val_pred = rfc_pred.predict(x_val)
    test_pred = rfc_pred.predict(x_test)

    val_auc = round(roc_auc_score(y_val, rfc_pred.predict_proba(x_val)[:, 1]), 3)
    test_auc = round(roc_auc_score(y_test, rfc_pred.predict_proba(x_test)[:, 1]), 3)

    val_accuracy = round(accuracy_score(y_val, val_pred), 3)
    test_accuracy = round(accuracy_score(y_test, test_pred), 3)

    val_conf_mat = confusion_matrix(y_val, val_pred)
    test_conf_mat = confusion_matrix(y_test, test_pred)

    t = time()
    trainTime = round(t - start, 3)

    print(f'Silhouette training took {trainTime} seconds')

    flat_conf_mat_val = (str(val_conf_mat.flatten(order='C')))[1:-1]
    flat_conf_mat_test = (str(test_conf_mat.flatten(order='C')))[1:-1]

    file.write(
        f"{filename}\t{silhouetteTime}\t{trainTime}\t{val_accuracy}\t{val_auc}\t{test_accuracy}\t{test_auc}\t{flat_conf_mat_val}\t"
        f"{flat_conf_mat_test}\n")

    file.flush()


def main(train_file, val_file, test_file):
    x_train, y_train, x_val, y_val, x_test, y_test, silhouetteTime = read_data(train_file, val_file, test_file)
    param_grid, num_cv = tuning_hyperparameter()
    random_forest(param_grid, x_train, x_val, y_train, y_val, x_test, y_test, num_cv, silhouetteTime, filename='molhiv')


if __name__ == "__main__":
    datapath = "path to silhouette_features"
    outputfile = "path to save silhouette training results"
    with open(outputfile, "w") as file:
        header = 'filename\tsilhouetteTime\ttrainTime\tval_acc\tval_auc\ttest_acc\ttest_auc\tconf_val\tconf_test\n'
        file.write(header)

        train_file = os.path.join(datapath, f"train_silhouette_features.csv")
        val_file = os.path.join(datapath, f"val_silhouette_features.csv")
        test_file = os.path.join(datapath, f"test_silhouette_features.csv")

        for duplication in np.arange(10):
            main(train_file, val_file, test_file)

    file.close()
