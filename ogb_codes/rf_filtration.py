"""
description: this code reads all the filtration files and trains the rf classifier using each of these files
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
    train_features = pd.read_csv(train_file)
    val_features = pd.read_csv(val_file)
    test_features = pd.read_csv(test_file)

    # Prepare data for training
    X_train = train_features.drop(columns=['dataset', 'graphId', 'graphLabel', 'num_nodes', 'num_edges', 'filtrTime'])
    y_train = train_features['graphLabel']
    X_val = val_features.drop(columns=['dataset', 'graphId', 'graphLabel', 'num_nodes', 'num_edges', 'filtrTime'])
    y_val = val_features['graphLabel']
    X_test = test_features.drop(columns=['dataset', 'graphId', 'graphLabel', 'num_nodes', 'num_edges', 'filtrTime'])
    y_test = test_features['graphLabel']

    # Calculate sum of filtrTime
    filtrTime = round(train_features["filtrTime"].sum(), 3)

    return X_train, y_train, X_val, y_val, X_test, y_test, filtrTime


def tuning_hyperparameter():
    n_estimators = [int(a) for a in np.linspace(start=200, stop=500, num=5)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=6)]
    num_cv = 10
    bootstrap = [True, False]
    gridlength = len(n_estimators) * len(max_depth) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

    return param_grid, num_cv


def random_forest(filename, param_grid, x_train, x_val, y_train, y_val, x_test, y_test, num_cv, filtrTime, file):
    print(filename[0] + " training started at", datetime.now().strftime("%H:%M:%S"))
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

    print(f'rips complex training took {trainTime} seconds')

    flat_conf_mat_val = (str(val_conf_mat.flatten(order='C')))[1:-1]
    flat_conf_mat_test = (str(test_conf_mat.flatten(order='C')))[1:-1]

    file.write(
        f"{filename[1]}\t{filename[2]}\t{filename[3]}\t{filename[4]}\t{filtrTime}\t"
        f"{trainTime}\t{val_accuracy}\t{val_auc}\t{test_accuracy}\t{test_auc}\t{flat_conf_mat_val}\t"
        f"{flat_conf_mat_test}\n")

    file.flush()


def main(train_file, val_file, test_file, filename, file):
    x_train, y_train, x_val, y_val, x_test, y_test, filtrTime = read_data(train_file, val_file, test_file)
    param_grid, num_cv = tuning_hyperparameter()
    random_forest(filename, param_grid, x_train, x_val, y_train, y_val, x_test, y_test, num_cv, filtrTime, file)


if __name__ == "__main__":
    step_sizes = [10, 20, 50, 100]
    percentages = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    distances = ['spd', 'resistance']

    datapath = "path to flitration feature data"

    outputfile = "path to save results"
    with open(outputfile, "w") as file:
        header = 'filtration\tpercent\tdist_type\tstep_size\tfiltrTime\ttrainTime\t' \
                 'val_acc\tval_auc\ttest_acc\ttest_auc\tconf_val\tconf_test\n'
        file.write(header)

        for step_size in step_sizes:
            for perc in percentages:
                for distance in distances:
                    train_file = os.path.join(datapath, f"train_rips_{perc}_{distance}_{step_size}.csv")
                    val_file = os.path.join(datapath, f"val_rips_{perc}_{distance}_{step_size}.csv")
                    test_file = os.path.join(datapath, f"test_rips_{perc}_{distance}_{step_size}.csv")

                    filename1 = os.path.basename(train_file).split(".csv")[0]
                    filename = filename1.split('_')
                    for duplication in np.arange(10):
                        main(train_file, val_file, test_file, filename, file)
