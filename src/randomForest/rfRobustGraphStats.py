import glob
import os
import random
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

random.seed(123)


def read_data(sub_data):
    # calculate sum of graphTime
    graphTime = sub_data["graphTime"].sum()
    features = sub_data[sub_data.columns.difference(['graph_label', 'graphTime'])]  # select features
    labels = sub_data["graph_label"]

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    return x_train, y_train, x_test, y_test, graphTime


def tuning_hyperparameter():
    n_estimators = [int(a) for a in np.linspace(start=200, stop=500, num=5)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=6)]
    num_cv = 10
    bootstrap = [True, False]
    gridlength = len(n_estimators) * len(max_depth) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

    return param_grid, num_cv


def random_forest(element, param_grid, x_train, x_test, y_train, y_test, num_cv, graphTime):
    print(element + " training started at", datetime.now().strftime("%H:%M:%S"))
    start = time()

    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=num_cv, n_jobs=10)
    grid.fit(x_train, y_train)
    param_choose = grid.best_params_
    if len(set(y_test)) > 2:  # multiclass case
        print(element + " requires multi class RF")
        forest = RandomForestClassifier(**param_choose, verbose=1).fit(x_train, y_train)
        y_pred = forest.predict(x_test)
        y_preda = forest.predict_proba(x_test)
        auc = roc_auc_score(y_test, y_preda, multi_class="ovr")
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
    else:  # binary case
        rfc_pred = RandomForestClassifier(**param_choose, verbose=1).fit(x_train, y_train)
        test_pred = rfc_pred.predict(x_test)
        auc = roc_auc_score(y_test, rfc_pred.predict_proba(x_test)[:, 1])
        accuracy = accuracy_score(y_test, test_pred)
        conf_mat = confusion_matrix(y_test, test_pred)
    print(element + " accuracy is " + str(accuracy) + ", AUC is " + str(auc))

    t = time()
    trainTime = t - start

    print(f'graphStats complex training took {trainTime} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[1:-1]

    file.write(f"{element}\t{graphTime}\t{trainTime}\t{accuracy}\t{auc}\t{flat_conf_mat}\n")

    file.flush()


def main():
    x_train, y_train, x_test, y_test, graphTime = read_data(sub_data)
    param_grid, num_cv = tuning_hyperparameter()
    random_forest(element, param_grid, x_train, x_test, y_train, y_test, num_cv, graphTime)


if __name__ == "__main__":
    # Set input and output file paths
    datapath = "path to robust graph statsistics data"
    collect_files = os.path.join(datapath + "/*.csv")
    list_files = glob.glob(collect_files)
    for files in list_files:
        outputfile = os.path.join("path to output file")
        # Open output file for writing with header
        with open(outputfile, "w") as file:
            header = 'dataset\tgraphTime\ttrainTime\taccuracy\tauc\tflat_conf_mat\n'
            file.write(header)
            # Read input file and loop over unique dataset values
            csv_tda = pd.read_csv(files, header=0)
            for element in csv_tda["dataset"].unique():
                sub_data = csv_tda[csv_tda["dataset"] == element].drop(
                    ['dataset', 'graph_id', 'numNodes', 'numEdges', 'mindegree', 'maxdegree', 'motif1', 'motif2'],
                    axis=1)
                sub_data.dropna(axis=0, inplace=True)  # Remove rows with NaN values

                # Run random forest with hyperparameter tuning and write results to file
                for duplication in np.arange(10):
                    main()
