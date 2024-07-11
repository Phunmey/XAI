import random
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from igraph import *
from ripser import ripser
import gudhi as gd
import gudhi.representations
import pickle

random.seed(42)


def silhoutte_train(graph_data, graph_ids, graph_label, dataset="molhiv", data_type="train"):  # this is for the train test
    train_silhouette = []

    for graph_id, label in zip(graph_ids, graph_label):
        graph = graph_data[graph_id]
        num_nodes = graph['num_nodes']
        num_edges = graph['num_edges']

        graph_edges = pd.DataFrame(graph['edge_index'].T, columns=[['from', 'to']])
        print(f"Shape of edgelist is {graph_edges.shape}")

        graph_label = label
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        create_dmatrix = np.asarray(Graph.distances(create_traingraph, algorithm='dijkstra'))
        norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix[create_dmatrix != np.inf])

        start = time()
        train_rips = ripser(norm_dmatrix, thresh=1, maxdim=1, distance_matrix=True)[
            'dgms']
        silhouette_init = gd.representations.Silhouette(resolution=1000, weight=lambda x: 1)
        sil_houette = silhouette_init.fit_transform([train_rips[1]])
        silhouette_time = time() - start

        train_silhouette.append(
            [dataset] + [graph_id] + [graph_label] + [num_nodes] + [num_edges] + [silhouette_time] + sil_houette.tolist())

    feature_data = pd.DataFrame(train_silhouette)

    #    giving column names
    columnnames = {}  # create an empty dict
    count = -6  # initialize count to -1
    for i in feature_data.columns:
        count += 1  # update count by 1
        columnnames[i] = f"res_{count}"  # index i in dictionary will be named res_count

    # rename first and last column in the dictionary
    columnnames.update(
        {(list(columnnames))[0]: 'dataset', (list(columnnames))[1]: 'graphId', (list(columnnames))[2]: 'graphLabel',
         (list(columnnames))[3]: 'num_nodes', (list(columnnames))[4]: 'num_edges',
         (list(columnnames))[5]: 'silhouetteTime', (list(columnnames))[6]: 'silhouetteList'})
    feature_data.rename(columns=columnnames, inplace=True)  # give column names to dataframe

    return feature_data

if __name__ == '__main__':
    readcsv = pd.read_csv('/home/taiwo/projects/def-cakcora/taiwo/src/ogbMolhiv/scaffold/molhiv_dataset/ogb_molhiv.csv')

    with open('/home/taiwo/projects/def-cakcora/taiwo/src/ogbMolhiv/scaffold/molhiv_dataset/graph_data.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    train_ids = pd.read_csv('/home/taiwo/projects/def-cakcora/taiwo/src/ogbMolhiv/scaffold/molhiv_dataset/ogbg_molhiv/split/scaffold/train.csv.gz').iloc[:, 0].tolist()
    val_ids = pd.read_csv('/home/taiwo/projects/def-cakcora/taiwo/src/ogbMolhiv/scaffold/molhiv_dataset/ogbg_molhiv/split/scaffold/valid.csv.gz').iloc[:, 0].tolist()
    test_ids = pd.read_csv('/home/taiwo/projects/def-cakcora/taiwo/src/ogbMolhiv/scaffold/molhiv_dataset/ogbg_molhiv/split/scaffold/test.csv.gz').iloc[:, 0].tolist()

    train_labels = readcsv[readcsv['graphId'].isin(train_ids)]['HIV_active'].tolist()
    val_labels = readcsv[readcsv['graphId'].isin(val_ids)]['HIV_active'].tolist()
    test_labels = readcsv[readcsv['graphId'].isin(test_ids)]['HIV_active'].tolist()

    train_features = silhoutte_train(graph_data, train_ids, train_labels, dataset="molhiv", data_type="train")
    val_features = silhoutte_train(graph_data, val_ids, val_labels, dataset="molhiv", data_type="val")
    test_features = silhoutte_train(graph_data, test_ids, test_labels, dataset="molhiv", data_type="test")

    train_features.to_csv(
        "/home/taiwo/projects/def-cakcora/taiwo/src/ogbMolhiv/ogbmolhiv_silhouette_features/train_silhouette_features.csv", index=False)
    val_features.to_csv(
        "/home/taiwo/projects/def-cakcora/taiwo/src/ogbMolhiv/ogbmolhiv_silhouette_features/val_silhouette_features.csv", index=False)
    test_features.to_csv(
        "/home/taiwo/projects/def-cakcora/taiwo/src/ogbMolhiv/ogbmolhiv_silhouette_features/test_silhouette_features.csv", index=False)
