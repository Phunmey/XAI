import random
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from igraph import *
from ripser import ripser
import gudhi as gd
import gudhi.representations


random.seed(42)


def reading_csv(dataset):
    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']
    print("Graph edges are loaded")
    csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    print("Graph indicators are loaded")
    csv.columns = ["ID"]
    graph_indicators = (csv["ID"].values.astype(int))
    read_csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graph_labels = (read_csv["ID"].values.astype(int))
    print("Graph labels are loaded")
    unique_graph_indicator = np.arange(min(graph_indicators),
                                       max(graph_indicators) + 1)  # list unique graph ids

    return unique_graph_indicator, graph_indicators, df_edges, graph_labels


def silhoutte_train(unique_graph_indicator, graph_indicators, df_edges, dataset,
                    graph_labels):  # this is for the train test
    train_silhouette = []

    for graph_id in unique_graph_indicator:
        id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_label = [ele for ind, ele in enumerate(graph_labels, start=1) if
                       ind == graph_id]  # obtain graph label corresponding to the graph_id
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        create_dmatrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
        norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix[create_dmatrix != np.inf])

        start = time()
        train_rips = ripser(norm_dmatrix, thresh=1, maxdim=1, distance_matrix=True)[
            'dgms']
        silhouette_init = gd.representations.Silhouette(resolution=1000, weight=lambda x: 1)
        sil_houette = silhouette_init.fit_transform([train_rips[1]])
        silhouette_time = time() - start

        train_silhouette.append([dataset] + [graph_id] + graph_label + [silhouette_time] + sil_houette.tolist())

    feature_data = pd.DataFrame(train_silhouette)

    #    giving column names
    columnnames = {}  # create an empty dict
    count = -4  # initialize count to -1
    for i in feature_data.columns:
        count += 1  # update count by 1
        columnnames[i] = f"res_{count}"  # index i in dictionary will be named thresh_count

    # rename first and last column in the dictionary
    columnnames.update(
        {(list(columnnames))[0]: 'dataset', (list(columnnames))[1]: 'graphId', (list(columnnames))[2]: 'graphLabel',
         (list(columnnames))[3]: 'silhouetteTime', (list(columnnames))[4]: 'silhouetteList'})
    feature_data.rename(columns=columnnames, inplace=True)  # give column names to dataframe

    # write dataframe to file
    feature_data.to_csv("save dataframe", index=False)


def main():
    unique_graph_indicator, graph_indicators, df_edges, graph_labels = reading_csv(dataset)
    silhoutte_train(unique_graph_indicator, graph_indicators, df_edges, dataset, graph_labels)


if __name__ == '__main__':
    data_path = "path to data"  # dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for dataset in data_list:
        main()

