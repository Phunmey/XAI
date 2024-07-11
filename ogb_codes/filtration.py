"""
Obtain betti features for ogb datasets using shortest-path and resistance distances respectively.
"""

import random
import pandas as pd
import numpy as np
import math
import networkx as nx
import pickle
from datetime import datetime
from time import time
from igraph import *
from ripser import ripser

random.seed(42)


def rips_filt(graph_data, graph_ids, graph_label, step_size, perc, dataset="molhiv", distance="spd", data_type="train"):
    betti_list = []
    dMatrix = None
    for graph_id, label in zip(graph_ids, graph_label):
        graph = graph_data[graph_id]
        num_nodes = graph['num_nodes']
        num_edges = graph['num_edges']

        graph_edges = pd.DataFrame(graph['edge_index'].T, columns=[['from', 'to']])
        print(f"Shape of edgelist is {graph_edges.shape}")

        graph_label = label

        if distance == "spd":
            createGraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
            deleteEdge = random.sample(createGraph.get_edgelist(), math.ceil(perc * (createGraph.ecount() / 2)))
            filtered_list = [edgeTuple for edgeTuple in createGraph.get_edgelist() if edgeTuple not in deleteEdge]
            graphPrime = Graph()
            graphPrime.add_vertices(num_nodes)
            graphPrime.add_edges(filtered_list)
            dMatrix = np.asarray(Graph.distances(graphPrime, algorithm='dijkstra'))
        elif distance == "resistance":
            createGraph = nx.from_pandas_edgelist(graph_edges, 'from', 'to')
            createGraph.remove_edges_from(
                random.sample(list(createGraph.edges()), math.ceil(perc * createGraph.number_of_edges())))
            graphNodes = createGraph.number_of_nodes()

            if not nx.is_connected(createGraph):
                restMatrices = []
                subgraphSizes = []
                for component in nx.connected_components(createGraph):
                    subgraph = createGraph.subgraph(component)
                    subgraphNodes = subgraph.number_of_nodes()
                    laplace = nx.laplacian_matrix(subgraph).toarray()
                    dMatrix = np.zeros((subgraphNodes, subgraphNodes))
                    auxMatrix = np.ones((subgraphNodes, subgraphNodes))
                    sumMatrix = laplace + ((1 / subgraphNodes) * auxMatrix)
                    invMatrix = np.linalg.pinv(sumMatrix)
                    for node1 in range(subgraphNodes):
                        for node2 in range(subgraphNodes):
                            if node1 != node2 and laplace[node1, node1] != 0 and laplace[node2, node2] != 0:
                                dMatrix[node1, node2] = invMatrix[node1, node1] + invMatrix[node2, node2] - (
                                        2 * invMatrix[node1, node2])

                    restMatrices.append(dMatrix)
                    subgraphSizes.append(subgraphNodes)

                dMatrix = np.zeros((sum(subgraphSizes), sum(subgraphSizes)))
                start = 0
                for subg in range(len(subgraphSizes)):
                    end = start + subgraphSizes[subg]
                    dMatrix[start:end, start:end] = restMatrices[subg]
                    if subg < len(subgraphSizes) - 1:
                        dMatrix[end:, :end] = np.full((sum(subgraphSizes) - end, end), np.inf)
                        dMatrix[:end, end:] = np.full((end, sum(subgraphSizes) - end), np.inf)
                    start = end
            else:
                laplace = nx.laplacian_matrix(createGraph).toarray()
                dMatrix = np.zeros((graphNodes, graphNodes))
                auxMatrix = np.ones((graphNodes, graphNodes))
                sumMatrix = laplace + ((1 / graphNodes) * auxMatrix)
                invMatrix = np.linalg.pinv(sumMatrix)
                for node1 in range(graphNodes):
                    for node2 in range(graphNodes):
                        if node1 != node2 and laplace[node1, node1] != 0 and laplace[node2, node2] != 0:
                            dMatrix[node1, node2] = invMatrix[node1, node1] + invMatrix[node2, node2] - (
                                    2 * invMatrix[node1, node2])
        else:
            print("this distance is not defined for:" + " " + str(graph_id))

        normalizedMatrix = (dMatrix - np.min(dMatrix, axis=0)) / (
                np.max(dMatrix[dMatrix != np.inf], axis=0) - np.min(dMatrix, axis=0))

        start = time()
        train_rips = ripser(normalizedMatrix, thresh=1, maxdim=1, distance_matrix=True)['dgms']
        filtr_time = time() - start

        train_dgm_0 = train_rips[0]
        train_dgm_1 = train_rips[1]

        filename = "save persistence dgms"
        with open(filename, "a") as f:
            f.write(f"{graph_id}: {train_dgm_0, train_dgm_1}\n")
            f.flush()
            f.close()

        train_betti_0 = []
        train_betti_1 = []

        for eps in np.linspace(0, 1, step_size):
            b_0 = 0
            for k in train_dgm_0:
                if k[0] <= eps and k[1] > eps:
                    b_0 += 1
            train_betti_0.append(b_0)

            b_1 = 0
            for l in train_dgm_1:
                if l[0] <= eps and l[1] > eps:
                    b_1 += 1
            train_betti_1.append(b_1)

        betti_list.append([dataset] + [graph_id] + [graph_label] + [num_nodes] + [num_edges] + [
            filtr_time] + train_betti_0 + train_betti_1)

    feature_data = pd.DataFrame(betti_list)

    columnnames = {}
    count = -6
    for i in feature_data.columns:
        count += 1
        columnnames[i] = f"eps_{count}"

    columnnames.update({
        (list(columnnames))[0]: 'dataset', (list(columnnames))[1]: 'graphId', (list(columnnames))[2]: 'graphLabel',
        (list(columnnames))[3]: 'num_nodes', (list(columnnames))[4]: 'num_edges', (list(columnnames))[5]: 'filtrTime'
    })

    feature_data.rename(columns=columnnames, inplace=True)

    return feature_data


if __name__ == '__main__':
    readcsv = pd.read_csv('path to ogb csv file')

    with open('path to graph_data.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    train_ids = pd.read_csv('path to train.csv.gz').iloc[:, 0].tolist()
    val_ids = pd.read_csv('path to valid.csv.gz').iloc[:, 0].tolist()
    test_ids = pd.read_csv('path to test.csv.gz').iloc[:, 0].tolist()

    train_labels = readcsv[readcsv['graphId'].isin(train_ids)]['HIV_active'].tolist()
    val_labels = readcsv[readcsv['graphId'].isin(val_ids)]['HIV_active'].tolist()
    test_labels = readcsv[readcsv['graphId'].isin(test_ids)]['HIV_active'].tolist()

    step_sizes = [10, 20, 50, 100]
    percentages = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    distances = ['spd', 'resistance']

    for step_size in step_sizes:
        for per in percentages:
            print(f"Processing with step size: {step_size} and perc: {per}")

            for distance in distances:
                train_features = rips_filt(graph_data, train_ids, train_labels, perc=per, step_size=step_size,
                                           dataset="molhiv", distance=distance, data_type="train")
                val_features = rips_filt(graph_data, val_ids, val_labels, perc=per, step_size=step_size,
                                         dataset="molhiv", distance=distance, data_type="val")
                test_features = rips_filt(graph_data, test_ids, test_labels, perc=per, step_size=step_size,
                                          dataset="molhiv", distance=distance, data_type="test")

                train_features.to_csv("save training features", index=False)
                val_features.to_csv("save validation features", index=False)
                test_features.to_csv("save test features", index=False)
