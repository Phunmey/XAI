"""
Compute the count of 0, 1, and 2-dimensional holes in the dataset
"""

import random
import sys
import numpy as np
import pandas as pd
from igraph import *
from ripser import ripser
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import networkx as nx

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
                                       max(graph_indicators) + 1)  # list unique graph ids in a dataset

    return unique_graph_indicator, graph_indicators, df_edges


def betti_stats(unique_graph_indicator, graph_indicators, df_edges, perc, dataset, distance="spd"):
    bettis = {}
    dMatrix = None
    min_degree = None
    max_degree = None
    for graph_id in unique_graph_indicator:
        id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        if distance == "spd":
            createGraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
            deleteEdge = random.sample(createGraph.get_edgelist(),
                                       math.ceil(perc * (createGraph.ecount() / 2)))
            filtered_list = [edgeTuple for edgeTuple in createGraph.get_edgelist() if edgeTuple not in deleteEdge]
            graphPrime = Graph()
            graphPrime.add_vertices(len(id_location))
            graphPrime.add_edges(filtered_list)
            min_degree = min(graphPrime.vs.degree())
            max_degree = max(graphPrime.vs.degree())
            dMatrix = np.asarray(Graph.shortest_paths_dijkstra(graphPrime))
        elif distance == "resistance":
            createGraph = nx.from_pandas_edgelist(graph_edges, 'from', 'to')
            createGraph.remove_edges_from(
                random.sample(list(createGraph.edges()), math.ceil(perc * createGraph.number_of_edges())))
            graphNodes = createGraph.number_of_nodes()
            degrees = dict(createGraph.degree())
            min_degree = min(degrees.values())
            max_degree = max(degrees.values())

            # check if the graph is disconnected
            if not nx.is_connected(createGraph):
                restMatrices = []
                subgraphSizes = []
                for component in nx.connected_components(createGraph):  # for each connected component
                    # obtain the subgraph and its size
                    subgraph = createGraph.subgraph(component)
                    subgraphNodes = subgraph.number_of_nodes()
                    # compute the Laplacian matrix and the auxiliary matrices
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

                    # store the auxiliary matrix in a list
                    restMatrices.append(dMatrix)
                    subgraphSizes.append(subgraphNodes)

                # create block diagonal matrix with off-diagonal entries as infinity
                dMatrix = np.zeros((sum(subgraphSizes), sum(subgraphSizes)))
                start = 0
                for subg in range(len(subgraphSizes)):
                    end = start + subgraphSizes[subg]
                    dMatrix[start:end, start:end] = restMatrices[subg]
                    if subg < len(subgraphSizes) - 1:
                        dMatrix[end:, :end] = np.full((sum(subgraphSizes) - end, end),
                                                      np.inf)  # fill lower block with infinity
                        dMatrix[:end, end:] = np.full((end, sum(subgraphSizes) - end),
                                                      np.inf)  # fill upper block with infinity
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
        train_rips = ripser(normalizedMatrix, thresh=1, maxdim=1, distance_matrix=True)[
            'dgms']

        B0 = 0 if len(train_rips[0]) == 0 else len(train_rips[0])
        B1 = 0 if len(train_rips[1]) == 0 else len(train_rips[1])
        # B2 = 0 if len(train_rips[2]) == 0 else len(train_rips[2])

        bettis.update({str(graph_id): [min_degree, max_degree, B0, B1]})  # append to external dictionary

    df = (pd.DataFrame.from_dict(bettis, orient='index',
                                 columns=['min_degree', 'max_degree', 'B0', 'B1'])).rename_axis(
        'id').reset_index()

    df.to_csv("path to result folder" + dataset + distance + "_" + str(
        perc) + "_betti.csv", index=False)


def main(distance, dataset, perc):
    unique_graph_indicator, graph_indicators, df_edges = reading_csv(dataset)
    betti_stats(unique_graph_indicator, graph_indicators, df_edges, perc, dataset, distance)


if __name__ == '__main__':
    data_path = "path to data"
    data_list = ('MUTAG', 'BZR', 'ENZYMES', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    outputfile = "path to result folder" + "predictive_compute.csv"
    for ele in data_list:
        for per in (0.05, 0.1, 0.15):
            main(distance="spd", dataset=ele, perc=per)
            main(distance="resistance", dataset=ele, perc=per)
