import random
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from igraph import *
from ripser import ripser
import networkx as nx
import matplotlib.pyplot as plt

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

    return unique_graph_indicator, graph_labels, df_edges, graph_indicators


def rips_filt(unique_graph_indicator, graph_indicators, df_edges, step_size, dataset,
               graph_labels, perc, distance="spd"):  # this is for the train data

    betti_list = []
    dMatrix = None
    for graph_id in unique_graph_indicator:
        id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_label = [ele for ind, ele in enumerate(graph_labels, start=1) if
                       ind == graph_id]  # obtain graph label corresponding to the graph_id
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        if distance == "spd":
            createGraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
            deleteEdge = random.sample(createGraph.get_edgelist(),
                                       math.ceil(perc * (createGraph.ecount() / 2)))
            filtered_list = [edgeTuple for edgeTuple in createGraph.get_edgelist() if edgeTuple not in deleteEdge]
            graphPrime = Graph()
            graphPrime.add_vertices(len(id_location))
            graphPrime.add_edges(filtered_list)
            dMatrix = np.asarray(Graph.shortest_paths_dijkstra(graphPrime))
        elif distance == "resistance":
            createGraph = nx.from_pandas_edgelist(graph_edges, 'from', 'to')
            createGraph.remove_edges_from(
                random.sample(list(createGraph.edges()), math.ceil(perc * createGraph.number_of_edges())))
            graphNodes = createGraph.number_of_nodes()

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

        start = time()
        train_rips = ripser(normalizedMatrix, thresh=1, maxdim=1, distance_matrix=True)[
            'dgms']
        filtr_time = time() - start

        #    splitting the dimension into 0 and 1
        train_dgm_0 = train_rips[0]

        # save the persistence diagrams
        filename = "save PD"
        with open(filename, "a") as f:
            f.write(f"{graph_id}: {train_dgm_0}\n")
            f.flush()  # ensures data is written to file
            f.close()

        #    obtain betti numbers for the unique dimensions
        train_betti_0 = []
        
        for eps in np.linspace(0, 1, step_size):
            b_0 = 0
            for k in train_dgm_0:
                if k[0] <= eps and k[1] > eps:
                    b_0 = b_0 + 1
            train_betti_0.append(b_0)

        betti_list.append([dataset] + [graph_id] + graph_label + [
            filtr_time] + train_betti_0)  # concatenate betti numbers

    feature_data = pd.DataFrame(betti_list)

    #    giving column names
    columnnames = {}  # create an empty dict
    count = -4  # initialize count to -1
    for i in feature_data.columns:
        count += 1  # update count by 1
        columnnames[i] = f"eps_{count}"  # index i in dictionary will be named thresh_count

    # rename first and last column in the dictionary
    columnnames.update(
        {(list(columnnames))[0]: 'dataset', (list(columnnames))[1]: 'graphId', (list(columnnames))[2]: 'graphLabel',
         (list(columnnames))[3]: 'filtrTime'})
    feature_data.rename(columns=columnnames, inplace=True)  # give column names to dataframe

    # write dataframe to file
    feature_data.to_csv("save dataframe", index=False)


def main(distance, step_size, dataset, perc):
    unique_graph_indicator, graph_labels, df_edges, graph_indicators = reading_csv(dataset)
    rips_filt(unique_graph_indicator, graph_indicators, df_edges, step_size, dataset, graph_labels, perc, distance)


if __name__ == '__main__':
    data_path = "path to data"  # dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for ele in data_list:
        for st in (10, 20, 50, 100):  # we will consider step size 100 for epsilon
            for per in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                main(step_size=st, distance="spd", dataset=ele, perc=per)
                #main(step_size=st, distance="resistance", dataset=ele, perc=per)
