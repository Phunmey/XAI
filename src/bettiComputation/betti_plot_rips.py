import random
from time import time
import numpy as np
import pandas as pd
from igraph import *
import matplotlib.pyplot as plt
from ripser import ripser

random.seed(42)
plt.rcParams['figure.figsize'] = (16, 12)


def read_csv(dataset):
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

    return unique_graph_indicator, graph_indicators, df_edges


def rips_betti(unique_graph_indicator, graph_indicators, df_edges, dataset):  # this is for the train test

    betti = []
    for i in unique_graph_indicator:
        graph_id = i
        id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
        normalize_matrix = distance_matrix / np.nanmax(distance_matrix[distance_matrix != np.inf])
        rips_diagrams = ripser(normalize_matrix, thresh=1, maxdim=1, distance_matrix=True)[
            'dgms']

        # obtain betti numbers for the unique dimensions
        betti0 = []
        betti1 = []
        betti2 = []

        for eps in np.linspace(0, 1, 100):
            b_0 = 0
            for k in rips_diagrams[0]:
                if k[0] <= eps and k[1] > eps:
                    b_0 = b_0 + 1
            betti0.append(b_0)  # concatenate betti numbers

            b_1 = 0
            for l in rips_diagrams[1]:
                if l[0] <= eps and l[1] > eps:
                    b_1 = b_1 + 1
            betti1.append(b_1)

            b_2 = 0
            for m in rips_diagrams[2]:
                if m[0] <= eps and m[1] > eps:
                    b_2 = b_2 + 1
            betti2.append(b_2)

        betti.append(betti0 + betti1)

    feature_data = pd.DataFrame(betti)
    feature_data.to_csv("save dataframe", index=False)
    obtain_mean = feature_data.mean(axis=0)  # obtain mean for each epsilon across all graphs in the data

    x = np.arange(0, 100)  # define axis which is the range of epsilon

    #    select the columns for each betti
    betti0 = obtain_mean.iloc[:100]
    betti1 = obtain_mean.iloc[100:]
    betti2 = obtain_mean.iloc[200:]

    #    normalizing using max
    norm_0 = betti0 / np.nanmax(betti0)
    norm_1 = betti1 / np.nanmax(betti1)
    norm_2 = betti2 / np.nanmax(betti2)

    #    plot using pyplot
    plt.plot(x, norm_0, color='b', label='B0', linestyle='--')
    plt.plot(x, norm_1, color='r', label='B1', linestyle='dotted')
    plt.plot(x, norm_2, color='k', label='B2')

    plt.box(False)
    plt.xlabel(r"$\epsilon$", fontdict={'fontsize': 32})
    plt.ylabel('mean frequency', fontdict={'fontsize': 32})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(bbox_to_anchor=(0.5, 0.6, 0.4, 0.5), loc='lower center', frameon=False, fontsize=32)

    plt.savefig("save figure")
    plt.clf()

    return


if __name__ == '__main__':
    data_path = "path to data"
    data_list = ('MUTAG', 'BZR', 'ENZYMES', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for dataset in data_list:
        unique_graph_indicator, graph_indicators, df_edges = read_csv(dataset)
        rips_betti(unique_graph_indicator, graph_indicators, df_edges, dataset)
