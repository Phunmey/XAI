import random
import numpy as np
import pandas as pd
from igraph import *
from time import time
from datetime import datetime
import math


def read_data(dataset, per):
    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)
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
                                       max(graph_indicators) + 1)

    random.seed(42)

    numNodes = []
    numEdges = []
    mindegree = []
    maxdegree = []
    graph_density = []
    graph_diameter = []
    clustering_coeff = []
    spectral_gap_ = []
    assortativity_ = []
    cliques = []
    motifs = []
    components = []
    graph_label_ = []
    graph_id_ = []
    graphTime = []

    for graph_id in unique_graph_indicator:
        id_loc = [index + 1 for index, element in enumerate(graph_indicators) if
                  element == graph_id]
        graph_edges = df_edges[df_edges['from'].isin(id_loc)]
        graph_label = [v for u, v in enumerate(graph_labels, start=1) if u == graph_id][0]
        initGraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False,
                                    weights=True)
        deleteEdge = random.sample(initGraph.get_edgelist(),
                                   math.ceil(per * (initGraph.ecount() / 2)))
        filtered_list = [edgeTuple for edgeTuple in initGraph.get_edgelist() if edgeTuple not in deleteEdge]
        createGraph = Graph()
        createGraph.add_vertices(len(id_loc))
        createGraph.add_edges(filtered_list)

        start = time()

        num_of_nodes = createGraph.vcount()
        num_of_edges = createGraph.ecount()
        minDegree = min(createGraph.vs.degree())
        maxDegree = max(createGraph.vs.degree())
        Density = createGraph.density()
        Diameter = createGraph.diameter()
        cluster_coeff = createGraph.transitivity_avglocal_undirected()
        laplacian = createGraph.laplacian()
        laplace_eigenvalue = np.linalg.eig(laplacian)
        sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
        spectral_gap = sort_eigenvalue[0] - sort_eigenvalue[1]
        assortativity = createGraph.assortativity_degree()
        clique_count = createGraph.clique_number()
        motifs_count = createGraph.motifs_randesu(size=3)
        count_components = len(createGraph.clusters())

        graphTime.append(time() - start)

        numNodes.append(num_of_nodes)
        numEdges.append(num_of_edges)
        mindegree.append(minDegree)
        maxdegree.append(maxDegree)
        graph_id_.append(graph_id)
        graph_label_.append(graph_label)
        graph_density.append(Density)
        graph_diameter.append(Diameter)
        clustering_coeff.append(cluster_coeff)
        spectral_gap_.append(spectral_gap)
        assortativity_.append(assortativity)
        cliques.append(clique_count)
        motifs.append(motifs_count)
        components.append(count_components)

    df1 = pd.DataFrame(motifs, columns=['motif1', 'motif2', 'motif3', 'motif4'])
    df2 = pd.DataFrame(
        data=zip(graph_id_, numNodes, numEdges, graph_label_, graphTime, mindegree, maxdegree, graph_density,
                 graph_diameter, clustering_coeff, spectral_gap_, assortativity_, cliques, components),
        columns=['graph_id', 'numNodes', 'numEdges', 'graph_label', 'graphTime', 'mindegree', 'maxdegree',
                 'graph_density', 'graph_diameter', 'clustering_coeff', 'spectral_gap', 'assortativity', 'cliques',
                 'components', ])

    feature_data = pd.concat([df2, df1], axis=1)
    feature_data.insert(loc=0, column='dataset', value=dataset)

    return feature_data


if __name__ == '__main__':
    data_path = "path to data"
    data_list = ('MUTAG', 'BZR', 'ENZYMES', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for per in [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        df = []
        for dataset in data_list:
            df.append(read_data(dataset, per))

        dfCombine = pd.concat(df)

        dfCombine.to_csv("save dataframe", index=False)
