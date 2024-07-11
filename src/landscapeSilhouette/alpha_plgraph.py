import random
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from igraph import *
import gudhi as gd
import gudhi.representations
from sklearn.decomposition import PCA

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
                                       max(graph_indicators) + 1)

    return unique_graph_indicator, graph_indicators, df_edges, graph_labels


def landscape_train(unique_graph_indicator, graph_indicators, df_edges, dataset,
                    graph_labels):
    start2 = time()

    train_landscape = []
    graph_density = []
    graph_diameter = []
    clustering_coeff = []
    spectral_gap = []
    assortativity_ = []
    cliques = []
    motifs = []
    components = []
    corrupted_graph = 0

    for graph_id in unique_graph_indicator:
        try:
            id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                           element == graph_id]
            graph_label = [ele for ind, ele in enumerate(graph_labels, start=1) if
                           ind == graph_id]
            graph_edges = df_edges[df_edges['from'].isin(id_location)]
            create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)

            if not create_traingraph.is_connected():
                graph_decompose = create_traingraph.decompose()
                pca_list = []
                for subg in graph_decompose:
                    create_subg = np.asarray(Graph.shortest_paths_dijkstra(subg))
                    norm_subg = create_subg / np.nanmax(create_subg)
                    pca = PCA(n_components=2).fit_transform(norm_subg)
                    pca_list.append(pca)
                matrix_pca = (np.vstack(pca_list))
            else:
                create_dmatrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
                norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix)
                matrix_pca = PCA(n_components=2).fit_transform(norm_dmatrix)

            train_ac = gd.AlphaComplex(points=matrix_pca).create_simplex_tree()
            train_dgm = train_ac.persistence()
            landscape_init = gd.representations.Landscape(num_landscapes=1, resolution=1000)
            land_scape = landscape_init.fit_transform([train_ac.persistence_intervals_in_dimension(1)])

            train_landscape.append([dataset] + [graph_id] + graph_label + land_scape.tolist())

            Density = create_traingraph.density()
            Diameter = create_traingraph.diameter()
            cluster_coeff = create_traingraph.transitivity_avglocal_undirected()
            laplacian = create_traingraph.laplacian()
            laplace_eigenvalue = np.linalg.eig(laplacian)
            sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
            spectral = sort_eigenvalue[0] - sort_eigenvalue[1]
            assortativity = create_traingraph.assortativity_degree()
            clique_count = create_traingraph.clique_number()
            motifs_count = create_traingraph.motifs_randesu(size=3)
            count_components = len(create_traingraph.clusters())

            graph_density.append(Density)
            graph_diameter.append(Diameter)
            clustering_coeff.append(cluster_coeff)
            spectral_gap.append(spectral)
            assortativity_.append(assortativity)
            cliques.append(clique_count)
            motifs.append(motifs_count)
            components.append(count_components)

        except:
            corrupted_graph += 1

    df1 = pd.DataFrame(train_landscape)
    df2 = pd.DataFrame(motifs)
    df3 = pd.DataFrame(
        list(zip(graph_density, graph_diameter, clustering_coeff, spectral_gap, assortativity_, cliques, components)))
    feature_data = pd.concat([df1, df2, df3], axis=1, ignore_index=True)

    t2 = time()
    landscape_time = t2 - start2

    # column names
    columnnames = {}
    count = -3
    for i in feature_data.columns:
        count += 1
        columnnames[i] = f"res_{count}"

    # rename first and last column in the dictionary
    columnnames.update(
        {(list(columnnames))[0]: 'dataset', (list(columnnames))[1]: 'graphId', (list(columnnames))[2]: 'graphLabel'})
    feature_data.rename(columns=columnnames, inplace=True)

    # write dataframe to file
    feature_data.to_csv("save to csv", index=False)


def main():
    unique_graph_indicator, graph_indicators, df_edges, graph_labels = reading_csv(dataset)
    landscape_train(unique_graph_indicator, graph_indicators, df_edges, dataset, graph_labels)


if __name__ == '__main__':
    data_path = "path to dataset"
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for dataset in data_list:
        main()
