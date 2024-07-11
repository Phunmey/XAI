import random
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from igraph import *
import gudhi as gd
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
                                       max(graph_indicators) + 1)  # list unique graph ids in a dataset

    return unique_graph_indicator, graph_labels, df_edges, graph_indicators


def alpha_filt(unique_graph_indicator, graph_indicators, df_edges, step_size, dataset,
               graph_labels):  # this is for the train data

    betti_list = []

    for graph_id in unique_graph_indicator:
        try:
            id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                           element == graph_id]  # list the index of the graph_id locations
            graph_label = [ele for ind, ele in enumerate(graph_labels, start=1) if
                           ind == graph_id]  # obtain graph label corresponding to the graph_id
            graph_edges = df_edges[df_edges['from'].isin(id_location)]
            create_graph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
            # plot(create_traingraph)

            if not create_graph.is_connected():
                graph_decompose = create_graph.decompose()
                mds_list = []
                for subg in graph_decompose:
                    create_subg = np.asarray(Graph.shortest_paths_dijkstra(subg))
                    norm_subg = create_subg / np.nanmax(create_subg)
                    mds = PCA(n_components=2).fit_transform(norm_subg)
                    mds_list.append(mds)
                matrix_mds = (np.vstack(mds_list))
            else:
                create_dmatrix = np.asarray(Graph.shortest_paths_dijkstra(create_graph))
                norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix)
                matrix_mds = PCA(n_components=2).fit_transform(norm_dmatrix)

            start = time()
            ac = gd.AlphaComplex(points=matrix_mds).create_simplex_tree()
            dgm = ac.persistence()  # obtain persistence values
            filtr_time = time() - start

            #    select dimensions 0 and 1
            dgm_0 = ac.persistence_intervals_in_dimension(0)
            dgm_1 = ac.persistence_intervals_in_dimension(1)

            # save the persistence diagrams
            filename = "save PD"
            with open(filename, "a") as f:
                f.write(f"{graph_id}: {dgm_0, dgm_1}\n")
                f.flush()  # ensures data is written to file
                f.close()

            #    obtain betti numbers for the unique dimensions
            betti_0 = []
            betti_1 = []

            for eps in np.linspace(0, 1, step_size):
                b_0 = 0
                for k in dgm_0:
                    if k[0] <= eps and k[1] > eps:
                        b_0 = b_0 + 1
                betti_0.append(b_0)

                b_1 = 0
                for l in dgm_1:
                    if l[0] <= eps and l[1] > eps:
                        b_1 = b_1 + 1
                betti_1.append(b_1)

            betti_list.append(
                [dataset] + [graph_id] + graph_label + [filtr_time] + betti_0 + betti_1)  # concatenate betti numbers

        except:
            pass

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


def main():
    unique_graph_indicator, graph_labels, df_edges, graph_indicators = reading_csv(dataset)
    alpha_filt(unique_graph_indicator, graph_indicators, df_edges, step_size, dataset, graph_labels)


if __name__ == '__main__':
    data_path = "path to data"  # dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for dataset in data_list:
        for step_size in (10, 20, 50, 100):  # we will consider step size 100 for epsilon
            main()
