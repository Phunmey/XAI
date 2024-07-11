import os
os.environ["OMP_NUM_THREADS"] = '1'  # this line is to prevent the warning I get wrt KMeans
import webbrowser
import kmapper as km
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


csv_tda = pd.read_csv("path to graph statistics data", header=0)
csv_tda['motif_3'] = csv_tda['motif3'] / (
            (csv_tda['numNodes'] * (csv_tda['numNodes'] - 1) * (csv_tda['numNodes'] - 2)) / 6)
csv_tda['motif_4'] = csv_tda['motif4'] / (
            (csv_tda['numNodes'] * (csv_tda['numNodes'] - 1) * (csv_tda['numNodes'] - 2)) / 6)
columns = ['graphTime', 'graph_id', 'numNodes', 'numEdges', 'motif1', 'motif2', 'motif3', 'motif4', 'mindegree',
               'maxdegree']
sub_data = csv_tda.drop(columns=columns).dropna(axis=0)

# group data based on the smallest data length which is mutag
df = sub_data.groupby(['dataset']).apply(lambda grp: grp.sample(n=188))
M = df[df.columns[1:]].apply(pd.to_numeric)
M = M.drop(['graph_label'], axis=1)

# mapper process
Xfilt = M  # input data
cls = len(pd.unique(df.iloc[:, 0]))
mapper = km.KeplerMapper()
scaler = MinMaxScaler(feature_range=(0, 1))
Xfilt = scaler.fit_transform(Xfilt)
lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE(verbose=1))
print("mapper started with " + str(len(pd.DataFrame(Xfilt).index)) + " data points," + str(cls) + " clusters")

graph = mapper.map(
    lens,
    Xfilt,
    clusterer=sklearn.cluster.KMeans(random_state=1618033),
    cover=km.Cover(n_cubes=10, perc_overlap=0.6)
)

print("mapper ended")
print(str(len(df['dataset'])) + " " + str(len(Xfilt)))

df['data_label'] = df['dataset'] + "-" + df['graph_label'].astype(str)
df = df.drop(['dataset'], axis=1)
y_visual = df.data_label

html = mapper.visualize(
    graph,
    path_html="save html",
    title="mapper html",
    custom_tooltips=y_visual
)

webbrowser.open('view mapper')
