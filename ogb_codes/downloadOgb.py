"""
Download Ogb dataset directly using this code
"""

import pandas as pd
import pickle

from ogb.graphproppred import GraphPropPredDataset
from smilestograph import smiles2graph


# Download the data to a folder.
dataset = GraphPropPredDataset(name="ogbg-molhiv", root='root folder')

readmol = pd.read_csv(".\mapping\mol.csv.gz")
readmol.to_csv("save as csv", index_label='graphId')

# Convert smiles to graph data
graph_data = {}
for index, row in readmol.iterrows():
    smiles = row['smiles']
    graph_data[index] = smiles2graph(smiles)

# Save the graph data to a pickle file
with open(".\graph_data.pkl", 'wb') as f:
    pickle.dump(graph_data, f)

print("smilestograph conversion and saving complete.")

