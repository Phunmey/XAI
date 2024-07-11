"""
Obtain average of the accuracy results
"""

import pandas as pd

data_path = "path to training results obtained"
read_file = pd.read_csv(data_path, delimiter='\t')

file_df = read_file.drop(['filtration', 'trainTime', 'conf_val', 'conf_test'], axis=1)  # drop columns
group_data = round(file_df.groupby(['percent', 'dist_type', 'step_size'])[
    ['val_acc', 'val_auc', 'test_acc', 'test_auc', 'filtrTime']].agg(['mean', 'std']), 3)

group_data.to_csv("save results")
