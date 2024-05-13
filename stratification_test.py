import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.CustomDataSet import CustomDataSet


data = pd.read_csv('data\processed\original_MS_profiles.csv', sep=';')

all_set = CustomDataSet(data.drop('group', axis=1).drop('ID', axis=1).to_numpy(dtype=float),
                          data['group'],
                          data['ID'])

X = np.array(data.drop(columns=['group', 'ID'], axis=1))
Y = data['group']

targets = Y.to_numpy()
unique_targets = np.unique(targets)

train_set, test_set = train_test_split(all_set, train_size=0.7, stratify=targets, shuffle=True)

train_t = pd.Series(data=np.zeros(len(unique_targets)), index=unique_targets)
test_t = pd.Series(data=np.zeros(len(unique_targets)), index=unique_targets)

# for i in train_t.columns:
#     train_t[i] = 0
#     test_t[i] = 0

for i in train_set:
    train_t[i[1]] = train_t[i[1]] + 1

for i in test_set:
    test_t[i[1]] = test_t[i[1]] + 1

for i in unique_targets:
    train_t[i] = train_t[i]/104
    test_t[i] = test_t[i]/45

stop = 1