# %%
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import datetime


# %% load files
train = pd.read_csv('../middle/train/train_4_takoi_feature.csv')
test = pd.read_csv('../middle/test/test_4_takoi_feature.csv')
maker = pd.read_csv('../middle/maker_all.csv')
technique = pd.read_csv('../middle/technique_2.csv')
production_place = pd.read_csv('../middle/production_place_2.csv')
object_collection = pd.read_csv('../middle/object_collection_2.csv')
#%%
train_base = pd.read_csv('../input/train.csv')
test_base = pd.read_csv('../input/test.csv')
#%%
train = pd.concat([train_base[['object_id']], train], axis=1)
test = pd.concat([test_base[['object_id']], test], axis=1)

del train_base, test_base
#%%
print(train.shape)
print(test.shape)
# %% merge tables
train = train.merge(technique, on='object_id', how='left')
train = train.merge(object_collection, on='object_id', how='left')
train = train.merge(production_place, on='object_id', how='left')
# print(train.columns.values)
print(train.shape)
print(len(train['object_id'].unique()))

# %% merge tables
test = test.merge(technique, on='object_id', how='left')
test = test.merge(object_collection, on='object_id', how='left')
test = test.merge(production_place, on='object_id', how='left')

print(test.shape)
print(len(test['object_id'].unique()))

# %%
train.to_csv(
    '../middle/train/train_4_x_technique_x_object_collection_x_production_collection.csv', index=False)
test.to_csv(
    '../middle/test/test_4_x_technique_x_object_collection_x_production_collection.csv', index=False)

# %%
