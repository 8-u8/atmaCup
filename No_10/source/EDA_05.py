# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# tsne_train = pd.read_csv(
#     '../middle/descriptions/principal_maker_bert_tsne_train.csv')
# tsne_test = pd.read_csv(
#     '../middle/descriptions/principal_maker_bert_tsne_test.csv')
material = pd.read_csv('../middle/material/material_onehot.csv')
# # %%
# plt.scatter(train['likes'], tsne_train['tsne_1'])
# plt.scatter(train['likes'], tsne_train['tsne_2'])

# # %%
# plt.hist(tsne_train['tsne_1'], bins=50)
# plt.hist(tsne_test['tsne_1'], bins=50)

# # %%
# plt.hist(tsne_train['tsne_2'], bins=50)
# plt.hist(tsne_test['tsne_2'], bins=50)

###
# %%
train_id = train[['object_id']]
train_mat = train_id.merge(material, on='object_id', how='left')
# %%

test_id = test[['object_id']]
test_mat = test_id.merge(material, on='object_id', how='left')
# %%
train_mat['material_na'] = train_mat.isnull().sum(axis=1)
test_mat['material_na'] = test_mat.isnull().sum(axis=1)
# %%
# train_mat.to_csv('../middle/material/material_train.csv', index=False)
# test_mat.to_csv('../middle/material/material_test.csv', index=False)

# %%
log_square_y = np.log1p(train['likes']) ** 2

# %%
log_root_y = np.sqrt(np.log1p(train['likes']))
# %%
