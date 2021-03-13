# %%
'''
以下特徴量の欲張りセット。
takoi feature
kaeru bert feature
principal_maker bert -> tsne(dim=2)
kien original features
    - train/test
    - technique
    - original_collection
    - production_place
    - color/palette
    - material
    - maker occupation
特にtrain/testにleft-joinする形のデータで
one hot encodingを採用しているが、
trainになくてtestにある(あるいはその逆)、みたいなデータがいくつかある。
そいつらは学習に効かないため以下で特徴量から消す。   
    - describe()をDataFrame化・counts行を削除
    - feature単位でsumをとる(定数ならここの和が0)
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os
import sys
import gc
import re

# %%
list_train = os.listdir('../middle/train/train_5')
list_train = sorted(list_train)
list_test = os.listdir('../middle/test/test_5')
list_test = sorted(list_test)
# %%
train = pd.read_csv('../middle/train/train_5/'+list_train[0])
test = pd.read_csv('../middle/test/test_5/'+list_test[0])

kien_tr = pd.read_csv('../middle/train/original/kien_origin_train_4.csv')
kien_te = pd.read_csv('../middle/test/original/kien_origin_test_4.csv')

kien_tr.fillna(-99999, inplace=True)
kien_te.fillna(-99999, inplace=True)

color = pd.read_csv('../middle/color/color_id_unique.csv')
palette = pd.read_csv('../middle/color/palette_id_unique.csv')

bert_tr = pd.read_csv('../middle/descriptions/kaeru_bert_train.csv')
bert_te = pd.read_csv('../middle/descriptions/kaeru_bert_test.csv')

mat_tr = pd.read_csv('../middle/material/material_train.csv')
mat_te = pd.read_csv('../middle/material/material_test.csv')

mak_tr = pd.read_csv('../middle/maker_occupation_train.csv')
mak_te = pd.read_csv('../middle/maker_occupation_test.csv')

# %%
tsne_tr = pd.read_csv(
    '../middle/descriptions/principal_maker_bert_tsne_train.csv', index_col=0)
tsne_te = pd.read_csv(
    '../middle/descriptions/principal_maker_bert_tsne_test.csv', index_col=0)

# %%
train = train.merge(kien_tr, on='object_id', how='inner')
train = train.merge(bert_tr, on='object_id', how='inner')
train = train.merge(color, on='object_id', how='left')
train = train.merge(palette, on='object_id', how='left')
train = train.merge(mat_tr, on='object_id', how='left')
train = train.merge(mak_tr, on='object_id', how='left')
train = train.merge(tsne_tr, on='object_id', how='inner')

test = test.merge(kien_te, on='object_id', how='inner')
test = test.merge(bert_te, on='object_id', how='inner')
test = test.merge(color, on='object_id', how='left')
test = test.merge(palette, on='object_id', how='left')
test = test.merge(mat_te, on='object_id', how='left')
test = test.merge(mak_te, on='object_id', how='left')
test = test.merge(tsne_te, on='object_id', how='inner')


# %%
plt.hist(train['name_China paper'], bins=50)
plt.hist(test['name_China paper'], bins=50)

# %%
# trainでdescribe()が全部0で、
# かつtestではそうでない特徴量を削る。
train_desc = pd.DataFrame(train.describe())
train_desc = train_desc[1:7]
# train_desc['index'] = train_desc.index

test_desc = pd.DataFrame(test.describe())
test_desc = test_desc[1:7]
# test_desc['index'] = test_desc.index
# %%
train_desc_sum = pd.DataFrame(train_desc.sum(), columns=['sum'])
test_desc_sum = pd.DataFrame(test_desc.sum(), columns=['sum'])
# %%
train_desc_sum['features'] = train_desc_sum.index
train_desc_sum.reset_index(drop=True, inplace=True)
# %%
test_desc_sum['features'] = test_desc_sum.index
test_desc_sum.reset_index(drop=True, inplace=True)

# %%
train_zero_list = train_desc_sum['features'][(
    train_desc_sum['sum'] == 0) & (test_desc_sum['sum'] != 0)].values
test_zero_list = train_desc_sum['features'][(
    train_desc_sum['sum'] != 0) & (test_desc_sum['sum'] == 0)].values
train_zero_list = list(np.array(train_zero_list))
test_zero_list = list(np.array(test_zero_list))
drop_list = train_zero_list + test_zero_list
drop_list = list(set(drop_list))  # 重複の削除
print(train_zero_list)
print(test_zero_list)
print(drop_list)
# %%
train.drop(drop_list, axis=1, inplace=True)
test.drop(drop_list, axis=1, inplace=True)

print(train.shape)
print(test.shape)
# %%
# lgbに突っ込む時にfeatureで起こられるときがあるのでクソ置換。
train = train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
test = test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
# %%
# どこかにUnnamed0がいるらしいので削除する。
# train.drop('Unnamed0', axis=1, inplace=True)
# test.drop('Unnamed0', axis=1, inplace=True)

# %%
train.to_csv('../middle/train/train_5/train_cleaned_5.csv', index=False)
test.to_csv('../middle/test/test_5/test_cleaned_5.csv', index=False)
# %%
