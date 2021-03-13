# %%

import os
import sys
import gc

import math

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# %% check likes(target) distribution
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.shape)
print(test.shape)
# %% long-tailed target??
plt.hist(train['likes'], bins=50)

# %% likes_segmentation
high_likes = train[train['likes'] >= 500]
mid_likes = train[(train['likes'] > 100) & (train['likes'] < 500)]
low_likes = train[train['likes'] <= 100]

high_likes.sort_values('likes', ascending=False, inplace=True)
mid_likes.sort_values('likes', ascending=False, inplace=True)
low_likes.sort_values('likes', ascending=False, inplace=True)

# %%
print(high_likes[['title', 'principal_maker']].head(n=10))
print(mid_likes[['title', 'principal_maker']].head(n=10))
print(low_likes[['title', 'principal_maker']].head(n=10))

# %%
print(high_likes[['title', 'principal_maker']].tail(n=10))
print(mid_likes[['title', 'principal_maker']].tail(n=10))
print(low_likes[['title', 'principal_maker']].tail(n=10))
# %% feature_1
# dating_periodがあるのでいらない。
# train['century'] = np.floor(train['dating_sorting_date'] / 100) + 1
# test['century'] = np.floor(train['dating_sorting_date'] / 100) + 1
# %%
train.head()
# %%

by_maker = pd.DataFrame()
by_maker['count_object'] = train.groupby('principal_maker')[
    'object_id'].agg("count")
by_maker['likes_sum'] = train.groupby('principal_maker')['likes'].sum()

# %% famous painter:Vermeer test 4, train 1.
print(train['principal_maker'].str.contains('Vermeer').sum())
print(test['principal_maker'].str.contains('Vermeer').sum())

# %% famous painter:Gogh
print(train['principal_maker'].str.contains('Gogh').sum())
print(test['principal_maker'].str.contains('Gogh').sum())

# %% famous painter:Rembrandt
print(train['principal_maker'].str.contains('Rembrandt').sum())
print(test['principal_maker'].str.contains('Rembrandt').sum())

# %% famous painter:Rembrandt
print(train['principal_maker'].str.contains('anonymous').sum())
print(test['principal_maker'].str.contains('anonymous').sum())

# %%
print(train['title'][train['principal_maker'].str.contains('Vermeer')])
print(test['title'][test['principal_maker'].str.contains('Vermeer')])
# %%
print(train[['title', 'likes']]
      [train['principal_maker'].str.contains('Rembrandt')])
print(test[['title']][test['principal_maker'].str.contains('Rembrandt')])

# %%
print(train[['title', 'likes']][train['principal_maker'].str.contains('Gogh')])
print(test[['title']][test['principal_maker'].str.contains('Gogh')])

# %%
print(train[['title', 'likes']]
      [train['title'].str.contains('Self', case=False)])
print(test[['title']][test['title'].str.contains('Self', case=False)])

# %%
train.columns.values
# %%
train['acquisition_method'].unique()
# %%
