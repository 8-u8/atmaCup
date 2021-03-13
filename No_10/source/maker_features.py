# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# %%
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

maker = pd.read_csv('../input/maker.csv')
principal_maker = pd.read_csv('../input/principal_maker.csv')
principal_maker_occupation = pd.read_csv(
    '../input/principal_maker_occupation.csv')
# %%
principal_maker.head()
# %%
principal_maker_occupation.head()
principal_maker_occupation.columns = ['id', 'occupation']
# %%
occupation_dummies = pd.get_dummies(principal_maker_occupation)

# %%
principal_maker = principal_maker[[
    'id', 'object_id', 'qualification', 'roles']]

# %%

maker_and_occup = occupation_dummies.merge(
    principal_maker, on='id', how='left')
# %%
maker_and_occup = pd.get_dummies(maker_and_occup,
                                 dummy_na=True,
                                 columns=['qualification', 'roles'],
                                 prefix=['qualification', 'roles'],
                                 prefix_sep='_')
# %%
maker_and_occup = maker_and_occup.groupby(
    ['object_id'], as_index=False).sum()
# %%
maker_and_occup.sort_values('id', ascending=True)
# %%

train = train[['object_id', 'likes']]
test = test[['object_id']]

# %%
train = train.merge(maker_and_occup, on='object_id', how='left')
test = test.merge(maker_and_occup, on='object_id', how='left')
# %%
# %%

plt.hist(train['qualification_nan'])
plt.hist(test['qualification_nan'])
# %%
train.drop(['likes', 'id'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# %%
train.to_csv('../middle/maker_occupation_train.csv')
test.to_csv('../middle/maker_occupation_test.csv')
# %%
