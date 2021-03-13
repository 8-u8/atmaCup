# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
train = pd.read_csv('../input/train.csv')
CV = pd.read_csv('../output/CV_score.csv')
historical_person = pd.read_csv('../input/historical_person.csv')
maker = pd.read_csv('../middle/maker_all.csv')

pca_50 = pd.read_csv('../middle/descriptions/pca_50_train.csv')
pca_100 = pd.read_csv('../middle/descriptions/pca_100_train.csv')
pca_150 = pd.read_csv('../middle/descriptions/pca_150_train.csv')
pca_200 = pd.read_csv('../middle/descriptions/pca_200_train.csv')

svd_50 = pd.read_csv('../middle/descriptions/svd_50_train.csv')
svd_100 = pd.read_csv('../middle/descriptions/svd_100_train.csv')
svd_150 = pd.read_csv('../middle/descriptions/svd_150_train.csv')
svd_200 = pd.read_csv('../middle/descriptions/svd_200_train.csv')

# %%
train['acquisition_date'] = pd.to_datetime(train['acquisition_date'])


# %%
train_CV = pd.concat([train, CV], axis=1)
train_CV = train_CV[['object_id', 'likes', 'pred']]

# %% historical_personのid数
historical_id_unique = historical_person.groupby(
    'object_id', as_index=False)['name'].agg("count")

# %%
train_CV = train_CV.merge(historical_id_unique, on='object_id', how='left')
# %%
plt.scatter(train_CV['pred'], svd_150['pca_2'])


# %%
plt.scatter(train_CV['likes'], train_CV['pred'])
# %%
plt.hist(train_CV['likes'][train_CV['likes'] >= 10000], bins=50)
# %%

prin_maker_tr = train[['principal_maker', 'likes']]
prin_maker_te = test[['principal_maker']]
# %%
prin_maker_tr.shape
# %%
prin_maker_te.shape
# %%
prin_maker_tr.sort_values('principal_maker', inplace=True)
prin_maker_te.sort_values('principal_maker', inplace=True)

# %%
prin_maker_tr.reset_index(drop=True, inplace=True)
prin_maker_te.reset_index(drop=True, inplace=True)

# %%
prin_maker = pd.concat([prin_maker_tr, prin_maker_te], axis=1).fillna(1300)
prin_maker.columns = ['train', 'likes', 'test']
# %%
plt.scatter(prin_maker['likes'][:200], prin_maker['test'][:200])
# %%
train['likes'].describe(
    percentiles=[.10, .15, .20, .25, .30, .35, .40, .50, .75, .90, .99, .995])
# %%
plt.plot(train[['likes']].sort_values(
    'likes', ascending=False).reset_index(drop=True)[1000:])
# %%
train['likes'].describe(
    percentiles=[
        .10,
        .15,
        .20,
        .25,
        .30,
        .35,
        .40,
        .45,
        .50,
        .55,
        .60,
        .65,
        .70,
        .75,
        .80,
        .85,
        .90,
        .95,
        .99
    ]
)
# %%
# init Stratified bin
train['stratify_bin'] = -1
# ~35 percentile
train['stratify_bin'][train['likes'] == 0] = 0
# 40~70 percentile
train['stratify_bin'][(train['likes'] > 0) & (train['likes'] <= 8)] = 1
# 70~80 percentile
train['stratify_bin'][(train['likes'] > 8) & (train['likes'] <= 18)] = 2
# 80~90 percentile
train['stratify_bin'][(train['likes'] > 18) & (train['likes'] <= 90)] = 3
# over 90 percentile
train['stratify_bin'][train['likes'] > 58] = 4


# %%
train.groupby('stratify_bin')['object_id'].count()

# %%
