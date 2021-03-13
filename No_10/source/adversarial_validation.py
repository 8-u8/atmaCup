# %%
from sklearn.metrics import mean_squared_log_error, mean_squared_log_error
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from takoi_lgb_fun import train_lgbm, calc_loss

import matplotlib.pyplot as plt
import datetime

import os
# %% custom metric


def rmsle(pred, data):
    true = data.get_label()
    loss = mean_squared_log_error(true, pred) ** 0.5
    return "rmsle", loss, False


# %% load files
list_train = os.listdir('../middle/train/train_5')
list_test = os.listdir('../middle/test/test_5')
# %%
train = pd.read_csv('../middle/train/train_5/train_cleaned.csv')
test = pd.read_csv('../middle/test/test_5/test_cleaned.csv')
# %%
# color = pd.read_csv('../middle/color/color_id_unique.csv')
# palette = pd.read_csv('../middle/color/palette_id_unique.csv')

# bert_tr = pd.read_csv('../middle/descriptions/kaeru_bert_train.csv')
# bert_te = pd.read_csv('../middle/descriptions/kaeru_bert_test.csv')

# mat_tr = pd.read_csv('../middle/material/material_train.csv')
# mat_te = pd.read_csv('../middle/material/material_test.csv')

# # %%
# train = train.merge(bert_tr, on='object_id', how='inner')
# train = train.merge(color, on='object_id', how='left')
# train = train.merge(palette, on='object_id', how='left')
# train = train.merge(mat_tr, on='object_id', how='left')

# test = test.merge(bert_te, on='object_id', how='inner')
# test = test.merge(color, on='object_id', how='left')
# test = test.merge(palette, on='object_id', how='left')
# test = test.merge(mat_te, on='object_id', how='left')

# %%
train['is_train'] = 1
test['is_train'] = 0

train = pd.concat([train, test], axis=0)
# %% init setting
SEED = 42
N_SPLITS = 5
SHUFFLE = False
LGBM_PARAMS = {
    'num_leaves': 31,
    # 'tree_learner': 'voting',
    'min_data_in_leaf': 20,
    'objective': 'binary',
    'max_depth': -1,
    'learning_rate': 0.05,
    "boosting": "gbdt",
    "bagging_freq": 1,
    "bagging_fraction": 0.8,
    "bagging_seed": SEED,
    "verbosity": -1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 0.8,
    'metric': "auc",
    'num_threads': 6,
}

LGBM_FIT_PARAMS = {
    'num_boost_round': 50000,
    'early_stopping_rounds': 500,
    'verbose_eval': 500,
}

# %% init setting
# kf = StratifiedKFold(n_splits=N_SPLITS,
#                      random_state=SEED, shuffle=SHUFFLE)
# kf = KFold(n_splits=N_SPLITS,
#         #    random_state=SEED,
#            shuffle=SHUFFLE)
train['stratify_bin'] = 0
train['stratify_bin'][train['likes'] > 0] = 1
# train['stratify_bin'][(train['likes'] > 0) & (train['likes'] < 50)] = 1
# train['stratify_bin'][(train['likes'] >= 50) & (train['likes'] < 500)] = 2
# train['stratify_bin'][(train['likes'] >= 500) & (train['likes'] < 2000)] = 3
# train['stratify_bin'][train['likes'] >= 2000] = 4
# train['stratify_bin'][train['likes'] == 0] = 0

kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

y = train['is_train']
# y = train['likes']
y_oof = np.empty([len(train), ])
y_test = np.empty([len(train)])
features = list(train.columns)
drop_cols = ['object_id', 'art_series_id', 'likes', 'is_train',
             'sub_title', 'more_title', 'acquisition_date',
             'dating_presenting_date', 'stratify_bin',
             'description_feature', 'info_omitted',
             'dating_period_y',
             #  'principal_maker', 'principal_or_first_maker'
             ]
# drop_cols = ['object_id', 'art_series_id', 'likes', 'title',
#              'description', 'long_title',
#              'sub_title', 'more_title', 'acquisition_date',
#              'dating_presenting_date']
#  'is_Rembrandt', 'is_Vermeer', 'is_Gogh']
features = [i for i in features if i not in drop_cols]
feature_importances = pd.DataFrame()

# %%
categorical_features = ["principal_maker", "principal_or_first_maker",
                        'acquisition_method']
for fold, (train_idx, valid_idx) in enumerate(kf.split(train[features], train['stratify_bin'])):
    print('Fold {}'.format(fold + 1))
    x_train, y_train = train.iloc[train_idx][features], y.iloc[train_idx]
    x_val, y_val = train.iloc[valid_idx][features], y.iloc[valid_idx]

    lgb_train = lgb.Dataset(x_train, label=y_train,
                            free_raw_data=False, feature_name=features,
                            categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(x_val, label=y_val,
                            free_raw_data=False, feature_name=features,
                            categorical_feature=categorical_features)

    eval_result = {}
    model = lgb.train(LGBM_PARAMS,
                      lgb_train,
                      valid_sets=[lgb_train, lgb_valid],
                      valid_names=['train', 'valid'],
                      evals_result=eval_result,
                      #   feval=rmsle,regression
                      **LGBM_FIT_PARAMS
                      )

    y_oof[valid_idx] = model.predict(x_val, num_iteration=model.best_iteration)
    # y_oof = np.expm1(y_oof)
    y_test += model.predict(train[features],
                            num_iteration=model.best_iteration) / kf.n_splits
    # y_test = np.expm1(y_test)
# %%
lgb.plot_metric(eval_result)

# %%
feature_importances['feature'] = features
feature_importances['gain'] = model.feature_importance(
    importance_type='gain', iteration=-1)
feature_importances['split'] = model.feature_importance(
    importance_type='split', iteration=-1)
feature_importances.sort_values('gain', ascending=False).head(n=10)
# %%
feature_importances.sort_values('split', ascending=False).head(n=10)
# %%
result = pd.DataFrame({'true': y,
                       'pred': y_oof})
# %%
pd.DataFrame(result.groupby('pred')['true'].count())
# %%
train.shape
# %%
