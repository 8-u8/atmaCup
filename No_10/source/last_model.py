# %%

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

import lightgbm as lgb

import matplotlib.pyplot as plt
from datetime import datetime
import pickle

import os
import gc

from reduce_mem_usage import reduce_mem_usage


# %% custom metric
# 結局使わなかったがこうやって関数定義できればkaggleでも使える。
# 完　全　に　理　解　し　た


def rmsle(pred, data):
    true = data.get_label()
    # loss = np.mean((np.log1p(pred) - np.log1p(data)) ^ 2) ** 5
    loss = mean_squared_log_error(true, pred) ** 0.5
    return 'rmsle', loss, False


def msle(pred, data):
    true = data.get_label()
    loss = mean_squared_log_error(true, pred)
    return 'msle', loss, False


def rmse(pred, data):
    true = data.get_label()
    loss = mean_squared_error(true, pred, squared=False)
    return 'rmse', loss, False


# %% initial value
# 実験管理のための時間を定義。
dt_now = datetime.now()

# use additional features
use_kien_features = True
kaeru_bert = True
kaeru_pca = False
kaeru_svd = False
kaeru_tsne = True
color = True
material = True

# init setting
# DO NOT EDIT!!
SEED = 42  # 魔法の数字
N_SPLITS = 5
SHUFFLE = True
kf = StratifiedKFold(n_splits=N_SPLITS,
                     random_state=SEED,
                     shuffle=SHUFFLE)

LGBM_PARAMS = {'num_leaves': 50,
               'min_data_in_leaf': 20,
               'objective': 'regression_l2',
               #    'first_metric_only': True,
               #    'tweedie_variance_power': 1.774438,
               'max_depth': 7,
               'learning_rate': 0.01,
               'boosting': 'dart',
               'drop_seed': SEED,
               'bagging_freq': 1,
               'bagging_fraction': 0.8,
               'bagging_seed': SEED,
               'verbosity': -1,
               'lambda_l1': 0.3,
               'lambda_l2': 0.1,
               'feature_fraction': 0.7,
               'metric': 'rmse',
               'max_bin': 70,
               'num_threads': 6,
               'seed': SEED
               }

LGBM_FIT_PARAMS = {
    'num_boost_round': 50000,
    'early_stopping_rounds': 200,
    'verbose_eval': 500,
}
drop_cols = ['object_id', 'art_series_id', 'likes', 'stratify_bin',
             'sub_title', 'more_title', 'acquisition_date',
             'dating_presenting_date',
             'description_feature',
             'description', 'title', 'long_title', 'weight',
             #  'is_Rembrandt','is_Vermeer','is_Gogh'
             ]
categorical_features = ['principal_maker', 'principal_or_first_maker',
                        'acquisition_method']


# %%
# load files
dt_now = datetime.now()
train_file = '../middle/train/train_5/train_cleaned_3.csv'
# '../middle/train/train_3.csv'
test_file = '../middle/test/test_5/test_cleaned_3.csv'
# '../middle/test/test_3.csv'

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
# train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
# test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
# %%
train.fillna({'size_d': -99999, 'size_t': -99999}, inplace=True)
test.fillna({'size_d': -99999, 'size_t': -99999}, inplace=True)


# train.drop(drop_cols_by_importance, axis=1, inplace=True)
# test.drop(drop_cols_by_importance, axis=1, inplace=True)

print(train.shape)
print(test.shape)
# %%
#  init Stratified bin
'''
Stratified KFoldに対するtarget分割
long tailed な分布なので
percentileでbinningを行う。
他にも95 percentile以上のlikesを持つ作品を削ることも考えたが
なんかtestにハイスコア作品がいそうで怖い。
- trainに1つしかないVermeerがtestに4作品ある
  - そのうち1つは「牛乳を注ぐ女」。コイツァ
正しいかどうかは俺ではなく世界が決めるので気にしない
'''
train['stratify_bin'] = -1
# ~35 percentile
train['stratify_bin'][train['likes'] == 0] = 0
# train['stratify_bin'][train['likes'] == 0] = 1
# 40~70 percentile
train['stratify_bin'][(train['likes'] > 0) & (train['likes'] <= 8)] = 1
# 70~80 percentile
train['stratify_bin'][(train['likes'] > 8) & (train['likes'] <= 18)] = 2
# 80~90 percentile
train['stratify_bin'][(train['likes'] > 18) & (train['likes'] <= 90)] = 3
# over 90 percentile
train['stratify_bin'][train['likes'] > 58] = 4
# MECE check
print(train['stratify_bin'].describe())

# %% 情報量の総合的な欠落
train['info_omitted_all'] = train.isnull().sum(axis=1)
test['info_omitted_all'] = test.isnull().sum(axis=1)

# %% weight。シグモイドつかってみっか。
# w_base = np.log1p(train['likes'])/(1+np.exp(-np.log1p(train['likes']))) + 1
# train['weight'] = w_base ** 2
# %%
# reduce mem usage
# 今回あまり意味ないかも知れない

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
gc.collect()
# %% init

'''
log1p target
以下discussionも参考
https://www.guruguru.science/competitions/16/discussions/d5a1a5dc-57a8-473e-8dd7-44ce24a7e777/
これにより生targetのrmsleでのvalidationと
rmseによるvalidationは同じとみなせる。
(証明は代入すればわかるね？)
'''
y = np.log1p(train['likes'])

# lgb output preprocessing
y_oof = np.zeros(len(train))
y_test = np.zeros(len(test))

# select features
features = list(train.columns)
features = [i for i in features if i not in drop_cols]
feature_importances = pd.DataFrame()

# %%
# run lgb.
for fold, (train_idx, valid_idx) in enumerate(kf.split(train[features], train['stratify_bin'])):
    print('Fold {}'.format(fold + 1))
    # print(train.shape)
    # w_train, w_valid = train['weight'][train_idx], train['weight'][valid_idx]
    x_train, y_train = train.iloc[train_idx][features], y.iloc[train_idx]
    x_valid, y_valid = train.iloc[valid_idx][features], y.iloc[valid_idx]

    # check train/valid volume.
    print(
        f'x_train rows:{x_train.shape[0]} \n x_valid rows:{x_valid.shape[0]}')

    # 気になるので可視化を挟んどく。
    # 分布ガバガバは怖い。
    plt.hist(y_train, bins=15)
    plt.hist(y_valid, bins=15)
    plt.show()

    lgb_train = lgb.Dataset(x_train,
                            label=y_train,
                            free_raw_data=False,
                            feature_name=features,
                            # weight=w_train,
                            categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(x_valid,
                            label=y_valid,
                            free_raw_data=False,
                            feature_name=features,
                            # weight=w_valid,
                            categorical_feature=categorical_features)

    eval_result = {}
    model = lgb.train(LGBM_PARAMS,
                      lgb_train,
                      valid_sets=[lgb_train, lgb_valid],
                      valid_names=['train', 'valid'],
                      evals_result=eval_result,
                      ** LGBM_FIT_PARAMS
                      )

    lgb.plot_metric(eval_result)
    plt.show()

    # predict out of fold
    print('oof predict')
    y_oof[valid_idx] = model.predict(
        x_valid, num_iteration=model.best_iteration)

    # predict test
    # add on each folds.
    print('test predict')
    y_test += model.predict(test[features],
                            num_iteration=model.best_iteration)  # / (fold + 1)

    del x_train, x_valid, y_train, y_valid
    gc.collect()

# 予測値は事後的に平均。
y_test = y_test/N_SPLITS

# %%
# imporanceを算出
feature_importances['feature'] = features
feature_importances['gain'] = model.feature_importance(
    importance_type='gain', iteration=-1)
feature_importances['split'] = model.feature_importance(
    importance_type='split', iteration=-1)
feature_importances.sort_values('gain', ascending=False).head(n=10)

# %%
# 特徴量重要度
lgb.plot_importance(model, max_num_features=10)

# 学習曲線
lgb.plot_metric(eval_result)
plt.show()
# %% 対数変換時の予実。
plt.scatter(y_oof, y)
plt.show()

# %%
#  local CVによるRSMLEの計算。
CV = pd.DataFrame({'true': train['likes'],
                   # sub.loc[sub.likes <= 0, 'likes'] = 0
                   #    'pred': np.expm1(y_oof)
                   'pred': y_oof
                   }
                  )
CV['pred'][CV['pred'] <= 0] = 0
CV['pred'] = np.expm1(CV['pred'])
CV_score = mean_squared_log_error(CV['true'], CV['pred']) ** 0.5
CV_score
# %% 逆変換後の出力
plt.scatter(CV['true'], CV['pred'])

# %% submission準備
# y_test_sub = y_test
sub = pd.read_csv('../input/atmacup10__sample_submission.csv')
sub['likes'] = y_test
sub['likes'].describe()

# %%
# 逆変換。
sub['likes'] = np.expm1(sub['likes'])
sub['likes'].describe()
# %%
# まれに出てくる負の値を0に押し込む。
sub['likes'][sub['likes'] < 0] = 0
sub.describe()


# %% いろいろのアウトプット
# write submission file.
model_date = dt_now.strftime('%Y%m%d_%H%M%S')
sub_filename = '../output/' + \
    model_date + '_submission.csv'
sub.to_csv(sub_filename, index=False)

# save model
model_name = '../model/LightGBM_' + model_date + '.pkl'
pickle.dump(model, open(model_name, 'wb'))

# save importance
feature_importances_name = '../result/importance/feature_importances_' + model_date + '.csv'
feature_importances.to_csv(feature_importances_name, index=False)

# save experiments table
result_table = pd.DataFrame({'experiment_time': [dt_now.strftime('%Y-%m-%d %H:%M:%S')],
                             #  'train_score': train_CV,
                             'CV_score': CV_score,
                             'train_table': train_file,
                             'test_table': test_file,
                             'model_file': model_name,
                             'sub_file': sub_filename,
                             'sub_max': sub['likes'].max(),
                             'bert_use': kaeru_bert,
                             'use_origin_feats': use_kien_features,
                             'onjective': LGBM_PARAMS['objective']
                             }
                            )
if 'experiment_table.csv' not in os.listdir('../result/'):
    result_table.to_csv('../result/experiment_table.csv', index=False)
else:
    exp_table = pd.read_csv('../result/experiment_table.csv')
    exp_table = pd.concat([exp_table, result_table], axis=0)
    exp_table = exp_table.sort_values(
        'experiment_time', ascending=True)
    exp_table.to_csv('../result/experiment_table.csv', index=False)

# for end

# %%
