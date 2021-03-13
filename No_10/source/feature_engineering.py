# %%
# from kaeru_BERT import BertSequenceVectorizer
import os
import sys
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from geopy.geocoders import Nominatim
from FE_utility_function import sub_title_rep, place2country
from FE_train_test import Feature_engineering

# %% loading main data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# dummiesとかのために全部くっつける。
train['is_train'] = 1
test['is_train'] = 0
all_data = pd.concat([train, test])

# %%
'''
info_omitted: 行単位の欠損値合計
acquisition_date -> year/month
dating_sorting_date categorise
    1500 < x :Gothic Art
    1500 <= x <= 1800: Renaisasse and Gold Nerderland era
    1800 < x <= 1945 : Gogh, and modern art
    x > 1945         : photography? modern art
is_Rembrandt
is_Vermeer
is_anonymous
in_follower_of: (long titleについてる何か。弟子の作品？)
sub_title: 高さと幅は使えそうだが、単位が揃っていない。
'''
all_data_fe = Feature_engineering(all_data, is_train=True)

train_fe = all_data_fe[all_data['is_train'] == 1]
test_fe = all_data_fe[all_data['is_train'] == 0]
# %%
train_fe.to_csv(
    '../middle/train/original/kien_origin_train_4.csv', index=False)
test_fe.to_csv('../middle/test/original/kien_origin_test_4.csv', index=False)


# %% technique FE
# 各技法をone hot encoding.
technique = pd.read_csv('../input/technique.csv')
technique_sep = pd.concat(
    [technique, technique['name'].str.split(" ", expand=True)], axis=1)
technique_sep.drop('name', axis=1, inplace=True)
technique_sep.columns = ['object_id', 'name_1',
                         'name_2', 'name_3', 'name_4', 'name_5']
technique_sep = technique_sep.melt(var_name="name", id_vars='object_id')
technique_sep.head()
# %%
technique_sep = pd.concat([technique_sep['object_id'], pd.get_dummies(
    technique_sep['value'], prefix="technique", prefix_sep="_")], axis=1)
technique_sep = technique_sep.groupby('object_id').sum()
technique_sep['object_id'] = technique_sep.index
technique_sep.to_csv('../middle/technique_2.csv', index=False)

# %% production_place FE
production_place = pd.read_csv('../input/production_place.csv')
production_place.describe()

production_place['question'] = 0
production_place['question'][production_place['name'].str.contains('\? ')] = 1
production_place['name'].replace('\? ', '', inplace=True)

# %%
# production placeを国名にする
# https://www.guruguru.science/competitions/16/discussions/970ced6d-f974-4979-8f04-dbcf1c2f51a0/
place_list = production_place['name'].unique()
print(len(place_list))
country_dict = {}

for place in place_list:
    try:
        country = place2country(place)
        country_dict[place] = country
    except:
        # 国名を取得できない場合はnan
        # print(place)
        country_dict[place] = np.nan

production_place['country_name'] = production_place['name'].map(country_dict)

# %%
production_place = pd.concat([production_place[['object_id', 'question']],
                              pd.get_dummies(production_place['country_name'],
                                             prefix='prod_place',
                                             prefix_sep='_')], axis=1)
production_place = production_place.groupby('object_id').sum()
production_place['object_id'] = production_place.index
print(production_place.columns.values)
# %%
production_place.to_csv('../middle/production_place_2.csv', index=False)

# %% object collection FE
object_collection = pd.read_csv('../input/object_collection.csv')
object_collection = pd.concat([object_collection['object_id'], pd.get_dummies(
    object_collection['name'], prefix='collection', prefix_sep='_')], axis=1)

object_collection = object_collection.groupby('object_id').sum()
object_collection['object_id'] = object_collection.index
object_collection.to_csv('../middle/object_collection_2.csv', index=False)

# %% maker, principal_maker, and principal_maker_occupation
principal_maker_occupation = pd.read_csv(
    '../input/principal_maker_occupation.csv')
maker = pd.read_csv('../input/maker.csv')
principal_maker = pd.read_csv('../input/principal_maker.csv')

# print(principal_maker_occupation.head())
# print(maker.head())
# print(principal_maker.head())

# print(principal_maker_occupation.shape)
# print(maker.shape)
# print(principal_maker.shape)

principal_maker = principal_maker.merge(
    principal_maker_occupation, on='id', how='right')

'''
16xx年がout of bounds nanosecond timestampらしいのでstr.split
'''

maker = pd.concat(
    [maker, maker['date_of_birth'].str.split('-', expand=True)], axis=1)
maker.rename(columns={0: 'year_of_birth', 1: 'month_of_birth',
                      2: 'day_of_birth'}, inplace=True)
maker.drop('date_of_birth', inplace=True, axis=1)

maker = pd.concat(
    [maker, maker['date_of_death'].str.split('-', expand=True)], axis=1)
maker.rename(columns={0: 'year_of_death', 1: 'month_of_death',
                      2: 'day_of_death'}, inplace=True)
maker.drop('date_of_death', inplace=True, axis=1)

principal_maker = principal_maker.merge(
    maker, left_on='maker_name', right_on='name', how='left')

principal_maker.to_csv('../middle/maker_all.csv', index=False)
