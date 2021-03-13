# %%

import os
import sys
import gc

import math

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# utility
from FE_utility_function import place2country

# %%
color = pd.read_csv('../input/color.csv')
palette = pd.read_csv('../input/palette.csv')
'''
palette: object_idでgroupby -> agg(mean/min/max/std/skew)
'''


# %%
# principal_maker_occupation = pd.read_csv(
#     '../input/principal_maker_occupation.csv')
# maker = pd.read_csv('../input/maker.csv')
# principal_maker = pd.read_csv('../input/principal_maker.csv')

# print(principal_maker_occupation.head())
# print(maker.head())
# print(principal_maker.head())

# print(principal_maker_occupation.shape)
# print(maker.shape)
# print(principal_maker.shape)

# # %%
# principal_maker = principal_maker.merge(
#     principal_maker_occupation, on='id', how='right')

# principal_maker.shape

# # %%
# '''
# 16xx年がout of bounds nanosecond timestampらしいのでstr.split

# '''
# maker = pd.concat(
#     [maker, maker['date_of_birth'].str.split('-', expand=True)], axis=1)
# maker.rename(columns={0: 'year_of_birth', 1: 'month_of_birth',
#                       2: 'day_of_birth'}, inplace=True)
# maker.drop('date_of_birth', inplace=True, axis=1)

# maker = pd.concat(
#     [maker, maker['date_of_death'].str.split('-', expand=True)], axis=1)
# maker.rename(columns={0: 'year_of_death', 1: 'month_of_death',
#                       2: 'day_of_death'}, inplace=True)
# maker.drop('date_of_death', inplace=True, axis=1)

# #%%
# principal_maker = principal_maker.merge(maker, left_on='maker_name', right_on='name', how='left')

# %%
# object_collection = pd.read_csv('../input/object_collection.csv')
# object_collection = pd.concat([object_collection['object_id'], pd.get_dummies(object_collection['name'], prefix='collection', prefix_sep='_')], axis=1)


# %%
# del object_collection
# gc.collect()
# technique = pd.read_csv('../input/technique.csv')
# technique_sep = pd.concat(
#     [technique, technique['name'].str.split(" ", expand=True)], axis=1)
# technique_sep.drop('name', axis=1, inplace=True)
# technique_sep.columns = ['object_id', 'name_1',
#                          'name_2', 'name_3', 'name_4', 'name_5']
# technique_sep = technique_sep.melt(var_name="name", id_vars='object_id')
# technique_sep = pd.concat([technique_sep['object_id'], pd.get_dummies(
#     technique_sep['value'], prefix="technique", prefix_sep="_")], axis=1)

# # %%
# technique_sep['tech_num'] = 5 - technique_sep[['name_1',
#                                                'name_2', 'name_3', 'name_4', 'name_5']].isnull().sum(axis=1)
#                   'name_2', 'name_3', 'name_4', 'name_5']].isnull().sum(axis=1)

# %% production_place
# production_place= pd.read_csv('../input/production_place.csv')
# production_place.describe()

# # %%
# production_place['question']= 0
# production_place['question'][production_place['name'].str.contains('\? ')]= 1
# production_place['name'].replace('\? ', '', inplace=True)

# # %% prodduction placeを国名にする
# # https://www.guruguru.science/competitions/16/discussions/970ced6d-f974-4979-8f04-dbcf1c2f51a0/
# place_list = production_place['name'].unique()
# print(len(place_list))
# country_dict = {}
# # i=0
# for place in place_list:
#     try:
#         country = place2country(place)
#         country_dict[place] = country
#     except:
#         # 国名を取得できない場合はnan
#         # print(place)
#         country_dict[place] = np.nan
#     # i += 1
#     # print(i)
# production_place['country_name'] = production_place['name'].map(country_dict)
