# %%
from kaeru_BERT import BertSequenceVectorizer
from kaeru_BERT import result_description_pca
from kaeru_BERT import result_description_svd
from kaeru_BERT import result_description_tsne

import os
import sys
import gc

import pandas as pd
import numpy as np

# %%
train = pd.read_csv('../middle/descriptions/kaeru_bert_train.csv')
test = pd.read_csv('../middle/descriptions/kaeru_bert_test.csv')

# %%
train_bert = train[train.columns[train.columns.str.contains('bert')]]
test_bert = test[test.columns[test.columns.str.contains('bert')]]
train_id = train[['object_id']]
test_id = test[['object_id']]
del train, test
gc.collect()
# %%
# pca_result_train = result_description_pca(train_bert, 200)
# %%
# svd_result_train = result_description_svd(train_bert, 200)
# %% 4次元以上はやべーらしい
# tsne_result_train = result_description_tsne(train_bert, 2)
# %%

comp_list = [50, 100, 150, 200]
for comp in comp_list:
    # %% 次元圧縮1. PCA 200次元
    pca_result_train = result_description_pca(train_bert, comp)
    pca_result_test = result_description_pca(test_bert, comp)
    pca_result_train.columns = ['pca_' + str(i) for i in range(comp)]
    pca_result_test.columns = ['pca_' + str(i) for i in range(comp)]

    pca_train = pd.concat([train_id, pca_result_train], axis=1)
    pca_test = pd.concat([test_id, pca_result_test], axis=1)

    pca_train.to_csv('../middle/descriptions/pca_' +
                     str(comp)+'_train.csv', index=False)
    pca_test.to_csv('../middle/descriptions/pca_' +
                    str(comp)+'_test.csv', index=False)

    # %% 次元圧縮2. SVD comp次元
    svd_result_train = result_description_svd(train_bert, comp)
    svd_result_test = result_description_pca(test_bert, comp)
    svd_result_train.columns = ['svd_' + str(i) for i in range(comp)]
    svd_result_test.columns = ['svd_' + str(i) for i in range(comp)]

    svd_train = pd.concat([train_id, pca_result_train], axis=1)
    svd_test = pd.concat([test_id, pca_result_test], axis=1)

    svd_train.to_csv('../middle/descriptions/svd_' +
                     str(comp)+'_train.csv', index=False)
    svd_test.to_csv('../middle/descriptions/svd_' +
                    str(comp)+'_test.csv', index=False)

    # %% 次元圧縮3. TSNE comp次元
    # tsne_result_train = result_description_tsne(train_bert, comp)
    # tsne_result_test = result_description_pca(test_bert, comp)
    # tsne_train = pd.concat([train_id, pca_result_train], axis=1)
    # tsne_test = pd.concat([test_id, pca_result_test], axis=1)

    # tsne_train.to_csv('../middle/descriptions/tsne' +
    #                   str(comp)+'train.csv', index=False)
    # tsne_test.to_csv('../middle/descriptions/tsne' +
    #                  str(comp)+'test.csv', index=False)
