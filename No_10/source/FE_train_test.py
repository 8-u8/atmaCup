import pandas as pd
import numpy as np

from FE_utility_function import sub_title_rep


def Feature_engineering(df, is_train=True):
    # train/testで振る舞いを変える
    # adversarial validationしてから考えてもいい。
    output = pd.DataFrame()
    output['object_id'] = df['object_id']

    # size情報
    output['sub_title'] = df['sub_title']
    output = sub_title_rep(output)
    print(output.columns.values)

    # 作品の欠落情報数
    # output = df[['object_id', 'art_series_id', 'dating_period']]
    output['info_omitted'] = df.isnull().sum(axis=1)

    # sub title
    output['sub_title_len'] = df['sub_title'].str.len()

    # more title
    output['more_title_len'] = df['more_title'].str.len()

    # output = output.merge(sub_title, how='inner', on='object_id')
    # タイトルの長さ
    output['title_length'] = df['title'].str.len()
    output['long_title_length'] = df['long_title'].str.len()
    output['diff_of_title_and_long_title'] = output['title_length'] - \
        output['long_title_length']

    # descriptionの長さ
    output['description_length'] = df['description'].str.len()

    # 自画像系
    output['Self_in_title'] = 0
    output['Self_in_title'][df['title'].str.contains('Self', case=False)] = 1

    # copyright_holder
    output['copyright_holder'] = df['copyright_holder']
    output = pd.get_dummies(
        output, columns=['copyright_holder'], dummy_na=True)
    # 収集方法
    output['acquisition_method'] = df['acquisition_method']
    output = pd.get_dummies(
        output, columns=['acquisition_method'], dummy_na=True)

    # 収集年
    df['acquisition_date'] = pd.to_datetime(df['acquisition_date'])
    output['acquisition_year'] = df['acquisition_date'].dt.year

    # 収集年と作成年の差
    output['diff_of_acqui_and_dating'] = output['acquisition_year'] - \
        df['dating_sorting_date']
    # 収集日の曜日
    output['acquisition_dayofweek'] = df['acquisition_date'].dt.dayofweek

    # 有名画家作品ダミー変数
    # レンブラント
    output['is_Rembrandt'] = 0
    output['is_Rembrandt'][df['principal_maker'].str.contains('Rembrandt')] = 1

    # ヒーム
    output['is_Heem'] = 0
    output['is_Heem'][df['principal_maker'].str.contains('Heem')] = 1

    # フェルメール
    output['is_Vermeer'] = 0
    output['is_Vermeer'][df['principal_maker'].str.contains('Vermeer')] = 1

    # ゴッホ
    output['is_Gogh'] = 0
    output['is_Gogh'][df['principal_maker'].str.contains('Gogh')] = 1

    # 作者不明
    output['is_anonymous'] = 0
    output['is_anonymous'][df['principal_maker'].str.contains('anonymous')] = 1

    # 製作期間
    # output['dating_year_early'] = df['dating_year_early']
    # output['dating_year_late'] = df['dating_year_late']
    output['production_interval'] = df['dating_year_late'] - \
        df['dating_year_early']

    # output['era'] もしかするとmeltったほうがいいかもしれん

    '''
    絵画の時代背景ベースでの年代ワケ。
    ポテトチップス(https://www.guruguru.science/competitions/16/discussions/14492ad2-fea7-4265-ba89-07a6b4d07d8b/)
    の自己消費
    '''
    output['Gothic'] = 0
    output['Gothic'][df['dating_sorting_date'] <= 1400] = 1

    # print(output['Gothic'])

    output['Renaisasse'] = 0
    output['Renaisasse'][(df['dating_sorting_date'] > 1400) & (
        df['dating_sorting_date'] <= 1500)] = 1

    output['Gold_Nerderland_era'] = 0
    output['Gold_Nerderland_era'][(df['dating_sorting_date'] > 1500) & (
        df['dating_sorting_date'] <= 1600)] = 1

    output['baroque_art'] = 0
    output['baroque_art'][(df['dating_sorting_date'] > 1600) & (
        df['dating_sorting_date'] <= 1700)] = 1

    output['impressionism_art'] = 0
    output['impressionism_art'][(df['dating_sorting_date'] > 1700) & (
        df['dating_sorting_date'] <= 1800)] = 1

    output['modern_art'] = 0
    output['modern_art'][(df['dating_sorting_date'] > 1800)
                         & (df['dating_sorting_date'] <= 1945)] = 1

    output['new_art'] = 0
    output['new_art'][(df['dating_sorting_date'] > 1945)] = 1

    print('fe done')
    print(output.columns.values)
    return output
