import pandas as pd
import numpy as np

from geopy.geocoders import Nominatim

# https://www.guruguru.science/competitions/16/discussions/970ced6d-f974-4979-8f04-dbcf1c2f51a0/
# 地名/国名置換


def place2country(address):
    geolocator = Nominatim(user_agent='sample', timeout=200)
    loc = geolocator.geocode(address, language='en')
    coordinates = (loc.latitude, loc.longitude)
    location = geolocator.reverse(coordinates, language='en')
    country = location.raw['address']['country']
    return country

# https://www.guruguru.science/competitions/16/discussions/556029f7-484d-40d4-ad6a-9d86337487e2/
# sub_titleから作品のサイズ情報抽出


def sub_title_rep(df):
    for axis in ['h', 'w', 't', 'd']:
        column_name = f'size_{axis}'
        # print(column_name)
        size_info = df['sub_title'].str.extract(
            r'{} (\d*|\d*\.\d*)(cm|mm)'.format(axis))  # 正規表現を使ってサイズを抽出
        size_info = size_info.rename(columns={0: column_name, 1: 'unit'})
        size_info[column_name] = size_info[column_name].replace(
            '', np.nan).astype(float)  # dtypeがobjectになってるのでfloatに直す
        size_info[column_name] = size_info.apply(
            lambda row: row[column_name] * 10 if row['unit'] == 'cm' else row[column_name], axis=1)  # 　単位をmmに統一する
        df[column_name] = size_info[column_name]  # dfにくっつける
    df.drop('sub_title', axis=1, inplace=True)
    return df
