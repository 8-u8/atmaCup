# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from PIL import ImageColor

# %%
color = pd.read_csv('../input/color.csv')


# %%
color = pd.concat([color, pd.DataFrame(color['hex'].str.strip().map(
    ImageColor.getrgb).values.tolist(), columns=['color_R', 'color_G', 'color_B'])], axis=1)

# %%color_id = color['object_id'].unique()

color['ratio'] = color['percentage']/100
color['RGB_sum'] = color[['color_R', 'color_G', 'color_B']].sum(axis=1)
color['R_ratio'] = color['color_R'] / color['RGB_sum']
color['G_ratio'] = color['color_G'] / color['RGB_sum']
color['B_ratio'] = color['color_B'] / color['RGB_sum']

# %%
color.drop(['percentage', 'hex'], axis=1, inplace=True)

# %%.agg('kurt')
color_gb = pd.DataFrame()
color_gb['color_ratio_mean'] = color.groupby('object_id')['ratio'].mean()
color_gb['R_ratio_mean'] = color.groupby('object_id')['R_ratio'].mean()
color_gb['G_ratio_mean'] = color.groupby('object_id')['G_ratio'].mean()
color_gb['B_ratio_mean'] = color.groupby('object_id')['B_ratio'].mean()
color_gb['color_RGB_mean'] = color.groupby('object_id')['RGB_sum'].mean()

# %%
color_gb['color_ratio_var'] = color.groupby('object_id')['ratio'].var()
color_gb['color_R_ratio_var'] = color.groupby('object_id')['R_ratio'].var()
color_gb['G_ratio_var'] = color.groupby('object_id')['G_ratio'].var()
color_gb['B_ratio_var'] = color.groupby('object_id')['B_ratio'].var()
color_gb['color_RGB_var'] = color.groupby('object_id')['RGB_sum'].var()
# %%
color_gb['color_ratio_skew'] = color.groupby('object_id')['ratio'].skew()
color_gb['R_ratio_skew'] = color.groupby('object_id')['R_ratio'].skew()
color_gb['G_ratio_skew'] = color.groupby('object_id')['G_ratio'].skew()
color_gb['B_ratio_skew'] = color.groupby('object_id')['B_ratio'].skew()
color_gb['color_RGB_skew'] = color.groupby('object_id')['RGB_sum'].skew()
# %%
color_gb['colot_ratio_kurt'] = color.groupby(
    'object_id')['ratio'].apply(pd.DataFrame.kurt)
color_gb['R_ratio_kurt'] = color.groupby(
    'object_id')['R_ratio'].apply(pd.DataFrame.kurt)
color_gb['G_ratio_kurt'] = color.groupby(
    'object_id')['G_ratio'].apply(pd.DataFrame.kurt)
color_gb['B_ratio_kurt'] = color.groupby(
    'object_id')['B_ratio'].apply(pd.DataFrame.kurt)

color_gb['color_RGB_kurt'] = color.groupby(
    'object_id')['RGB_sum'].apply(pd.DataFrame.kurt)

# %%
color_gb.head()

# %%
color_gb['object_id'] = color_gb.index
color_gb.reset_index(drop=True)

# %%
color_gb.to_csv('../middle/color/color_id_unique.csv', index=False)

# %% ===========palette============
palette = pd.read_csv('../input/palette.csv')
# %%
palette['rgb_sum'] = palette[['color_r', 'color_g', 'color_b']].sum(axis=1)
palette['r_x_ratio'] = palette['color_r'] / palette['rgb_sum']
palette['g_x_ratio'] = palette['color_g'] / palette['rgb_sum']
palette['b_x_ratio'] = palette['color_b'] / palette['rgb_sum']
# %%
palette_gb = pd.DataFrame
# %%.agg('kurt')
palette_gb = pd.DataFrame()
palette_gb['palette_ratio_mean'] = palette.groupby('object_id')['ratio'].mean()
palette_gb['r_x_ratio_mean'] = palette.groupby('object_id')['r_x_ratio'].mean()
palette_gb['g_x_ratio_mean'] = palette.groupby('object_id')['g_x_ratio'].mean()
palette_gb['b_x_ratio_mean'] = palette.groupby('object_id')['b_x_ratio'].mean()
palette_gb['palette_RGB_mean'] = palette.groupby('object_id')['rgb_sum'].mean()

# %%
palette_gb['palette_ratio_var'] = palette.groupby('object_id')['ratio'].var()
palette_gb['r_x_ratio_var'] = palette.groupby('object_id')['r_x_ratio'].var()
palette_gb['g_x_ratio_var'] = palette.groupby('object_id')['g_x_ratio'].var()
palette_gb['b_x_ratio_var'] = palette.groupby('object_id')['b_x_ratio'].var()
palette_gb['palette_RGB_var'] = palette.groupby('object_id')['rgb_sum'].var()
# %%
palette_gb['palette_ratio_skew'] = palette.groupby('object_id')['ratio'].skew()
palette_gb['r_x_ratio_skew'] = palette.groupby('object_id')['r_x_ratio'].skew()
palette_gb['g_x_ratio_skew'] = palette.groupby('object_id')['g_x_ratio'].skew()
palette_gb['b_x_ratio_skew'] = palette.groupby('object_id')['b_x_ratio'].skew()
palette_gb['paletteRGB_skew'] = palette.groupby('object_id')['rgb_sum'].skew()
# %%
palette_gb['palette_ratio_kurt'] = palette.groupby(
    'object_id')['ratio'].apply(pd.DataFrame.kurt)
palette_gb['r_x_ratio_kurt'] = palette.groupby(
    'object_id')['r_x_ratio'].apply(pd.DataFrame.kurt)
palette_gb['g_x_ratio_kurt'] = palette.groupby(
    'object_id')['g_x_ratio'].apply(pd.DataFrame.kurt)
palette_gb['b_x_ratio_kurt'] = palette.groupby(
    'object_id')['b_x_ratio'].apply(pd.DataFrame.kurt)

palette_gb['palette_RGB_kurt'] = palette.groupby(
    'object_id')['rgb_sum'].apply(pd.DataFrame.kurt)

# %%
palette_gb['object_id'] = palette_gb.index
palette_gb.reset_index(drop=True)

# %%
palette_gb.to_csv('../middle/color/palette_id_unique.csv', index=False)

# %%
