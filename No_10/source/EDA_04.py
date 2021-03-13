# %%
import pandas as pd

# %%
material = pd.read_csv('../input/material.csv')

# %%
material_oh = pd.get_dummies(material, columns=['name'])

# %%
material_oh = material_oh.groupby('object_id').sum()

# %%
material_oh['object_id'] = material_oh.index
material_oh.reset_index(drop=True, inplace=True)
# %%
material_oh.to_csv('../middle/material/material_onehot.csv', index=False)
