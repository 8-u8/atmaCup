# %%
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

sub = pd.read_csv('../input/atmacup10__sample_submission.csv')

sub['likes'] = 0


# all zero sub: 2.3799
sub.to_csv('../output/all_zero_sub.csv', index=False)


# %%

train = pd.read_csv('../input/train.csv')

# %%

train_likes = (np.log1p(train['likes'])) ** 2

# %% 仮にtrainがtestだった時にall zero subをしたらどうなるか
# 2.4263
np.sqrt(np.mean(train_likes))

# %% 仮にランダムサンプルであった場合どうなるか
train_likes_LB = train_likes.sample(frac=0.5)
print(len(train_likes_LB))
np.sqrt(np.mean(train_likes_LB))

# %%
best_pred = pd.read_csv('../output/20210311_130740_submission.csv')
y_pred_likes = (np.log1p(best_pred['likes']) ** 2)

np.sqrt(np.mean(y_pred_likes))  # 2.16256

# %%
# ランダムにtrainを50%サンプリングして、
# All_zero_subの計算を行う。
# 中心極限定理すげー。
mean_result = []
for i in range(20000):
    seed = random.randint(0, 999)
    train_likes_LB = train_likes.sample(frac=0.5)
    # print(np.sqrt(np.mean(train_likes_LB)))
    mean_result.append(np.sqrt(np.mean(train_likes_LB)))
# %% 大体正規分布
plt.hist(mean_result, bins=50)

# %%
# all_zero_subの結果が50%サンプリングした時の平均である
# 前提で、乱数をつくる。
psuedo_test_dist = []
for i in range(20000):
    psuedo_test_dist.append(random.gauss(
        2.3799, np.sqrt(0.0004133775934827785)))

# %%
# all_zero subした時の結果の比較。
# testのほうがちいさい。
# ちいさいということは、すくなくとも、
# testのlikesの最大値は、trainよりも小さい？
# zeroの多さはわからない。でも上を狙うのは悪手。
# 当たり前のように平均周りをあてていく？

plt.hist(mean_result, bins=50)
plt.hist(psuedo_test_dist, bins=50)
# %%


def rmsle(true, pred):
    # true = data.get_label()
    loss = mean_squared_log_error(true, pred) ** 0.5
    return loss
