"""合成数据"""

import pandas as pd
from sdv.tabular import GaussianCopula


data = pd.read_csv("/Users/sunxiaopeng9/PycharmProjects/synthetic_data/data/train_data_2.csv", header=None)
model = GaussianCopula()
data = data.head(20000)
data.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
print(data)
model.fit(data)

syn_data = model.sample(data.shape[0])
syn_data.to_csv("/Users/sunxiaopeng9/PycharmProjects/synthetic_data/data/train_data_2_syn.csv", header=None, index=False)
