import pandas as pd
from sdv.demo import load_tabular_demo
from sdv.tabular import GaussianCopula


pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

data = load_tabular_demo('student_placements')
print(data)
print(data.experience_years.value_counts())

model = GaussianCopula()
model.fit(data)

new_data = model.sample(300)
print(new_data)
print(new_data.experience_years.value_counts())
