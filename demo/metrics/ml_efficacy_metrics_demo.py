import pandas as pd
from sdv.metrics.demos import load_single_table_demo
from sdv.metrics.tabular import LinearRegression


pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

real_data, synthetic_data, metadata = load_single_table_demo()
real_data = real_data[['second_perc', 'high_perc', 'degree_perc', 'experience_years', 'employability_perc', 'mba_perc', 'salary']].fillna(0)
synthetic_data = synthetic_data[['second_perc', 'high_perc', 'degree_perc', 'experience_years', 'employability_perc', 'mba_perc', 'salary']].fillna(0)
print(real_data)
print(synthetic_data)
print(metadata)

print(LinearRegression.compute(synthetic_data, real_data, target='mba_perc'))
