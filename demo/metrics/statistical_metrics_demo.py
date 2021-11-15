import pandas as pd
from sdv.metrics.demos import load_single_table_demo
from sdv.metrics.tabular import CSTest, KSTest


pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

real_data, synthetic_data, metadata = load_single_table_demo()
print(real_data)
print(synthetic_data)
print(metadata)

print(CSTest.compute(real_data, synthetic_data))
print(KSTest.compute(real_data, synthetic_data))
