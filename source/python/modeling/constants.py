import pandas as pd
from pathlib import Path

# Import dataset
root_path = Path("C:\\Dev_env\\Projects\\wisard_mlp_regression")
df = pd.read_csv(root_path / 'data' / 'void_fraction_dataset.csv', sep=';')
X = df.iloc[:, [1, 2, 3]].values.tolist()
y = df.iloc[:, 4].values.tolist()
