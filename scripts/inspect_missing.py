import pandas as pd
from pathlib import Path
p=Path('data/final/fish_mapping_master_v2.csv')
df=pd.read_csv(p, na_values=['', 'NA', 'None', 'nan', 'NaN'])
missing = df[df['family'].isna()]['scientific_name'].dropna().head(50).tolist()
print('Missing family (sample):')
for i, n in enumerate(missing, 1):
    print(i, n)