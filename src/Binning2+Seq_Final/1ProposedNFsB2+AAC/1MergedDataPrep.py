import pandas as pd
import numpy as np

Sdf = pd.read_csv('SeqFV.csv')
NDf = pd.read_csv('NetFV.csv')
NDf['pdbId'] = NDf['pdbId'].str.upper()
MergedDf = pd.merge(Sdf, NDf, how='inner', on=['pdbId', 'label'])
MergedDf.to_csv('MergedData.csv', index=False)
print('Done')
