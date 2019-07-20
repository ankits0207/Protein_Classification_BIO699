import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

plt.rc('font', family='serif', size=14, weight='bold')

idx = ['DEG', 'CLC', 'CS', 'DC1', 'DC2', 'SGC', 'EBC']
df = pd.read_csv('HeatMap.csv')
df.index = idx
sns.heatmap(df, annot=True)
plt.yticks(rotation=0)

plt.show()
