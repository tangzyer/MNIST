import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('log_csv_2.csv')
legends = []
for col in df.columns:
    if 'Unnamed' not in col and col != 'n':
        plt.plot(df['n'].values, df[col].values, '.-')
        legends.append(col)

plt.legend(legends)
plt.savefig('test_3.png')