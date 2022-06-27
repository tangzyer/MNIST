import pandas as pd

df = pd.read_csv('student-mat.csv')

features = df.copy().drop(['G1', 'G2', 'G3'], axis=1)
target = df.copy()['G1']

ohe_columns = []
for col in features.columns:
    if col not in ['age', 'absences']:
        ohe_columns.append(col)

features = pd.get_dummies(features, drop_first=True, columns=ohe_columns)

