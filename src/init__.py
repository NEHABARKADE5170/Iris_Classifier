import pandas as pd

df = pd.read_csv('data/iris_classification.csv')

# Separate features and target
self.X = df.drop('species', axis=1).values
self.y = pd.Categorical(df['species']).codes
self.feature_names = df.drop('species', axis=1).columns.tolist()
self.target_names = pd.Categorical(df['species']).categories.tolist()