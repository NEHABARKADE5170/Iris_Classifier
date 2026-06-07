import numpy as np
import pandas as pd


class DataLoader:

    def __init__(self):
        # Load from CSV file
        df = pd.read_csv('data/isris_classification.csv')

        # Drop the Id column (not needed)
        df = df.drop('Id', axis=1)

        # Separate measurements and species
        self.X = df.drop('Species', axis=1).values
        self.feature_names = df.drop('Species', axis=1).columns.tolist()

        # Convert species names to numbers (0, 1, 2)
        species_categorical = pd.Categorical(df['Species'])
        self.y = species_categorical.codes
        self.target_names = species_categorical.categories.tolist()

    def get_dataset_info(self):
        print("=" * 60)
        print("IRIS DATASET INFORMATION")
        print("=" * 60)
        print(f"\nTotal flowers: {self.X.shape[0]}")
        print(f"Measurements per flower: {self.X.shape[1]}")
        print(f"\nSpecies:")
        for i, name in enumerate(self.target_names):
            count = np.sum(self.y == i)
            print(f"  {i}. {name} - {count} flowers")
        print(f"\nMeasurements:")
        for i, name in enumerate(self.feature_names):
            print(f"  {i+1}. {name}")
        print("=" * 60)

    def show_sample_data(self, n_samples=5):
        print(f"\nFirst {n_samples} Flowers:")
        print("-" * 60)
        for i in range(n_samples):
            print(f"\nFlower {i+1}:")
            print(f"  Measurements: {self.X[i]}")
            print(f"  Species: {self.target_names[self.y[i]]}")

    def get_data(self):
        return self.X, self.y, self.feature_names, self.target_names