from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['dataset']
    df = pd.read_csv(file)

    print("Columns:", df.columns.tolist())
    print("First row:", df.iloc[0].tolist())

    # Drop Id column if exists
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    # Get last column as species
    species_col = df.columns[-1]
    print("Species column:", species_col)
    print("Sample species values:", df[species_col].unique())

    # Count each species
    species_counts = {}
    for name in sorted(df[species_col].astype(str).unique()):
        species_counts[name] = int(np.sum(df[species_col].astype(str) == name))

    print("Species counts:", species_counts)

    # Features
    X = df.drop(species_col, axis=1).values
    feature_names = df.drop(species_col, axis=1).columns.tolist()

    # Convert species to numbers
    species_categorical = pd.Categorical(df[species_col].astype(str))
    y = species_categorical.codes
    target_names = species_categorical.categories.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = round(accuracy * 100, 2)

    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    best_model = list(results.keys())[0]
    best_accuracy = list(results.values())[0]

    return render_template('result.html',
        species_counts=species_counts,
        results=results,
        best_model=best_model,
        best_accuracy=best_accuracy,
        total_flowers=len(df),
        feature_names=feature_names,
        target_names=target_names
    )

if __name__ == '__main__':
    app.run(debug=True)