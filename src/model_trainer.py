from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib


class ModelTrainer:

    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}

        print("Model Trainer Ready!")
        self._split_data()
        self._scale_features()

    def _split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        print(f"Training flowers: {len(self.X_train)}")
        print(f"Testing flowers: {len(self.X_test)}")

    def _scale_features(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.scaler = scaler
        print("Features scaled!")

    def train_decision_tree(self):
        print("\nTraining Decision Tree...")
        model = DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = model
        print("Done!")
        return model

    def train_random_forest(self):
        print("\nTraining Random Forest...")
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        print("Done!")
        return model

    def train_logistic_regression(self):
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        print("Done!")
        return model

    def train_svm(self):
        print("\nTraining SVM...")
        model = SVC(kernel='rbf', random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        self.models['SVM'] = model
        print("Done!")
        return model

    def train_all_models(self):
        print("\n--- TRAINING ALL 4 MODELS ---")
        self.train_decision_tree()
        self.train_random_forest()
        self.train_logistic_regression()
        self.train_svm()
        print("\nAll models trained!")

    def get_model(self, name):
        return self.models.get(name)

    def get_all_models(self):
        return self.models

    def save_model(self, name, filepath):
        model = self.models.get(name)
        if model:
            joblib.dump(model, filepath)
            print(f"Model saved: {filepath}")

    def save_scaler(self, filepath):
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved: {filepath}")

    def get_train_test_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test