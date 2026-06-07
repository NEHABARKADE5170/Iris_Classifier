from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


class ModelEvaluator:

    def __init__(self, X_test, y_test, target_names):
        self.X_test = X_test
        self.y_test = y_test
        self.target_names = target_names
        self.results = {}

    def evaluate_model(self, model, model_name):
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, labels=range(len(self.target_names))
        )
        self.results[model_name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return accuracy

    def print_evaluation(self, model_name):
        result = self.results[model_name]
        accuracy = result['accuracy']

        print("\n" + "="*60)
        print(f"EVALUATION: {model_name}")
        print("="*60)
        print(f"\nAccuracy: {accuracy:.2%}")
        print(f"{int(accuracy * len(self.y_test))} out of {len(self.y_test)} flowers correct")

        print(f"\nMetrics Per Species:")
        print("-"*60)
        print(f"{'Species':<15} {'Precision':<15} {'Recall':<15} {'F1':<15}")
        print("-"*60)
        for i, target in enumerate(self.target_names):
            print(f"{target:<15} {result['precision'][i]:<15.4f} {result['recall'][i]:<15.4f} {result['f1_score'][i]:<15.4f}")

        print(f"\nConfusion Matrix:")
        print("-"*60)
        cm = result['confusion_matrix']
        print(f"{'Actual/Pred':<15}", end="")
        for target in self.target_names:
            print(f"{target:<15}", end="")
        print()
        print("-"*60)
        for i, target in enumerate(self.target_names):
            print(f"{target:<15}", end="")
            for j in range(len(self.target_names)):
                print(f"{cm[i][j]:<15}", end="")
            print()
        print("="*60)

    def compare_models(self, model_names=None):
        if model_names is None:
            model_names = list(self.results.keys())

        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        accuracies = [(name, self.results[name]['accuracy']) for name in model_names if name in self.results]
        accuracies.sort(key=lambda x: x[1], reverse=True)

        accuracies = [(name, self.results[name]['accuracy']) for name in model_names if name in self.results]
        accuracies.sort(key=lambda x: x[1], reverse=True)

        medals = ["1st", "2nd", "3rd", "4th"]
        print(f"\n{'Model':<25} {'Accuracy':<15} {'Rank'}")
        print("-"*60)
        for i, (name, accuracy) in enumerate(accuracies):
            print(f"{name:<25} {accuracy:<15.2%} {medals[i]}")

        print("="*60)