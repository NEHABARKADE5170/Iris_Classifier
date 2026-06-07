from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.predictor import Predictor
import os


def main():

    print("\n" + "="*60)
    print("IRIS FLOWER CLASSIFICATION")
    print("="*60)

    # Create output folders
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # STEP 1: Load Data
    print("\nSTEP 1: Loading Data")
    print("-"*60)
    loader = DataLoader()
    loader.get_dataset_info()
    loader.show_sample_data(n_samples=3)
    X, y, feature_names, target_names = loader.get_data()

    # STEP 2: Train Models
    print("\nSTEP 2: Training Models")
    print("-"*60)
    trainer = ModelTrainer(X, y, test_size=0.2)
    trainer.train_all_models()
    X_train, X_test, y_train, y_test = trainer.get_train_test_data()

    # STEP 3: Evaluate Models
    print("\nSTEP 3: Evaluating Models")
    print("-"*60)
    evaluator = ModelEvaluator(X_test, y_test, target_names)
    for model_name in trainer.get_all_models().keys():
        model = trainer.get_model(model_name)
        evaluator.evaluate_model(model, model_name)
        evaluator.print_evaluation(model_name)
    evaluator.compare_models()

    # STEP 4: Save Best Model
    print("\nSTEP 4: Saving Model")
    print("-"*60)
    best_model = trainer.get_model('Random Forest')
    trainer.save_model('Random Forest', 'models/iris_model.pkl')
    trainer.save_scaler('models/iris_scaler.pkl')

    # STEP 5: Make Predictions
    print("\nSTEP 5: Making Predictions")
    print("-"*60)
    predictor = Predictor(
        model=best_model,
        scaler=trainer.scaler,
        feature_names=feature_names,
        target_names=target_names
    )

    test_flowers = [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3],
        [7.7, 2.8, 6.7, 2.0],
    ]

    for i, measurements in enumerate(test_flowers, 1):
        print(f"\nFlower {i}:")
        result = predictor.predict_single(measurements)
        predictor.print_prediction(result)

    print("\n" + "="*60)
    print("DONE! Check models/ folder for saved model.")
    print("="*60)


if __name__ == "__main__":
    main()