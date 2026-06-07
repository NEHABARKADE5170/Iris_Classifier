import numpy as np


class Predictor:

    def __init__(self, model, scaler, feature_names, target_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.target_names = target_names

    def predict_single(self, measurements):
        measurements = np.array(measurements).reshape(1, -1)
        measurements_scaled = self.scaler.transform(measurements)
        prediction = self.model.predict(measurements_scaled)[0]
        species = self.target_names[prediction]

        try:
            probabilities = self.model.predict_proba(measurements_scaled)[0]
            confidence = max(probabilities)
        except AttributeError:
            probabilities = None
            confidence = None

        return {
            'species': species,
            'confidence': confidence,
            'probabilities': probabilities,
            'measurements': measurements[0]
        }

    def print_prediction(self, result):
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\nMeasurements:")
        for i, (feature, value) in enumerate(zip(self.feature_names, result['measurements'])):
            print(f"  {i+1}. {feature}: {value:.2f} cm")
        print(f"\nPredicted Species: {result['species'].upper()}")
        if result['confidence'] is not None:
            print(f"Confidence: {result['confidence']:.2%}")
            if result['probabilities'] is not None:
                print(f"\nProbability breakdown:")
                for species, prob in zip(self.target_names, result['probabilities']):
                    bar = "#" * int(prob * 25)
                    print(f"  {species:<15} {bar} {prob:.2%}")
        print("="*60)