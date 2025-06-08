import pandas as pd
import numpy as np
import joblib
from train import preprocess_data

def make_predictions():
    # Load test data
    test_df = pd.read_csv('../datasets/spaceship-titanic/test.csv')
    
    # Load trained models and transformers
    models = joblib.load('spaceship_titanic_models.joblib')
    
    # Preprocess test data
    X_test, _, _, _ = preprocess_data(test_df, is_training=False, 
                                    scaler=models['scaler'],
                                    pca=models['pca'],
                                    feature_selector=models['feature_selector'])
    
    # Get predictions from each model
    xgb_pred = models['xgb'].predict_proba(X_test)[:, 1]
    lgb_pred = models['lgb'].predict_proba(X_test)[:, 1]
    cat_pred = models['cat'].predict_proba(X_test)[:, 1]
    
    # Get weights
    weights = models['weights']
    threshold = models['threshold']
    
    # Create weighted ensemble predictions
    predictions_proba = (
        weights['xgb'] * xgb_pred +
        weights['lgb'] * lgb_pred +
        weights['cat'] * cat_pred
    )
    
    # Apply threshold
    predictions = (predictions_proba > threshold).astype(bool)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': predictions
    })
    
    # Save predictions
    submission.to_csv('submission.csv', index=False)
    print("\nPredictions saved to submission.csv")
    print("\nFirst few predictions:")
    print(submission.head())

if __name__ == "__main__":
    make_predictions()
