import pandas as pd
import numpy as np
import torch
import joblib
from train import preprocess_data
from train_nn import SpaceshipDataset, SpaceshipNN

def make_predictions():
    # Load test data
    test_df = pd.read_csv('../datasets/spaceship-titanic/test.csv')
    
    # Load trained models and transformers
    models = joblib.load('spaceship_titanic_nn_models.joblib')
    
    # Preprocess test data
    X_test, _, _, _ = preprocess_data(test_df, is_training=False, 
                                    scaler=models['scaler'],
                                    pca=models['pca'],
                                    feature_selector=models['feature_selector'])
    
    # Create dataset and dataloader
    test_dataset = SpaceshipDataset(X_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpaceshipNN(input_size=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            predictions.extend((outputs > 0.5).cpu().numpy())
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': predictions
    })
    
    # Save predictions
    submission.to_csv('submission_nn.csv', index=False)
    print("\nPredictions saved to submission_nn.csv")
    print("\nFirst few predictions:")
    print(submission.head())

if __name__ == "__main__":
    make_predictions() 