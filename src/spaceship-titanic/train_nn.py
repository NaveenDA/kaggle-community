import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from train import preprocess_data

class SpaceshipDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)
        self.y = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class SpaceshipNN(nn.Module):
    def __init__(self, input_size):
        super(SpaceshipNN, self).__init__()
        
        # Define the architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),

           nn.Dropout(0.3),
            
            nn.Linear(400, 212),
            nn.BatchNorm1d(212),
            nn.ReLU(),
           nn.Dropout(0.3),
            
            nn.Linear(212, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
           nn.Dropout(0.3),
            
            nn.Linear(100, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
           nn.Dropout(0.3),
            
            nn.Linear(25, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

def train_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load the data
    train_df = pd.read_csv('../datasets/spaceship-titanic/train.csv')
    y = train_df['Transported'].astype(int)
    
    # Preprocess the data
    X, scaler, pca, feature_selector = preprocess_data(train_df, y=y, is_training=True)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)
    print("\nClass distribution in training set:")
    print(y_train.value_counts(normalize=True))
    
    # Create datasets and dataloaders
    train_dataset = SpaceshipDataset(X_train, y_train)
    val_dataset = SpaceshipDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpaceshipNN(input_size=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    n_epochs = 200
    best_val_acc = 0
    
    print("\nTraining neural network...")
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
        
        # Calculate metrics
        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}:')
            print(f'Training Loss: {train_loss/len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
            print(f'Validation Accuracy: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with accuracy: {best_val_acc:.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Final evaluation
    model.eval()
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            val_preds.extend((outputs > 0.5).cpu().numpy())
            val_true.extend(y_batch.cpu().numpy())
    
    # Print final performance
    print("\nFinal Validation Set Performance:")
    print("Accuracy:", accuracy_score(val_true, val_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_true, val_preds))
    print("\nClassification Report:")
    print(classification_report(val_true, val_preds))
    
    # Save models and transformers
    models = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'feature_selector': feature_selector
    }
    joblib.dump(models, 'spaceship_titanic_nn_models.joblib')
    print("\nModels saved as spaceship_titanic_nn_models.joblib")
    
    return models

if __name__ == "__main__":
    train_model() 