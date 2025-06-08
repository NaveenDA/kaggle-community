import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, StackingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, y=None, is_training=True, scaler=None, pca=None, feature_selector=None):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Extract features from Cabin
    df[['Deck', 'Cabin_num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    
    # Convert Cabin_num to numeric, replacing non-numeric values with NaN
    df['Cabin_num'] = pd.to_numeric(df['Cabin_num'], errors='coerce')
    
    # Extract group and number from PassengerId
    df[['Group', 'Number']] = df['PassengerId'].str.split('_', expand=True)
    df['Group'] = df['Group'].astype(int)
    df['Number'] = df['Number'].astype(int)
    
    # Calculate spending features
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['SpendingPerService'] = df['TotalSpending'] / df[spending_cols].astype(bool).sum(axis=1)
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    
    # Create spending categories with handling for duplicate values
    try:
        df['SpendingCategory'] = pd.qcut(df['TotalSpending'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    except ValueError:
        # If qcut fails, create categories based on spending ranges
        spending_ranges = [0, 100, 500, 1000, 5000, float('inf')]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['SpendingCategory'] = pd.cut(df['TotalSpending'], bins=spending_ranges, labels=labels)
    
    # Create luxury spending indicator
    df['LuxurySpending'] = df[['Spa', 'VRDeck']].sum(axis=1)
    df['IsLuxurySpender'] = (df['LuxurySpending'] > df['LuxurySpending'].median()).astype(int)
    
    # Create deck categories
    deck_mapping = {
        'A': 'Upper',
        'B': 'Upper',
        'C': 'Upper',
        'D': 'Middle',
        'E': 'Middle',
        'F': 'Middle',
        'G': 'Lower',
        'T': 'Lower'
    }
    df['DeckCategory'] = df['Deck'].map(deck_mapping)
    
    # Calculate family size and group features
    df['FamilySize'] = df.groupby('Group')['Group'].transform('count')
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['GroupSize'] = df.groupby('Group')['Group'].transform('size')
    
    # Add more sophisticated features
    df['AgeGroup'] = pd.qcut(df['Age'], q=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Very Senior'], duplicates='drop')
    df['SpendingRatio'] = df['TotalSpending'] / (df['Age'] + 1)  # Add 1 to avoid division by zero
    df['LuxuryRatio'] = df['LuxurySpending'] / (df['TotalSpending'] + 1)
    df['ServiceDiversity'] = df[spending_cols].astype(bool).sum(axis=1)
    
    # Create spending patterns
    df['IsHighSpender'] = (df['TotalSpending'] > df['TotalSpending'].quantile(0.75)).astype(int)
    df['IsLowSpender'] = (df['TotalSpending'] < df['TotalSpending'].quantile(0.25)).astype(int)
    df['SpendingBalance'] = df['TotalSpending'] - df['TotalSpending'].mean()
    
    # Add more sophisticated features
    df['SpendingPerPerson'] = df['TotalSpending'] / (df['FamilySize'] + 1)
    df['LuxuryPerPerson'] = df['LuxurySpending'] / (df['FamilySize'] + 1)
    df['ServiceUtilization'] = df['ServiceDiversity'] / len(spending_cols)
    df['SpendingEfficiency'] = df['TotalSpending'] / (df['ServiceDiversity'] + 1)
    
    # Create deck-based features
    df['IsUpperDeck'] = df['DeckCategory'].isin(['Upper']).astype(int)
    df['IsMiddleDeck'] = df['DeckCategory'].isin(['Middle']).astype(int)
    df['IsLowerDeck'] = df['DeckCategory'].isin(['Lower']).astype(int)
    
    # Create group-based features
    df['GroupSpending'] = df.groupby('Group')['TotalSpending'].transform('sum')
    df['GroupLuxurySpending'] = df.groupby('Group')['LuxurySpending'].transform('sum')
    df['GroupSize'] = df.groupby('Group')['Group'].transform('size')
    df['IsLargeGroup'] = (df['GroupSize'] > df['GroupSize'].median()).astype(int)
    
    # Add new features
    df['Age_Group'] = df['Age'] * df['Group']
    df['Spending_Group'] = df['TotalSpending'] * df['Group']
    df['Luxury_Group'] = df['LuxurySpending'] * df['Group']
    df['Service_Group'] = df['ServiceDiversity'] * df['Group']
    
    # Create spending patterns by deck
    deck_spending = df.groupby('Deck')['TotalSpending'].transform('mean')
    df['DeckSpendingDiff'] = df['TotalSpending'] - deck_spending
    df['IsHighSpenderForDeck'] = (df['DeckSpendingDiff'] > 0).astype(int)
    
    # Create spending patterns by destination
    dest_spending = df.groupby('Destination')['TotalSpending'].transform('mean')
    df['DestSpendingDiff'] = df['TotalSpending'] - dest_spending
    df['IsHighSpenderForDest'] = (df['DestSpendingDiff'] > 0).astype(int)
    
    # Create spending patterns by home planet
    planet_spending = df.groupby('HomePlanet')['TotalSpending'].transform('mean')
    df['PlanetSpendingDiff'] = df['TotalSpending'] - planet_spending
    df['IsHighSpenderForPlanet'] = (df['PlanetSpendingDiff'] > 0).astype(int)
    
    # Fill missing values
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckCategory', 'SpendingCategory', 'AgeGroup']
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num']
    
    # Fill categorical columns with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Fill numerical columns with median
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Convert boolean columns to int
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    df['VIP'] = df['VIP'].astype(int)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in ['HomePlanet', 'Destination', 'Deck', 'Side', 'DeckCategory', 'SpendingCategory', 'AgeGroup']:
        df[col] = le.fit_transform(df[col])
    
    # Create interaction features
    df['Age_VIP'] = df['Age'] * df['VIP']
    df['CryoSleep_Spending'] = df['CryoSleep'] * df['TotalSpending']
    df['FamilySize_Spending'] = df['FamilySize'] * df['TotalSpending']
    df['Age_FamilySize'] = df['Age'] * df['FamilySize']
    df['VIP_Spending'] = df['VIP'] * df['TotalSpending']
    df['Age_Deck'] = df['Age'] * df['DeckCategory']
    df['VIP_Deck'] = df['VIP'] * df['DeckCategory']
    df['CryoSleep_Deck'] = df['CryoSleep'] * df['DeckCategory']
    
    # Create more interaction features
    df['Age_SpendingRatio'] = df['Age'] * df['SpendingRatio']
    df['VIP_ServiceDiversity'] = df['VIP'] * df['ServiceDiversity']
    df['CryoSleep_ServiceDiversity'] = df['CryoSleep'] * df['ServiceDiversity']
    df['Age_ServiceDiversity'] = df['Age'] * df['ServiceDiversity']
    df['FamilySize_ServiceDiversity'] = df['FamilySize'] * df['ServiceDiversity']
    
    # Create polynomial features for important numerical columns
    numerical_features = ['Age', 'TotalSpending', 'FamilySize', 'SpendingRatio', 'LuxuryRatio', 'ServiceDiversity']
    
    # Ensure all numerical features are filled
    for feature in numerical_features:
        if df[feature].isnull().any():
            df[feature] = df[feature].fillna(df[feature].median())
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[numerical_features])
    poly_feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
    df = pd.concat([df, df_poly], axis=1)
    
    # Select features for training
    features = [
        'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'Deck', 'Cabin_num', 'Side', 'Group', 'Number', 'TotalSpending',
        'SpendingPerService', 'HasSpending', 'LuxurySpending', 'IsLuxurySpender',
        'DeckCategory', 'FamilySize', 'IsAlone', 'GroupSize', 'SpendingCategory',
        'AgeGroup', 'SpendingRatio', 'LuxuryRatio', 'ServiceDiversity',
        'IsHighSpender', 'IsLowSpender', 'SpendingBalance', 'SpendingPerPerson',
        'LuxuryPerPerson', 'ServiceUtilization', 'SpendingEfficiency',
        'IsUpperDeck', 'IsMiddleDeck', 'IsLowerDeck', 'GroupSpending',
        'GroupLuxurySpending', 'IsLargeGroup', 'Age_Group', 'Spending_Group',
        'Luxury_Group', 'Service_Group', 'DeckSpendingDiff', 'IsHighSpenderForDeck',
        'DestSpendingDiff', 'IsHighSpenderForDest', 'PlanetSpendingDiff',
        'IsHighSpenderForPlanet', 'Age_VIP', 'CryoSleep_Spending', 'FamilySize_Spending',
        'Age_FamilySize', 'VIP_Spending', 'Age_Deck', 'VIP_Deck', 'CryoSleep_Deck',
        'Age_SpendingRatio', 'VIP_ServiceDiversity', 'CryoSleep_ServiceDiversity',
        'Age_ServiceDiversity', 'FamilySize_ServiceDiversity'
    ] + poly_feature_names
    
    X = df[features]
    
    # Fill any remaining NaN values with median
    X = X.fillna(X.median())
    
    # Scale the features
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Apply PCA
    if is_training:
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = pca.transform(X_scaled)
    
    # Feature selection
    if is_training:
        feature_selector = SelectFromModel(
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            prefit=False
        )
        X_selected = feature_selector.fit_transform(X_pca, y)
    else:
        X_selected = feature_selector.transform(X_pca)
    
    return X_selected, scaler, pca, feature_selector

def train_model():
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
    
    # Initialize base models with optimized parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        eval_metric='logloss'
    )
    
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        verbose=-1
    )
    
    cat_model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_strength=0.1,
        reg_lambda=1,
        random_state=42,
        verbose=False
    )
    
    # Train individual models and get their predictions
    print("\nTraining individual models...")
    
    # XGBoost
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_train_pred = xgb_model.predict_proba(X_train)[:, 1]
    
    # LightGBM
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_train_pred = lgb_model.predict_proba(X_train)[:, 1]
    
    # CatBoost
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    cat_train_pred = cat_model.predict_proba(X_train)[:, 1]
    
    # Calculate individual model accuracies
    xgb_acc = accuracy_score(y_val, (xgb_pred > 0.5).astype(int))
    lgb_acc = accuracy_score(y_val, (lgb_pred > 0.5).astype(int))
    cat_acc = accuracy_score(y_val, (cat_pred > 0.5).astype(int))
    
    # Calculate weights based on accuracies
    total_acc = xgb_acc + lgb_acc + cat_acc
    xgb_weight = xgb_acc / total_acc
    lgb_weight = lgb_acc / total_acc
    cat_weight = cat_acc / total_acc
    
    print("\nModel Weights:")
    print(f"XGBoost: {xgb_weight:.4f}")
    print(f"LightGBM: {lgb_weight:.4f}")
    print(f"CatBoost: {cat_weight:.4f}")
    
    # Create weighted ensemble predictions
    val_predictions_proba = (
        xgb_weight * xgb_pred +
        lgb_weight * lgb_pred +
        cat_weight * cat_pred
    )
    
    # Optimize threshold
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_threshold = 0.5
    best_accuracy = 0
    
    for threshold in thresholds:
        predictions = (val_predictions_proba > threshold).astype(int)
        accuracy = accuracy_score(y_val, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.3f}")
    val_predictions = (val_predictions_proba > best_threshold).astype(int)
    
    # Print model performance
    print("\nValidation Set Performance:")
    print("Accuracy:", accuracy_score(y_val, val_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, val_predictions))
    print("\nClassification Report:")
    print(classification_report(y_val, val_predictions))
    
    # Print individual model performances
    print("\nIndividual Model Performances:")
    print("XGBoost Accuracy:", xgb_acc)
    print("LightGBM Accuracy:", lgb_acc)
    print("CatBoost Accuracy:", cat_acc)
    
    # Save models and transformers
    models = {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'cat': cat_model,
        'weights': {
            'xgb': xgb_weight,
            'lgb': lgb_weight,
            'cat': cat_weight
        },
        'threshold': best_threshold,
        'scaler': scaler,
        'pca': pca,
        'feature_selector': feature_selector
    }
    joblib.dump(models, 'spaceship_titanic_models.joblib')
    print("\nModels saved as spaceship_titanic_models.joblib")
    
    return models

if __name__ == "__main__":
    models = train_model()
