import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

class SklearnNeuralPipeline:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
        self.models = []
        self.feature_names = []
        self.best_model = None
        
    def load_data(self, train_path, test_path=None):
        """Load and initial inspection of data"""
        self.train_df = pd.read_csv(train_path)
        if test_path:
            self.test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_df.shape}")
        if hasattr(self, 'test_df'):
            print(f"Test data shape: {self.test_df.shape}")
        
        print("\nSurvival distribution:")
        print(self.train_df['Survived'].value_counts(normalize=True))
    
    def minimal_feature_engineering(self, df, is_train=True):
        """Minimal feature engineering - let neural networks learn patterns"""
        df = df.copy()
        
        # Basic title extraction
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Simplified title mapping
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'
        }
        df['Title'] = df['Title'].map(title_mapping).fillna('Other')
        
        # Basic family features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Simple age processing
        df['Age'].fillna(df['Age'].median(), inplace=True)
        
        # Simple fare processing
        df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'), inplace=True)
        df['Fare_Log'] = np.log1p(df['Fare'])
        
        # Basic cabin feature
        df['Has_Cabin'] = df['Cabin'].notna().astype(int)
        
        # Embarked
        df['Embarked'].fillna('S', inplace=True)
        
        # Core interaction (most important for survival)
        df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
        
        return df
    
    def neural_preprocessing(self, df, is_train=True):
        """Preprocessing optimized for neural networks"""
        df = df.copy()
        
        # Select core features
        categorical_features = ['Sex', 'Embarked', 'Title', 'Sex_Pclass']
        numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Fare_Log', 
                             'FamilySize', 'IsAlone', 'Has_Cabin']
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in df.columns:
                if is_train:
                    self.label_encoders[feature] = LabelEncoder()
                    df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    if feature in self.label_encoders:
                        df[feature] = df[feature].astype(str)
                        unknown_mask = ~df[feature].isin(self.label_encoders[feature].classes_)
                        if unknown_mask.any():
                            df.loc[unknown_mask, feature] = self.label_encoders[feature].classes_[0]
                        df[feature] = self.label_encoders[feature].transform(df[feature])
        
        # Return all features
        all_features = categorical_features + numerical_features
        available_features = [f for f in all_features if f in df.columns]
        self.feature_names = available_features
        
        return df[available_features]
    
    def create_neural_models(self):
        """Create diverse neural network models"""
        models = []
        
        # Model 1: Deep network
        mlp1 = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        models.append(('deep_mlp', mlp1))
        
        # Model 2: Wide network
        mlp2 = MLPClassifier(
            hidden_layer_sizes=(300, 150),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=123,
            early_stopping=True,
            validation_fraction=0.1
        )
        models.append(('wide_mlp', mlp2))
        
        # Model 3: Complex network
        mlp3 = MLPClassifier(
            hidden_layer_sizes=(150, 100, 75, 50),
            activation='tanh',
            solver='adam',
            alpha=0.005,
            learning_rate='adaptive',
            learning_rate_init=0.002,
            max_iter=600,
            random_state=456,
            early_stopping=True,
            validation_fraction=0.1
        )
        models.append(('complex_mlp', mlp3))
        
        # Model 4: Conservative network
        mlp4 = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='lbfgs',
            alpha=0.01,
            max_iter=300,
            random_state=789
        )
        models.append(('conservative_mlp', mlp4))
        
        # Model 5: L-BFGS solver (good for small datasets)
        mlp5 = MLPClassifier(
            hidden_layer_sizes=(80, 40),
            activation='tanh',
            solver='lbfgs',
            alpha=0.005,
            max_iter=200,
            random_state=101112
        )
        models.append(('lbfgs_mlp', mlp5))
        
        return models
    
    def create_hybrid_ensemble(self):
        """Create hybrid ensemble with neural networks and traditional ML"""
        
        # Neural networks
        neural_models = self.create_neural_models()
        
        # Traditional ML models for comparison/ensemble
        traditional_models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
            ('lr', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
        ]
        
        # Combine all models
        all_models = neural_models + traditional_models
        return all_models
    
    def train_neural_ensemble(self, X_train, y_train, X_val, y_val):
        """Train ensemble of neural networks"""
        print("\n=== TRAINING NEURAL NETWORK ENSEMBLE ===")
        
        # Scale features (critical for neural networks)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Get models
        models = self.create_neural_models()
        trained_models = []
        
        print(f"Training {len(models)} neural network models...")
        
        for name, model in models:
            print(f"\n--- Training {name} ---")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            print(f"Train accuracy: {train_acc:.4f}")
            print(f"Validation accuracy: {val_acc:.4f}")
            
            # Check for convergence (if applicable)
            if hasattr(model, 'n_iter_'):
                print(f"Converged in {model.n_iter_} iterations")
            
            trained_models.append((name, model))
        
        # Create voting ensemble
        print(f"\n--- Creating Voting Ensemble ---")
        voting_ensemble = VotingClassifier(
            estimators=trained_models,
            voting='soft'  # Use probabilities
        )
        
        # Train ensemble
        voting_ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate ensemble
        ensemble_train_pred = voting_ensemble.predict(X_train_scaled)
        ensemble_val_pred = voting_ensemble.predict(X_val_scaled)
        
        ensemble_train_acc = accuracy_score(y_train, ensemble_train_pred)
        ensemble_val_acc = accuracy_score(y_val, ensemble_val_pred)
        
        print(f"Ensemble train accuracy: {ensemble_train_acc:.4f}")
        print(f"Ensemble validation accuracy: {ensemble_val_acc:.4f}")
        
        self.models = trained_models
        self.best_model = voting_ensemble
        
        return voting_ensemble, ensemble_val_acc
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Quick hyperparameter tuning for best neural network"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Parameter grid for MLPClassifier
        param_grid = {
            'hidden_layer_sizes': [(100, 50), (150, 75), (200, 100), (100, 100, 50)],
            'alpha': [0.001, 0.005, 0.01],
            'learning_rate_init': [0.001, 0.005, 0.01]
        }
        
        # Base model
        mlp = MLPClassifier(
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True
        )
        
        # Grid search
        grid_search = GridSearchCV(
            mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def run_neural_pipeline(self, train_path, test_path=None):
        """Run complete neural network pipeline"""
        print("=== SKLEARN NEURAL NETWORK TITANIC PIPELINE ===")
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Feature engineering
        print("\n=== MINIMAL FEATURE ENGINEERING ===")
        train_engineered = self.minimal_feature_engineering(self.train_df, is_train=True)
        
        # Neural preprocessing
        print("\n=== NEURAL PREPROCESSING ===")
        X = self.neural_preprocessing(train_engineered, is_train=True)
        y = self.train_df['Survived']
        
        print(f"Features for neural network: {X.shape[1]}")
        print(f"Feature names: {self.feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training set: {X_train_final.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train neural ensemble
        best_model, ensemble_accuracy = self.train_neural_ensemble(
            X_train_final, y_train_final, X_val, y_val
        )
        
        # Hyperparameter tuning for comparison
        tuned_model = self.hyperparameter_tuning(X_train, y_train)
        
        # Final evaluation on test set
        print("\n=== FINAL EVALUATION ===")
        
        # Scale test set
        X_test_scaled = self.scaler.transform(X_test)
        
        # Ensemble prediction
        ensemble_test_pred = best_model.predict(X_test_scaled)
        ensemble_test_acc = accuracy_score(y_test, ensemble_test_pred)
        
        # Tuned model prediction
        tuned_scaler = StandardScaler()
        X_train_for_tuned = tuned_scaler.fit_transform(X_train)
        X_test_for_tuned = tuned_scaler.transform(X_test)
        
        tuned_model.fit(X_train_for_tuned, y_train)
        tuned_test_pred = tuned_model.predict(X_test_for_tuned)
        tuned_test_acc = accuracy_score(y_test, tuned_test_pred)
        
        print(f"Ensemble test accuracy: {ensemble_test_acc:.4f}")
        print(f"Tuned model test accuracy: {tuned_test_acc:.4f}")
        
        # Cross-validation for server estimate
        cv_scores = []
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
            y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale
            cv_scaler = StandardScaler()
            X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = cv_scaler.transform(X_cv_val)
            
            # Simple neural network for CV
            cv_model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=300,
                random_state=42,
                early_stopping=True
            )
            
            cv_model.fit(X_cv_train_scaled, y_cv_train)
            cv_pred = cv_model.predict(X_cv_val_scaled)
            cv_score = accuracy_score(y_cv_val, cv_pred)
            cv_scores.append(cv_score)
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Server estimate (neural networks can be less stable)
        server_estimate = cv_mean * 0.96  # Slightly conservative
        
        print(f"\n=== NEURAL NETWORK RESULTS ===")
        print(f"Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"Best test accuracy: {max(ensemble_test_acc, tuned_test_acc):.4f}")
        print(f"Server estimate: {server_estimate:.4f}")
        
        # Choose best approach
        if ensemble_test_acc >= tuned_test_acc:
            self.final_model = best_model
            self.final_scaler = self.scaler
            best_approach = "ensemble"
            best_test_acc = ensemble_test_acc
        else:
            self.final_model = tuned_model
            self.final_scaler = tuned_scaler
            best_approach = "tuned"
            best_test_acc = tuned_test_acc
        
        print(f"Best approach: {best_approach}")
        
        if server_estimate >= 0.80:
            print("🎯 HIGH CONFIDENCE FOR 80%+ SERVER PERFORMANCE!")
        elif server_estimate >= 0.78:
            print("✅ GOOD CONFIDENCE FOR BEATING 78% THRESHOLD")
        else:
            print("📈 Solid neural network approach")
        
        return {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'ensemble_accuracy': ensemble_accuracy,
            'best_test_accuracy': best_test_acc,
            'server_estimate': server_estimate,
            'best_approach': best_approach
        }
    
    def create_neural_submission(self, test_path, output_path="neural_submission.csv"):
        """Create submission using neural networks"""
        print(f"\n=== CREATING NEURAL NETWORK SUBMISSION ===")
        
        if not hasattr(self, 'final_model'):
            print("Error: No final model found. Run pipeline first.")
            return
        
        # Process test data
        test_df = pd.read_csv(test_path)
        test_engineered = self.minimal_feature_engineering(test_df, is_train=False)
        X_test = self.neural_preprocessing(test_engineered, is_train=False)
        
        # Scale features
        X_test_scaled = self.final_scaler.transform(X_test)
        
        # Predictions
        if hasattr(self.final_model, 'predict_proba'):
            test_probabilities = self.final_model.predict_proba(X_test_scaled)[:, 1]
        else:
            test_probabilities = self.final_model.predict(X_test_scaled)
        
        final_predictions = (test_probabilities > 0.5).astype(int)
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': final_predictions
        })
        
        submission.to_csv(output_path, index=False)
        print(f"Neural network submission saved to: {output_path}")
        print(f"Total predictions: {len(final_predictions)}")
        print(f"Predicted survivors: {final_predictions.sum()} ({final_predictions.mean():.1%})")
        
        if hasattr(self.final_model, 'predict_proba'):
            print(f"Prediction confidence - Mean: {test_probabilities.mean():.3f}, Std: {test_probabilities.std():.3f}")
        
        return submission

# Run the neural network pipeline
if __name__ == "__main__":
    pipeline = SklearnNeuralPipeline()
    
    # Set paths
    train_path = "../datasets/titanic/train.csv"
    test_path = "../datasets/titanic/test.csv"
    
    # Run neural pipeline
    results = pipeline.run_neural_pipeline(train_path, test_path)
    
    print(f"\n=== NEURAL NETWORK PIPELINE SUMMARY ===")
    print(f"Cross-validation: {results['cv_mean']:.1%} ± {results['cv_std']:.1%}")
    print(f"Best Test Accuracy: {results['best_test_accuracy']:.1%}")
    print(f"Server Estimate: {results['server_estimate']:.1%}")
    print(f"Best Approach: {results['best_approach']}")
    
    # Create neural submission
    submission = pipeline.create_neural_submission(test_path, "neural_submission.csv")
    
    print(f"\n=== NEURAL NETWORK SUBMISSION READY ===")
    print("🧠 Neural network submission 'neural_submission.csv' created!")
    print("Key advantages of this approach:")
    print("✅ Multiple neural network architectures")
    print("✅ Ensemble voting")
    print("✅ Hyperparameter optimization")
    print("✅ Minimal feature engineering (let NN learn)")
    print("✅ Proper scaling and regularization")
    print("✅ Early stopping to prevent overfitting")
    
    if results['server_estimate'] >= 0.78:
        print(f"\n🚀 NEURAL NETWORKS: Targeting {results['server_estimate']*100:.1f}%+ performance! 🚀") 