import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DeepTitanicPipeline:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
        self.models = []
        self.feature_names = []
        
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
    
    def smart_feature_engineering(self, df, is_train=True):
        """Smart feature engineering optimized for neural networks"""
        df = df.copy()
        
        # Basic title extraction
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Simplified title mapping for neural networks
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'
        }
        df['Title'] = df['Title'].map(title_mapping).fillna('Other')
        
        # Family features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Age groups (let NN learn the nuances)
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Age_Binned'] = pd.cut(df['Age'], bins=5, labels=[0,1,2,3,4]).astype(int)
        
        # Fare processing
        df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'), inplace=True)
        df['Fare_Log'] = np.log1p(df['Fare'])  # Log transformation
        df['Fare_Binned'] = pd.qcut(df['Fare'], q=5, labels=[0,1,2,3,4], duplicates='drop').astype(int)
        
        # Cabin features
        df['Has_Cabin'] = df['Cabin'].notna().astype(int)
        df['Cabin_Deck'] = df['Cabin'].str[0].fillna('Unknown')
        
        # Embarked
        df['Embarked'].fillna('S', inplace=True)
        
        # Ticket features
        df['Ticket_Len'] = df['Ticket'].str.len()
        df['Ticket_Num'] = pd.to_numeric(df['Ticket'].str.extract(r'(\d+)')[0], errors='coerce').fillna(0)
        
        # Interaction features (key for survival)
        df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
        df['Title_Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
        
        return df
    
    def neural_preprocessing(self, df, is_train=True):
        """Preprocessing optimized for neural networks"""
        df = df.copy()
        
        # Advanced age imputation
        if is_train:
            age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
            age_data = df[age_features + ['Age']].copy()
            age_data['Sex'] = LabelEncoder().fit_transform(age_data['Sex'])
            
            self.age_imputer = KNNImputer(n_neighbors=5)
            age_data_imputed = self.age_imputer.fit_transform(age_data)
            df['Age'] = age_data_imputed[:, -1]
        else:
            age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
            age_data = df[age_features + ['Age']].copy()
            age_data['Sex'] = LabelEncoder().fit_transform(age_data['Sex'])
            age_data_imputed = self.age_imputer.transform(age_data)
            df['Age'] = age_data_imputed[:, -1]
        
        # Recalculate derived features after imputation
        df['Age_Binned'] = pd.cut(df['Age'], bins=5, labels=[0,1,2,3,4]).astype(int)
        df['Fare_Log'] = np.log1p(df['Fare'])
        
        # Select features for neural network
        categorical_features = ['Sex', 'Embarked', 'Title', 'Cabin_Deck', 'Sex_Pclass', 'Title_Pclass']
        numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone',
                             'Age_Binned', 'Fare_Log', 'Fare_Binned', 'Has_Cabin', 'Ticket_Len', 'Ticket_Num']
        
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
    
    def scale_features(self, X_train, X_val=None, X_test=None):
        """Scale features for neural networks"""
        # Use RobustScaler for neural networks (handles outliers better)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            results.append(X_test_scaled)
        
        return results if len(results) > 1 else results[0]
    
    def create_deep_network_v1(self, input_dim, name="deep_v1"):
        """Create first deep network architecture"""
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid', name='output')
        ], name=name)
        
        return model
    
    def create_deep_network_v2(self, input_dim, name="deep_v2"):
        """Create second deep network architecture with residual connections"""
        inputs = Input(shape=(input_dim,))
        
        # First block
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Second block with residual
        residual = x
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = layers.Add()([x, residual])  # Residual connection
        
        # Third block
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Fourth block
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def create_wide_network(self, input_dim, name="wide_net"):
        """Create wide network architecture"""
        model = Sequential([
            Dense(1024, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(256, activation='relu'),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid', name='output')
        ], name=name)
        
        return model
    
    def create_ensemble_model(self, input_dim, name="ensemble"):
        """Create ensemble model with multiple branches"""
        inputs = Input(shape=(input_dim,))
        
        # Branch 1: Deep narrow
        branch1 = Dense(128, activation='relu')(inputs)
        branch1 = BatchNormalization()(branch1)
        branch1 = Dropout(0.3)(branch1)
        branch1 = Dense(64, activation='relu')(branch1)
        branch1 = Dropout(0.2)(branch1)
        branch1 = Dense(32, activation='relu')(branch1)
        
        # Branch 2: Wide shallow
        branch2 = Dense(256, activation='relu')(inputs)
        branch2 = BatchNormalization()(branch2)
        branch2 = Dropout(0.4)(branch2)
        branch2 = Dense(128, activation='relu')(branch2)
        
        # Branch 3: Medium depth
        branch3 = Dense(192, activation='relu')(inputs)
        branch3 = BatchNormalization()(branch3)
        branch3 = Dropout(0.3)(branch3)
        branch3 = Dense(96, activation='relu')(branch3)
        branch3 = Dropout(0.2)(branch3)
        branch3 = Dense(48, activation='relu')(branch3)
        
        # Combine branches
        combined = Concatenate()([branch1, branch2, branch3])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.2)(combined)
        
        outputs = Dense(1, activation='sigmoid', name='output')(combined)
        
        model = Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def train_neural_ensemble(self, X_train, y_train, X_val, y_val):
        """Train ensemble of neural networks"""
        print("\n=== TRAINING NEURAL NETWORK ENSEMBLE ===")
        
        input_dim = X_train.shape[1]
        models = []
        
        # Create different architectures
        architectures = [
            (self.create_deep_network_v1, {'learning_rate': 0.001, 'optimizer': 'adam'}),
            (self.create_deep_network_v2, {'learning_rate': 0.0015, 'optimizer': 'adam'}),
            (self.create_wide_network, {'learning_rate': 0.0008, 'optimizer': 'rmsprop'}),
            (self.create_ensemble_model, {'learning_rate': 0.001, 'optimizer': 'adam'})
        ]
        
        for i, (arch_func, config) in enumerate(architectures):
            print(f"\n--- Training Model {i+1}/4: {arch_func.__name__} ---")
            
            # Create model
            model = arch_func(input_dim, name=f"model_{i+1}")
            
            # Choose optimizer
            if config['optimizer'] == 'adam':
                optimizer = Adam(learning_rate=config['learning_rate'])
            else:
                optimizer = RMSprop(learning_rate=config['learning_rate'])
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks_list = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
                ModelCheckpoint(f'best_model_{i+1}.h5', save_best_only=True, monitor='val_accuracy')
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                callbacks=callbacks_list,
                verbose=0
            )
            
            # Evaluate model
            val_pred = model.predict(X_val, verbose=0)
            val_accuracy = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            
            print(f"Model {i+1} validation accuracy: {val_accuracy:.4f}")
            print(f"Best epoch: {np.argmax(history.history['val_accuracy']) + 1}")
            
            models.append(model)
        
        self.models = models
        return models
    
    def ensemble_predict(self, X, method='average'):
        """Make predictions using ensemble of models"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        if method == 'average':
            return np.mean(predictions, axis=0)
        elif method == 'weighted':
            # Simple weighting based on model complexity
            weights = [0.3, 0.3, 0.2, 0.2]  # Favor the first two models
            return np.average(predictions, axis=0, weights=weights)
        
        return np.mean(predictions, axis=0)
    
    def run_deep_pipeline(self, train_path, test_path=None):
        """Run complete deep learning pipeline"""
        print("=== DEEP NEURAL NETWORK TITANIC PIPELINE ===")
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Feature engineering
        print("\n=== SMART FEATURE ENGINEERING ===")
        train_engineered = self.smart_feature_engineering(self.train_df, is_train=True)
        
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
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train_final, X_val, X_test
        )
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Validation set: {X_val_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        # Train neural ensemble
        models = self.train_neural_ensemble(
            X_train_scaled, y_train_final, X_val_scaled, y_val
        )
        
        # Final evaluation
        print("\n=== ENSEMBLE EVALUATION ===")
        
        # Average ensemble
        test_pred_avg = self.ensemble_predict(X_test_scaled, method='average')
        test_accuracy_avg = accuracy_score(y_test, (test_pred_avg > 0.5).astype(int))
        
        # Weighted ensemble
        test_pred_weighted = self.ensemble_predict(X_test_scaled, method='weighted')
        test_accuracy_weighted = accuracy_score(y_test, (test_pred_weighted > 0.5).astype(int))
        
        print(f"Average ensemble accuracy: {test_accuracy_avg:.4f}")
        print(f"Weighted ensemble accuracy: {test_accuracy_weighted:.4f}")
        
        # Cross-validation estimate
        cv_scores = []
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            print(f"\nCV Fold {fold+1}/5")
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Scale
            cv_scaler = RobustScaler()
            X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = cv_scaler.transform(X_cv_val)
            
            # Simple model for CV (faster)
            cv_model = Sequential([
                Dense(128, activation='relu', input_shape=(X_cv_train_scaled.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            cv_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            cv_model.fit(
                X_cv_train_scaled, y_cv_train,
                validation_data=(X_cv_val_scaled, y_cv_val),
                epochs=50, batch_size=32, verbose=0,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            )
            
            cv_pred = cv_model.predict(X_cv_val_scaled, verbose=0)
            cv_score = accuracy_score(y_cv_val, (cv_pred > 0.5).astype(int))
            cv_scores.append(cv_score)
            print(f"Fold {fold+1} accuracy: {cv_score:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Server estimate (conservative for neural networks)
        server_estimate = cv_mean * 0.95  # Conservative for NN generalization
        
        print(f"\n=== DEEP LEARNING RESULTS ===")
        print(f"Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"Test accuracy (avg): {test_accuracy_avg:.4f}")
        print(f"Test accuracy (weighted): {test_accuracy_weighted:.4f}")
        print(f"Server estimate: {server_estimate:.4f}")
        
        # Choose best ensemble method
        best_method = 'weighted' if test_accuracy_weighted >= test_accuracy_avg else 'average'
        best_accuracy = max(test_accuracy_avg, test_accuracy_weighted)
        
        print(f"Best ensemble method: {best_method}")
        print(f"Best test accuracy: {best_accuracy:.4f}")
        
        if server_estimate >= 0.80:
            print("🎯 HIGH CONFIDENCE FOR 80%+ SERVER PERFORMANCE!")
        elif server_estimate >= 0.78:
            print("✅ GOOD CONFIDENCE FOR BEATING 78% THRESHOLD")
        else:
            print("📈 Solid neural network approach")
        
        return {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy_avg': test_accuracy_avg,
            'test_accuracy_weighted': test_accuracy_weighted,
            'server_estimate': server_estimate,
            'best_method': best_method,
            'best_accuracy': best_accuracy
        }
    
    def create_deep_submission(self, test_path, output_path="deep_submission.csv", method='weighted'):
        """Create submission using deep neural networks"""
        print(f"\n=== CREATING DEEP NEURAL NETWORK SUBMISSION ===")
        
        if not self.models:
            print("Error: No models found. Run pipeline first.")
            return
        
        # Process test data
        test_df = pd.read_csv(test_path)
        test_engineered = self.smart_feature_engineering(test_df, is_train=False)
        X_test = self.neural_preprocessing(test_engineered, is_train=False)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Ensemble predictions
        test_probabilities = self.ensemble_predict(X_test_scaled, method=method)
        final_predictions = (test_probabilities > 0.5).astype(int)
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': final_predictions
        })
        
        submission.to_csv(output_path, index=False)
        print(f"Deep learning submission saved to: {output_path}")
        print(f"Total predictions: {len(final_predictions)}")
        print(f"Predicted survivors: {final_predictions.sum()} ({final_predictions.mean():.1%})")
        print(f"Prediction confidence - Mean: {test_probabilities.mean():.3f}, Std: {test_probabilities.std():.3f}")
        
        # Confidence analysis
        high_conf = (test_probabilities > 0.7).sum() + (test_probabilities < 0.3).sum()
        med_conf = ((test_probabilities >= 0.3) & (test_probabilities <= 0.7)).sum()
        
        print(f"Confidence distribution:")
        print(f"  High confidence (>0.7 or <0.3): {high_conf} ({high_conf/len(test_probabilities)*100:.1f}%)")
        print(f"  Medium confidence (0.3-0.7): {med_conf} ({med_conf/len(test_probabilities)*100:.1f}%)")
        
        return submission

# Run the deep learning pipeline
if __name__ == "__main__":
    pipeline = DeepTitanicPipeline()
    
    # Set paths
    train_path = "../datasets/titanic/train.csv"
    test_path = "../datasets/titanic/test.csv"
    
    # Run deep pipeline
    results = pipeline.run_deep_pipeline(train_path, test_path)
    
    print(f"\n=== DEEP LEARNING PIPELINE SUMMARY ===")
    print(f"Cross-validation: {results['cv_mean']:.1%} ± {results['cv_std']:.1%}")
    print(f"Best Test Accuracy: {results['best_accuracy']:.1%}")
    print(f"Server Estimate: {results['server_estimate']:.1%}")
    print(f"Best Method: {results['best_method']}")
    
    # Create deep submission
    submission = pipeline.create_deep_submission(test_path, "deep_submission.csv", method=results['best_method'])
    
    print(f"\n=== DEEP NEURAL NETWORK SUBMISSION READY ===")
    print("🧠 Deep learning submission 'deep_submission.csv' created!")
    print("Key advantages of neural networks:")
    print("✅ Automatic feature learning")
    print("✅ Complex pattern recognition")
    print("✅ Ensemble of 4 different architectures")
    print("✅ Advanced regularization (dropout, batch norm)")
    print("✅ Residual connections")
    print("✅ Multiple optimization strategies")
    
    if results['server_estimate'] >= 0.78:
        print(f"\n🚀 NEURAL NETWORKS: Targeting {results['server_estimate']*100:.1f}%+ performance! 🚀") 