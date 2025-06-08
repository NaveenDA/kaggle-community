import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MultiSeedSimplePipeline:
    def __init__(self):
        self.scalers = []
        self.models = []
        self.random_seeds = [42, 123, 456, 789, 101112]  # 5 different seeds
        
    def load_data(self, train_path, test_path=None):
        """Load data"""
        self.train_df = pd.read_csv(train_path)
        if test_path:
            self.test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_df.shape}")
        if hasattr(self, 'test_df'):
            print(f"Test data shape: {self.test_df.shape}")
    
    def basic_features_only(self, df, is_train=True):
        """Same winning 13 features from simple approach"""
        df = df.copy()
        
        # Fill missing values with simple strategies
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Embarked'].fillna('S', inplace=True)
        
        # Same 13 features that got us 78.229%
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Core survival rules (exact same as winning approach)
        df['IsWoman'] = (df['Sex'] == 'female').astype(int)
        df['IsChild'] = (df['Age'] < 16).astype(int) 
        df['IsFirstClass'] = (df['Pclass'] == 1).astype(int)
        df['IsThirdClass'] = (df['Pclass'] == 3).astype(int)
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Simple feature selection - only the essentials
        features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize',
                   'IsWoman', 'IsChild', 'IsFirstClass', 'IsThirdClass', 'IsAlone']
        
        # Add embarked as numeric
        df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
        df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
        features.extend(['Embarked_S', 'Embarked_C'])
        
        return df[features]
    
    def create_simple_models(self, seed):
        """Create simple models with given random seed"""
        models = [
            ('LogReg', LogisticRegression(
                C=1.0, 
                penalty='l2',
                solver='liblinear',
                random_state=seed,
                max_iter=1000
            )),
            ('SimpleRF', RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=seed
            )),
            ('SimpleGB', GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=seed
            ))
        ]
        return models
    
    def train_multi_seed_ensemble(self, X, y):
        """Train ensemble with multiple seeds"""
        print(f"\n=== TRAINING MULTI-SEED SIMPLE ENSEMBLE ===")
        print(f"Using {len(self.random_seeds)} different random seeds")
        print(f"Training 3 simple models per seed = {len(self.random_seeds) * 3} total models")
        
        all_models = []
        all_scalers = []
        seed_accuracies = []
        
        for i, seed in enumerate(self.random_seeds):
            print(f"\n--- Seed {seed} ({i+1}/{len(self.random_seeds)}) ---")
            
            # Create scaler for this seed
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create models for this seed
            models = self.create_simple_models(seed)
            seed_models = []
            
            for name, model in models:
                # Train model
                model.fit(X_scaled, y)
                
                # Cross-validation score for this model
                cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                cv_mean = cv_scores.mean()
                
                print(f"  {name}: {cv_mean:.4f}")
                seed_models.append((name, model))
            
            # Calculate average accuracy for this seed
            seed_cv_scores = []
            for name, model in seed_models:
                cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                seed_cv_scores.append(cv_scores.mean())
            
            seed_avg_accuracy = np.mean(seed_cv_scores)
            seed_accuracies.append(seed_avg_accuracy)
            print(f"  Seed {seed} average CV: {seed_avg_accuracy:.4f}")
            
            all_models.append(seed_models)
            all_scalers.append(scaler)
        
        # Overall ensemble performance estimate
        ensemble_cv_mean = np.mean(seed_accuracies)
        ensemble_cv_std = np.std(seed_accuracies)
        
        print(f"\n=== ENSEMBLE RESULTS ===")
        print(f"Multi-seed ensemble CV: {ensemble_cv_mean:.4f} ± {ensemble_cv_std:.4f}")
        print(f"Individual seed accuracies: {[f'{acc:.4f}' for acc in seed_accuracies]}")
        
        self.models = all_models
        self.scalers = all_scalers
        
        return ensemble_cv_mean, ensemble_cv_std
    
    def ensemble_predict(self, X_test):
        """Make predictions using multi-seed ensemble"""
        all_predictions = []
        
        # Get predictions from all models
        for i, (seed_models, scaler) in enumerate(zip(self.models, self.scalers)):
            X_test_scaled = scaler.transform(X_test)
            
            # Average predictions from models in this seed
            seed_predictions = []
            for name, model in seed_models:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    pred_proba = model.predict(X_test_scaled)
                seed_predictions.append(pred_proba)
            
            # Average across models for this seed
            seed_avg = np.mean(seed_predictions, axis=0)
            all_predictions.append(seed_avg)
        
        # Average across all seeds
        final_predictions = np.mean(all_predictions, axis=0)
        return final_predictions
    
    def run_multi_seed_pipeline(self, train_path, test_path=None):
        """Run multi-seed simple pipeline"""
        print("=== MULTI-SEED SIMPLE ENSEMBLE PIPELINE ===")
        print("Building on the winning 78.229% approach with variance reduction!")
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Same basic features that worked
        print("\n=== SAME WINNING 13 FEATURES ===")
        X = self.basic_features_only(self.train_df, is_train=True)
        y = self.train_df['Survived']
        
        print(f"Using exact same {X.shape[1]} features that got 78.229%:")
        print(f"Features: {list(X.columns)}")
        
        # Train multi-seed ensemble
        ensemble_cv_mean, ensemble_cv_std = self.train_multi_seed_ensemble(X, y)
        
        # Conservative server estimate (same multiplier as simple approach)
        server_estimate = ensemble_cv_mean * 0.94  # Slightly less conservative since we're reducing variance
        
        print(f"\n=== MULTI-SEED RESULTS ===")
        print(f"Ensemble CV score: {ensemble_cv_mean:.4f} ± {ensemble_cv_std:.4f}")
        print(f"Server estimate: {server_estimate:.4f}")
        print(f"Expected improvement: Variance reduction should boost performance")
        
        print(f"\n=== STRATEGY ===")
        print("✅ Same 13 winning features")
        print("✅ Same simple models (LogReg, SimpleRF, SimpleGB)")
        print("✅ 5 different random seeds for variance reduction")
        print("✅ 15 total models (3 models × 5 seeds)")
        print("✅ Averaged predictions for stability")
        
        if server_estimate >= 0.785:
            print("🎯 HIGH CONFIDENCE FOR BEATING 78.229%!")
        elif server_estimate >= 0.780:
            print("✅ GOOD CONFIDENCE FOR MATCHING 78.229%")
        else:
            print("📈 Conservative estimate - variance reduction should help")
        
        return {
            'ensemble_cv_mean': ensemble_cv_mean,
            'ensemble_cv_std': ensemble_cv_std,
            'server_estimate': server_estimate,
            'total_models': len(self.random_seeds) * 3
        }
    
    def create_multi_seed_submission(self, test_path, output_path="multiseed_submission.csv"):
        """Create submission using multi-seed ensemble"""
        print(f"\n=== CREATING MULTI-SEED SUBMISSION ===")
        
        if not self.models:
            print("Error: No models trained. Run pipeline first.")
            return
        
        # Process test data (same way as winning approach)
        test_df = pd.read_csv(test_path)
        X_test = self.basic_features_only(test_df, is_train=False)
        
        # Multi-seed ensemble predictions
        test_probabilities = self.ensemble_predict(X_test)
        final_predictions = (test_probabilities > 0.5).astype(int)
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': final_predictions
        })
        
        submission.to_csv(output_path, index=False)
        print(f"Multi-seed submission saved to: {output_path}")
        print(f"Total predictions: {len(final_predictions)}")
        print(f"Predicted survivors: {final_predictions.sum()} ({final_predictions.mean():.1%})")
        print(f"Mean confidence: {test_probabilities.mean():.3f}")
        print(f"Prediction stability (std): {test_probabilities.std():.3f}")
        
        # Compare with simple approach patterns
        print(f"\nSurvival patterns in predictions:")
        test_analysis = test_df.copy()
        test_analysis['Predicted_Survival'] = final_predictions
        test_analysis['Confidence'] = test_probabilities
        
        women_survival = test_analysis[test_analysis['Sex']=='female']['Predicted_Survival'].mean()
        men_survival = test_analysis[test_analysis['Sex']=='male']['Predicted_Survival'].mean()
        first_class_survival = test_analysis[test_analysis['Pclass']==1]['Predicted_Survival'].mean()
        third_class_survival = test_analysis[test_analysis['Pclass']==3]['Predicted_Survival'].mean()
        
        print(f"Women survival rate: {women_survival:.1%}")
        print(f"Men survival rate: {men_survival:.1%}")
        print(f"1st class survival rate: {first_class_survival:.1%}")
        print(f"3rd class survival rate: {third_class_survival:.1%}")
        
        # Variance analysis
        print(f"\nVariance reduction analysis:")
        print(f"Prediction confidence range: [{test_probabilities.min():.3f}, {test_probabilities.max():.3f}]")
        print(f"High confidence predictions (>0.8 or <0.2): {((test_probabilities > 0.8) | (test_probabilities < 0.2)).sum()}")
        
        return submission

# Run the multi-seed pipeline
if __name__ == "__main__":
    pipeline = MultiSeedSimplePipeline()
    
    # Set paths
    train_path = "../datasets/titanic/train.csv"
    test_path = "../datasets/titanic/test.csv"
    
    # Run multi-seed pipeline
    results = pipeline.run_multi_seed_pipeline(train_path, test_path)
    
    print(f"\n=== MULTI-SEED PIPELINE SUMMARY ===")
    print(f"Ensemble CV: {results['ensemble_cv_mean']:.1%} ± {results['ensemble_cv_std']:.1%}")
    print(f"Server estimate: {results['server_estimate']:.1%}")
    print(f"Total models: {results['total_models']}")
    
    # Create multi-seed submission
    submission = pipeline.create_multi_seed_submission(test_path, "multiseed_submission.csv")
    
    print(f"\n=== MULTI-SEED SUBMISSION READY ===")
    print("🎯 Multi-seed submission 'multiseed_submission.csv' created!")
    print("Strategy: Same winning formula + variance reduction")
    print("Expected: Beat 78.229% through ensemble stability")
    
    if results['server_estimate'] >= 0.785:
        print(f"\n🚀 HIGH CONFIDENCE: Should beat 78.229% and reach {results['server_estimate']*100:.1f}%+ 🚀")
    else:
        print(f"\n📈 CONSERVATIVE ESTIMATE: {results['server_estimate']*100:.1f}% but variance reduction should help") 