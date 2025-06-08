import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class OptimizedMultiSeedPipeline:
    def __init__(self):
        self.scalers = []
        self.models = []
        self.random_seeds = [42, 123, 456, 789, 101112]  # Same 5 seeds that worked
        self.best_params = {}
        
    def load_data(self, train_path, test_path=None):
        """Load data"""
        self.train_df = pd.read_csv(train_path)
        if test_path:
            self.test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_df.shape}")
        if hasattr(self, 'test_df'):
            print(f"Test data shape: {self.test_df.shape}")
    
    def optimized_features(self, df, is_train=True):
        """Same winning 13 features + 1-2 ultra-simple additions"""
        df = df.copy()
        
        # Fill missing values with simple strategies
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Embarked'].fillna('S', inplace=True)
        
        # CORE 13 features that got us 78.468%
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Core survival rules (exact same as winning approach)
        df['IsWoman'] = (df['Sex'] == 'female').astype(int)
        df['IsChild'] = (df['Age'] < 16).astype(int) 
        df['IsFirstClass'] = (df['Pclass'] == 1).astype(int)
        df['IsThirdClass'] = (df['Pclass'] == 3).astype(int)
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # NEW FEATURE 1: Ultra-simple Title (most basic extraction)
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['IsMaster'] = (df['Title'] == 'Master').astype(int)  # Boys (important for survival)
        df['IsMrs'] = (df['Title'] == 'Mrs').astype(int)       # Married women
        
        # NEW FEATURE 2: Age threshold optimization (test 15 vs 16)
        df['IsVeryYoung'] = (df['Age'] < 15).astype(int)  # Even younger children
        
        # NEW FEATURE 3: Fare categories (simple quantiles)
        df['HighFare'] = (df['Fare'] > df['Fare'].quantile(0.75)).astype(int)
        df['LowFare'] = (df['Fare'] < df['Fare'].quantile(0.25)).astype(int)
        
        # Core features (same 13 that worked)
        core_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize',
                        'IsWoman', 'IsChild', 'IsFirstClass', 'IsThirdClass', 'IsAlone']
        
        # Embarked features
        df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
        df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
        core_features.extend(['Embarked_S', 'Embarked_C'])
        
        # New optimized features (only 5 additions)
        new_features = ['IsMaster', 'IsMrs', 'IsVeryYoung', 'HighFare', 'LowFare']
        
        all_features = core_features + new_features
        print(f"Using {len(core_features)} core + {len(new_features)} new = {len(all_features)} total features")
        
        return df[all_features]
    
    def hyperparameter_tuning(self, X, y):
        """Optimize hyperparameters for our simple models"""
        print("\n=== HYPERPARAMETER OPTIMIZATION ===")
        
        X_scaled = StandardScaler().fit_transform(X)
        
        # Optimize Logistic Regression
        print("Optimizing Logistic Regression...")
        lr_params = {
            'C': [0.5, 1.0, 2.0, 5.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            lr_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        lr_grid.fit(X_scaled, y)
        self.best_params['LogReg'] = lr_grid.best_params_
        print(f"Best LogReg params: {lr_grid.best_params_}")
        print(f"Best LogReg score: {lr_grid.best_score_:.4f}")
        
        # Optimize Random Forest
        print("Optimizing Random Forest...")
        rf_params = {
            'n_estimators': [30, 50, 75],
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 8]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        rf_grid.fit(X_scaled, y)
        self.best_params['SimpleRF'] = rf_grid.best_params_
        print(f"Best RF params: {rf_grid.best_params_}")
        print(f"Best RF score: {rf_grid.best_score_:.4f}")
        
        # Optimize Gradient Boosting
        print("Optimizing Gradient Boosting...")
        gb_params = {
            'n_estimators': [30, 50, 75],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [2, 3, 4],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [3, 5]
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        gb_grid.fit(X_scaled, y)
        self.best_params['SimpleGB'] = gb_grid.best_params_
        print(f"Best GB params: {gb_grid.best_params_}")
        print(f"Best GB score: {gb_grid.best_score_:.4f}")
        
        return self.best_params
    
    def create_optimized_models(self, seed):
        """Create optimized models with best hyperparameters"""
        models = [
            ('LogReg', LogisticRegression(
                random_state=seed,
                max_iter=1000,
                **self.best_params.get('LogReg', {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'})
            )),
            ('SimpleRF', RandomForestClassifier(
                random_state=seed,
                **self.best_params.get('SimpleRF', {
                    'n_estimators': 50, 'max_depth': 5, 
                    'min_samples_split': 10, 'min_samples_leaf': 5
                })
            )),
            ('SimpleGB', GradientBoostingClassifier(
                random_state=seed,
                **self.best_params.get('SimpleGB', {
                    'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3,
                    'min_samples_split': 10, 'min_samples_leaf': 5
                })
            ))
        ]
        return models
    
    def train_optimized_ensemble(self, X, y):
        """Train optimized multi-seed ensemble"""
        print(f"\n=== TRAINING OPTIMIZED MULTI-SEED ENSEMBLE ===")
        print(f"Using {len(self.random_seeds)} seeds with optimized hyperparameters")
        print(f"Training 3 optimized models per seed = {len(self.random_seeds) * 3} total models")
        
        all_models = []
        all_scalers = []
        seed_accuracies = []
        
        for i, seed in enumerate(self.random_seeds):
            print(f"\n--- Seed {seed} ({i+1}/{len(self.random_seeds)}) ---")
            
            # Create scaler for this seed
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create optimized models for this seed
            models = self.create_optimized_models(seed)
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
        
        print(f"\n=== OPTIMIZED ENSEMBLE RESULTS ===")
        print(f"Optimized multi-seed ensemble CV: {ensemble_cv_mean:.4f} ± {ensemble_cv_std:.4f}")
        print(f"Individual seed accuracies: {[f'{acc:.4f}' for acc in seed_accuracies]}")
        
        # Compare with previous result (78.468% server)
        improvement = ensemble_cv_mean - 0.8014  # Previous CV was 0.8014
        print(f"CV improvement over previous: {improvement:+.4f}")
        
        self.models = all_models
        self.scalers = all_scalers
        
        return ensemble_cv_mean, ensemble_cv_std
    
    def ensemble_predict(self, X_test):
        """Make predictions using optimized multi-seed ensemble"""
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
    
    def run_optimized_pipeline(self, train_path, test_path=None):
        """Run optimized multi-seed pipeline"""
        print("=== OPTIMIZED MULTI-SEED PIPELINE ===")
        print("Building on 78.468% with feature + hyperparameter optimization!")
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Optimized features (13 core + 5 new = 18 total)
        print("\n=== OPTIMIZED FEATURES ===")
        X = self.optimized_features(self.train_df, is_train=True)
        y = self.train_df['Survived']
        
        print(f"Features: {list(X.columns)}")
        
        # Hyperparameter optimization
        best_params = self.hyperparameter_tuning(X, y)
        
        # Train optimized multi-seed ensemble
        ensemble_cv_mean, ensemble_cv_std = self.train_optimized_ensemble(X, y)
        
        # Less conservative server estimate (we're getting better at this)
        server_estimate = ensemble_cv_mean * 0.96  # Less conservative since we're optimizing
        
        print(f"\n=== OPTIMIZED RESULTS ===")
        print(f"Optimized ensemble CV: {ensemble_cv_mean:.4f} ± {ensemble_cv_std:.4f}")
        print(f"Server estimate: {server_estimate:.4f}")
        print(f"Target: Beat 78.468% and reach 79%+")
        
        print(f"\n=== OPTIMIZATION STRATEGY ===")
        print("✅ Same 13 core features that got 78.468%")
        print("✅ Added 5 ultra-simple new features")
        print("✅ Hyperparameter optimization for all models")
        print("✅ Same proven 5-seed ensemble approach")
        print("✅ 18 total features (still simple!)")
        
        if server_estimate >= 0.79:
            print("🎯 HIGH CONFIDENCE FOR 79%+ BREAKTHROUGH!")
        elif server_estimate >= 0.785:
            print("✅ EXCELLENT CONFIDENCE FOR BEATING 78.468%")
        elif server_estimate >= 0.780:
            print("📈 GOOD CONFIDENCE FOR IMPROVEMENT")
        else:
            print("🔧 Conservative estimate - optimizations should help")
        
        return {
            'ensemble_cv_mean': ensemble_cv_mean,
            'ensemble_cv_std': ensemble_cv_std,
            'server_estimate': server_estimate,
            'best_params': best_params,
            'total_features': X.shape[1]
        }
    
    def create_optimized_submission(self, test_path, output_path="optimized_submission.csv"):
        """Create submission using optimized ensemble"""
        print(f"\n=== CREATING OPTIMIZED SUBMISSION ===")
        
        if not self.models:
            print("Error: No models trained. Run pipeline first.")
            return
        
        # Process test data with optimized features
        test_df = pd.read_csv(test_path)
        X_test = self.optimized_features(test_df, is_train=False)
        
        # Optimized ensemble predictions
        test_probabilities = self.ensemble_predict(X_test)
        final_predictions = (test_probabilities > 0.5).astype(int)
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': final_predictions
        })
        
        submission.to_csv(output_path, index=False)
        print(f"Optimized submission saved to: {output_path}")
        print(f"Total predictions: {len(final_predictions)}")
        print(f"Predicted survivors: {final_predictions.sum()} ({final_predictions.mean():.1%})")
        print(f"Mean confidence: {test_probabilities.mean():.3f}")
        print(f"Prediction stability (std): {test_probabilities.std():.3f}")
        
        # Survival pattern analysis
        print(f"\nOptimized survival patterns:")
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
        
        # Optimization impact analysis
        print(f"\nOptimization analysis:")
        print(f"High confidence predictions (>0.8 or <0.2): {((test_probabilities > 0.8) | (test_probabilities < 0.2)).sum()}")
        print(f"Model consensus range: [{test_probabilities.min():.3f}, {test_probabilities.max():.3f}]")
        
        return submission

# Run the optimized pipeline
if __name__ == "__main__":
    pipeline = OptimizedMultiSeedPipeline()
    
    # Set paths
    train_path = "../datasets/titanic/train.csv"
    test_path = "../datasets/titanic/test.csv"
    
    # Run optimized pipeline
    results = pipeline.run_optimized_pipeline(train_path, test_path)
    
    print(f"\n=== OPTIMIZED PIPELINE SUMMARY ===")
    print(f"Total features: {results['total_features']}")
    print(f"Optimized CV: {results['ensemble_cv_mean']:.1%} ± {results['ensemble_cv_std']:.1%}")
    print(f"Server estimate: {results['server_estimate']:.1%}")
    print(f"Best hyperparameters: {results['best_params']}")
    
    # Create optimized submission
    submission = pipeline.create_optimized_submission(test_path, "optimized_submission.csv")
    
    print(f"\n=== OPTIMIZED SUBMISSION READY ===")
    print("🎯 Optimized submission 'optimized_submission.csv' created!")
    print("Strategy: 78.468% winning approach + feature & hyperparameter optimization")
    print("Goal: Push toward 79%+ with methodical improvements")
    
    if results['server_estimate'] >= 0.79:
        print(f"\n🚀 BREAKTHROUGH TARGET: {results['server_estimate']*100:.1f}%+ for 79% milestone! 🚀")
    elif results['server_estimate'] >= 0.785:
        print(f"\n✅ HIGH CONFIDENCE: Should beat 78.468% and reach {results['server_estimate']*100:.1f}%+")
    else:
        print(f"\n📈 OPTIMIZATION: {results['server_estimate']*100:.1f}% estimate, improvements should help") 