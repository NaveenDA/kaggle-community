import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class UltraSimpleTitanicPipeline:
    def __init__(self):
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self, train_path, test_path=None):
        """Load data"""
        self.train_df = pd.read_csv(train_path)
        if test_path:
            self.test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_df.shape}")
        if hasattr(self, 'test_df'):
            print(f"Test data shape: {self.test_df.shape}")
    
    def basic_features_only(self, df, is_train=True):
        """Extract only the most basic, proven survival features"""
        df = df.copy()
        
        # Fill missing values with simple strategies
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Embarked'].fillna('S', inplace=True)
        
        # ONLY the most basic features that are proven to matter
        # 1. Sex (biggest factor)
        # 2. Pclass (social status)
        # 3. Age (children first)
        # 4. Family size (small families better)
        
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Core survival rules
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
    
    def train_simple_model(self, X, y):
        """Train ultra-simple model"""
        print("\n=== TRAINING ULTRA-SIMPLE MODEL ===")
        
        # Use logistic regression - simplest, most interpretable
        self.model = LogisticRegression(
            C=1.0,  # No strong regularization
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Print feature importance (coefficients)
        feature_names = X.columns
        coefficients = self.model.coef_[0]
        
        print("Feature importance (coefficients):")
        for name, coef in zip(feature_names, coefficients):
            print(f"  {name}: {coef:.3f}")
        
        return self.model
    
    def cross_validate_simple(self, X, y):
        """Simple cross-validation"""
        print("\n=== SIMPLE CROSS-VALIDATION ===")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Simple models for comparison
        models = {
            'Logistic Regression': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
            'Simple RF': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
            'Conservative RF': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        }
        
        cv_results = {}
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            cv_results[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
            print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Choose the most stable model (lowest std)
        best_model_name = min(cv_results.keys(), key=lambda k: cv_results[k]['std'])
        best_score = cv_results[best_model_name]['mean']
        
        print(f"\nMost stable model: {best_model_name}")
        print(f"Score: {best_score:.4f} ± {cv_results[best_model_name]['std']:.4f}")
        
        return cv_results, best_model_name, best_score
    
    def run_simple_pipeline(self, train_path, test_path=None):
        """Run ultra-simple pipeline"""
        print("=== ULTRA-SIMPLE TITANIC PIPELINE ===")
        print("Back to basics - only proven survival patterns!")
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Basic features only
        print("\n=== BASIC FEATURES ONLY ===")
        X = self.basic_features_only(self.train_df, is_train=True)
        y = self.train_df['Survived']
        
        print(f"Using only {X.shape[1]} basic features:")
        print(f"Features: {list(X.columns)}")
        
        # Cross-validation
        cv_results, best_model_name, best_score = self.cross_validate_simple(X, y)
        
        # Train final simple model
        final_model = self.train_simple_model(X, y)
        
        # Conservative server estimate
        server_estimate = best_score * 0.92  # Very conservative given the overfitting pattern
        
        print(f"\n=== SIMPLE RESULTS ===")
        print(f"Best CV model: {best_model_name}")
        print(f"CV score: {best_score:.4f}")
        print(f"Conservative server estimate: {server_estimate:.4f}")
        
        print(f"\n=== STRATEGY ===")
        print("✅ Ultra-simple features (13 basic features)")
        print("✅ Logistic regression (most interpretable)")
        print("✅ Minimal regularization")
        print("✅ Conservative server estimate")
        print("✅ Focus on proven survival patterns only")
        
        if server_estimate >= 0.77:
            print("🎯 Targeting recovery above 77%")
        else:
            print("📈 Conservative but stable approach")
        
        return {
            'cv_results': cv_results,
            'best_score': best_score,
            'server_estimate': server_estimate,
            'features_used': X.shape[1]
        }
    
    def create_simple_submission(self, test_path, output_path="simple_submission.csv"):
        """Create submission with ultra-simple approach"""
        print(f"\n=== CREATING SIMPLE SUBMISSION ===")
        
        if self.model is None:
            print("Error: No model trained. Run pipeline first.")
            return
        
        # Process test data
        test_df = pd.read_csv(test_path)
        X_test = self.basic_features_only(test_df, is_train=False)
        
        # Scale and predict
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': predictions
        })
        
        submission.to_csv(output_path, index=False)
        print(f"Simple submission saved to: {output_path}")
        print(f"Total predictions: {len(predictions)}")
        print(f"Predicted survivors: {predictions.sum()} ({predictions.mean():.1%})")
        print(f"Mean confidence: {probabilities.mean():.3f}")
        
        # Survival rate analysis
        print(f"\nSurvival patterns in predictions:")
        test_analysis = test_df.copy()
        test_analysis['Predicted_Survival'] = predictions
        test_analysis['Confidence'] = probabilities
        
        print(f"Women survival rate: {test_analysis[test_analysis['Sex']=='female']['Predicted_Survival'].mean():.1%}")
        print(f"Men survival rate: {test_analysis[test_analysis['Sex']=='male']['Predicted_Survival'].mean():.1%}")
        print(f"1st class survival rate: {test_analysis[test_analysis['Pclass']==1]['Predicted_Survival'].mean():.1%}")
        print(f"3rd class survival rate: {test_analysis[test_analysis['Pclass']==3]['Predicted_Survival'].mean():.1%}")
        
        return submission

# Run the ultra-simple pipeline
if __name__ == "__main__":
    pipeline = UltraSimpleTitanicPipeline()
    
    # Set paths
    train_path = "../datasets/titanic/train.csv"
    test_path = "../datasets/titanic/test.csv"
    
    # Run simple pipeline
    results = pipeline.run_simple_pipeline(train_path, test_path)
    
    print(f"\n=== ULTRA-SIMPLE PIPELINE SUMMARY ===")
    print(f"Features used: {results['features_used']}")
    print(f"Best CV score: {results['best_score']:.1%}")
    print(f"Server estimate: {results['server_estimate']:.1%}")
    
    # Create simple submission
    submission = pipeline.create_simple_submission(test_path, "simple_submission.csv")
    
    print(f"\n=== BACK TO BASICS SUBMISSION READY ===")
    print("🎯 Ultra-simple submission 'simple_submission.csv' created!")
    print("Strategy: Abandon complexity, focus on core survival patterns")
    print("Goal: Stop the declining trend with stable, simple approach")
    
    if results['server_estimate'] >= 0.77:
        print(f"\n🚀 SIMPLE BUT EFFECTIVE: Targeting {results['server_estimate']*100:.1f}%+ 🚀")
    else:
        print(f"\n📈 CONSERVATIVE: {results['server_estimate']*100:.1f}% but should be stable") 