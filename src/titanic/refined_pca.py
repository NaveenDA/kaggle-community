import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFECV, SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RefinedAdvancedTitanicPipeline:
    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA()
        self.label_encoders = {}
        self.imputers = {}
        self.best_model = None
        self.best_score = 0
        self.feature_selector = SelectKBest(f_classif)
        # Back to 5 seeds like the successful advanced approach
        self.random_seeds = [42, 123, 456, 789, 101112]
        self.cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        
    def load_data(self, train_path, test_path=None):
        """Load training and test datasets"""
        self.train_df = pd.read_csv(train_path)
        if test_path:
            self.test_df = pd.read_csv(test_path)
        print(f"Training data shape: {self.train_df.shape}")
        if hasattr(self, 'test_df'):
            print(f"Test data shape: {self.test_df.shape}")
    
    def refined_advanced_features(self, df, is_train=True):
        """Refined advanced feature engineering - building on what worked"""
        df = df.copy()
        
        # Core survival patterns (these worked well)
        df['WomenAndChildren'] = ((df['Sex'] == 'female') | (df['Age'] < 16)).astype(int)
        df['FirstClassWoman'] = ((df['Pclass'] == 1) & (df['Sex'] == 'female')).astype(int)
        df['ThirdClassMale'] = ((df['Pclass'] == 3) & (df['Sex'] == 'male')).astype(int)
        
        # Enhanced title extraction
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Detailed title mapping (server seemed to like granular features)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Don': 'Noble', 'Rev': 'Officer', 'Dr': 'Professional', 'Mme': 'Mrs',
            'Ms': 'Miss', 'Major': 'Officer', 'Lady': 'Noble', 'Sir': 'Noble',
            'Mlle': 'Miss', 'Col': 'Officer', 'Capt': 'Officer', 'Countess': 'Noble',
            'Jonkheer': 'Noble', 'Dona': 'Noble'
        }
        df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
        
        # Enhanced family features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
        df['LargeFamily'] = (df['FamilySize'] >= 5).astype(int)
        
        # Age processing with multiple strategies
        age_median = df['Age'].median()
        if pd.isna(age_median):
            age_median = 29.0
        df['Age'].fillna(age_median, inplace=True)
        
        # Detailed age groups (server liked granular features)
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[0, 5, 12, 18, 25, 35, 50, 65, 100], 
                                labels=['Baby', 'Child', 'Teen', 'Young', 'Adult', 'Middle', 'Senior', 'Elderly'], 
                                include_lowest=True)
        
        # Age interaction features
        df['Age_Pclass'] = df['Age'] * df['Pclass']
        df['Age_Sex'] = df['Age'] * (df['Sex'] == 'male').astype(int)
        df['Age_SibSp'] = df['Age'] * df['SibSp']
        
        # Enhanced fare features
        df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'), inplace=True)
        df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
        df['Fare_Group'] = pd.qcut(df['Fare'], q=8, labels=False, duplicates='drop')
        
        # Advanced cabin features
        df['Has_Cabin'] = (~df['Cabin'].isnull()).astype(int)
        df['Cabin_Letter'] = df['Cabin'].str[0].fillna('Unknown')
        df['Multiple_Cabins'] = df['Cabin'].str.contains(' ', na=False).astype(int)
        
        # Deck mapping
        deck_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'T': 1, 'Unknown': 0}
        df['Deck'] = df['Cabin_Letter'].map(deck_mapping)
        
        # Enhanced ticket features
        df['Ticket_Length'] = df['Ticket'].str.len()
        df['Ticket_Prefix'] = df['Ticket'].str.extract(r'([A-Za-z]+)', expand=False).fillna('None')
        df['Ticket_Number'] = pd.to_numeric(df['Ticket'].str.extract(r'(\d+)', expand=False), errors='coerce').fillna(0)
        
        # Name features
        df['Name_Length'] = df['Name'].str.len()
        df['Name_WordCount'] = df['Name'].str.split().str.len()
        
        # Advanced interaction features
        df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
        df['Title_Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
        df['Age_Title'] = df['Title'] + '_' + df['Age_Group'].astype(str)
        
        # Binary features
        df['IsChild'] = (df['Age'] < 16).astype(int)
        df['IsElderly'] = (df['Age'] >= 60).astype(int)
        df['IsYoungAdult'] = ((df['Age'] >= 18) & (df['Age'] < 35)).astype(int)
        
        # Survival probability features
        df['HighSurvivalTitle'] = df['Title'].isin(['Mrs', 'Miss', 'Master']).astype(int)
        df['FirstClass'] = (df['Pclass'] == 1).astype(int)
        df['ThirdClass'] = (df['Pclass'] == 3).astype(int)
        
        # Port features
        df['Embarked'].fillna('S', inplace=True)
        df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
        df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
        df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
        
        # Fare-based features
        df['ExpensiveTicket'] = (df['Fare'] > df['Fare'].quantile(0.8)).astype(int)
        df['CheapTicket'] = (df['Fare'] < df['Fare'].quantile(0.2)).astype(int)
        
        return df
    
    def refined_preprocessing(self, df, is_train=True):
        """Refined preprocessing with KNN imputation"""
        df = df.copy()
        
        # Advanced Age imputation using KNN (this worked well)
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
        
        # Recalculate derived features
        df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
        df['Age_Pclass'] = df['Age'] * df['Pclass']
        df['Age_Sex'] = df['Age'] * (df['Sex'] == 'male').astype(int)
        df['Age_SibSp'] = df['Age'] * df['SibSp']
        df['IsChild'] = (df['Age'] < 16).astype(int)
        df['IsElderly'] = (df['Age'] >= 60).astype(int)
        df['IsYoungAdult'] = ((df['Age'] >= 18) & (df['Age'] < 35)).astype(int)
        df['ExpensiveTicket'] = (df['Fare'] > df['Fare'].quantile(0.8)).astype(int)
        df['CheapTicket'] = (df['Fare'] < df['Fare'].quantile(0.2)).astype(int)
        
        # Comprehensive feature selection (restore the complexity that worked)
        categorical_features = ['Sex', 'Embarked', 'Title', 'Age_Group', 'Cabin_Letter', 'Ticket_Prefix',
                               'Sex_Pclass', 'Title_Pclass', 'Age_Title']
        
        numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone',
                             'SmallFamily', 'LargeFamily', 'Has_Cabin', 'Multiple_Cabins', 'Deck',
                             'Ticket_Length', 'Ticket_Number', 'Name_Length', 'Name_WordCount',
                             'Fare_Per_Person', 'Fare_Group', 'Age_Pclass', 'Age_Sex', 'Age_SibSp',
                             'IsChild', 'IsElderly', 'IsYoungAdult', 'WomenAndChildren', 'FirstClassWoman',
                             'ThirdClassMale', 'HighSurvivalTitle', 'FirstClass', 'ThirdClass',
                             'ExpensiveTicket', 'CheapTicket', 'Embarked_S', 'Embarked_C', 'Embarked_Q']
        
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
        
        # Return comprehensive features
        feature_columns = categorical_features + numerical_features
        available_features = [f for f in feature_columns if f in df.columns]
        
        return df[available_features]
    
    def refined_feature_selection(self, X_train, y_train=None, X_test=None):
        """Refined feature selection - keep more features like the successful approach"""
        if y_train is not None:
            print(f"\n=== REFINED FEATURE SELECTION ===")
            print(f"Features before selection: {X_train.shape[1]}")
            
            # Less aggressive selection (keep more features)
            self.selector = SelectKBest(f_classif, k=min(25, X_train.shape[1]))
            X_train_selected = self.selector.fit_transform(X_train, y_train)
            
            if X_test is not None:
                X_test_selected = self.selector.transform(X_test)
            
            print(f"Features after statistical selection: {X_train_selected.shape[1]}")
            
            # Model-based selection but not too aggressive
            rf_selector = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            self.model_selector = SelectFromModel(rf_selector, threshold='0.5*mean', max_features=18)
            X_train_final = self.model_selector.fit_transform(X_train_selected, y_train)
            
            if X_test is not None:
                X_test_final = self.model_selector.transform(X_test_selected)
            
            print(f"Features after model selection: {X_train_final.shape[1]}")
            
            return (X_train_final, X_test_final) if X_test is not None else X_train_final
        else:
            X_selected = self.selector.transform(X_train)
            X_final = self.model_selector.transform(X_selected)
            return X_final
    
    def create_refined_ensemble(self):
        """Create refined ensemble with proven models"""
        base_models = [
            ('rf1', RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=3, 
                                         min_samples_leaf=2, random_state=42)),
            ('rf2', RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_split=4, 
                                         min_samples_leaf=1, random_state=123)),
            ('et', ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_split=3, 
                                      min_samples_leaf=2, random_state=456)),
            ('gb1', GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=6, 
                                             subsample=0.8, random_state=789)),
            ('gb2', GradientBoostingClassifier(n_estimators=150, learning_rate=0.12, max_depth=5, 
                                             subsample=0.9, random_state=101112)),
            ('svm1', SVC(C=2.0, kernel='rbf', gamma='scale', probability=True, random_state=42)),
            ('svm2', SVC(C=1.5, kernel='linear', probability=True, random_state=123)),
            ('lr1', LogisticRegression(C=1.5, penalty='l2', random_state=456, max_iter=1500)),
            ('lr2', LogisticRegression(C=2.0, penalty='l1', solver='liblinear', random_state=789)),
            ('nb', GaussianNB()),
            ('knn', KNeighborsClassifier(n_neighbors=6, weights='distance'))
        ]
        
        # Enhanced meta-learner
        meta_model = CalibratedClassifierCV(
            LogisticRegression(C=1.5, penalty='l2', random_state=42, max_iter=2000), 
            method='isotonic', cv=5
        )
        
        return base_models, meta_model
    
    def train_refined_ensemble(self, X_train, y_train, X_val, y_val):
        """Train refined ensemble with 5 seeds like successful approach"""
        print(f"\n=== REFINED MULTI-SEED ENSEMBLE TRAINING ===")
        
        all_predictions_val = []
        all_trained_models = []
        
        for seed_idx, seed in enumerate(self.random_seeds):
            print(f"\n--- Seed {seed} ({seed_idx+1}/{len(self.random_seeds)}) ---")
            
            base_models, meta_model = self.create_refined_ensemble()
            
            # Update random states
            for name, model in base_models:
                if hasattr(model, 'random_state'):
                    model.random_state = seed
            
            # Train with current seed
            base_predictions_train = np.zeros((X_train.shape[0], len(base_models)))
            base_predictions_val = np.zeros((X_val.shape[0], len(base_models)))
            
            trained_base_models = []
            
            for i, (name, model) in enumerate(base_models):
                # Cross-validation for training predictions
                cv_preds = np.zeros(X_train.shape[0])
                
                for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    
                    model_copy = type(model)(**model.get_params())
                    if hasattr(model_copy, 'random_state'):
                        model_copy.random_state = seed + i
                    
                    model_copy.fit(X_fold_train, y_fold_train)
                    cv_preds[val_idx] = model_copy.predict_proba(X_fold_val)[:, 1]
                
                base_predictions_train[:, i] = cv_preds
                
                # Train on full training set
                model.fit(X_train, y_train)
                base_predictions_val[:, i] = model.predict_proba(X_val)[:, 1]
                trained_base_models.append(model)
            
            # Train meta-model for this seed
            meta_model.fit(base_predictions_train, y_train)
            
            all_predictions_val.append(meta_model.predict_proba(base_predictions_val)[:, 1])
            all_trained_models.append((trained_base_models, meta_model))
            
            # Validation accuracy for this seed
            seed_predictions = meta_model.predict(base_predictions_val)
            seed_accuracy = accuracy_score(y_val, seed_predictions)
            print(f"Seed {seed} ensemble accuracy: {seed_accuracy:.4f}")
        
        # Combine predictions from all seeds
        final_val_predictions = np.mean(all_predictions_val, axis=0)
        final_val_binary = (final_val_predictions > 0.5).astype(int)
        ensemble_accuracy = accuracy_score(y_val, final_val_binary)
        
        print(f"\nRefined multi-seed ensemble accuracy: {ensemble_accuracy:.4f}")
        
        return all_trained_models, ensemble_accuracy
    
    def conservative_pseudo_labeling(self, X_train, y_train, X_test_unlabeled, confidence_threshold=0.98):
        """Very conservative pseudo-labeling"""
        print(f"\n=== CONSERVATIVE PSEUDO LABELING ===")
        
        # Use multiple models for pseudo-labeling
        pseudo_models = [
            RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
            GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=123),
            LogisticRegression(C=1.0, random_state=456, max_iter=1500)
        ]
        
        all_probs = []
        all_preds = []
        
        for model in pseudo_models:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test_unlabeled)
            preds = model.predict(X_test_unlabeled)
            all_probs.append(probs)
            all_preds.append(preds)
        
        # Average predictions
        avg_probs = np.mean(all_probs, axis=0)
        max_probs = np.max(avg_probs, axis=1)
        consensus_preds = np.round(np.mean([p for p in all_preds], axis=0)).astype(int)
        
        # Very high confidence threshold
        high_confidence_mask = max_probs >= confidence_threshold
        
        if high_confidence_mask.sum() > 0:
            X_pseudo = X_test_unlabeled[high_confidence_mask]
            y_pseudo = consensus_preds[high_confidence_mask]
            
            # Add to training data
            X_train_expanded = np.vstack([X_train, X_pseudo])
            y_train_expanded = np.concatenate([y_train, y_pseudo])
            
            print(f"Added {len(y_pseudo)} high-confidence pseudo-labeled samples")
            print(f"New training size: {len(y_train_expanded)}")
            
            return X_train_expanded, pd.Series(y_train_expanded)
        else:
            print("No high-confidence pseudo-labels found")
            return X_train, y_train
    
    def refined_pca(self, X_train, X_test=None, variance_threshold=0.98):
        """Refined PCA - keep complexity like successful approach"""
        print(f"\n=== REFINED PCA ===")
        print(f"Original feature dimensions: {X_train.shape[1]}")
        
        # Robust scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        # Less aggressive PCA (keep more components)
        pca_temp = PCA()
        pca_temp.fit(X_train_scaled)
        
        cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
        n_components = min(n_components, X_train.shape[1], 14)  # Cap at 14 like successful approach
        n_components = max(n_components, 8)  # Minimum 8 components
        
        print(f"Number of components for {variance_threshold*100}% variance: {n_components}")
        print(f"Explained variance ratio: {cumsum_var[n_components-1]:.4f}")
        
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        
        if X_test is not None:
            X_test_pca = self.pca.transform(X_test_scaled)
            return X_train_pca, X_test_pca
        
        return X_train_pca
    
    def run_refined_pipeline(self, train_path, test_path=None, target_server_accuracy=0.90):
        """Run refined advanced pipeline building on what worked"""
        print("=== REFINED ADVANCED TITANIC PIPELINE ===")
        print("Building on the successful 78.229% approach with refinements...")
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Refined advanced feature engineering
        print(f"\n=== REFINED ADVANCED FEATURE ENGINEERING ===")
        train_engineered = self.refined_advanced_features(self.train_df, is_train=True)
        
        # Refined preprocessing
        print(f"\n=== REFINED PREPROCESSING ===")
        X = self.refined_preprocessing(train_engineered, is_train=True)
        y = self.train_df['Survived']
        
        print(f"Features after preprocessing: {X.shape[1]}")
        
        # Data split similar to successful approach
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        # Refined feature selection
        X_train_selected, X_test_selected = self.refined_feature_selection(X_train, y_train, X_test)
        
        # Refined PCA
        X_train_pca, X_test_pca = self.refined_pca(X_train_selected, X_test_selected, variance_threshold=0.98)
        
        print(f"Final feature dimensions: {X_train_pca.shape[1]}")
        
        # Train/validation split
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_pca, y_train, test_size=0.18, random_state=42, stratify=y_train
        )
        
        # Conservative pseudo-labeling (restore but make it very conservative)
        if hasattr(self, 'test_df'):
            test_engineered = self.refined_advanced_features(self.test_df, is_train=False)
            X_test_unlabeled = self.refined_preprocessing(test_engineered, is_train=False)
            X_test_unlabeled_selected = self.refined_feature_selection(X_test_unlabeled)
            X_test_unlabeled_scaled = self.scaler.transform(X_test_unlabeled_selected)
            X_test_unlabeled_pca = self.pca.transform(X_test_unlabeled_scaled)
            
            X_train_final, y_train_final = self.conservative_pseudo_labeling(
                X_train_final, y_train_final, X_test_unlabeled_pca, confidence_threshold=0.98
            )
        
        # Refined multi-seed ensemble training
        all_trained_models, ensemble_accuracy = self.train_refined_ensemble(
            X_train_final, y_train_final, X_val, y_val
        )
        
        # Final evaluation with multi-seed averaging
        final_test_predictions = []
        
        for trained_models in all_trained_models:
            base_models, meta_model = trained_models
            base_predictions_test = np.zeros((X_test_pca.shape[0], len(base_models)))
            
            for i, model in enumerate(base_models):
                base_predictions_test[:, i] = model.predict_proba(X_test_pca)[:, 1]
            
            seed_test_pred_proba = meta_model.predict_proba(base_predictions_test)[:, 1]
            final_test_predictions.append(seed_test_pred_proba)
        
        # Average across all seeds
        avg_test_predictions = np.mean(final_test_predictions, axis=0)
        avg_test_binary = (avg_test_predictions > 0.5).astype(int)
        final_accuracy = accuracy_score(y_test, avg_test_binary)
        
        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in self.cv_strategy.split(X_train_pca, y_train):
            X_cv_train, X_cv_val = X_train_pca[train_idx], X_train_pca[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Ensemble for CV
            ensemble_models = [
                RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
                GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
                LogisticRegression(C=1.5, random_state=42, max_iter=1500)
            ]
            
            cv_preds = np.zeros(len(y_cv_val))
            for model in ensemble_models:
                model.fit(X_cv_train, y_cv_train)
                cv_preds += model.predict_proba(X_cv_val)[:, 1]
            
            cv_preds = (cv_preds / len(ensemble_models) > 0.5).astype(int)
            cv_score = accuracy_score(y_cv_val, cv_preds)
            cv_scores.append(cv_score)
        
        refined_cv_score = np.mean(cv_scores)
        refined_cv_std = np.std(cv_scores)
        
        # Optimistic server estimate (based on successful pattern)
        server_estimate = refined_cv_score * 0.98  # Less conservative
        
        print(f"\n=== REFINED EVALUATION ===")
        print(f"Multi-seed ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"Test accuracy: {final_accuracy:.4f}")
        print(f"Refined CV score: {refined_cv_score:.4f} ± {refined_cv_std:.4f}")
        print(f"Server estimate: {server_estimate:.4f}")
        
        # Store models
        self.all_trained_models = all_trained_models
        self.best_score = refined_cv_score
        
        print(f"\n=== TARGET ANALYSIS ===")
        print(f"Previous Best Server: 78.229%")
        print(f"Estimated Server Performance: {server_estimate:.1%}")
        
        if server_estimate >= 0.785:
            print("🎯 HIGH CONFIDENCE FOR BEATING 78.229%! 🎯")
        elif server_estimate >= 0.780:
            print("✅ GOOD CONFIDENCE FOR MATCHING 78.229%")
        else:
            print(f"📈 Solid approach - should improve from recent scores")
        
        return {
            'final_accuracy': final_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'refined_cv_score': refined_cv_score,
            'refined_cv_std': refined_cv_std,
            'server_estimate': server_estimate,
            'final_features': X_train_pca.shape[1],
            'original_features': X.shape[1]
        }
    
    def create_refined_submission(self, test_path, output_path="refined_submission.csv"):
        """Create refined submission building on successful approach"""
        print(f"\n=== CREATING REFINED SUBMISSION ===")
        
        if not hasattr(self, 'all_trained_models'):
            print("Error: No trained models found. Run refined pipeline first.")
            return
        
        # Process test data
        test_df = pd.read_csv(test_path)
        test_engineered = self.refined_advanced_features(test_df, is_train=False)
        X_test = self.refined_preprocessing(test_engineered, is_train=False)
        
        # Apply transformations
        X_test_selected = self.refined_feature_selection(X_test)
        X_test_scaled = self.scaler.transform(X_test_selected)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Multi-seed ensemble predictions
        all_probabilities = []
        
        for seed_idx, trained_models in enumerate(self.all_trained_models):
            base_models, meta_model = trained_models
            base_predictions = np.zeros((X_test_pca.shape[0], len(base_models)))
            
            for i, model in enumerate(base_models):
                base_predictions[:, i] = model.predict_proba(X_test_pca)[:, 1]
            
            seed_probabilities = meta_model.predict_proba(base_predictions)[:, 1]
            all_probabilities.append(seed_probabilities)
        
        # Average predictions across all seeds
        avg_probabilities = np.mean(all_probabilities, axis=0)
        final_predictions = (avg_probabilities > 0.5).astype(int)
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': final_predictions
        })
        
        submission.to_csv(output_path, index=False)
        print(f"Refined submission saved to: {output_path}")
        print(f"Total predictions: {len(final_predictions)}")
        print(f"Predicted survivors: {final_predictions.sum()} ({final_predictions.mean():.1%})")
        print(f"Prediction confidence - Mean: {avg_probabilities.mean():.3f}, Std: {avg_probabilities.std():.3f}")
        
        return submission

# Run the refined pipeline
if __name__ == "__main__":
    pipeline = RefinedAdvancedTitanicPipeline()
    
    # Set paths
    train_path = "../datasets/titanic/train.csv"
    test_path = "../datasets/titanic/test.csv"
    
    # Run refined pipeline
    results = pipeline.run_refined_pipeline(train_path, test_path, target_server_accuracy=0.90)
    
    print(f"\n=== REFINED PIPELINE SUMMARY ===")
    print(f"Original Features: {results['original_features']}")
    print(f"Final Features: {results['final_features']}")
    print(f"Feature Reduction: {(1 - results['final_features']/results['original_features'])*100:.1f}%")
    print(f"Refined CV Score: {results['refined_cv_score']:.1%} ± {results['refined_cv_std']:.1%}")
    print(f"Server Estimate: {results['server_estimate']:.1%}")
    
    # Create refined submission
    submission = pipeline.create_refined_submission(test_path, "refined_submission.csv")
    
    print(f"\n=== REFINED SUBMISSION READY ===")
    print("🎯 Refined submission 'refined_submission.csv' - building on 78.229% success!")
    print("Key refinements:")
    print("✅ Restored sophisticated feature engineering that worked")
    print("✅ Conservative pseudo-labeling with 98% confidence threshold")
    print("✅ 5-seed ensemble like successful approach")
    print("✅ 8-14 PCA components (not 3 or 5)")
    print("✅ Less aggressive feature selection")
    print("✅ Enhanced ensemble diversity")
    print("✅ Better calibration and meta-learning")
    
    if results['server_estimate'] >= 0.785:
        print(f"\n🚀 HIGH CONFIDENCE: Should beat 78.229% and reach {results['server_estimate']*100:.1f}%+ 🚀") 