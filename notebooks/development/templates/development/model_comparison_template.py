"""
Model Comparison Template for Polymer Prediction
==============================================

This template provides a comprehensive framework for comparing
different machine learning models for the polymer prediction task.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import joblib
import time

# ============================================================================
# Model Definitions
# ============================================================================

def get_models():
    """Define all models to compare"""
    models = {
        # Linear models
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=1.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
        
        # Tree-based models
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        
        # Support Vector Machine
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
    }
    
    return models

def get_advanced_models():
    """Define advanced models (if libraries are available)"""
    advanced_models = {}
    
    # XGBoost
    try:
        import xgboost as xgb
        advanced_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    except ImportError:
        print("XGBoost not available")
    
    # LightGBM
    try:
        import lightgbm as lgb
        advanced_models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    except ImportError:
        print("LightGBM not available")
    
    # CatBoost
    try:
        import catboost as cb
        advanced_models['CatBoost'] = cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
    except ImportError:
        print("CatBoost not available")
    
    return advanced_models

# ============================================================================
# Model Evaluation
# ============================================================================

class ModelComparator:
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
        self.scalers = {}
        
    def evaluate_models(self, X, y, target_name="target"):
        """Evaluate all models using cross-validation"""
        print(f"Evaluating models for {target_name}...")
        print(f"Dataset shape: {X.shape}")
        print(f"Target range: {y.min():.4f} to {y.max():.4f}")
        
        # Get all models
        models = {**get_models(), **get_advanced_models()}
        
        results = []
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Time the evaluation
                start_time = time.time()
                
                # Check if model needs scaling
                if model_name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR']:
                    X_scaled = self._scale_features(X, model_name)
                    X_eval = X_scaled
                else:
                    X_eval = X
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_eval, y, 
                    cv=kf, 
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
                
                end_time = time.time()
                
                # Calculate metrics
                rmse_scores = -cv_scores
                rmse_mean = rmse_scores.mean()
                rmse_std = rmse_scores.std()
                
                results.append({
                    'model': model_name,
                    'target': target_name,
                    'rmse_mean': rmse_mean,
                    'rmse_std': rmse_std,
                    'training_time': end_time - start_time,
                    'cv_scores': rmse_scores.tolist()
                })
                
                print(f"  RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
                print(f"  Time: {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                continue
        
        # Store results
        self.results[target_name] = results
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('rmse_mean')
            print(f"\n{target_name} Model Ranking:")
            print(summary_df[['model', 'rmse_mean', 'rmse_std', 'training_time']].to_string(index=False))
        
        return summary_df
    
    def _scale_features(self, X, model_name):
        """Scale features for models that require it"""
        if model_name not in self.scalers:
            if model_name == 'SVR':
                self.scalers[model_name] = RobustScaler()
            else:
                self.scalers[model_name] = StandardScaler()
            
            X_scaled = self.scalers[model_name].fit_transform(X)
        else:
            X_scaled = self.scalers[model_name].transform(X)
        
        return X_scaled
    
    def train_best_models(self, X, y, target_cols):
        """Train best performing models for each target"""
        best_models = {}
        
        for target in target_cols:
            if target in self.results and self.results[target]:
                # Find best model for this target
                best_result = min(self.results[target], key=lambda x: x['rmse_mean'])
                best_model_name = best_result['model']
                
                print(f"\nTraining best model for {target}: {best_model_name}")
                
                # Get model
                all_models = {**get_models(), **get_advanced_models()}
                model = all_models[best_model_name]
                
                # Prepare data
                if best_model_name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR']:
                    X_train = self._scale_features(X, best_model_name)
                else:
                    X_train = X
                
                # Train model
                model.fit(X_train, y[target])
                
                best_models[target] = {
                    'model': model,
                    'model_name': best_model_name,
                    'scaler': self.scalers.get(best_model_name),
                    'cv_rmse': best_result['rmse_mean']
                }
                
                print(f"  CV RMSE: {best_result['rmse_mean']:.4f}")
        
        return best_models
    
    def create_ensemble(self, X, y, target_cols, top_n=3):
        """Create ensemble of top N models for each target"""
        ensemble_models = {}
        
        for target in target_cols:
            if target in self.results and len(self.results[target]) >= top_n:
                # Get top N models
                sorted_results = sorted(self.results[target], key=lambda x: x['rmse_mean'])
                top_models = sorted_results[:top_n]
                
                print(f"\nCreating ensemble for {target} with top {top_n} models:")
                
                ensemble_predictions = []
                ensemble_weights = []
                
                all_models = {**get_models(), **get_advanced_models()}
                
                for result in top_models:
                    model_name = result['model']
                    model = all_models[model_name]
                    
                    # Prepare data
                    if model_name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR']:
                        X_train = self._scale_features(X, model_name)
                    else:
                        X_train = X
                    
                    # Train model
                    model.fit(X_train, y[target])
                    
                    # Weight based on inverse of RMSE
                    weight = 1.0 / result['rmse_mean']
                    ensemble_weights.append(weight)
                    
                    print(f"  {model_name}: RMSE {result['rmse_mean']:.4f}, Weight: {weight:.4f}")
                
                # Normalize weights
                total_weight = sum(ensemble_weights)
                ensemble_weights = [w / total_weight for w in ensemble_weights]
                
                ensemble_models[target] = {
                    'models': [(result['model'], all_models[result['model']], ensemble_weights[i]) 
                              for i, result in enumerate(top_models)],
                    'weights': ensemble_weights
                }
        
        return ensemble_models

# ============================================================================
# Hyperparameter Optimization
# ============================================================================

def optimize_hyperparameters(X, y, model_type='RandomForest', cv_folds=3):
    """Basic hyperparameter optimization"""
    print(f"Optimizing hyperparameters for {model_type}...")
    
    if model_type == 'RandomForest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    elif model_type == 'GradientBoosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = GradientBoostingRegressor(random_state=42)
    
    else:
        print(f"Optimization not implemented for {model_type}")
        return None
    
    # Simple grid search (can be replaced with RandomizedSearchCV for efficiency)
    from sklearn.model_selection import GridSearchCV
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_folds,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# ============================================================================
# Main Comparison Function
# ============================================================================

def run_model_comparison(train_df, features, target_cols):
    """Run complete model comparison pipeline"""
    print("=" * 80)
    print("Model Comparison for Polymer Prediction")
    print("=" * 80)
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Prepare feature matrix
    X = train_df[features].values
    y = train_df[target_cols]
    
    print(f"Features: {len(features)}")
    print(f"Targets: {target_cols}")
    print(f"Data shape: {X.shape}")
    
    # Evaluate models for each target
    all_results = {}
    for target in target_cols:
        if target in train_df.columns:
            results_df = comparator.evaluate_models(X, y[target].values, target)
            all_results[target] = results_df
    
    # Train best models
    best_models = comparator.train_best_models(X, y, target_cols)
    
    # Create ensembles
    ensemble_models = comparator.create_ensemble(X, y, target_cols)
    
    # Save results
    results_summary = {
        'comparison_results': all_results,
        'best_models': best_models,
        'ensemble_models': ensemble_models
    }
    
    # Save to file
    joblib.dump(results_summary, 'model_comparison_results.pkl')
    print(f"\nResults saved to model_comparison_results.pkl")
    
    return results_summary

if __name__ == "__main__":
    # Example usage
    train = pd.read_csv('data/raw/train.csv')
    
    # Assume features have been created
    feature_cols = [col for col in train.columns if col not in ['id', 'polymer_smiles']]
    target_cols = ['target1', 'target2']  # Replace with actual target columns
    
    results = run_model_comparison(train, feature_cols, target_cols)