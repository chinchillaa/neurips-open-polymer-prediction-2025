"""
Simple Baseline for NeurIPS Open Polymer Prediction 2025
========================================================

This notebook provides a simple yet effective baseline using Random Forest
with molecular fingerprints. It's designed to be fast, reliable, and
easy to understand for code competition.

Key Features:
- Uses RDKit Morgan fingerprints for feature extraction
- Random Forest regressor for each target property
- Handles missing values appropriately
- Optimized for 9-hour execution limit
- Produces submission.csv file
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for the baseline model"""
    # Paths (adjust for Kaggle environment)
    COMP_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025'
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test.csv'
    SAMPLE_SUB_FILE = 'sample_submission.csv'
    SUBMISSION_FILE = 'submission.csv'
    
    # Model parameters
    RANDOM_STATE = 42
    N_ESTIMATORS = 100
    MAX_DEPTH = 10
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2
    N_JOBS = -1
    
    # Feature parameters
    MORGAN_RADIUS = 2
    MORGAN_NBITS = 1024
    
    # Target columns
    TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# ============================================================================
# Utility Functions
# ============================================================================

def timer_decorator(func):
    """Decorator to time function execution"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} completed in {end - start:.2f} seconds")
        return result
    return wrapper

def reduce_mem_usage(df):
    """Reduce memory usage of dataframe"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

# ============================================================================
# Feature Engineering
# ============================================================================

@timer_decorator
def extract_molecular_features(smiles_list):
    """Extract Morgan fingerprints from SMILES strings"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print("RDKit not available. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit"])
        from rdkit import Chem
        from rdkit.Chem import AllChem
    
    print(f"Extracting molecular features from {len(smiles_list)} SMILES strings...")
    
    def smiles_to_fp(smiles):
        """Convert SMILES to Morgan fingerprint"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Return zero vector for invalid SMILES
                return np.zeros(Config.MORGAN_NBITS)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, Config.MORGAN_RADIUS, Config.MORGAN_NBITS
            )
            return np.array(fp)
        except:
            # Return zero vector for any errors
            return np.zeros(Config.MORGAN_NBITS)
    
    # Extract features with progress indication
    features = []
    for i, smiles in enumerate(smiles_list):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(smiles_list)} molecules...")
        features.append(smiles_to_fp(smiles))
    
    feature_matrix = np.vstack(features)
    print(f"Feature extraction complete. Shape: {feature_matrix.shape}")
    
    return feature_matrix

# ============================================================================
# Model Training and Prediction
# ============================================================================

class PolymerPredictor:
    """Simple Random Forest predictor for polymer properties"""
    
    def __init__(self):
        self.models = {}
        self.feature_names = None
        
    @timer_decorator
    def fit(self, X, y):
        """Train separate Random Forest for each target"""
        print("Training Random Forest models...")
        
        for i, target in enumerate(Config.TARGET_COLS):
            print(f"Training model for {target}...")
            
            # Get target values and mask for non-NaN values
            y_target = y[:, i]
            mask = ~np.isnan(y_target)
            
            if mask.sum() == 0:
                print(f"Warning: No valid data for {target}")
                continue
            
            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=Config.N_ESTIMATORS,
                max_depth=Config.MAX_DEPTH,
                min_samples_split=Config.MIN_SAMPLES_SPLIT,
                min_samples_leaf=Config.MIN_SAMPLES_LEAF,
                random_state=Config.RANDOM_STATE,
                n_jobs=Config.N_JOBS,
                verbose=0
            )
            
            rf.fit(X[mask], y_target[mask])
            self.models[target] = rf
            
            print(f"  {target}: trained on {mask.sum()} samples")
        
        print(f"Training complete. Trained {len(self.models)} models.")
    
    @timer_decorator
    def predict(self, X):
        """Make predictions for all targets"""
        print("Making predictions...")
        
        predictions = np.zeros((X.shape[0], len(Config.TARGET_COLS)))
        
        for i, target in enumerate(Config.TARGET_COLS):
            if target in self.models:
                predictions[:, i] = self.models[target].predict(X)
                print(f"  {target}: predictions generated")
            else:
                # Use mean value if model wasn't trained
                predictions[:, i] = 0.0
                print(f"  {target}: using zero (no model available)")
        
        return predictions
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        print("\nModel Evaluation:")
        print("-" * 40)
        
        mse_scores = {}
        for i, target in enumerate(Config.TARGET_COLS):
            mask = ~np.isnan(y_true[:, i])
            if mask.sum() > 0:
                mse = mean_squared_error(y_true[mask, i], y_pred[mask, i])
                mse_scores[target] = mse
                print(f"{target:10s}: MSE = {mse:.4f} (n={mask.sum()})")
            else:
                mse_scores[target] = np.nan
                print(f"{target:10s}: No valid data")
        
        # Overall MSE
        mask_all = ~np.isnan(y_true)
        if mask_all.sum() > 0:
            overall_mse = mean_squared_error(y_true[mask_all], y_pred[mask_all])
            print(f"{'Overall':10s}: MSE = {overall_mse:.4f}")
        
        return mse_scores

# ============================================================================
# Main Pipeline
# ============================================================================

@timer_decorator
def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    
    # Load training data
    train_path = f"{Config.COMP_PATH}/{Config.TRAIN_FILE}"
    train_df = pd.read_csv(train_path)
    print(f"Train data shape: {train_df.shape}")
    
    # Load test data
    test_path = f"{Config.COMP_PATH}/{Config.TEST_FILE}"
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    # Load sample submission
    sample_path = f"{Config.COMP_PATH}/{Config.SAMPLE_SUB_FILE}"
    sample_sub = pd.read_csv(sample_path)
    print(f"Sample submission shape: {sample_sub.shape}")
    
    # Display basic info
    print("\nTarget columns info:")
    for col in Config.TARGET_COLS:
        if col in train_df.columns:
            valid_count = train_df[col].notna().sum()
            print(f"  {col}: {valid_count}/{len(train_df)} valid values")
    
    return train_df, test_df, sample_sub

@timer_decorator
def create_validation_split(train_df, test_size=0.2):
    """Create validation split for model evaluation"""
    from sklearn.model_selection import train_test_split
    
    print(f"Creating validation split ({test_size*100:.0f}% for validation)...")
    
    train_split, val_split = train_test_split(
        train_df, 
        test_size=test_size, 
        random_state=Config.RANDOM_STATE,
        shuffle=True
    )
    
    print(f"Training split: {len(train_split)} samples")
    print(f"Validation split: {len(val_split)} samples")
    
    return train_split, val_split

def main():
    """Main execution pipeline"""
    print("=" * 60)
    print("NeurIPS Open Polymer Prediction 2025 - Simple Baseline")
    print("=" * 60)
    
    # Load data
    train_df, test_df, sample_sub = load_data()
    
    # Reduce memory usage
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    
    # Create validation split
    train_split, val_split = create_validation_split(train_df)
    
    # Extract features for all splits
    print("\nExtracting features...")
    X_train = extract_molecular_features(train_split['SMILES'].tolist())
    X_val = extract_molecular_features(val_split['SMILES'].tolist())
    X_test = extract_molecular_features(test_df['SMILES'].tolist())
    
    # Prepare targets
    y_train = train_split[Config.TARGET_COLS].values
    y_val = val_split[Config.TARGET_COLS].values
    
    # Train model
    print("\n" + "="*40)
    print("MODEL TRAINING")
    print("="*40)
    
    predictor = PolymerPredictor()
    predictor.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("\n" + "="*40)
    print("VALIDATION EVALUATION")
    print("="*40)
    
    val_scores = predictor.evaluate(X_val, y_val)
    
    # Retrain on full dataset
    print("\n" + "="*40)
    print("FINAL MODEL TRAINING")
    print("="*40)
    
    X_full = extract_molecular_features(train_df['SMILES'].tolist())
    y_full = train_df[Config.TARGET_COLS].values
    
    final_predictor = PolymerPredictor()
    final_predictor.fit(X_full, y_full)
    
    # Make predictions on test set
    print("\n" + "="*40)
    print("TEST PREDICTIONS")
    print("="*40)
    
    test_predictions = final_predictor.predict(X_test)
    
    # Create submission
    print("\nCreating submission file...")
    submission_df = sample_sub.copy()
    submission_df[Config.TARGET_COLS] = test_predictions
    
    # Save submission
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    
    # Display submission summary
    print("\nSubmission Summary:")
    print(f"Shape: {submission_df.shape}")
    print(submission_df.head())
    
    print("\nTarget statistics:")
    for col in Config.TARGET_COLS:
        values = submission_df[col]
        print(f"  {col}: mean={values.mean():.4f}, std={values.std():.4f}")
    
    print("\n" + "="*60)
    print("BASELINE COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()