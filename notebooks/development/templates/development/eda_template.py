"""
Exploratory Data Analysis Template for NeurIPS Polymer Prediction
================================================================

This notebook template provides a comprehensive EDA framework for the competition.
Use this for local development before adapting to Kaggle notebook format.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# Data Loading
# ============================================================================

def load_competition_data(data_path: str = "data/raw"):
    """Load competition data"""
    data_path = Path(data_path)
    
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    sample_sub = pd.read_csv(data_path / "sample_submission.csv")
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample submission shape: {sample_sub.shape}")
    
    return train, test, sample_sub

# ============================================================================
# Basic Data Exploration
# ============================================================================

def basic_info(df, name="Dataset"):
    """Display basic information about the dataset"""
    print(f"\n{'='*50}")
    print(f"{name} Basic Information")
    print(f"{'='*50}")
    
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing)
    else:
        print("No missing values")
    
    print(f"\nDuplicated rows: {df.duplicated().sum()}")

def explore_columns(df):
    """Explore each column in detail"""
    print(f"\n{'='*50}")
    print("Column Exploration")
    print(f"{'='*50}")
    
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  Type: {df[col].dtype}")
        print(f"  Unique values: {df[col].nunique()}")
        
        if df[col].dtype in ['object']:
            print(f"  Sample values: {df[col].value_counts().head(3).to_dict()}")
        else:
            print(f"  Range: {df[col].min():.4f} to {df[col].max():.4f}")
            print(f"  Mean: {df[col].mean():.4f}, Std: {df[col].std():.4f}")

# ============================================================================
# Target Variable Analysis
# ============================================================================

def analyze_targets(train_df, target_cols=None):
    """Analyze target variables"""
    if target_cols is None:
        # Auto-detect target columns (numeric columns except id)
        target_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in target_cols:
            target_cols.remove('id')
    
    print(f"\n{'='*50}")
    print("Target Variable Analysis")
    print(f"{'='*50}")
    
    print(f"Target columns: {target_cols}")
    
    # Statistical summary
    print(f"\nTarget statistics:")
    print(train_df[target_cols].describe())
    
    # Correlation matrix
    if len(target_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = train_df[target_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Target Variables Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    # Distribution plots
    n_targets = len(target_cols)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_targets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, target in enumerate(target_cols):
        row = i // n_cols
        col = i % n_cols
        
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Histogram
        ax.hist(train_df[target].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribution of {target}')
        ax.set_xlabel(target)
        ax.set_ylabel('Frequency')
        
        # Add statistics
        mean_val = train_df[target].mean()
        median_val = train_df[target].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_targets, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# Feature Analysis
# ============================================================================

def analyze_polymer_smiles(df, smiles_col='polymer_smiles'):
    """Analyze polymer SMILES strings"""
    if smiles_col not in df.columns:
        print(f"Column {smiles_col} not found")
        return
    
    print(f"\n{'='*50}")
    print("Polymer SMILES Analysis")
    print(f"{'='*50}")
    
    # Basic statistics
    smiles_lengths = df[smiles_col].str.len()
    print(f"SMILES length statistics:")
    print(smiles_lengths.describe())
    
    # Character frequency
    all_chars = ''.join(df[smiles_col].fillna(''))
    char_freq = pd.Series(list(all_chars)).value_counts().head(20)
    
    plt.figure(figsize=(15, 5))
    
    # Length distribution
    plt.subplot(1, 2, 1)
    plt.hist(smiles_lengths.dropna(), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of SMILES Length')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    
    # Character frequency
    plt.subplot(1, 2, 2)
    char_freq.plot(kind='bar')
    plt.title('Most Common Characters in SMILES')
    plt.xlabel('Character')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nMost common characters:")
    print(char_freq.head(10))

def correlation_with_targets(train_df, target_cols, feature_cols=None):
    """Analyze correlation between features and targets"""
    if feature_cols is None:
        # Auto-detect numeric feature columns
        feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove targets and id
        feature_cols = [col for col in feature_cols if col not in target_cols + ['id']]
    
    if len(feature_cols) == 0:
        print("No numeric features found for correlation analysis")
        return
    
    print(f"\n{'='*50}")
    print("Feature-Target Correlation Analysis")
    print(f"{'='*50}")
    
    # Calculate correlations
    corr_data = []
    for target in target_cols:
        for feature in feature_cols:
            corr = train_df[feature].corr(train_df[target])
            corr_data.append({
                'target': target,
                'feature': feature,
                'correlation': corr
            })
    
    corr_df = pd.DataFrame(corr_data)
    
    # Top correlations for each target
    for target in target_cols:
        target_corrs = corr_df[corr_df['target'] == target].copy()
        target_corrs['abs_correlation'] = target_corrs['correlation'].abs()
        target_corrs = target_corrs.sort_values('abs_correlation', ascending=False)
        
        print(f"\nTop correlations with {target}:")
        print(target_corrs.head(10)[['feature', 'correlation']].to_string(index=False))

# ============================================================================
# Main EDA Function
# ============================================================================

def run_eda():
    """Run complete EDA pipeline"""
    print("=" * 80)
    print("NeurIPS Open Polymer Prediction 2025 - Exploratory Data Analysis")
    print("=" * 80)
    
    # Load data
    train, test, sample_sub = load_competition_data()
    
    # Basic exploration
    basic_info(train, "Training Data")
    basic_info(test, "Test Data")
    basic_info(sample_sub, "Sample Submission")
    
    explore_columns(train)
    
    # Target analysis
    target_cols = [col for col in sample_sub.columns if col != 'id']
    analyze_targets(train, target_cols)
    
    # SMILES analysis
    analyze_polymer_smiles(train)
    
    # Feature correlation
    correlation_with_targets(train, target_cols)
    
    print("\n" + "=" * 80)
    print("EDA completed!")
    print("=" * 80)

if __name__ == "__main__":
    run_eda()