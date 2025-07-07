"""
Feature Engineering Template for Polymer Prediction
=================================================

This template provides comprehensive feature engineering approaches
specifically designed for polymer SMILES data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Basic Molecular Features
# ============================================================================

def extract_basic_features(smiles_series):
    """Extract basic molecular features from SMILES strings"""
    features = pd.DataFrame()
    
    # Length features
    features['smiles_length'] = smiles_series.str.len()
    features['smiles_word_count'] = smiles_series.str.split().str.len()
    
    # Character counts
    features['carbon_count'] = smiles_series.str.count('C')
    features['nitrogen_count'] = smiles_series.str.count('N')
    features['oxygen_count'] = smiles_series.str.count('O')
    features['sulfur_count'] = smiles_series.str.count('S')
    features['fluorine_count'] = smiles_series.str.count('F')
    features['chlorine_count'] = smiles_series.str.count('Cl')
    features['bromine_count'] = smiles_series.str.count('Br')
    
    # Bond features
    features['single_bond_count'] = smiles_series.str.count('-')
    features['double_bond_count'] = smiles_series.str.count('=')
    features['triple_bond_count'] = smiles_series.str.count('#')
    features['aromatic_bond_count'] = smiles_series.str.count(':')
    
    # Ring features
    features['ring_count'] = smiles_series.str.count(r'\d+').fillna(0)
    features['bracket_count'] = smiles_series.str.count('\[') + smiles_series.str.count('\]')
    features['parentheses_count'] = smiles_series.str.count('\(') + smiles_series.str.count('\)')
    
    return features

def extract_advanced_molecular_features(smiles_series):
    """Extract advanced molecular features"""
    features = pd.DataFrame()
    
    # Complexity measures
    features['unique_chars'] = smiles_series.apply(lambda x: len(set(str(x))) if pd.notna(x) else 0)
    features['char_diversity'] = features['unique_chars'] / smiles_series.str.len()
    
    # Branching indicators
    features['branch_points'] = smiles_series.str.count('\(')
    features['branch_complexity'] = features['branch_points'] / smiles_series.str.len()
    
    # Functional group patterns
    features['carboxyl_groups'] = smiles_series.str.count('C\(=O\)O')
    features['hydroxyl_groups'] = smiles_series.str.count('O')  # Simplified
    features['amino_groups'] = smiles_series.str.count('N')    # Simplified
    features['ester_groups'] = smiles_series.str.count('C\(=O\)O')
    
    # Ring system complexity
    features['aromatic_rings'] = smiles_series.str.count('c')  # Aromatic carbon
    features['cyclic_structures'] = smiles_series.str.count('\d')
    
    return features

# ============================================================================
# Statistical Features
# ============================================================================

def create_statistical_features(df, numeric_cols):
    """Create statistical features from existing numeric columns"""
    stat_features = pd.DataFrame()
    
    if len(numeric_cols) > 1:
        # Cross-column statistics
        stat_features['mean_all'] = df[numeric_cols].mean(axis=1)
        stat_features['std_all'] = df[numeric_cols].std(axis=1)
        stat_features['max_all'] = df[numeric_cols].max(axis=1)
        stat_features['min_all'] = df[numeric_cols].min(axis=1)
        stat_features['range_all'] = stat_features['max_all'] - stat_features['min_all']
        
        # Ratios
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if (df[col2] != 0).all():
                    stat_features[f'{col1}_{col2}_ratio'] = df[col1] / df[col2]
    
    return stat_features

# ============================================================================
# Polymer-Specific Features
# ============================================================================

def extract_polymer_features(smiles_series):
    """Extract features specific to polymer structures"""
    features = pd.DataFrame()
    
    # Monomer patterns (simplified detection)
    features['repeating_patterns'] = smiles_series.apply(detect_repeating_patterns)
    features['polymer_backbone_length'] = smiles_series.apply(estimate_backbone_length)
    features['side_chain_complexity'] = smiles_series.apply(estimate_side_chain_complexity)
    
    # Molecular weight estimation (very rough)
    features['estimated_mw'] = smiles_series.apply(estimate_molecular_weight)
    
    # Flexibility indicators
    features['rotatable_bonds'] = smiles_series.str.count('-') - smiles_series.str.count('=')
    features['rigidity_score'] = (smiles_series.str.count('=') + 
                                  smiles_series.str.count('#') + 
                                  smiles_series.str.count('c')) / smiles_series.str.len()
    
    return features

def detect_repeating_patterns(smiles):
    """Detect potential repeating patterns in SMILES"""
    if pd.isna(smiles):
        return 0
    
    # Simple pattern detection (can be enhanced)
    patterns = []
    smiles_str = str(smiles)
    
    # Look for repeated subsequences
    for length in range(2, min(10, len(smiles_str) // 2)):
        for start in range(len(smiles_str) - length + 1):
            pattern = smiles_str[start:start + length]
            if smiles_str.count(pattern) > 1:
                patterns.append(pattern)
    
    return len(set(patterns))

def estimate_backbone_length(smiles):
    """Estimate polymer backbone length"""
    if pd.isna(smiles):
        return 0
    
    # Count carbon atoms as proxy for backbone length
    return str(smiles).count('C') + str(smiles).count('c')

def estimate_side_chain_complexity(smiles):
    """Estimate side chain complexity"""
    if pd.isna(smiles):
        return 0
    
    # Use branching points as proxy
    return str(smiles).count('(') + str(smiles).count('[')

def estimate_molecular_weight(smiles):
    """Rough molecular weight estimation"""
    if pd.isna(smiles):
        return 0
    
    # Atomic weights (simplified)
    weights = {
        'C': 12, 'c': 12, 'N': 14, 'n': 14, 'O': 16, 'o': 16,
        'S': 32, 's': 32, 'F': 19, 'P': 31, 'p': 31,
        'Cl': 35.5, 'Br': 80, 'I': 127
    }
    
    total_weight = 0
    smiles_str = str(smiles)
    
    for char in smiles_str:
        if char in weights:
            total_weight += weights[char]
    
    return total_weight

# ============================================================================
# Feature Engineering Pipeline
# ============================================================================

class PolymerFeatureEngineer:
    def __init__(self):
        self.feature_names = []
        self.scaler = None
        
    def fit_transform(self, df, smiles_col='polymer_smiles'):
        """Fit and transform features"""
        features = self._extract_all_features(df, smiles_col)
        self.feature_names = features.columns.tolist()
        return features
    
    def transform(self, df, smiles_col='polymer_smiles'):
        """Transform new data using fitted parameters"""
        features = self._extract_all_features(df, smiles_col)
        
        # Ensure same feature order
        missing_cols = set(self.feature_names) - set(features.columns)
        for col in missing_cols:
            features[col] = 0
            
        return features[self.feature_names]
    
    def _extract_all_features(self, df, smiles_col):
        """Extract all features"""
        all_features = pd.DataFrame(index=df.index)
        
        if smiles_col in df.columns:
            # Basic molecular features
            basic_features = extract_basic_features(df[smiles_col])
            all_features = pd.concat([all_features, basic_features], axis=1)
            
            # Advanced molecular features
            advanced_features = extract_advanced_molecular_features(df[smiles_col])
            all_features = pd.concat([all_features, advanced_features], axis=1)
            
            # Polymer-specific features
            polymer_features = extract_polymer_features(df[smiles_col])
            all_features = pd.concat([all_features, polymer_features], axis=1)
        
        # Statistical features from numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
            
        if len(numeric_cols) > 0:
            stat_features = create_statistical_features(df, numeric_cols)
            all_features = pd.concat([all_features, stat_features], axis=1)
        
        # Handle missing values
        all_features = all_features.fillna(0)
        
        # Remove infinite values
        all_features = all_features.replace([np.inf, -np.inf], 0)
        
        return all_features

# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """Example of how to use the feature engineering pipeline"""
    
    # Load data
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    
    # Initialize feature engineer
    fe = PolymerFeatureEngineer()
    
    # Fit and transform training data
    train_features = fe.fit_transform(train)
    print(f"Created {len(train_features.columns)} features for training data")
    
    # Transform test data
    test_features = fe.transform(test)
    print(f"Applied features to test data: {test_features.shape}")
    
    # Feature importance analysis (if target available)
    if 'target_col' in train.columns:  # Replace with actual target column
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectKBest, f_regression
        
        # Quick feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train_features, train['target_col'])
        
        feature_importance = pd.DataFrame({
            'feature': train_features.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
    
    return train_features, test_features

if __name__ == "__main__":
    example_usage()