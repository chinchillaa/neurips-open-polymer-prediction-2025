#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - ローカル実行用
高度なアンサンブルモデルによるポリマー特性予測

このスクリプトはKaggleノートブックをローカル環境で実行できるように変換したものです。
実験管理とローカルデータ読み込みに対応しています。
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ML関連ライブラリ
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')

# プロジェクトルートの設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "polymer_prediction_baseline"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"

# ディレクトリ作成
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 実験管理設定
EXPERIMENT_NAME = f"polymer_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EXPERIMENT_DIR = EXPERIMENTS_DIR / "experiments_results" / EXPERIMENT_NAME
EXPERIMENT_DIR.mkdir(exist_ok=True)

print(f"🚀 実験開始: {EXPERIMENT_NAME}")
print(f"📁 実験ディレクトリ: {EXPERIMENT_DIR}")

# RDKit関連のインポート（フォールバック対応）
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, MACCSkeys
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
    print("✅ RDKit利用可能 - 高精度分子特徴量を使用")
except ImportError:
    print("⚠️  RDKit利用不可 - 基本特徴量のみを使用")
    print("   pip install rdkit-pypi でインストール可能")

print("ライブラリインポート完了")

def load_local_data():
    """ローカルデータの読み込み"""
    print("\n📂 ローカルデータ読み込み中...")
    
    # データファイルパス
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    sample_submission_path = DATA_DIR / "sample_submission.csv"
    
    # ファイル存在確認
    if not train_path.exists():
        raise FileNotFoundError(f"訓練データが見つかりません: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"テストデータが見つかりません: {test_path}")
    if not sample_submission_path.exists():
        raise FileNotFoundError(f"サンプル提出ファイルが見つかりません: {sample_submission_path}")
    
    # データ読み込み
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(sample_submission_path)
    
    print(f"✅ 訓練データ: {train.shape}")
    print(f"✅ テストデータ: {test.shape}")
    print(f"✅ サンプル提出: {submission.shape}")
    
    return train, test, submission

def analyze_data(train, target_cols):
    """データ分析"""
    print("\n📊 データ分析実行中...")
    
    # 欠損値確認
    print("訓練データの欠損値:")
    missing_values = train.isnull().sum()
    print(missing_values)
    
    # 各特性の利用可能サンプル数計算
    available_samples = {col: train.shape[0] - missing_values[col] for col in target_cols}
    print("\n各特性の利用可能サンプル数:")
    for col, count in available_samples.items():
        print(f"{col}: {count} ({count/train.shape[0]*100:.2f}%)")
    
    # ターゲット特性の統計情報
    print("\nターゲット特性統計:")
    print(train[target_cols].describe())
    
    # 特性の範囲計算（wMAE計算に必要）
    property_ranges = {}
    for col in target_cols:
        property_ranges[col] = train[col].dropna().max() - train[col].dropna().min()
        
    print("\n特性の推定範囲:")
    for col, range_val in property_ranges.items():
        print(f"{col}: {range_val:.4f}")
    
    # wMAEメトリック用の重み計算
    weights = {}
    for col in target_cols:
        weights[col] = (1 / np.sqrt(available_samples[col])) / property_ranges[col]
    
    # 重みを正規化
    weight_sum = sum(weights.values())
    for col in target_cols:
        weights[col] = weights[col] / weight_sum * len(target_cols)
    
    print("\nwMAEメトリック用推定重み:")
    for col, weight in weights.items():
        print(f"{col}: {weight:.4f}")
    
    return available_samples, property_ranges, weights

def get_safe_polymer_features(mol):
    """RDKit使用時のポリマー固有特徴量抽出"""
    features = {}
    
    # 基本的なカウント
    features['num_atoms'] = mol.GetNumAtoms()
    features['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
    features['num_bonds'] = mol.GetNumBonds()
    
    # 原子タイプのカウント
    atom_types = {'C': 0, 'N': 0, 'O': 0, 'S': 0, 'F': 0, 'Cl': 0, 'Br': 0, 'I': 0}
    aromatic_atoms = 0
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_types:
            atom_types[symbol] += 1
        
        if atom.GetIsAromatic():
            aromatic_atoms += 1
    
    features['aromatic_atoms'] = aromatic_atoms
    
    # 原子タイプカウントを特徴量に追加
    for atom_type, count in atom_types.items():
        features[f'num_{atom_type}'] = count
        
        if mol.GetNumHeavyAtoms() > 0:
            features[f'ratio_{atom_type}'] = count / mol.GetNumHeavyAtoms()
        else:
            features[f'ratio_{atom_type}'] = 0
    
    # 結合タイプのカウント
    bond_types = {Chem.rdchem.BondType.SINGLE: 0, 
                  Chem.rdchem.BondType.DOUBLE: 0, 
                  Chem.rdchem.BondType.TRIPLE: 0,
                  Chem.rdchem.BondType.AROMATIC: 0}
    
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type in bond_types:
            bond_types[bond_type] += 1
    
    features['num_single_bonds'] = bond_types[Chem.rdchem.BondType.SINGLE]
    features['num_double_bonds'] = bond_types[Chem.rdchem.BondType.DOUBLE]
    features['num_triple_bonds'] = bond_types[Chem.rdchem.BondType.TRIPLE]
    features['num_aromatic_bonds'] = bond_types[Chem.rdchem.BondType.AROMATIC]
    
    # 信頼性の高い記述子の計算
    try:
        features['mw'] = Descriptors.MolWt(mol)
        features['logp'] = Descriptors.MolLogP(mol)
        features['tpsa'] = Descriptors.TPSA(mol)
        features['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        features['num_h_donors'] = Descriptors.NumHDonors(mol)
        features['num_h_acceptors'] = Descriptors.NumHAcceptors(mol)
        features['num_rings'] = Descriptors.RingCount(mol)
        features['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
        features['num_aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
    except:
        for desc in ['mw', 'logp', 'tpsa', 'num_rotatable_bonds', 'num_h_donors', 
                     'num_h_acceptors', 'num_rings', 'num_aromatic_rings', 'num_aliphatic_rings']:
            if desc not in features:
                features[desc] = 0
    
    # ポリマー関連のカスタム比率
    if mol.GetNumHeavyAtoms() > 0:
        features['rotatable_per_heavy'] = features['num_rotatable_bonds'] / mol.GetNumHeavyAtoms()
        features['rings_per_heavy'] = features.get('num_rings', 0) / mol.GetNumHeavyAtoms()
        features['aromatic_atom_ratio'] = features.get('aromatic_atoms', 0) / mol.GetNumHeavyAtoms()
    else:
        features['rotatable_per_heavy'] = 0
        features['rings_per_heavy'] = 0
        features['aromatic_atom_ratio'] = 0
    
    return features

def generate_basic_smiles_features(smiles_list):
    """RDKitなしでのSMILES基本特徴量生成"""
    print("RDKitなしでの基本SMILES特徴量生成中...")
    
    features = []
    feature_names = [
        'smiles_length', 'num_C', 'num_N', 'num_O', 'num_S', 'num_F', 'num_Cl', 'num_Br',
        'num_equals', 'num_hash', 'num_parens', 'num_brackets', 'num_rings_estimated',
        'aromatic_estimated', 'double_bonds_estimated', 'triple_bonds_estimated'
    ]
    
    for smiles in smiles_list:
        try:
            feat = {}
            feat['smiles_length'] = len(smiles)
            feat['num_C'] = smiles.count('C')
            feat['num_N'] = smiles.count('N')
            feat['num_O'] = smiles.count('O')
            feat['num_S'] = smiles.count('S')
            feat['num_F'] = smiles.count('F')
            feat['num_Cl'] = smiles.count('Cl')
            feat['num_Br'] = smiles.count('Br')
            feat['num_equals'] = smiles.count('=')
            feat['num_hash'] = smiles.count('#')
            feat['num_parens'] = smiles.count('(') + smiles.count(')')
            feat['num_brackets'] = smiles.count('[') + smiles.count(']')
            feat['num_rings_estimated'] = smiles.count('1') + smiles.count('2') + smiles.count('3')
            feat['aromatic_estimated'] = smiles.count('c') + smiles.count('n') + smiles.count('o')
            feat['double_bonds_estimated'] = smiles.count('=')
            feat['triple_bonds_estimated'] = smiles.count('#')
            
            features.append([feat[name] for name in feature_names])
        except:
            features.append([0] * len(feature_names))
    
    return pd.DataFrame(features, columns=feature_names), list(range(len(smiles_list)))

def generate_rdkit_features(smiles_list, calculator):
    """RDKit使用時の高精度分子特徴量生成"""
    print("RDKit使用での高精度分子特徴量生成中...")
    
    # サンプル分子で特徴量構造を作成
    sample_mol = Chem.MolFromSmiles('CC')
    sample_descriptors = list(calculator.CalcDescriptors(sample_mol))
    sample_polymer_features = get_safe_polymer_features(sample_mol)
    
    # フィンガープリント
    sample_morgan_fp = AllChem.GetMorganFingerprintAsBitVect(sample_mol, 2, nBits=256)
    sample_morgan_features = np.zeros((256,))
    AllChem.DataStructs.ConvertToNumpyArray(sample_morgan_fp, sample_morgan_features)
    
    sample_maccs_fp = MACCSkeys.GenMACCSKeys(sample_mol)
    sample_maccs_features = np.zeros((167,))
    AllChem.DataStructs.ConvertToNumpyArray(sample_maccs_fp, sample_maccs_features)
    
    # 特徴量名作成
    descriptor_names = [x[0] for x in Descriptors._descList]
    polymer_feature_names = list(sample_polymer_features.keys())
    morgan_feature_names = [f'morgan_{i}' for i in range(len(sample_morgan_features))]
    maccs_feature_names = [f'maccs_{i}' for i in range(len(sample_maccs_features))]
    
    feature_names = descriptor_names + polymer_feature_names + morgan_feature_names + maccs_feature_names
    
    features = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None and '*' in smiles:
                modified_smiles = smiles.replace('*', 'C')
                mol = Chem.MolFromSmiles(modified_smiles)
            
            if mol is None:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    except:
                        pass
            
            if mol is not None:
                # 記述子計算
                descriptors = list(calculator.CalcDescriptors(mol))
                polymer_features = list(get_safe_polymer_features(mol).values())
                
                # フィンガープリント計算
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
                    morgan_features = np.zeros((256,))
                    AllChem.DataStructs.ConvertToNumpyArray(morgan_fp, morgan_features)
                    
                    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                    maccs_features = np.zeros((167,))
                    AllChem.DataStructs.ConvertToNumpyArray(maccs_fp, maccs_features)
                
                all_features = descriptors + polymer_features + list(morgan_features) + list(maccs_features)
                
                if len(all_features) != len(feature_names):
                    if len(all_features) < len(feature_names):
                        all_features = all_features + [0] * (len(feature_names) - len(all_features))
                    else:
                        all_features = all_features[:len(feature_names)]
                
                features.append(all_features)
                valid_indices.append(i)
            else:
                if i < 5:
                    print(f"警告: SMILES解析不可: {smiles}")
        except Exception as e:
            if i < 5:
                print(f"SMILES {smiles} 処理エラー: {str(e)}")
    
    print(f"{len(smiles_list)} 中 {len(valid_indices)} のSMILES文字列を正常に処理")
    print(f"特徴量ベクトル長: {len(feature_names)}")
    
    return pd.DataFrame(features, index=valid_indices, columns=feature_names), valid_indices

def generate_molecule_features(smiles_list):
    """SMILES文字列から分子特徴量を生成"""
    if RDKIT_AVAILABLE:
        descriptor_names = [x[0] for x in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        return generate_rdkit_features(smiles_list, calculator)
    else:
        return generate_basic_smiles_features(smiles_list)

def clean_features(df, feature_cols):
    """特徴量クリーニング"""
    df_clean = df.copy()
    
    # 無限値をNaNで置換
    df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # 有効な列を特定
    valid_cols = []
    for col in feature_cols:
        if df_clean[col].notna().sum() > 0 and df_clean[col].nunique() > 1:
            valid_cols.append(col)
    
    print(f"無効列除去後: {len(valid_cols)} / {len(feature_cols)} 特徴量を保持")
    
    # 外れ値処理
    for col in valid_cols:
        if df_clean[col].dtype.kind in 'fc':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            if std > 0:
                lower_bound = mean - 5 * std
                upper_bound = mean + 5 * std
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean, valid_cols

def select_features_for_target(df, target_col, all_features, max_features=500):
    """ターゲット別特徴量選択"""
    df_valid = df[df[target_col].notna()].copy()
    
    if len(df_valid) < 50:
        return all_features[:min(len(all_features), max_features)]
    
    correlations = []
    for col in all_features:
        if df_valid[col].dtype.kind in 'fc':
            corr = df_valid[col].corr(df_valid[target_col])
            if not pd.isna(corr):
                correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [x[0] for x in correlations[:max_features]]
    
    print(f"{target_col} 用に相関ベースで {len(top_features)} 特徴量を選択")
    return top_features

def get_property_hyperparams(property_name):
    """各特性の最適化済みハイパーパラメータ"""
    params = {
        'Tg': {
            'xgb': {'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 6,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
                   'reg_alpha': 0.01, 'reg_lambda': 1.0},
            'cat': {'iterations': 1000, 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 3}
        },
        'FFV': {
            'xgb': {'n_estimators': 2000, 'learning_rate': 0.005, 'max_depth': 8,
                   'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 2,
                   'reg_alpha': 0.1, 'reg_lambda': 0.5},
            'cat': {'iterations': 1500, 'learning_rate': 0.02, 'depth': 7, 'l2_leaf_reg': 2}
        },
        'Tc': {
            'xgb': {'n_estimators': 1500, 'learning_rate': 0.01, 'max_depth': 7,
                   'subsample': 0.85, 'colsample_bytree': 0.75, 'min_child_weight': 3,
                   'reg_alpha': 0.05, 'reg_lambda': 1.0},
            'cat': {'iterations': 1200, 'learning_rate': 0.02, 'depth': 5, 'l2_leaf_reg': 4}
        },
        'Density': {
            'xgb': {'n_estimators': 1200, 'learning_rate': 0.01, 'max_depth': 6,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 2,
                   'reg_alpha': 0.1, 'reg_lambda': 0.5},
            'cat': {'iterations': 1000, 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 3}
        },
        'Rg': {
            'xgb': {'n_estimators': 1000, 'learning_rate': 0.02, 'max_depth': 7,
                   'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 3,
                   'reg_alpha': 0.05, 'reg_lambda': 1.0},
            'cat': {'iterations': 1200, 'learning_rate': 0.02, 'depth': 7, 'l2_leaf_reg': 3}
        }
    }
    return params.get(property_name, params['Tg'])

class AveragingEnsemble:
    """アンサンブル予測用クラス"""
    def __init__(self, models, imputer, scaler):
        self.models = models
        self.imputer = imputer
        self.scaler = scaler
        
    def predict(self, X):
        try:
            X = X.copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            X_imputed = self.imputer.transform(X)
            X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.transform(X_imputed)
            
            preds = np.column_stack([model.predict(X_scaled) for model in self.models])
            return np.mean(preds, axis=1)
        except Exception as e:
            print(f"予測エラー: {str(e)}")
            return np.zeros(X.shape[0])

class MeanPredictor:
    """フォールバック用平均予測器"""
    def __init__(self, value):
        self.value = value
        
    def predict(self, X):
        return np.full(len(X), self.value)

def train_advanced_model(df, target_col, feature_cols, n_splits=5, seed=42):
    """高度なアンサンブルモデル訓練"""
    valid_idx = df[target_col].notna()
    df_valid = df.loc[valid_idx]
    
    if len(df_valid) < 30:
        print(f"{target_col} のデータ不足、平均値予測を使用")
        mean_val = df_valid[target_col].mean() if len(df_valid) > 0 else 0
        return {'model': MeanPredictor(mean_val), 'cv_score': 0.0}
    
    X = df_valid[feature_cols].copy()
    y = df_valid[target_col].values
    
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    
    scaler = PowerTransformer(method='yeo-johnson')
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"{target_col} 用の高度なモデルを {len(y)} サンプルで訓練")
    
    n_splits = min(n_splits, len(y) // 10 or 2)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    hyperparams = get_property_hyperparams(target_col)
    
    all_models = []
    oof_preds = np.zeros(len(y))
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        fold_models = []
        fold_preds = []
        
        # XGBoost
        try:
            xgb_model = xgb.XGBRegressor(**hyperparams['xgb'], random_state=seed+fold)
            xgb_model.fit(X_train, y_train)
            xgb_preds = xgb_model.predict(X_val)
            fold_preds.append(xgb_preds)
            fold_models.append(xgb_model)
            print(f"  フォールド {fold+1} XGB MAE: {mean_absolute_error(y_val, xgb_preds):.6f}")
        except Exception as e:
            print(f"  XGBモデル失敗: {str(e)}")
        
        # CatBoost
        try:
            cat_model = CatBoostRegressor(**hyperparams['cat'], random_seed=seed+fold, verbose=False)
            cat_model.fit(X_train, y_train)
            cat_preds = cat_model.predict(X_val)
            fold_preds.append(cat_preds)
            fold_models.append(cat_model)
            print(f"  フォールド {fold+1} CatBoost MAE: {mean_absolute_error(y_val, cat_preds):.6f}")
        except Exception as e:
            print(f"  CatBoostモデル失敗: {str(e)}")
        
        # Random Forest
        try:
            rf_model = RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, random_state=seed+fold, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            rf_preds = rf_model.predict(X_val)
            fold_preds.append(rf_preds)
            fold_models.append(rf_model)
            print(f"  フォールド {fold+1} RF MAE: {mean_absolute_error(y_val, rf_preds):.6f}")
        except Exception as e:
            print(f"  RFモデル失敗: {str(e)}")
        
        # Gradient Boosting
        try:
            gb_model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                subsample=0.8, random_state=seed+fold
            )
            gb_model.fit(X_train, y_train)
            gb_preds = gb_model.predict(X_val)
            fold_preds.append(gb_preds)
            fold_models.append(gb_model)
            print(f"  フォールド {fold+1} GB MAE: {mean_absolute_error(y_val, gb_preds):.6f}")
        except Exception as e:
            print(f"  GBモデル失敗: {str(e)}")
        
        # KNN（小データセット用）
        if len(y_train) < 1000:
            try:
                knn_model = KNeighborsRegressor(n_neighbors=7)
                knn_model.fit(X_train, y_train)
                knn_preds = knn_model.predict(X_val)
                fold_preds.append(knn_preds)
                fold_models.append(knn_model)
                print(f"  フォールド {fold+1} KNN MAE: {mean_absolute_error(y_val, knn_preds):.6f}")
            except Exception as e:
                print(f"  KNNモデル失敗: {str(e)}")
        
        all_models.extend(fold_models)
        
        if fold_preds:
            ensemble_preds = np.mean(np.column_stack(fold_preds), axis=1)
            oof_preds[val_idx] = ensemble_preds
            fold_score = mean_absolute_error(y_val, ensemble_preds)
            cv_scores.append(fold_score)
            print(f"  フォールド {fold+1} アンサンブル MAE: {fold_score:.6f}")
    
    if not all_models:
        print("  全モデル失敗。平均値予測を使用。")
        mean_val = np.mean(y)
        return {'model': MeanPredictor(mean_val), 'cv_score': 0.0}
    
    if cv_scores:
        cv_score = mean_absolute_error(y[~np.isnan(oof_preds)], oof_preds[~np.isnan(oof_preds)])
        print(f"{target_col} のクロスバリデーション MAE: {cv_score:.6f}")
    else:
        cv_score = 0.0
    
    final_model = AveragingEnsemble(all_models, imputer, scaler)
    
    return {'model': final_model, 'cv_score': cv_score}

def save_experiment_results(experiment_dir, models, submission, weights, property_ranges, elapsed_time):
    """実験結果の保存"""
    print(f"\n💾 実験結果を保存中: {experiment_dir}")
    
    # モデル情報
    model_info = {}
    for col, model_data in models.items():
        model_info[col] = {
            'cv_score': model_data['cv_score'],
            'model_type': type(model_data['model']).__name__
        }
    
    # 実験メタデータ
    metadata = {
        'experiment_name': experiment_dir.name,
        'timestamp': datetime.now().isoformat(),
        'rdkit_available': RDKIT_AVAILABLE,
        'model_info': model_info,
        'weights': weights,
        'property_ranges': property_ranges,
        'elapsed_time_minutes': elapsed_time / 60
    }
    
    # 保存
    import json
    with open(experiment_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    submission.to_csv(experiment_dir / 'submission.csv', index=False)
    
    print(f"✅ 実験結果保存完了")

def main():
    """メイン実行関数"""
    start_time = time.time()
    
    print(f"高度なポリマー特性予測パイプライン開始...")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 再現性のためのランダムシード設定
    SEED = 42
    np.random.seed(SEED)
    
    # データ読み込み
    train, test, submission = load_local_data()
    
    # データ分析
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    available_samples, property_ranges, weights = analyze_data(train, target_cols)
    
    # 分子特徴量生成
    print("\n🧬 分子特徴量生成中...")
    train_features, train_valid_idx = generate_molecule_features(train['SMILES'])
    test_features, test_valid_idx = generate_molecule_features(test['SMILES'])
    
    # 特徴量結合
    train_with_features = pd.concat([
        train.iloc[train_valid_idx].reset_index(drop=True),
        train_features.reset_index(drop=True)
    ], axis=1)
    
    test_with_features = pd.concat([
        test.iloc[test_valid_idx].reset_index(drop=True),
        test_features.reset_index(drop=True)
    ], axis=1)
    
    print(f"特徴量付き訓練データ形状: {train_with_features.shape}")
    print(f"特徴量付きテストデータ形状: {test_with_features.shape}")
    
    # 特徴量クリーニング
    all_feature_cols = train_features.columns.tolist()
    train_with_features, valid_feature_cols = clean_features(train_with_features, all_feature_cols)
    test_with_features, _ = clean_features(test_with_features, all_feature_cols)
    
    # ターゲット別特徴量選択
    target_features = {}
    for col in target_cols:
        target_features[col] = select_features_for_target(train_with_features, col, valid_feature_cols)
    
    # モデル訓練
    print("\n🤖 モデル訓練開始...")
    models = {}
    for col in target_cols:
        print(f"\n{col} 用の高度なモデル訓練中...")
        models[col] = train_advanced_model(
            train_with_features, col, target_features[col], seed=SEED
        )
    
    print("\n✅ 全モデル訓練完了")
    
    # 予測生成
    print("\n🔮 テスト予測生成中...")
    test_preds = {}
    fallbacks = {'Tg': 400, 'FFV': 0.2, 'Tc': 0.2, 'Density': 1.0, 'Rg': 10.0}
    
    for col in target_cols:
        print(f"{col} の予測生成中...")
        try:
            test_preds[col] = models[col]['model'].predict(test_with_features[target_features[col]])
        except Exception as e:
            print(f"{col} の予測生成エラー: {str(e)}")
            test_preds[col] = np.full(len(test_with_features), fallbacks[col])
    
    # 提出ファイル作成
    submission_df = pd.DataFrame({'id': test_with_features['id']})
    for col in target_cols:
        submission_df[col] = test_preds[col]
    
    print("\n📋 提出ファイルプレビュー:")
    print(submission_df.head())
    
    # 欠損行の補完
    if len(submission_df) < len(test):
        print(f"警告: 提出ファイルに {len(submission_df)} 行、テストセットに {len(test)} 行")
        missing_ids = set(test['id']) - set(submission_df['id'])
        print(f"{len(missing_ids)} 不足行を追加")
        
        for missing_id in missing_ids:
            row = {'id': missing_id}
            row.update(fallbacks)
            submission_df = pd.concat([submission_df, pd.DataFrame([row])], ignore_index=True)
    
    # CVベースwMAE計算
    weighted_scores = []
    for col in target_cols:
        if 'cv_score' in models[col] and models[col]['cv_score'] > 0:
            weighted_score = models[col]['cv_score'] * weights[col] / property_ranges[col]
            weighted_scores.append(weighted_score)
            print(f"{col} の重み付きスコア: {weighted_score:.6f}")
    
    if weighted_scores:
        estimated_wmae = sum(weighted_scores)
        print(f"\n🎯 推定重み付きMAE: {estimated_wmae:.6f}")
    else:
        print("\n⚠️  重み付きMAE推定不可（CVスコア利用不可）")
    
    # 実行時間
    elapsed_time = time.time() - start_time
    print(f"⏱️  総実行時間: {elapsed_time/60:.2f} 分")
    
    # 実験結果保存
    save_experiment_results(EXPERIMENT_DIR, models, submission_df, weights, property_ranges, elapsed_time)
    
    print("\n🎉 高度なポリマー特性予測パイプライン完了")
    print(f"📁 実験結果: {EXPERIMENT_DIR}")

if __name__ == "__main__":
    main()