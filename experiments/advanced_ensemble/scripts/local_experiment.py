#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - WandB統合ローカル実行版
高度なアンサンブルモデル + WandB実験管理によるポリマー特性予測

このスクリプトはKaggleノートブックをローカル環境で実行できるように変換し、
WandBによる実験追跡機能を統合したものです。
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
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
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # 4つ上がプロジェクトルート
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXPERIMENTS_DIR = Path(__file__).parent.parent  # advanced_ensemble ディレクトリ
MODELS_DIR = EXPERIMENTS_DIR / "results" / "models"

# ディレクトリ作成
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 実験管理設定
EXPERIMENT_NAME = f"advanced_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EXPERIMENT_DIR = EXPERIMENTS_DIR / "results" / "runs" / EXPERIMENT_NAME
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🚀 実験開始: {EXPERIMENT_NAME}")
print(f"📁 実験ディレクトリ: {EXPERIMENT_DIR}")

# 設定
SEED = 42
np.random.seed(SEED)

# WandB設定
WANDB_PROJECT = "neurips-polymer-prediction-2025"
WANDB_ENTITY = None  # チーム名があれば設定

# RDKit利用可能性チェック
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit import DataStructs
    rdkit_available = True
    print("✅ RDKit利用可能 - 高精度分子特徴量を使用")
except ImportError:
    rdkit_available = False
    print("⚠️  RDKit利用不可 - 基本SMILES特徴量を使用")

def init_wandb(offline_mode=False):
    """WandB初期化"""
    try:
        mode = "offline" if offline_mode else "online"
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=EXPERIMENT_NAME,
            mode=mode,
            config={
                "experiment_name": EXPERIMENT_NAME,
                "seed": SEED,
                "rdkit_available": rdkit_available,
                "model_types": ["xgboost", "catboost", "random_forest", "gradient_boosting", "knn"],
                "cv_folds": 5,
                "max_features": 500,
                "hyperparameters": {
                    "xgboost": {
                        "n_estimators": 200,
                        "max_depth": 8,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8
                    },
                    "catboost": {
                        "iterations": 200,
                        "depth": 7,
                        "learning_rate": 0.08
                    },
                    "random_forest": {
                        "n_estimators": 300,
                        "max_depth": 15
                    },
                    "gradient_boosting": {
                        "n_estimators": 200,
                        "max_depth": 8,
                        "learning_rate": 0.1
                    },
                    "knn": {
                        "n_neighbors": 10
                    }
                }
            }
        )
        print(f"✅ WandB初期化成功（{mode}モード）")
        return True, run
    except Exception as e:
        print(f"⚠️  WandB初期化失敗: {e}")
        print("📝 WandBなしで実験継続")
        return False, None

def load_local_data():
    """ローカルデータの読み込み"""
    print("📂 ローカルデータ読み込み中...")
    print(f"📁 データディレクトリ: {DATA_DIR}")
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    sample_submission_path = DATA_DIR / "sample_submission.csv"
    
    print(f"🔍 訓練データ存在確認: {train_path} -> {train_path.exists()}")
    print(f"🔍 テストデータ存在確認: {test_path} -> {test_path.exists()}")
    print(f"🔍 サンプル提出存在確認: {sample_submission_path} -> {sample_submission_path.exists()}")
    
    if not all([train_path.exists(), test_path.exists(), sample_submission_path.exists()]):
        raise FileNotFoundError("必要なデータファイルが見つかりません")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(sample_submission_path)
    
    print(f"✅ 訓練データ形状: {train.shape}")
    print(f"✅ テストデータ形状: {test.shape}")
    
    return train, test, submission

def basic_smiles_features(smiles):
    """基本的なSMILES特徴量（RDKit不使用時）"""
    if pd.isna(smiles) or smiles == '':
        return [0] * 16
    
    features = [
        len(smiles),                    # SMILES文字列長
        smiles.count('C'),             # 炭素数
        smiles.count('N'),             # 窒素数
        smiles.count('O'),             # 酸素数
        smiles.count('S'),             # 硫黄数
        smiles.count('P'),             # リン数
        smiles.count('F'),             # フッ素数
        smiles.count('Cl'),            # 塩素数
        smiles.count('='),             # 二重結合数
        smiles.count('#'),             # 三重結合数
        smiles.count('('),             # 分岐数
        smiles.count('['),             # 特殊原子数
        smiles.count('@'),             # キラル中心数
        smiles.count('c'),             # 芳香族炭素数
        smiles.count(':'),             # 芳香族結合数
        smiles.count('-'),             # 単結合数
    ]
    return features

def rdkit_molecular_features(smiles):
    """RDKitを使用した分子特徴量"""
    if pd.isna(smiles) or smiles == '':
        return [0] * 100  # デフォルト特徴量数
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 100
    
    features = []
    
    # 基本記述子（30個）
    basic_descriptors = [
        Descriptors.MolWt,              # 分子量
        Descriptors.NumHDonors,         # 水素結合ドナー数
        Descriptors.NumHAcceptors,      # 水素結合アクセプター数
        Descriptors.TPSA,               # トポロジカル極性表面積
        Descriptors.MolLogP,            # 分配係数
        Descriptors.NumRotatableBonds,  # 回転可能結合数
        Descriptors.NumAromaticRings,   # 芳香環数
        Descriptors.NumSaturatedRings,  # 飽和環数
        Descriptors.NumAliphaticRings,  # 脂肪族環数
        Descriptors.RingCount,          # 環数
        Descriptors.NumHeteroatoms,     # ヘテロ原子数
        Descriptors.FractionCSP3,       # sp3炭素の割合
        Descriptors.BalabanJ,           # Balaban J指数
        Descriptors.BertzCT,            # Bertz分子複雑度
        Descriptors.Chi0,               # 分子連結性指数 0次
        Descriptors.Chi1,               # 分子連結性指数 1次
        Descriptors.Chi0n,              # 正規化分子連結性指数 0次
        Descriptors.Chi1n,              # 正規化分子連結性指数 1次
        Descriptors.HallKierAlpha,      # Hall-Kier α
        Descriptors.Kappa1,             # Kappa形状指数 1
        Descriptors.Kappa2,             # Kappa形状指数 2
        Descriptors.Kappa3,             # Kappa形状指数 3
        Descriptors.LabuteASA,          # Labute接触面積
        Descriptors.PEOE_VSA1,          # 部分電荷加重表面積 1
        Descriptors.SMR_VSA1,           # SMR 加重表面積 1
        Descriptors.SlogP_VSA1,         # SlogP 加重表面積 1
        Descriptors.EState_VSA1,        # EState 加重表面積 1
        Descriptors.VSA_EState1,        # VSA EState 1
        Descriptors.Ipc,                # 情報含有量
        Descriptors.BertzCT            # 再度Bertz複雑度
    ]
    
    for desc_func in basic_descriptors:
        try:
            features.append(desc_func(mol))
        except:
            features.append(0)
    
    # Morganフィンガープリント（70個のビット）
    try:
        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=70)
        features.extend(list(morgan_fp))
    except:
        features.extend([0] * 70)
    
    return features[:100]  # 100個の特徴量に制限

def calculate_weighted_mae(y_true_df, y_pred_df, target_columns, train_df):
    """
    Calculate weighted MAE (wMAE) score according to the competition formula

    Formula: wMAE = Σ(w_i × MAE_i)
    where w_i = (1/r_i) × (K × √(1/n_i)) / Σ(√(1/n_j))

    Args:
        y_true_df: DataFrame with true values
        y_pred_df: DataFrame with predicted values  
        target_columns: List of target column names
        train_df: Training DataFrame to calculate ranges and sample counts

    Returns:
        wMAE score and individual weights
    """
    # Calculate sample counts and ranges from training data
    K = len(target_columns)  # Total number of tasks
    sample_counts = {}
    value_ranges = {}

    for target in target_columns:
        if target in train_df.columns:
            valid_data = train_df[target].dropna()
            sample_counts[target] = len(valid_data)
            value_ranges[target] = valid_data.max() - valid_data.min()
        else:
            sample_counts[target] = 1  # Default if not available
            value_ranges[target] = 1.0  # Default if not available

    # Calculate sqrt(1/n_i) for each target
    sqrt_inv_n = {}
    for target in target_columns:
        sqrt_inv_n[target] = np.sqrt(1.0 / sample_counts[target])

    # Calculate sum of sqrt(1/n_j) for all j
    sum_sqrt_inv_n = sum(sqrt_inv_n.values())

    # Calculate weights w_i = (1/r_i) × (K × √(1/n_i)) / Σ(√(1/n_j))
    weights = {}
    for target in target_columns:
        weights[target] = (1.0 / value_ranges[target]) * (K * sqrt_inv_n[target]) / sum_sqrt_inv_n

    # Calculate MAE for each target and weighted sum
    individual_maes = {}
    weighted_sum = 0.0

    for target in target_columns:
        if target in y_true_df.columns and target in y_pred_df.columns:
            # Get valid (non-null) predictions and true values
            valid_mask = y_true_df[target].notna() & y_pred_df[target].notna()
            if valid_mask.sum() > 0:
                y_true_valid = y_true_df[target][valid_mask]
                y_pred_valid = y_pred_df[target][valid_mask]
                mae = mean_absolute_error(y_true_valid, y_pred_valid)
                individual_maes[target] = mae
                weighted_sum += weights[target] * mae
            else:
                individual_maes[target] = 0.0
        else:
            individual_maes[target] = 0.0

    return weighted_sum, weights, individual_maes, sample_counts, value_ranges

def feature_engineering(df, wandb_available=False):
    """特徴量エンジニアリング"""
    print("🧬 分子特徴量生成中...")
    
    if rdkit_available:
        print("  RDKitベース分子記述子とフィンガープリントを使用")
        features_list = []
        for i, smiles in enumerate(df['SMILES']):
            if i % 1000 == 0:
                print(f"  進捗: {i}/{len(df)}")
            features = rdkit_molecular_features(smiles)
            features_list.append(features)
        
        feature_names = [f'rdkit_feature_{i}' for i in range(100)]
        
        if wandb_available:
            wandb.log({"feature_engineering/method": "rdkit", "feature_engineering/feature_count": 100})
    else:
        print("  基本SMILES特徴量を使用")
        features_list = []
        for smiles in df['SMILES']:
            features = basic_smiles_features(smiles)
            features_list.append(features)
        
        feature_names = [
            'smiles_length', 'carbon_count', 'nitrogen_count', 'oxygen_count', 'sulfur_count',
            'phosphorus_count', 'fluorine_count', 'chlorine_count', 'double_bond_count', 
            'triple_bond_count', 'branch_count', 'special_atom_count', 'chiral_count',
            'aromatic_carbon_count', 'aromatic_bond_count', 'single_bond_count'
        ]
        
        if wandb_available:
            wandb.log({"feature_engineering/method": "basic_smiles", "feature_engineering/feature_count": 16})
    
    features_df = pd.DataFrame(features_list, columns=feature_names)
    
    print(f"✅ 特徴量生成完了: {features_df.shape[1]}個の特徴量")
    return features_df

def train_models_for_target(X, y, target_name, wandb_available=False, n_splits=5):
    """特定の特性に対するモデル訓練とアンサンブル"""
    print(f"🤖 {target_name}用の高度なモデル訓練中...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    models_performance = {}
    
    # モデル定義（ハイパーパラメータ調整版）
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=200, depth=7, learning_rate=0.08,
            random_seed=SEED, verbose=False,
            train_dir=str(EXPERIMENT_DIR / "catboost_info")
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=15, random_state=SEED, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1, random_state=SEED
        ),
        'KNN': KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
    }
    
    cv_results = {}
    fold_predictions = {model_name: [] for model_name in models.keys()}
    fold_true_values = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # データ前処理
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_fold), 
            columns=X_train_fold.columns, 
            index=X_train_fold.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val_fold), 
            columns=X_val_fold.columns, 
            index=X_val_fold.index
        )
        
        fold_true_values.append(y_val_fold)
        
        for model_name, model in models.items():
            # モデル訓練
            model.fit(X_train_scaled, y_train_fold)
            y_pred = model.predict(X_val_scaled)
            fold_predictions[model_name].append(y_pred)
            
            mae = mean_absolute_error(y_val_fold, y_pred)
            
            if model_name not in cv_results:
                cv_results[model_name] = {
                    'cv_scores': [],
                    'predictions': []
                }
            
            cv_results[model_name]['cv_scores'].append(mae)
            cv_results[model_name]['predictions'].append(y_pred)
            
            print(f"    フォールド {fold+1} {model_name} MAE: {mae:.6f}")
            
            if wandb_available:
                wandb.log({
                    f"{target_name}/{model_name}/fold_{fold+1}_mae": mae,
                    f"{target_name}/{model_name}/fold": fold+1
                })
    
    # 各モデルの平均性能を計算
    for model_name in models.keys():
        avg_mae = np.mean(cv_results[model_name]['cv_scores'])
        std_mae = np.std(cv_results[model_name]['cv_scores'])
        cv_results[model_name]['cv_mae'] = float(avg_mae)
        cv_results[model_name]['cv_std'] = float(std_mae)
        cv_results[model_name]['cv_scores'] = [float(score) for score in cv_results[model_name]['cv_scores']]
        
        print(f"    {model_name} 平均 CV MAE: {avg_mae:.6f} (±{std_mae:.6f})")
        
        if wandb_available:
            wandb.log({
                f"{target_name}/{model_name}/cv_mae": avg_mae,
                f"{target_name}/{model_name}/cv_std": std_mae
            })
    
    # アンサンブル予測の計算
    print(f"\n  🎯 {target_name}のアンサンブル予測を計算中...")
    
    # 1. 単純平均アンサンブル
    ensemble_predictions_simple = []
    for fold in range(n_splits):
        fold_ensemble = np.mean([fold_predictions[model][fold] for model in models.keys()], axis=0)
        ensemble_predictions_simple.append(fold_ensemble)
    
    # 単純平均アンサンブルのMAE計算
    simple_ensemble_maes = []
    for fold in range(n_splits):
        mae = mean_absolute_error(fold_true_values[fold], ensemble_predictions_simple[fold])
        simple_ensemble_maes.append(mae)
    
    simple_ensemble_avg_mae = np.mean(simple_ensemble_maes)
    simple_ensemble_std_mae = np.std(simple_ensemble_maes)
    
    print(f"    単純平均アンサンブル MAE: {simple_ensemble_avg_mae:.6f} (±{simple_ensemble_std_mae:.6f})")
    
    # 2. 加重平均アンサンブル（性能ベースの重み）
    # 各モデルの重みを性能（MAE）の逆数に基づいて計算
    model_weights = {}
    total_inv_mae = sum(1.0 / cv_results[model]['cv_mae'] for model in models.keys())
    for model in models.keys():
        model_weights[model] = (1.0 / cv_results[model]['cv_mae']) / total_inv_mae
    
    print(f"\n  📊 最適化された重み:")
    for model, weight in model_weights.items():
        print(f"    {model}: {weight:.4f}")
    
    # 加重平均アンサンブル予測
    ensemble_predictions_weighted = []
    for fold in range(n_splits):
        fold_ensemble = sum(
            fold_predictions[model][fold] * model_weights[model] 
            for model in models.keys()
        )
        ensemble_predictions_weighted.append(fold_ensemble)
    
    # 加重平均アンサンブルのMAE計算
    weighted_ensemble_maes = []
    for fold in range(n_splits):
        mae = mean_absolute_error(fold_true_values[fold], ensemble_predictions_weighted[fold])
        weighted_ensemble_maes.append(mae)
    
    weighted_ensemble_avg_mae = np.mean(weighted_ensemble_maes)
    weighted_ensemble_std_mae = np.std(weighted_ensemble_maes)
    
    print(f"    加重平均アンサンブル MAE: {weighted_ensemble_avg_mae:.6f} (±{weighted_ensemble_std_mae:.6f})")
    
    # アンサンブル結果をcv_resultsに追加
    cv_results['SimpleEnsemble'] = {
        'cv_mae': float(simple_ensemble_avg_mae),
        'cv_std': float(simple_ensemble_std_mae),
        'cv_scores': [float(score) for score in simple_ensemble_maes]
    }
    
    cv_results['WeightedEnsemble'] = {
        'cv_mae': float(weighted_ensemble_avg_mae),
        'cv_std': float(weighted_ensemble_std_mae),
        'cv_scores': [float(score) for score in weighted_ensemble_maes],
        'weights': {model: float(weight) for model, weight in model_weights.items()}
    }
    
    if wandb_available:
        wandb.log({
            f"{target_name}/SimpleEnsemble/cv_mae": simple_ensemble_avg_mae,
            f"{target_name}/SimpleEnsemble/cv_std": simple_ensemble_std_mae,
            f"{target_name}/WeightedEnsemble/cv_mae": weighted_ensemble_avg_mae,
            f"{target_name}/WeightedEnsemble/cv_std": weighted_ensemble_std_mae,
            f"{target_name}/WeightedEnsemble/weights": model_weights
        })
    
    # 最良のモデルを選択（個別モデルとアンサンブルを含む）
    all_models = list(cv_results.keys())
    best_model = min(all_models, key=lambda m: cv_results[m]['cv_mae'])
    print(f"\n  🏆 {target_name}の最良モデル: {best_model} (MAE: {cv_results[best_model]['cv_mae']:.6f})")
    
    # 全データで最終モデルを訓練
    print(f"\n  💾 {target_name}の最終モデルを訓練中...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_scaled, y)
        trained_models[model_name] = model
    
    # スケーラーと訓練済みモデルを返す
    return cv_results, trained_models, scaler, model_weights

def main_experiment(offline_wandb=True):
    """メイン実験関数"""
    start_time = time.time()
    
    # WandB初期化
    wandb_available, wandb_run = init_wandb(offline_mode=offline_wandb)
    
    # データ読み込み
    train, test, submission = load_local_data()
    
    if wandb_available:
        wandb.log({
            "data/train_size": len(train),
            "data/test_size": len(test),
            "data/train_columns": list(train.columns),
            "data/test_columns": list(test.columns)
        })
    
    # 特徴量エンジニアリング
    train_features = feature_engineering(train, wandb_available)
    test_features = feature_engineering(test, wandb_available)
    
    # 特性列の特定
    target_columns = [col for col in train.columns if col not in ['SMILES', 'Id']]
    print(f"🎯 対象特性: {target_columns}")
    
    if wandb_available:
        wandb.log({"experiment/target_properties": target_columns})
    
    # 各特性に対してモデル訓練（サンプル実行のため制限）
    results = {}
    # すべての特性を処理（idを除く）
    actual_targets = [col for col in target_columns if col != 'id']
    
    for target_col in actual_targets:
        if target_col in train.columns:
            # 欠損値除去
            valid_mask = ~train[target_col].isna()
            if valid_mask.sum() < 10:
                print(f"⚠️  {target_col}: データ不足（{valid_mask.sum()}件）- スキップ")
                continue
            
            X_valid = train_features[valid_mask]
            y_valid = train[target_col][valid_mask]
            
            print(f"\n📊 {target_col} - 有効データ: {len(X_valid)}件")
            
            # モデル訓練とアンサンブル
            cv_results, trained_models, scaler, ensemble_weights = train_models_for_target(
                X_valid, y_valid, target_col, wandb_available, n_splits=3  # 高速化のため3-fold
            )
            results[target_col] = {
                'cv_results': cv_results,
                'trained_models': trained_models,
                'scaler': scaler,
                'ensemble_weights': ensemble_weights
            }
    
    # 実験結果保存
    elapsed_time = time.time() - start_time
    
    experiment_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "rdkit_available": rdkit_available,
        "elapsed_time": elapsed_time,
        "results": {
            target: {
                model: {
                    'cv_mae': data['cv_results'][model]['cv_mae'],
                    'cv_std': data['cv_results'][model]['cv_std'],
                    'cv_scores': data['cv_results'][model]['cv_scores']
                } if model not in ['WeightedEnsemble'] else {
                    'cv_mae': data['cv_results'][model]['cv_mae'],
                    'cv_std': data['cv_results'][model]['cv_std'],
                    'cv_scores': data['cv_results'][model]['cv_scores'],
                    'weights': data['cv_results'][model]['weights']
                }
                for model in data['cv_results'] if 'predictions' not in data['cv_results'][model]
            }
            for target, data in results.items()
        },
        "ensemble_info": {
            target: {
                "best_model": min(data['cv_results'].keys(), key=lambda m: data['cv_results'][m]['cv_mae']),
                "weighted_ensemble_weights": data['ensemble_weights']
            } for target, data in results.items()
        },
        "hyperparameters": {
            "xgboost": {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            "catboost": {
                "iterations": 200,
                "depth": 7,
                "learning_rate": 0.08
            },
            "random_forest": {
                "n_estimators": 300,
                "max_depth": 15
            },
            "gradient_boosting": {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.1
            },
            "knn": {
                "n_neighbors": 10
            }
        }
    }
    
    metadata_path = EXPERIMENT_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 実験結果サマリー:")
    for target, target_data in results.items():
        print(f"  {target}:")
        for model, performance in target_data['cv_results'].items():
            if model == 'WeightedEnsemble' and 'weights' in performance:
                print(f"    {model}: {performance['cv_mae']:.6f} (±{performance['cv_std']:.6f})")
                print(f"      重み: {performance['weights']}")
            else:
                print(f"    {model}: {performance['cv_mae']:.6f} (±{performance['cv_std']:.6f})")
    
    # wMAE計算のための準備
    print(f"\n📊 wMAE（重み付き平均絶対誤差）計算:")
    
    # 各モデルの予測値を格納するDataFrame（CVのMAEを使った推定）
    # アンサンブルを含むすべてのモデル名を取得
    model_names = list(next(iter(results.values()))['cv_results'].keys())
    
    for model_name in model_names:
        print(f"\n  {model_name}モデルのwMAE計算:")
        
        # CVのMAEを使って疑似的なDataFrameを作成
        y_true_dict = {}
        y_pred_dict = {}
        
        for target in actual_targets:
            if target in results and model_name in results[target]['cv_results']:
                # CVのMAEを持つダミーデータを作成（実際の予測ではなく推定）
                cv_mae = results[target]['cv_results'][model_name]['cv_mae']
                y_true_dict[target] = [0.0]  # ダミー値
                y_pred_dict[target] = [cv_mae]  # MAEを予測誤差として使用
        
        y_true_df = pd.DataFrame(y_true_dict)
        y_pred_df = pd.DataFrame(y_pred_dict)
        
        # wMAE計算
        try:
            wmae, weights, individual_maes, sample_counts, value_ranges = calculate_weighted_mae(
                y_true_df, y_pred_df, actual_targets, train
            )
            
            print(f"    重み:")
            for target in actual_targets:
                if target in weights:
                    print(f"      {target}: {weights[target]:.4f}")
            
            print(f"    個別MAE:")
            for target in actual_targets:
                if target in individual_maes:
                    print(f"      {target}: {individual_maes[target]:.4f}")
            
            print(f"    推定wMAE: {wmae:.4f}")
            
            if wandb_available:
                wandb.log({
                    f"wmae/{model_name}/estimated_wmae": wmae,
                    f"wmae/{model_name}/weights": weights,
                    f"wmae/{model_name}/individual_maes": individual_maes
                })
        except Exception as e:
            print(f"    wMAE計算エラー: {e}")
    
    if wandb_available:
        # 総合指標をWandBに記録
        avg_performance = {}
        for target, target_data in results.items():
            for model, performance in target_data['cv_results'].items():
                if model not in avg_performance:
                    avg_performance[model] = []
                avg_performance[model].append(performance['cv_mae'])
        
        for model, maes in avg_performance.items():
            avg_mae = np.mean(maes)
            wandb.log({f"overall/{model}_avg_mae": avg_mae})
        
        wandb.log({
            "experiment/elapsed_time": elapsed_time,
            "experiment/completed_targets": len(results)
        })
        
        # メタデータファイルもWandBにアップロード
        wandb.save(str(metadata_path))
        
        wandb.finish()
        print("✅ WandB実験記録完了")
    
    print(f"⏱️  総実行時間: {elapsed_time:.2f} 秒")
    print("🎉 WandB統合実験完了!")
    
    return results

if __name__ == "__main__":
    results = main_experiment(offline_wandb=True)
    print(f"\n🎯 最終結果: {len(results)}個の特性で実験完了")