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
                "max_features": 500
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
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    sample_submission_path = DATA_DIR / "sample_submission.csv"
    
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
    """特定の特性に対するモデル訓練"""
    print(f"🤖 {target_name}用の高度なモデル訓練中...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    models_performance = {}
    
    # モデル定義
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=100, depth=6, learning_rate=0.1,
            random_seed=SEED, verbose=False
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=SEED
        ),
        'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    }
    
    cv_results = {}
    
    for model_name, model in models.items():
        print(f"  {model_name}モデル訓練中...")
        cv_scores = []
        
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
            
            # モデル訓練
            model.fit(X_train_scaled, y_train_fold)
            y_pred = model.predict(X_val_scaled)
            mae = mean_absolute_error(y_val_fold, y_pred)
            cv_scores.append(mae)
            
            print(f"    フォールド {fold+1} {model_name} MAE: {mae:.6f}")
            
            if wandb_available:
                wandb.log({
                    f"{target_name}/{model_name}/fold_{fold+1}_mae": mae,
                    f"{target_name}/{model_name}/fold": fold+1
                })
        
        avg_mae = np.mean(cv_scores)
        std_mae = np.std(cv_scores)
        cv_results[model_name] = {
            'cv_mae': avg_mae,
            'cv_std': std_mae,
            'cv_scores': cv_scores
        }
        
        print(f"    {model_name} 平均 CV MAE: {avg_mae:.6f} (±{std_mae:.6f})")
        
        if wandb_available:
            wandb.log({
                f"{target_name}/{model_name}/cv_mae": avg_mae,
                f"{target_name}/{model_name}/cv_std": std_mae
            })
    
    return cv_results

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
    sample_targets = target_columns[:2]  # 最初の2つの特性のみテスト
    
    for target_col in sample_targets:
        if target_col in train.columns:
            # 欠損値除去
            valid_mask = ~train[target_col].isna()
            if valid_mask.sum() < 10:
                print(f"⚠️  {target_col}: データ不足（{valid_mask.sum()}件）- スキップ")
                continue
            
            X_valid = train_features[valid_mask]
            y_valid = train[target_col][valid_mask]
            
            print(f"\n📊 {target_col} - 有効データ: {len(X_valid)}件")
            
            # モデル訓練
            cv_results = train_models_for_target(
                X_valid, y_valid, target_col, wandb_available, n_splits=3  # 高速化のため3-fold
            )
            results[target_col] = cv_results
    
    # 実験結果保存
    elapsed_time = time.time() - start_time
    
    experiment_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "rdkit_available": rdkit_available,
        "elapsed_time": elapsed_time,
        "results": results
    }
    
    metadata_path = EXPERIMENT_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 実験結果サマリー:")
    for target, target_results in results.items():
        print(f"  {target}:")
        for model, performance in target_results.items():
            print(f"    {model}: {performance['cv_mae']:.6f} (±{performance['cv_std']:.6f})")
    
    if wandb_available:
        # 総合指標をWandBに記録
        avg_performance = {}
        for target, target_results in results.items():
            for model, performance in target_results.items():
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