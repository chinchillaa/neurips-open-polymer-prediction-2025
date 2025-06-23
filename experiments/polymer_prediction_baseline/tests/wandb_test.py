#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - WandB実験管理テスト
WandBを使った実験追跡とモデル管理のテスト
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from datetime import datetime

# ML関連ライブラリ
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

# プロジェクトルートの設定
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "polymer_prediction_baseline"

print(f"🚀 WandB実験管理テスト開始")
print(f"📁 プロジェクトルート: {PROJECT_ROOT}")

# RDKit利用可能性チェック
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    rdkit_available = True
    print("✅ RDKit利用可能 - 高精度分子特徴量を使用")
except ImportError:
    rdkit_available = False
    print("⚠️  RDKit利用不可 - 基本SMILES特徴量を使用")

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
    """基本的なSMILES特徴量"""
    if pd.isna(smiles) or smiles == '':
        return [0] * 10
    
    features = [
        len(smiles),                    # SMILES文字列長
        smiles.count('C'),             # 炭素数
        smiles.count('N'),             # 窒素数
        smiles.count('O'),             # 酸素数
        smiles.count('S'),             # 硫黄数
        smiles.count('='),             # 二重結合数
        smiles.count('#'),             # 三重結合数
        smiles.count('('),             # 分岐数
        smiles.count('['),             # 特殊原子数
        smiles.count('@'),             # キラル中心数
    ]
    return features

def quick_feature_engineering(df):
    """クイック特徴量エンジニアリング"""
    print("🧬 基本特徴量生成中...")
    
    # SMILES基本特徴量
    smiles_features = []
    for smiles in df['SMILES']:
        features = basic_smiles_features(smiles)
        smiles_features.append(features)
    
    feature_names = [
        'smiles_length', 'carbon_count', 'nitrogen_count', 'oxygen_count', 'sulfur_count',
        'double_bond_count', 'triple_bond_count', 'branch_count', 'special_atom_count', 'chiral_count'
    ]
    
    features_df = pd.DataFrame(smiles_features, columns=feature_names)
    
    print(f"✅ 特徴量生成完了: {features_df.shape[1]}個の特徴量")
    return features_df

def wandb_test():
    """WandB実験管理テスト"""
    start_time = time.time()
    
    # WandB初期化（オフラインモードでテスト）
    print("🔧 WandB初期化中...")
    try:
        # オフラインモードで初期化（ログインなしでテスト可能）
        run = wandb.init(
            project="neurips-polymer-prediction-test",
            name=f"baseline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode="offline",  # オフラインモードでテスト
            config={
                "model_type": "random_forest",
                "n_estimators": 10,
                "max_depth": 5,
                "random_state": 42,
                "rdkit_available": rdkit_available,
                "feature_count": 10
            }
        )
        print("✅ WandB初期化成功（オフラインモード）")
        wandb_available = True
    except Exception as e:
        print(f"⚠️  WandB初期化失敗: {e}")
        print("📝 WandBなしでテスト継続")
        wandb_available = False
    
    # データ読み込み
    train, test, submission = load_local_data()
    
    # 特徴量生成
    train_features = quick_feature_engineering(train)
    test_features = quick_feature_engineering(test)
    
    # データサイズをWandBに記録
    if wandb_available:
        wandb.log({
            "data/train_size": len(train),
            "data/test_size": len(test),
            "data/feature_count": train_features.shape[1]
        })
    
    # 簡単なモデル訓練（複数の特性でテスト）
    target_cols = ['Tg', 'Tm', 'Density']  # 複数の特性をテスト
    results = {}
    
    for target_col in target_cols:
        if target_col in train.columns:
            print(f"🤖 {target_col}用のモデル訓練中...")
            
            # 欠損値除去
            valid_mask = ~train[target_col].isna()
            X_valid = train_features[valid_mask]
            y_valid = train[target_col][valid_mask]
            
            print(f"✅ {target_col} 有効データ: {len(X_valid)}件")
            
            if len(X_valid) > 10:  # 最小限のデータがある場合のみ実行
                # 簡単なRandomForestモデル
                model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
                
                # 2-Fold CVでMAE計算
                kf = KFold(n_splits=2, shuffle=True, random_state=42)
                cv_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_valid)):
                    X_train_fold = X_valid.iloc[train_idx]
                    X_val_fold = X_valid.iloc[val_idx]
                    y_train_fold = y_valid.iloc[train_idx]
                    y_val_fold = y_valid.iloc[val_idx]
                    
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    mae = mean_absolute_error(y_val_fold, y_pred)
                    cv_scores.append(mae)
                    
                    print(f"  フォールド {fold+1} MAE: {mae:.3f}")
                    
                    # WandBにフォールド結果を記録
                    if wandb_available:
                        wandb.log({
                            f"{target_col}/fold_{fold+1}_mae": mae,
                            f"{target_col}/fold": fold+1
                        })
                
                avg_mae = np.mean(cv_scores)
                results[target_col] = avg_mae
                
                # 全データで再訓練
                model.fit(X_valid, y_valid)
                
                # テスト予測
                test_pred = model.predict(test_features)
                
                # 結果表示
                print(f"✅ {target_col} CV MAE: {avg_mae:.3f}")
                print(f"✅ {target_col} 予測範囲: {np.min(test_pred):.2f} - {np.max(test_pred):.2f}")
                
                # WandBに最終結果を記録
                if wandb_available:
                    wandb.log({
                        f"{target_col}/cv_mae": avg_mae,
                        f"{target_col}/pred_mean": np.mean(test_pred),
                        f"{target_col}/pred_std": np.std(test_pred),
                        f"{target_col}/pred_min": np.min(test_pred),
                        f"{target_col}/pred_max": np.max(test_pred)
                    })
            else:
                print(f"⚠️  {target_col} データが不足しています")
    
    elapsed_time = time.time() - start_time
    
    # 実験サマリー
    print(f"\n📊 実験結果サマリー:")
    for target, mae in results.items():
        print(f"  {target}: CV MAE = {mae:.3f}")
    
    # WandBに総合結果を記録
    if wandb_available:
        wandb.log({
            "experiment/elapsed_time": elapsed_time,
            "experiment/target_count": len(results),
            "experiment/avg_mae": np.mean(list(results.values())) if results else 0
        })
        
        # 実験終了
        wandb.finish()
        print("✅ WandB実験記録完了")
    
    print(f"⏱️  実行時間: {elapsed_time:.2f} 秒")
    print("🎉 WandB実験管理テスト完了!")
    
    return results

if __name__ == "__main__":
    results = wandb_test()
    print(f"\n🎯 最終結果: {len(results)}個の特性で実験完了")