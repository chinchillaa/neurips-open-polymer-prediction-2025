#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - クイックテスト用
基本機能のテストを短時間で実行
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
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

print(f"🚀 クイックテスト開始")
print(f"📁 プロジェクトルート: {PROJECT_ROOT}")
print(f"📁 データディレクトリ: {DATA_DIR}")

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

def quick_test():
    """クイックテスト実行"""
    start_time = time.time()
    
    # データ読み込み
    train, test, submission = load_local_data()
    
    # 特徴量生成
    train_features = quick_feature_engineering(train)
    test_features = quick_feature_engineering(test)
    
    # 簡単なモデル訓練（1つの特性のみ）
    target_col = 'Tg'  # ガラス転移温度
    if target_col in train.columns:
        print(f"🤖 {target_col}用の簡単なモデル訓練中...")
        
        # 欠損値除去
        valid_mask = ~train[target_col].isna()
        X_valid = train_features[valid_mask]
        y_valid = train[target_col][valid_mask]
        
        print(f"✅ 有効データ: {len(X_valid)}件")
        
        if len(X_valid) > 10:  # 最小限のデータがある場合のみ実行
            # 簡単なRandomForestモデル
            model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
            model.fit(X_valid, y_valid)
            
            # テスト予測
            test_pred = model.predict(test_features)
            
            # 結果表示
            print(f"✅ 予測完了: 平均値 = {np.mean(test_pred):.2f}")
            print(f"✅ 予測範囲: {np.min(test_pred):.2f} - {np.max(test_pred):.2f}")
        else:
            print("⚠️  データが不足しています")
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  実行時間: {elapsed_time:.2f} 秒")
    print("🎉 クイックテスト完了!")

if __name__ == "__main__":
    quick_test()