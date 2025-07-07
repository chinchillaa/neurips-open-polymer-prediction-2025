#!/usr/bin/env python
# coding: utf-8

"""
NeurIPS Open Polymer Prediction 2025 - Advanced Ensemble v2
高度なアンサンブルモデルによるポリマー特性予測

このノートブックは以下の特徴を持ちます：
1. RDKitベースの高度な分子特徴量（100特徴量）
2. 5つのMLモデル（XGBoost, CatBoost, RandomForest, GradientBoosting, KNN）
3. 加重平均アンサンブルによる性能向上
4. 全5特性（Tg, FFV, Tc, Density, Rg）の予測
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# 設定
SEED = 42
np.random.seed(SEED)

# Kaggle環境判定
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    # Kaggle環境
    INPUT_DIR = '/kaggle/input/neurips-open-polymer-prediction-2025'
    
    # RDKit wheel fileをインストール
    print("🔧 RDKitインストール中...")
    try:
        import subprocess
        # RDKit wheel datasetからインストール（複数のパスを試す）
        rdkit_paths = [
            '/kaggle/input/rdkit-install-whl/rdkit_wheel',
            '/kaggle/input/rdkit-install-whl',
            '/kaggle/input/rdkit-whl',
            '/kaggle/input/rdkit'
        ]
        
        installed = False
        for rdkit_dataset in rdkit_paths:
            if os.path.exists(rdkit_dataset):
                print(f"  📁 RDKitデータセット発見: {rdkit_dataset}")
                whl_files = [f for f in os.listdir(rdkit_dataset) if f.endswith('.whl')]
                if whl_files:
                    # 最新のwhlファイルを使用
                    whl_file = os.path.join(rdkit_dataset, sorted(whl_files)[-1])
                    print(f"  📦 インストール中: {whl_file}")
                    result = subprocess.run([sys.executable, '-m', 'pip', 'install', whl_file, '--no-deps'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"  ✅ RDKit installed from: {whl_file}")
                        installed = True
                        break
                    else:
                        print(f"  ⚠️ インストール失敗: {result.stderr}")
        
        if not installed:
            print("  ⚠️ RDKitデータセットが見つかりません")
            # 通常のpipインストールを試す
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'rdkit-pypi'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("  ✅ RDKit installed via pip")
            else:
                print(f"  ⚠️ pip install failed: {result.stderr}")
                
    except Exception as e:
        print(f"⚠️ RDKit installation failed: {e}")
else:
    # ローカル環境
    INPUT_DIR = '../input/neurips-open-polymer-prediction-2025'

# RDKit利用可能性チェック
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    rdkit_available = True
    print("✅ RDKit利用可能 - 高精度分子特徴量を使用")
except ImportError:
    rdkit_available = False
    print("⚠️ RDKit利用不可 - 基本SMILES特徴量を使用")

print("ライブラリインポート完了")

# データ読み込み
def load_data():
    """データの読み込み"""
    train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
    submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
    
    print(f"訓練データ形状: {train.shape}")
    print(f"テストデータ形状: {test.shape}")
    
    return train, test, submission

# 基本的なSMILES特徴量
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

# RDKitベースの分子特徴量
def rdkit_molecular_features(smiles):
    """RDKitを使用した分子特徴量"""
    if pd.isna(smiles) or smiles == '':
        return [0] * 100
    
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

# 特徴量エンジニアリング
def feature_engineering(df):
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
    
    features_df = pd.DataFrame(features_list, columns=feature_names)
    
    print(f"✅ 特徴量生成完了: {features_df.shape[1]}個の特徴量")
    return features_df

# モデル訓練とアンサンブル
def train_models_for_target(X, y, target_name, n_splits=3):
    """特定の特性に対するモデル訓練とアンサンブル"""
    print(f"\n🤖 {target_name}用の高度なモデル訓練中...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    # モデル定義（ハイパーパラメータ調整版）
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=200, depth=7, learning_rate=0.08,
            random_seed=SEED, verbose=False
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
    
    # クロスバリデーション
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
                cv_results[model_name] = {'cv_scores': []}
            
            cv_results[model_name]['cv_scores'].append(mae)
            print(f"    フォールド {fold+1} {model_name} MAE: {mae:.6f}")
    
    # 各モデルの平均性能を計算
    for model_name in models.keys():
        avg_mae = np.mean(cv_results[model_name]['cv_scores'])
        std_mae = np.std(cv_results[model_name]['cv_scores'])
        cv_results[model_name]['cv_mae'] = avg_mae
        cv_results[model_name]['cv_std'] = std_mae
        print(f"  {model_name} 平均 CV MAE: {avg_mae:.6f} (±{std_mae:.6f})")
    
    # アンサンブル予測の計算
    print(f"\n  🎯 {target_name}のアンサンブル予測を計算中...")
    
    # 加重平均アンサンブル（性能ベースの重み）
    model_weights = {}
    total_inv_mae = sum(1.0 / cv_results[model]['cv_mae'] for model in models.keys())
    for model in models.keys():
        model_weights[model] = (1.0 / cv_results[model]['cv_mae']) / total_inv_mae
    
    print(f"  📊 最適化された重み:")
    for model, weight in model_weights.items():
        print(f"    {model}: {weight:.4f}")
    
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
    
    return trained_models, scaler, model_weights

# 予測関数
def predict_ensemble(models, scaler, weights, X):
    """アンサンブル予測"""
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X_scaled)
    
    # 加重平均
    ensemble_pred = sum(
        predictions[model_name] * weights[model_name] 
        for model_name in models.keys()
    )
    
    return ensemble_pred

# メイン処理
def main():
    """メイン処理"""
    start_time = time.time()
    
    # データ読み込み
    train, test, submission = load_data()
    
    # 特徴量エンジニアリング
    train_features = feature_engineering(train)
    test_features = feature_engineering(test)
    
    # 特性列の特定
    target_columns = [col for col in train.columns if col not in ['SMILES', 'Id', 'id']]
    print(f"\n🎯 対象特性: {target_columns}")
    
    # 各特性に対してモデル訓練と予測
    all_predictions = {}
    
    for target_col in target_columns:
        if target_col in train.columns:
            # 欠損値除去
            valid_mask = ~train[target_col].isna()
            if valid_mask.sum() < 10:
                print(f"⚠️  {target_col}: データ不足（{valid_mask.sum()}件）- スキップ")
                # テストデータに対してデフォルト値を設定
                all_predictions[target_col] = np.zeros(len(test))
                continue
            
            X_valid = train_features[valid_mask]
            y_valid = train[target_col][valid_mask]
            
            print(f"\n📊 {target_col} - 有効データ: {len(X_valid)}件")
            
            # モデル訓練とアンサンブル
            trained_models, scaler, weights = train_models_for_target(
                X_valid, y_valid, target_col, n_splits=3
            )
            
            # テストデータの予測
            predictions = predict_ensemble(trained_models, scaler, weights, test_features)
            all_predictions[target_col] = predictions
    
    # 提出ファイル作成
    print("\n📝 提出ファイル作成中...")
    for col in submission.columns:
        if col != 'Id' and col in all_predictions:
            submission[col] = all_predictions[col]
    
    # 保存
    submission.to_csv('submission.csv', index=False)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  総実行時間: {elapsed_time:.2f} 秒")
    print("🎉 アンサンブル予測完了!")
    
    return submission

# 実行
if __name__ == "__main__":
    submission = main()
    print("\n提出ファイル作成完了！")
    print(submission.head())