#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - シンプルなベースラインモデル

このスクリプトは：
1. SMILES文字列から分子記述子を抽出
2. シンプルなモデルで複数のターゲット（Tg, FFV, Tc, Density, Rg）を予測
3. WandBで実験を追跡
4. 提出用ファイルを生成
"""

import os
import yaml
import pandas as pd
import numpy as np
import wandb
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import joblib

# RDKitを使用して分子記述子を計算
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    print("警告: RDKitが利用できません。基本的な特徴量のみ使用します。")
    RDKIT_AVAILABLE = False


def load_config(config_path: str = "config/config.yaml") -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: dict) -> None:
    """WandBを初期化"""
    wandb_config = config.get('wandb', {})
    
    if not wandb_config.get('enabled', True):
        return
    
    wandb.init(
        project=wandb_config.get('project', 'neurips-polymer-prediction'),
        entity=wandb_config.get('entity'),
        tags=wandb_config.get('tags', []) + ['baseline'],
        notes="シンプルなベースラインモデル",
        save_code=wandb_config.get('save_code', True),
        config=config
    )


def calculate_simple_features(smiles: str) -> dict:
    """SMILES文字列から基本的な特徴量を計算"""
    features = {
        'length': len(smiles),
        'aromatic_atoms': smiles.count('c') + smiles.count('n') + smiles.count('o') + smiles.count('s'),
        'rings': smiles.count('1') + smiles.count('2') + smiles.count('3'),
        'branches': smiles.count('('),
        'double_bonds': smiles.count('='),
        'triple_bonds': smiles.count('#'),
    }
    return features


def calculate_rdkit_features(smiles: str) -> dict:
    """RDKitを使用して分子記述子を計算"""
    if not RDKIT_AVAILABLE:
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        features = {
            'mol_weight': Descriptors.MolWt(mol),
            'log_p': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'num_hbd': rdMolDescriptors.CalcNumHBD(mol),
            'num_hba': rdMolDescriptors.CalcNumHBA(mol),
        }
        
        # モルガンフィンガープリント（最初の10ビットのみ使用）
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_array = np.array(fp)
        for i in range(min(10, len(fp_array))):
            features[f'morgan_fp_{i}'] = fp_array[i]
        
        return features
    except Exception as e:
        print(f"SMILES処理エラー: {smiles[:50]}... - {e}")
        return {}


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームから特徴量を抽出"""
    print("特徴量を抽出中...")
    
    features_list = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"処理中: {idx}/{len(df)}")
        
        smiles = row['SMILES']
        
        # 基本的な特徴量
        simple_features = calculate_simple_features(smiles)
        
        # RDKit特徴量
        rdkit_features = calculate_rdkit_features(smiles)
        
        # 統合
        all_features = {**simple_features, **rdkit_features}
        all_features['id'] = row['id']
        
        features_list.append(all_features)
    
    features_df = pd.DataFrame(features_list)
    features_df = features_df.fillna(0)  # 欠損値を0で埋める
    
    print(f"特徴量抽出完了。形状: {features_df.shape}")
    return features_df


def train_baseline_model(train_df: pd.DataFrame, config: dict) -> tuple:
    """ベースラインモデルを訓練"""
    print("ベースラインモデルを訓練中...")
    
    # 特徴量を抽出
    train_features = extract_features(train_df)
    
    # ターゲット変数
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # 有効なサンプルのみを使用（全てのターゲットが欠損していないもの）
    valid_mask = train_df[target_columns].notna().any(axis=1)
    valid_train_df = train_df[valid_mask].copy()
    valid_features = train_features[valid_mask].copy()
    
    print(f"有効なサンプル数: {len(valid_train_df)}")
    
    # 特徴量とターゲットを準備
    feature_columns = [col for col in valid_features.columns if col != 'id']
    X = valid_features[feature_columns].values
    
    # 各ターゲットに対してNaNでないサンプルで訓練
    models = {}
    scalers = {}
    results = {}
    
    for target in target_columns:
        print(f"\n{target}のモデルを訓練中...")
        
        # 現在のターゲットが欠損していないサンプル
        target_mask = valid_train_df[target].notna()
        if target_mask.sum() == 0:
            print(f"警告: {target}の有効なデータがありません")
            continue
        
        X_target = X[target_mask]
        y_target = valid_train_df[target][target_mask].values
        
        print(f"  {target}の有効サンプル数: {len(y_target)}")
        
        if len(y_target) < 10:  # 最小限のサンプル数チェック
            print(f"警告: {target}のサンプル数が少なすぎます")
            continue
        
        # データ分割
        X_train, X_val, y_train, y_val = train_test_split(
            X_target, y_target, test_size=0.2, random_state=42
        )
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # モデル訓練
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # 評価
        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        
        print(f"  訓練RMSE: {train_rmse:.4f}, 検証RMSE: {val_rmse:.4f}")
        print(f"  訓練R²: {train_r2:.4f}, 検証R²: {val_r2:.4f}")
        
        # WandBにログ
        if wandb.run is not None:
            wandb.log({
                f'{target}/train_rmse': train_rmse,
                f'{target}/val_rmse': val_rmse,
                f'{target}/train_r2': train_r2,
                f'{target}/val_r2': val_r2,
            })
        
        models[target] = model
        scalers[target] = scaler
        results[target] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2
        }
    
    return models, scalers, feature_columns, results


def generate_predictions(test_df: pd.DataFrame, models: dict, scalers: dict, 
                        feature_columns: list) -> pd.DataFrame:
    """テストデータで予測を生成"""
    print("予測を生成中...")
    
    # テストデータの特徴量を抽出
    test_features = extract_features(test_df)
    X_test = test_features[feature_columns].values
    
    # 予測結果を格納
    predictions = {'id': test_df['id'].values}
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    for target in target_columns:
        if target in models:
            # スケーリングと予測
            X_test_scaled = scalers[target].transform(X_test)
            pred = models[target].predict(X_test_scaled)
            predictions[target] = pred
        else:
            # モデルがない場合は0で埋める
            print(f"警告: {target}のモデルがないため0で埋めます")
            predictions[target] = np.zeros(len(test_df))
    
    return pd.DataFrame(predictions)


def save_models(models: dict, scalers: dict, feature_columns: list, config: dict):
    """モデルとスケーラーを保存"""
    models_dir = Path(config['models']['trained_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # モデルを保存
    model_path = models_dir / "baseline_models.joblib"
    joblib.dump({
        'models': models,
        'scalers': scalers,
        'feature_columns': feature_columns
    }, model_path)
    
    print(f"モデルを保存しました: {model_path}")
    
    # WandBに保存
    if wandb.run is not None:
        wandb.save(str(model_path))


def main():
    """メイン実行関数"""
    # 設定の読み込み
    config = load_config()
    
    # WandBの初期化
    setup_wandb(config)
    
    try:
        print("NeurIPS Open Polymer Prediction 2025 - ベースラインモデル")
        print("=" * 60)
        
        # データの読み込み
        print("データを読み込み中...")
        train_df = pd.read_csv('data/raw/train.csv')
        test_df = pd.read_csv('data/raw/test.csv')
        
        print(f"訓練データ: {train_df.shape}")
        print(f"テストデータ: {test_df.shape}")
        
        # データの確認
        target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        print("\nターゲット変数の欠損状況:")
        for col in target_columns:
            missing_count = train_df[col].isna().sum()
            valid_count = train_df[col].notna().sum()
            print(f"  {col}: 有効 {valid_count}, 欠損 {missing_count}")
        
        # モデル訓練
        models, scalers, feature_columns, results = train_baseline_model(train_df, config)
        
        # モデル保存
        save_models(models, scalers, feature_columns, config)
        
        # 予測生成
        predictions_df = generate_predictions(test_df, models, scalers, feature_columns)
        
        # 提出ファイルの保存
        submission_path = Path(config['models']['submissions_dir']) / "baseline_submission.csv"
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(submission_path, index=False)
        
        print(f"\n提出ファイルを保存しました: {submission_path}")
        print(f"予測ファイルの形状: {predictions_df.shape}")
        
        # WandBに提出ファイルを保存
        if wandb.run is not None:
            wandb.save(str(submission_path))
            print(f"WandBダッシュボード: {wandb.run.url}")
        
        print("\nベースラインモデルの訓練が完了しました！")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # WandBセッションを終了
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()