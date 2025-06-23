#!/usr/bin/env python3
"""
WandBを使用したモデル訓練のサンプルスクリプト
"""

import os
import yaml
import wandb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path


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
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
        save_code=wandb_config.get('save_code', True),
        config=config
    )


def create_sample_data(n_samples: int = 1000, n_features: int = 10) -> tuple:
    """サンプルデータを生成（実際のコンペでは実データを使用）"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # 分子特性のサンプル予測タスク
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + 
         np.sin(X[:, 2]) + np.random.normal(0, 0.1, n_samples))
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    return X, y, feature_names


def train_model(config: dict) -> dict:
    """モデルを訓練し、結果を返す"""
    
    # サンプルデータの生成（実際はデータ読み込み）
    X, y, feature_names = create_sample_data()
    
    # 訓練・テストデータの分割
    training_config = config.get('training', {})
    test_size = training_config.get('test_size', 0.2)
    random_state = config.get('random_seed', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # モデルの設定
    model_config = config.get('model', {})
    model_params = model_config.get('parameters', {})
    
    # モデルの訓練
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    
    # 予測と評価
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # クロスバリデーション
    cv_config = training_config.get('cross_validation', {})
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv_config.get('n_splits', 5),
        scoring='neg_root_mean_squared_error'
    )
    cv_rmse_mean = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()
    
    results = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_rmse_mean': cv_rmse_mean,
        'cv_rmse_std': cv_rmse_std,
        'feature_importance': dict(zip(feature_names, model.feature_importances_))
    }
    
    # WandBにメトリクスをログ
    if wandb.run is not None:
        wandb.log({
            'train/rmse': train_rmse,
            'train/r2': train_r2,
            'test/rmse': test_rmse,
            'test/r2': test_r2,
            'cv/rmse_mean': cv_rmse_mean,
            'cv/rmse_std': cv_rmse_std,
        })
        
        # 特徴量重要度をログ
        wandb.log({f'feature_importance/{name}': importance 
                  for name, importance in results['feature_importance'].items()})
    
    return model, results


def save_model(model, config: dict) -> str:
    """モデルを保存"""
    models_dir = Path(config['models']['trained_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / f"model_{wandb.run.id if wandb.run else 'local'}.joblib"
    joblib.dump(model, model_path)
    
    # WandBにモデルを保存
    if wandb.run is not None and config.get('wandb', {}).get('log_model', True):
        wandb.save(str(model_path))
        
        # アーティファクトとしても保存
        artifact = wandb.Artifact(
            name=f"model_{wandb.run.id}",
            type="model",
            description="訓練済みRandomForestモデル"
        )
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
    
    return str(model_path)


def main():
    """メイン実行関数"""
    
    # 設定の読み込み
    config = load_config()
    
    # WandBの初期化
    setup_wandb(config)
    
    try:
        print("モデル訓練を開始します...")
        
        # モデル訓練
        model, results = train_model(config)
        
        # 結果の表示
        print(f"訓練RMSE: {results['train_rmse']:.4f}")
        print(f"テストRMSE: {results['test_rmse']:.4f}")
        print(f"訓練R²: {results['train_r2']:.4f}")
        print(f"テストR²: {results['test_r2']:.4f}")
        print(f"CV RMSE: {results['cv_rmse_mean']:.4f} ± {results['cv_rmse_std']:.4f}")
        
        # モデルの保存
        model_path = save_model(model, config)
        print(f"モデルを保存しました: {model_path}")
        
        if wandb.run is not None:
            print(f"WandBダッシュボード: {wandb.run.url}")
        
    finally:
        # WandBセッションを終了
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()