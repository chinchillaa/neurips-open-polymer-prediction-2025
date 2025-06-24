#!/usr/bin/env python3
"""
クイックテスト - NeurIPS Polymer Advanced Ensemble
基本的な動作確認用テストスクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

def test_imports():
    """必要なライブラリのインポートテスト"""
    print("📦 ライブラリインポートテスト...")
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError:
        print("❌ pandas - pip install pandas")
        
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError:
        print("❌ numpy - pip install numpy")
        
    try:
        import xgboost
        print("✅ xgboost")
    except ImportError:
        print("❌ xgboost - pip install xgboost")
        
    try:
        import catboost
        print("✅ catboost")
    except ImportError:
        print("❌ catboost - pip install catboost")
        
    try:
        from rdkit import Chem
        print("✅ rdkit")
    except ImportError:
        print("⚠️  rdkit（オプション） - pip install rdkit-pypi")
        
    try:
        import yaml
        print("✅ yaml")
    except ImportError:
        print("❌ yaml - pip install pyyaml")

def test_data_files():
    """データファイルの存在確認"""
    print("\n📂 データファイルテスト...")
    
    data_dir = project_root / "data" / "raw"
    required_files = ["train.csv", "test.csv", "sample_submission.csv"]
    
    all_exist = True
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - {file_path}が見つかりません")
            all_exist = False
    
    return all_exist

def test_config():
    """設定ファイルの読み込みテスト"""
    print("\n⚙️  設定ファイルテスト...")
    
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        print(f"✅ config.yaml が存在します")
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print(f"✅ 設定ファイル読み込み成功")
            print(f"   実験名: {config.get('experiment', {}).get('name', 'N/A')}")
        except Exception as e:
            print(f"❌ 設定ファイル読み込みエラー: {e}")
    else:
        print(f"❌ config.yaml が見つかりません: {config_path}")

def test_directories():
    """必要なディレクトリの確認"""
    print("\n📁 ディレクトリ構造テスト...")
    
    experiment_dir = Path(__file__).parent.parent
    dirs_to_check = [
        "scripts",
        "tests", 
        "experiments_results",
        "results"  # 旧形式も確認
    ]
    
    for dir_name in dirs_to_check:
        dir_path = experiment_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/ が存在しません（自動作成されます）")

def main():
    """メインテスト実行"""
    print("🧪 NeurIPS Polymer Advanced Ensemble クイックテスト")
    print("=" * 50)
    
    # 各種テスト実行
    test_imports()
    data_ok = test_data_files()
    test_config()
    test_directories()
    
    print("\n" + "=" * 50)
    if data_ok:
        print("✅ 基本的な環境は整っています！")
        print("💡 実験を実行するには:")
        print("   ./scripts/run_experiment.sh")
    else:
        print("⚠️  データファイルが不足しています")
        print("💡 Kaggleからデータをダウンロードしてください:")
        print("   https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data")

if __name__ == "__main__":
    main()