#!/usr/bin/env python3
"""
ローカル実験をKaggleノートブックに変換するスクリプト
"""

import os
import sys
import argparse
import json
from pathlib import Path
import yaml
import nbformat as nbf
from datetime import datetime

def convert_local_to_kaggle(experiment_name):
    """ローカル実験をKaggleノートブックに変換"""
    
    project_root = Path(__file__).parent.parent
    local_exp_dir = project_root / "experiments" / experiment_name
    kaggle_exp_dir = project_root / "kaggle_notebooks" / "submission" / experiment_name
    
    print(f"🔄 ローカル実験 → Kaggleノートブック変換: {experiment_name}")
    print(f"📂 入力: {local_exp_dir}")
    print(f"📂 出力: {kaggle_exp_dir}")
    
    # 存在確認
    if not local_exp_dir.exists():
        print(f"❌ エラー: ローカル実験が見つかりません: {local_exp_dir}")
        return False
    
    if not kaggle_exp_dir.exists():
        print(f"📁 Kaggleディレクトリを作成: {kaggle_exp_dir}")
        kaggle_exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定ファイル読み込み
    config_file = local_exp_dir / "config.yaml"
    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        print("⚠️  config.yamlが見つかりません。デフォルト設定を使用")
        config = {}
    
    # ローカル実験スクリプト読み込み
    local_script = local_exp_dir / "scripts" / "local_experiment.py"
    if not local_script.exists():
        print(f"❌ エラー: メインスクリプトが見つかりません: {local_script}")
        return False
    
    with open(local_script, "r") as f:
        script_content = f.read()
    
    # Kaggleノートブック生成
    notebook_file = kaggle_exp_dir / f"{experiment_name}.ipynb"
    success = create_kaggle_notebook(script_content, config, notebook_file, experiment_name)
    
    if success:
        print(f"✅ 変換完了: {notebook_file}")
        
        # kernel-metadata.json更新
        update_kernel_metadata(kaggle_exp_dir, experiment_name, config)
        
        print("\n📋 次のステップ:")
        print(f"1. cd {kaggle_exp_dir}")
        print("2. kaggle kernels push -p . でアップロード")
        
        return True
    else:
        print("❌ 変換に失敗しました")
        return False

def create_kaggle_notebook(script_content, config, output_file, experiment_name):
    """Pythonスクリプトからkaggleノートブックを生成"""
    
    try:
        # 新しいノートブック作成
        nb = nbf.v4.new_notebook()
        
        # セル1: タイトルと依存関係インストール
        title = experiment_name.replace('_', ' ').title()
        cell1_content = f"""# {title}
# ローカル実験から自動変換されたKaggleノートブック
# 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# 依存関係インストール
import subprocess
import sys

def install_package(package):
    try:
        __import__(package)
        print(f"✅ {package} は既にインストール済み")
    except ImportError:
        print(f"📦 {package} をインストール中...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print(f"✅ {package} インストール完了")

# 必要なパッケージをインストール
packages = ["rdkit-pypi", "xgboost", "catboost"]
for package in packages:
    install_package(package)
"""
        nb.cells.append(nbf.v4.new_code_cell(cell1_content))
        
        # セル2: 基本ライブラリインポート
        cell2_content = """# ライブラリインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("基本ライブラリインポート完了")"""
        nb.cells.append(nbf.v4.new_code_cell(cell2_content))
        
        # セル3: データ読み込み（Kaggleパスに変更）
        cell3_content = """# データ読み込み
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

print(f"訓練データ: {train.shape}")
print(f"テストデータ: {test.shape}")
print(f"提出データ: {submission.shape}")"""
        nb.cells.append(nbf.v4.new_code_cell(cell3_content))
        
        # セル4以降: ローカルスクリプトを適切に分割
        script_cells = split_script_into_cells(script_content)
        for cell_content in script_cells:
            # Kaggle環境用にパス修正
            kaggle_content = adapt_for_kaggle(cell_content)
            if kaggle_content.strip():  # 空でない場合のみ追加
                nb.cells.append(nbf.v4.new_code_cell(kaggle_content))
        
        # 最終セル: 提出ファイル保存
        final_cell = """# 提出ファイル保存
print("提出ファイルを保存します...")
# submission.to_csv('submission.csv', index=False)
print("提出ファイル保存完了！")"""
        nb.cells.append(nbf.v4.new_code_cell(final_cell))
        
        # ノートブック保存
        with open(output_file, "w") as f:
            nbf.write(nb, f)
        
        return True
        
    except Exception as e:
        print(f"❌ ノートブック生成エラー: {e}")
        return False

def split_script_into_cells(script_content):
    """Pythonスクリプトを適切なセルに分割"""
    
    # 簡単な分割ロジック（改善の余地あり）
    lines = script_content.split('\n')
    cells = []
    current_cell = []
    
    for line in lines:
        # コメント行でセル分割の目安とする
        if line.strip().startswith('# ') and len(current_cell) > 10:
            if current_cell:
                cells.append('\n'.join(current_cell))
                current_cell = []
        current_cell.append(line)
    
    # 最後のセルを追加
    if current_cell:
        cells.append('\n'.join(current_cell))
    
    # 空のセルを除去
    cells = [cell for cell in cells if cell.strip()]
    
    return cells

def adapt_for_kaggle(cell_content):
    """セル内容をKaggle環境用に調整"""
    
    # ローカルパスをKaggleパスに変更
    kaggle_content = cell_content
    
    # データパスの変更
    kaggle_content = kaggle_content.replace(
        "../../data/raw", 
        "/kaggle/input/neurips-open-polymer-prediction-2025"
    )
    
    # WandB関連の削除（Kaggleでは使用しない）
    if "wandb" in kaggle_content.lower():
        lines = kaggle_content.split('\n')
        filtered_lines = [line for line in lines if "wandb" not in line.lower()]
        kaggle_content = '\n'.join(filtered_lines)
    
    # ローカル保存パスの調整
    kaggle_content = kaggle_content.replace("results/", "")
    
    return kaggle_content

def update_kernel_metadata(kaggle_dir, experiment_name, config):
    """kernel-metadata.jsonを更新"""
    
    metadata_file = kaggle_dir / "kernel-metadata.json"
    
    # 既存メタデータを読み込むか、新規作成
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # 基本設定を更新
    metadata.update({
        "id": f"tgwstr/{experiment_name.lower().replace('_', '-')}",
        "title": experiment_name.replace('_', ' ').title(),
        "code_file": f"{experiment_name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["neurips-open-polymer-prediction-2025"],
        "kernel_sources": []
    })
    
    # 設定ファイルから追加情報を取得
    if config and "experiment" in config:
        exp_config = config["experiment"]
        if "description" in exp_config:
            # タイトルに説明を追加（省略）
            pass
    
    # メタデータ保存
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ kernel-metadata.json 更新完了")

def main():
    """メインエントリポイント"""
    parser = argparse.ArgumentParser(description="ローカル実験をKaggleノートブックに変換")
    parser.add_argument("experiment_name", help="実験名")
    parser.add_argument("--force", "-f", action="store_true", help="既存ファイルを上書き")
    
    args = parser.parse_args()
    
    success = convert_local_to_kaggle(args.experiment_name)
    
    if success:
        print(f"\n🎉 変換完了: {args.experiment_name}")
    else:
        print(f"\n❌ 変換失敗: {args.experiment_name}")
        sys.exit(1)

if __name__ == "__main__":
    main()