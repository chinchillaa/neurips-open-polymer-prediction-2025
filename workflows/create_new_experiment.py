#!/usr/bin/env python3
"""
新しい実験セットアップスクリプト
ローカル実験とKaggleノートブックの1対1対応セットを作成
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml
import json

def create_experiment_pair(experiment_name, description="", base_template="baseline"):
    """新しい実験ペア（ローカル + Kaggle）を作成"""
    
    project_root = Path(__file__).parent.parent
    
    # パス設定
    local_exp_dir = project_root / "experiments" / experiment_name
    kaggle_exp_dir = project_root / "kaggle_notebooks" / "submission" / experiment_name
    
    print(f"🚀 新しい実験ペアを作成: {experiment_name}")
    print(f"📂 ローカル: {local_exp_dir}")
    print(f"📂 Kaggle: {kaggle_exp_dir}")
    
    # ディレクトリが既に存在するかチェック
    if local_exp_dir.exists() or kaggle_exp_dir.exists():
        print("❌ エラー: 実験名が既に存在します")
        return False
    
    # 1. ローカル実験ディレクトリ作成
    create_local_experiment(local_exp_dir, experiment_name, description, base_template)
    
    # 2. Kaggleノートブックディレクトリ作成  
    create_kaggle_notebook(kaggle_exp_dir, experiment_name, description)
    
    print(f"✅ 実験ペア '{experiment_name}' の作成完了")
    print("\n📋 次のステップ:")
    print(f"1. cd experiments/{experiment_name}")
    print(f"2. config.yamlを編集")
    print(f"3. scripts/local_experiment.pyを実装")
    print(f"4. python scripts/local_experiment.py でテスト実行")
    
    return True

def create_local_experiment(exp_dir, name, description, base_template):
    """ローカル実験ディレクトリを作成"""
    
    # ディレクトリ構造作成
    dirs_to_create = [
        exp_dir,
        exp_dir / "scripts",
        exp_dir / "results" / "models",
        exp_dir / "results" / "predictions", 
        exp_dir / "results" / "logs",
        exp_dir / "wandb"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # config.yaml作成
    config = {
        "experiment": {
            "name": name,
            "description": description or f"{name} experiment",
            "version": "v1.0",
            "corresponding_kaggle_notebook": f"kaggle_notebooks/submission/{name}/"
        },
        "data": {
            "raw_data_dir": "../../data/raw",
            "processed_data_dir": "../../data/processed",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "sample_submission_file": "sample_submission.csv"
        },
        "model": {
            "type": "ensemble",
            "cross_validation": {
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42
            }
        },
        "logging": {
            "use_wandb": True,
            "wandb_project": "neurips-polymer-prediction-2025",
            "wandb_run_name": f"{name}_local"
        },
        "output": {
            "results_dir": "results",
            "models_dir": "results/models",
            "predictions_dir": "results/predictions",
            "logs_dir": "results/logs",
            "submission_file": "results/predictions/submission.csv"
        }
    }
    
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # README.md作成
    readme_content = f"""# {name} - ローカル実験

## 🎯 実験概要
{description or f"{name} による実験"}

## 🔗 対応関係
- **Kaggleノートブック**: `kaggle_notebooks/submission/{name}/`
- **実験名**: {name}
- **バージョン**: v1.0

## 🚀 実行方法
```bash
cd experiments/{name}
python scripts/local_experiment.py
```

## 📁 ディレクトリ構造
```
{name}/
├── config.yaml                    # 実験設定
├── README.md                      # このファイル
├── scripts/
│   └── local_experiment.py        # メインスクリプト
├── results/
│   ├── models/                    # 訓練済みモデル
│   ├── predictions/               # 予測結果
│   └── logs/                      # ログファイル
└── wandb/                         # WandB実験ログ
```

## 🔄 Kaggleノートブック変換
```bash
cd ../../workflows
python local_to_kaggle.py {name}
```
"""
    
    with open(exp_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # テンプレートスクリプトをコピー
    template_script = Path(__file__).parent / "templates" / "local_experiment_template.py"
    if template_script.exists():
        shutil.copy2(template_script, exp_dir / "scripts" / "local_experiment.py")
    else:
        # シンプルなテンプレートを作成
        script_content = f'''#!/usr/bin/env python3
"""
{name} - ローカル実験スクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_config():
    """設定ファイルを読み込み"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    """データを読み込み"""
    data_dir = Path(config["data"]["raw_data_dir"])
    
    train = pd.read_csv(data_dir / config["data"]["train_file"])
    test = pd.read_csv(data_dir / config["data"]["test_file"])
    
    print(f"訓練データ: {{train.shape}}")
    print(f"テストデータ: {{test.shape}}")
    
    return train, test

def main():
    """メイン実行関数"""
    print(f"🚀 {{config['experiment']['name']}} 実験開始")
    
    # 設定読み込み
    config = load_config()
    
    # データ読み込み
    train, test = load_data(config)
    
    # TODO: ここに実験ロジックを実装
    print("⚠️  実験ロジックを実装してください")
    
    print("✅ 実験完了")

if __name__ == "__main__":
    main()
'''
        
        with open(exp_dir / "scripts" / "local_experiment.py", "w") as f:
            f.write(script_content)
    
    print(f"📂 ローカル実験ディレクトリ作成完了: {exp_dir}")

def create_kaggle_notebook(kaggle_dir, name, description):
    """Kaggleノートブックディレクトリを作成"""
    
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # kernel-metadata.json作成
    metadata = {
        "id": f"tgwstr/{name.lower().replace('_', '-')}",
        "title": name.replace('_', ' ').title(),
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["neurips-open-polymer-prediction-2025"],
        "kernel_sources": []
    }
    
    with open(kaggle_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 基本的なnotebook構造をコピー（TODO: 実装）
    print(f"📔 Kaggleノートブックディレクトリ作成完了: {kaggle_dir}")
    print("⚠️  ノートブック(.ipynb)ファイルは手動で作成してください")

def main():
    """メインエントリポイント"""
    parser = argparse.ArgumentParser(description="新しい実験ペアを作成")
    parser.add_argument("name", help="実験名（例: polymer_prediction_v2）")
    parser.add_argument("--description", "-d", help="実験の説明", default="")
    parser.add_argument("--template", "-t", help="ベーステンプレート", 
                       choices=["baseline", "advanced"], default="baseline")
    
    args = parser.parse_args()
    
    # 実験名のバリデーション
    if not args.name.replace('_', '').replace('-', '').isalnum():
        print("❌ エラー: 実験名は英数字とアンダースコア/ハイフンのみ使用可能")
        return
    
    success = create_experiment_pair(args.name, args.description, args.template)
    
    if success:
        print(f"\n🎉 実験ペア '{args.name}' の作成が完了しました！")
    else:
        print(f"\n❌ 実験ペア '{args.name}' の作成に失敗しました")

if __name__ == "__main__":
    main()