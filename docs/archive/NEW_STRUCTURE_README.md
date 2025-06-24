# 🎯 実験-Kaggleノートブック 1対1対応構造

## 📋 新構造の概要

このプロジェクトでは、ローカル実験とKaggleノートブックの1対1対応を実現し、管理しやすい構造に変更しました。

## 🔄 実験-Kaggleノートブック 1対1対応

| 実験名 | ローカル実験パス | Kaggleノートブックパス | 説明 |
|--------|-----------------|-------------------|------|
| **高度なアンサンブル** | `experiments/neurips_polymer_advanced_ensemble/` | `kaggle_notebooks/submission/neurips_polymer_advanced_ensemble/` | RDKit + XGBoost + CatBoost高度アンサンブル |
| **ベースライン** | `experiments/polymer_prediction_baseline/` | `kaggle_notebooks/submission/polymer_prediction_baseline/` | 基本的なML手法ベースライン |

## 🎯 対応の原則

1. **同じ名前 = 同じ実験**: ディレクトリ名が同じなら1対1対応
2. **双方向変換**: ローカル ⇔ Kaggle の変換が可能
3. **設定共有**: 実験設定を両環境で共有
4. **結果同期**: 実験結果の相互参照が可能

## 📁 新ディレクトリ構造

```
neurips-open-polymer-prediction-2025/
├── experiments/                              # ローカル実験環境
│   ├── neurips_polymer_advanced_ensemble/    # 高度なアンサンブルモデル実験
│   │   ├── scripts/local_experiment.py      # ローカル実行スクリプト
│   │   ├── config.yaml                      # 実験設定
│   │   ├── results/                         # 実験結果
│   │   │   ├── models/                      # 訓練済みモデル
│   │   │   ├── predictions/                 # 予測結果
│   │   │   └── logs/                        # ログファイル
│   │   ├── wandb/                           # WandB実験ログ
│   │   └── README.md                        # 実験説明
│   │
│   └── polymer_prediction_baseline/          # ベースラインモデル実験
│       ├── scripts/local_polymer_prediction.py  # 既存スクリプト
│       ├── results/experiments_results/          # 実験結果
│       ├── wandb/                               # 既存WandBログ
│       └── README.md                            # 実験ドキュメント
│
├── kaggle_notebooks/                         # Kaggle提出用
│   ├── submission/
│   │   ├── neurips_polymer_advanced_ensemble/  # ⟷ experiments/neurips_polymer_advanced_ensemble/
│   │   │   ├── neurips_polymer_advanced_ensemble.ipynb
│   │   │   ├── kernel-metadata.json
│   │   │   └── install_dependencies.py
│   │   │
│   │   └── polymer_prediction_baseline/        # ⟷ experiments/polymer_prediction_baseline/
│   │       ├── polymer_prediction_baseline.ipynb  # 新規作成
│   │       └── kernel-metadata.json               # 新規作成
│   │
│   ├── templates/                           # 既存テンプレート
│   └── references/                          # 参考ノートブック
│
└── workflows/                               # 🆕 実験⇔Kaggle変換ツール
    ├── local_to_kaggle.py                  # ローカル実験 → Kaggleノートブック変換
    ├── create_new_experiment.py            # 新実験セットアップ
    └── templates/                          # 変換用テンプレート
```

## 🚀 使用方法

### 新しい実験を作成
```bash
cd workflows
python create_new_experiment.py my_new_experiment --description "新しい実験の説明"
```

### ローカル実験をKaggleノートブックに変換
```bash
cd workflows
python local_to_kaggle.py neurips_polymer_advanced_ensemble
```

### 既存の実験を実行
```bash
# 高度なアンサンブル実験
cd experiments/neurips_polymer_advanced_ensemble
python scripts/local_experiment.py

# ベースライン実験
cd experiments/polymer_prediction_baseline
python scripts/local_polymer_prediction_with_wandb.py
```

## 📊 期待される効果

1. **明確な対応関係**: 同じ名前 = 同じ実験で混乱を回避
2. **簡単な変換**: ワンコマンドでローカル ⇔ Kaggle変換
3. **実験管理の向上**: どの実験がどこにあるか一目瞭然
4. **開発効率向上**: ローカルで開発 → Kaggleで実行の流れが簡単
5. **バージョン管理**: 実験の履歴が追跡しやすい

## 🔧 変換ツール機能

### `local_to_kaggle.py`
- ローカル実験スクリプトをJupyterノートブック形式に変換
- 設定ファイルをノートブック内に埋め込み
- Kaggle用パスに変更
- 依存関係インストールセルを追加

### `create_new_experiment.py`
- 新しい実験ペア（ローカル + Kaggle）を一括作成
- テンプレートベースの設定ファイル生成
- 対応関係を自動で設定

## 📝 移行完了項目

✅ **新ディレクトリ構造作成**
- `experiments/neurips_polymer_advanced_ensemble/` 作成
- `kaggle_notebooks/submission/polymer_prediction_baseline/` 作成
- `workflows/` フォルダと変換ツール作成

✅ **既存ファイル整理**
- 既存実験スクリプトのコピーと配置
- 設定ファイルの作成
- READMEファイルの更新

✅ **変換ツール実装**
- ローカル → Kaggle変換スクリプト
- 新実験作成スクリプト
- メタデータ自動生成機能

## 🎉 完了！

新しい1対1対応構造により、実験管理が大幅に向上しました。これで：

- **ローカルで開発・実験** → **Kaggleで提出・検証** の流れが簡単
- **実験の対応関係が明確** → 混乱や重複作業を回避
- **自動化ツール** → 手作業によるミスを削減

実験の効率的な管理と実行が可能になりました！