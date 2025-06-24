# 実験-Kaggleノートブック 1対1対応構造設計

## 🎯 目的
ローカル実験とKaggleノートブックの1対1対応を明確にし、管理しやすいディレクトリ構造を実現する

## 📊 現在の問題点
- `experiments/polymer_prediction_baseline` と `kaggle_notebooks/submission/neurips_polymer_advanced_ensemble` の対応が不明確
- 命名規則が統一されていない
- 派生関係がわからない

## 🎯 新しい構造設計

### 基本原則
1. **同じ名前 = 1対1対応**: `experiments/X/` ⟷ `kaggle_notebooks/submission/X/`
2. **明確な命名規則**: 実験内容がわかる名前を使用
3. **双方向同期**: ローカル ⇔ Kaggle の変換を簡単に

### 📁 新ディレクトリ構造

```
kaggle/pjt/neurips-open-polymer-prediction-2025/
├── experiments/                              # ローカル実験環境
│   ├── neurips_polymer_advanced_ensemble/    # 高度なアンサンブルモデル実験
│   │   ├── local_experiment.py              # ローカル実行スクリプト
│   │   ├── config.yaml                      # 実験設定
│   │   ├── results/                         # 実験結果
│   │   │   ├── models/                      # 訓練済みモデル
│   │   │   ├── predictions/                 # 予測結果
│   │   │   └── logs/                        # ログファイル
│   │   ├── wandb/                           # WandB実験ログ
│   │   └── README.md                        # 実験説明
│   │
│   ├── polymer_prediction_baseline/          # ベースラインモデル実験
│   │   ├── local_experiment.py              # 既存スクリプトをリネーム
│   │   ├── config.yaml                      
│   │   ├── results/                         
│   │   ├── wandb/                           # 既存WandBログ
│   │   └── README.md                        
│   │
│   └── template/                            # 新実験用テンプレート
│       ├── local_experiment_template.py
│       ├── config_template.yaml
│       └── README_template.md
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
│   │       ├── kernel-metadata.json               # 新規作成
│   │       └── install_dependencies.py            # 共通
│   │
│   ├── templates/                           # 既存テンプレート
│   └── references/                          # 参考ノートブック
│
├── workflows/                               # 🆕 実験⇔Kaggle変換ツール
│   ├── local_to_kaggle.py                  # ローカル実験 → Kaggleノートブック変換
│   ├── kaggle_to_local.py                  # Kaggleノートブック → ローカル実験変換
│   ├── sync_experiments.py                 # 双方向同期
│   ├── create_new_experiment.py            # 新実験セットアップ
│   └── templates/                          # 変換用テンプレート
│       ├── local_experiment_template.py
│       └── kaggle_notebook_template.ipynb
```

## 🔄 対応表

| 実験名 | ローカルパス | Kaggleパス | 説明 |
|--------|-------------|-----------|------|
| 高度なアンサンブル | `experiments/neurips_polymer_advanced_ensemble/` | `kaggle_notebooks/submission/neurips_polymer_advanced_ensemble/` | RDKit + XGBoost + CatBoost |
| ベースライン | `experiments/polymer_prediction_baseline/` | `kaggle_notebooks/submission/polymer_prediction_baseline/` | 基本的なML手法 |

## 📋 移行計画

### Phase 1: 新構造作成
1. `experiments/neurips_polymer_advanced_ensemble/` フォルダ作成
2. `kaggle_notebooks/submission/polymer_prediction_baseline/` 作成
3. `workflows/` フォルダと変換ツール作成

### Phase 2: 既存ファイル移動
1. 既存実験ファイルの整理とリネーム
2. 設定ファイルの統一
3. READMEファイルの更新

### Phase 3: 変換ツール実装
1. ローカル → Kaggle変換スクリプト
2. Kaggle → ローカル変換スクリプト
3. 双方向同期機能

### Phase 4: ドキュメント更新
1. 全READMEファイルの更新
2. ワークフロー説明書作成
3. 使用例の追加

## 🛠️ 変換ツールの機能

### `local_to_kaggle.py`
- ローカル実験スクリプトをJupyterノートブック形式に変換
- 設定ファイルをノートブック内に埋め込み
- Kaggle用パスに変更
- 依存関係インストールセルを追加

### `kaggle_to_local.py`
- Kaggleノートブックをローカル実行用Pythonスクリプトに変換
- WandB設定の追加
- ローカルデータパスに変更
- 実験結果保存機能の追加

### `sync_experiments.py`
- 両方向の変更を検出して同期
- コンフリクト解決機能
- バックアップ作成

## 📈 期待される効果

1. **明確な対応関係**: 同じ名前 = 同じ実験
2. **簡単な変換**: ワンコマンドでローカル ⇔ Kaggle変換
3. **実験管理の向上**: どの実験がどこにあるか一目瞭然
4. **開発効率向上**: ローカルで開発 → Kaggleで実行の流れが簡単
5. **バージョン管理**: 実験の履歴が追跡しやすい

## 🚀 実装開始

この設計に基づいて実装を開始します。まず新しいディレクトリ構造を作成し、既存ファイルを移動・整理します。