# NeurIPS Polymer Advanced Ensemble Experiment

## 概要

このディレクトリは、NeurIPS Open Polymer Prediction 2025コンペティション用の高度なアンサンブル実験を管理します。最先端のアンサンブル手法と分子特徴量を組み合わせた高精度予測システムです。

## ファイル構成

```
experiments/neurips_polymer_advanced_ensemble/
├── README.md                           # このファイル
├── config.yaml                         # 実験設定ファイル
├── scripts/                           # 実行スクリプト
│   ├── local_experiment.py            # メイン実験スクリプト
│   ├── local_polymer_prediction.py    # ベースライン互換版
│   └── run_experiment.sh              # 実行シェルスクリプト
├── tests/                             # テスト用スクリプト
│   ├── quick_test.py                   # クイックテスト
│   ├── wandb_test.py                   # WandBテスト
│   └── model_test.py                   # モデル検証テスト
├── experiments_results/               # 実験結果
│   └── advanced_ensemble_[タイムスタンプ]/  # 実行時に自動生成
│       ├── metadata.json             # 実験メタデータ
│       ├── config_used.yaml          # 使用した設定
│       ├── models/                   # 訓練済みモデル
│       ├── predictions/              # 予測結果
│       │   └── submission.csv        # Kaggle提出用ファイル
│       └── logs/                     # 詳細ログ
├── results/                          # 旧形式の結果（移行中）
└── wandb/                            # WandB実験ログ
    └── [WandB実験データ]
```

## 主な機能

### 🧬 高度な分子特徴量生成
- **RDKit完全特徴セット**: 
  - 全記述子（500+特徴量）
  - Morganフィンガープリント（256ビット）
  - MACCSキー（167ビット）
  - カスタムポリマー特徴量
- **特徴量選択**: 
  - 特性別の重要特徴量自動選択
  - 相関分析による冗長性削減

### 🤖 最先端アンサンブルモデル
- **XGBoost**: 特性別最適化済みハイパーパラメータ
- **CatBoost**: カテゴリカル特徴量対応
- **Random Forest**: 安定性重視のベースモデル
- **Gradient Boosting**: 追加の多様性確保
- **K-NN**: 小データセット用補完モデル
- **アンサンブル戦略**: 
  - 加重平均アンサンブル
  - スタッキングアンサンブル（オプション）

### 📊 実験管理
- **自動ディレクトリ作成**: タイムスタンプ付き実験フォルダ
- **包括的メタデータ**: 
  - モデル性能（CV、特性別）
  - 使用設定の完全記録
  - 実行環境情報
- **結果の整理**: 
  - モデル、予測、ログの分離保存
  - 再現可能な実験設定

## 使用方法

### 前提条件

1. **データ準備**: `data/raw/`に以下のファイルを配置
   ```
   data/raw/
   ├── train.csv
   ├── test.csv
   └── sample_submission.csv
   ```

2. **依存関係**: 
   ```bash
   # 必須
   pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn pyyaml
   
   # 強く推奨（高精度予測必須）
   pip install rdkit-pypi
   
   # オプション（実験管理）
   pip install wandb
   ```

### 実行

```bash
# プロジェクトルートから実行
cd neurips-open-polymer-prediction-2025

# 基本実験
./experiments/neurips_polymer_advanced_ensemble/scripts/run_experiment.sh

# 依存関係インストール後実行
./experiments/neurips_polymer_advanced_ensemble/scripts/run_experiment.sh --install

# RDKit含む完全インストール後実行
./experiments/neurips_polymer_advanced_ensemble/scripts/run_experiment.sh --rdkit

# WandB統合実験
./experiments/neurips_polymer_advanced_ensemble/scripts/run_experiment.sh --wandb

# 直接Python実行
python experiments/neurips_polymer_advanced_ensemble/scripts/local_experiment.py
python experiments/neurips_polymer_advanced_ensemble/scripts/local_experiment.py --use-wandb
```

### 出力

実行後、以下が自動生成されます：

```
experiments/neurips_polymer_advanced_ensemble/experiments_results/advanced_ensemble_YYYYMMDD_HHMMSS/
├── metadata.json        # 実験設定・結果の完全記録
├── config_used.yaml     # 使用した設定のコピー
├── models/              # 訓練済みモデル（.pkl形式）
│   ├── Tg_models.pkl
│   ├── e_models.pkl
│   └── ...
├── predictions/         # 予測結果
│   ├── submission.csv   # Kaggle提出用予測ファイル
│   └── cv_predictions.csv # クロスバリデーション予測
└── logs/                # 詳細ログ
    ├── experiment.log   # 実行ログ
    └── feature_importance.json # 特徴量重要度
```

## 設定オプション

`config.yaml`で以下の設定をカスタマイズ可能：

```yaml
experiment:
  name: "advanced_ensemble"
  seed: 42
  cv_folds: 5

features:
  use_rdkit: true
  max_features: 500
  feature_selection: true

models:
  xgboost:
    enabled: true
    n_estimators: 1000
    learning_rate: 0.01
  catboost:
    enabled: true
    iterations: 1000
    learning_rate: 0.03
  # ... その他モデル設定

ensemble:
  method: "weighted_average"  # or "stacking"
  optimize_weights: true
```

## 性能指標

- **評価指標**: wMAE（重み付き平均絶対誤差）
- **検証**: 5-Fold クロスバリデーション
- **特性別評価**: 各ポリマー特性の個別性能
- **期待性能**: 
  - CV wMAE: ~2.0-2.5（ベースラインより改善）
  - 実行時間: 30-60分（ローカル環境、RDKit使用時）

## ログ出力例

```
🚀 実験開始: advanced_ensemble_20250624_150000
✅ 設定ファイル読み込み完了: config.yaml
✅ RDKit利用可能 - 高精度分子特徴量を使用
📂 ローカルデータ読み込み中...
🧬 高度な分子特徴量生成中...
  - RDKit記述子: 500 特徴量
  - Morganフィンガープリント: 256 ビット
  - MACCSキー: 167 ビット
🤖 アンサンブルモデル訓練開始...
  Tg 用の高度なモデル訓練中...
  フォールド 1:
    - XGBoost MAE: 43.567
    - CatBoost MAE: 42.890
    - アンサンブル MAE: 42.234
🎯 推定重み付きMAE: 2.123
💾 モデルと結果を保存中...
⏱️  総実行時間: 45.67 分
```

## トラブルシューティング

### RDKitインストールエラー
```bash
# 代替インストール方法
conda install -c conda-forge rdkit
# または環境を分けて
conda create -n polymer-pred python=3.9 rdkit
conda activate polymer-pred
```

### メモリエラー
- 設定ファイルで特徴量数を削減：`max_features: 200`
- モデルのパラメータを調整：`n_estimators`を削減
- 一部のモデルを無効化：`enabled: false`

### 実行時間が長い
- 並列処理の確認：`n_jobs: -1`
- CVフォールド数の削減：`cv_folds: 3`
- 早期停止の有効化：`early_stopping_rounds: 50`

## 実験の比較・分析

実験結果は`metadata.json`に詳細に記録され、複数実験の比較が可能：

```json
{
  "experiment_name": "advanced_ensemble_20250624_150000",
  "config_file": "config.yaml",
  "rdkit_available": true,
  "feature_stats": {
    "total_features": 923,
    "selected_features": 500
  },
  "model_info": {
    "Tg": {
      "cv_scores": [43.2, 42.8, 43.5, 42.1, 43.0],
      "mean_cv_score": 42.92,
      "ensemble_method": "weighted_average",
      "model_weights": {
        "xgboost": 0.35,
        "catboost": 0.40,
        "random_forest": 0.15,
        "gradient_boosting": 0.10
      }
    }
  },
  "estimated_wmae": 2.123,
  "execution_time_minutes": 45.67
}
```

## Kaggleノートブック変換

ローカル実験をKaggleノートブックに変換：

```bash
# ワークフローディレクトリから実行
cd workflows
python local_to_kaggle.py neurips_polymer_advanced_ensemble
```

## 次のステップ

1. **ハイパーパラメータ最適化**: 
   - Optunaによる自動調整
   - ベイズ最適化の実装

2. **特徴量エンジニアリング改善**: 
   - ドメイン知識に基づく新特徴量
   - 外部化学データベースの活用

3. **アンサンブル手法の高度化**: 
   - ニューラルネットワークベースのスタッキング
   - 動的重み付け戦略

4. **実験自動化**: 
   - グリッドサーチの並列実行
   - 結果の自動分析・レポート生成

## 貢献者向けガイドライン

- 新しい実験はこのテンプレートに従って作成
- 実験結果は必ず`experiments_results/`に保存
- 設定変更は`config.yaml`で管理
- 重要な変更は`metadata.json`に記録