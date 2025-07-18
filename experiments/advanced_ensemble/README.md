# NeurIPS Polymer Advanced Ensemble Experiment

## 概要

このディレクトリは、NeurIPS Open Polymer Prediction 2025コンペティション用の高度なアンサンブル実験を管理します。最先端のアンサンブル手法と分子特徴量を組み合わせた高精度予測システムです。

## ファイル構成

```
experiments/advanced_ensemble/
├── README.md                           # このファイル
├── config.yaml                         # 実験設定ファイル
├── scripts/                           # 実行スクリプト
│   ├── local_experiment.py            # メイン実験スクリプト（WandB統合）
│   ├── local_polymer_prediction.py    # ベースライン互換版
│   └── run_experiment.sh              # 実行シェルスクリプト
├── results/                           # 実験結果
│   ├── runs/                         # 個別実験実行結果
│   │   ├── advanced_ensemble_YYYYMMDD_HHMMSS/
│   │   │   ├── metadata.json        # 実験メタデータ
│   │   │   └── catboost_info/      # CatBoostログ
│   │   └── latest -> [最新の実験へのシンボリックリンク]
│   ├── models/                       # 共有訓練済みモデル
│   ├── predictions/                  # 予測結果
│   └── submissions/                  # Kaggle提出ファイル
├── logs/                             # ログファイル
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

# 基本実験（推奨方法）
cd experiments/advanced_ensemble
python scripts/local_experiment.py

# 実験シェルスクリプト使用
./scripts/run_experiment.sh

# 依存関係インストール後実行
./scripts/run_experiment.sh --install

# RDKit含む完全インストール後実行
./scripts/run_experiment.sh --rdkit

# WandB統合実験（オンラインモード）
python scripts/local_experiment.py --online-wandb
```

### 出力

実行後、以下が自動生成されます：

```
experiments/advanced_ensemble/results/runs/advanced_ensemble_YYYYMMDD_HHMMSS/
├── metadata.json        # 実験設定・結果の完全記録
├── catboost_info/      # CatBoostの訓練ログ
└── [その他の実験結果ファイル]

# WandB実行ログ（オフラインモード）
experiments/advanced_ensemble/wandb/offline-run-YYYYMMDD_HHMMSS-xxxxxxxx/
├── files/
│   └── metadata.json   # WandBにアップロードされるメタデータ
├── run-xxxxxxxx.wandb  # 実験データ
└── logs/               # WandBログ
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

### NumPy互換性問題
RDKitとNumPyのバージョン互換性に注意が必要です：
- **問題**: RDKit 2022.9.5はNumPy 1.xでコンパイルされているため、NumPy 2.0以降では動作しません
- **エラー例**: `AttributeError: _ARRAY_API not found`
- **解決方法**:
  ```bash
  # NumPyを1.x系にダウングレード
  pip install "numpy<2"
  # またはuvを使用
  uv pip install "numpy<2"
  ```
- **推奨バージョン**: numpy==1.26.4（2024年1月時点の最新1.x系）

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

## 実験運用の知見

### 📝 Kaggle環境でのRDKit導入

Kaggleノートブックでは、RDKitのインストールが課題となります。以下の方法で解決：

1. **kernel-metadata.jsonの設定**:
   ```json
   {
     "dataset_sources": ["richolson/rdkit-install-whl"],
     ...
   }
   ```
   
2. **ノートブック内でのインストール**:
   ```python
   # RDKitデータセットからwheelファイルをインストール
   import subprocess
   import sys
   import os
   
   # 複数のパスパターンを試す
   rdkit_paths = [
       '/kaggle/input/rdkit-install-whl/rdkit_wheel',
       '/kaggle/input/rdkit-install-whl',
       '/kaggle/input/rdkit-whl',
       '/kaggle/input/rdkit'
   ]
   
   for path in rdkit_paths:
       if os.path.exists(path):
           whl_files = [f for f in os.listdir(path) if f.endswith('.whl')]
           if whl_files:
               subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                                    os.path.join(path, whl_files[0])])
               break
   ```

### 🗂️ 実験結果の管理

新しいディレクトリ構造での管理方法：

1. **実験結果の配置**:
   - 全ての実験結果は`results/runs/`以下に保存
   - 最新の実験は`latest`シンボリックリンクで参照可能
   - 例: `results/runs/advanced_ensemble_20250707_153814/`

2. **WandB統合**:
   ```python
   # オフラインモードで実行（Kaggle環境用）
   wandb_available, wandb_run = init_wandb(offline_mode=True)
   
   # 実験後のアップロード
   cd wandb && wandb sync offline-run-YYYYMMDD_HHMMSS-xxxxxxxx
   ```

3. **実験の再現性**:
   - `metadata.json`に全ての設定と結果を記録
   - 実行時のRDKit可用性も記録
   - ハイパーパラメータと重みも保存

### 🎯 実験実行のベストプラクティス

1. **環境確認**:
   ```bash
   # 仮想環境の有効化
   source .venv/bin/activate
   
   # パッケージ確認
   python -c "import pandas, numpy, sklearn, xgboost, catboost, wandb; print('✅ OK')"
   ```

2. **実験実行**:
   ```bash
   cd experiments/advanced_ensemble
   python scripts/local_experiment.py
   ```

3. **結果の確認**:
   ```bash
   # 最新の実験結果
   ls -la results/runs/latest/
   
   # WandBアップロード
   cd wandb && wandb sync offline-run-*
   ```

### 📊 実行結果例（2025年7月7日）

```
実験名: advanced_ensemble_20250707_153814
実行時間: 467.62秒（約7分47秒）
RDKit: 利用可能（100特徴量）

推定wMAE結果:
- WeightedEnsemble: 0.2953（最良）
- XGBoost: 0.2960
- SimpleEnsemble: 0.2976
- CatBoost: 0.2977

特性別最良モデル:
- Tg: WeightedEnsemble (MAE: 53.13)
- FFV: XGBoost (MAE: 0.0074)
- Tc: CatBoost (MAE: 0.0289)
- Density: XGBoost (MAE: 0.0352)
- Rg: WeightedEnsemble (MAE: 1.677)
```

### 🔧 トラブルシューティング（追加）

1. **ディレクトリ構造の確認**:
   ```bash
   # 実験ディレクトリ構造の確認
   tree experiments/advanced_ensemble -d -L 3
   ```

2. **WandBオフライン同期エラー**:
   ```bash
   # API認証の確認
   wandb login --verify
   
   # 手動でのアップロード
   wandb sync --sync-all wandb/
   ```

3. **メモリ効率化**:
   - 特徴量を段階的に生成
   - バッチ処理の実装
   - 不要な中間データの削除

## 貢献者向けガイドライン

- 新しい実験はこのテンプレートに従って作成
- 実験結果は必ず`results/runs/`に保存
- 設定変更は`config.yaml`で管理
- 重要な変更は`metadata.json`に記録
- Kaggleノートブックとの1対1対応を維持