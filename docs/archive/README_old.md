# NeurIPS Open Polymer Prediction 2025

このリポジトリはNeurIPS Open Polymer Prediction 2025 Kaggleコンペティションのコードとドキュメントを含んでいます。

## コンペティション概要

NeurIPS Open Polymer Prediction 2025コンペティションは、分子構造データを用いてポリマーの性質を予測することに焦点を当てています。これは化学、材料科学、データサイエンスを組み合わせた機械学習チャレンジです。

## プロジェクト構造

```
neurips-open-polymer-prediction-2025/
├── config/                    # 設定ファイル
│   └── config.yaml           # メイン設定
├── data/                     # データディレクトリ
│   ├── raw/                 # Kaggleからの生データ
│   ├── interim/             # 中間データ
│   ├── processed/           # 最終処理済みデータ
│   └── external/            # 外部データソース
├── docs/                    # ドキュメント
│   └── CODE_COMPETITION_GUIDE.md  # コードコンペ完全ガイド
├── experiments/             # ローカル実験管理
│   └── polymer_prediction_baseline/  # ベースライン実験
│       ├── local_polymer_prediction.py  # ローカル実行スクリプト
│       ├── run_experiment.sh           # 実験実行スクリプト
│       └── README.md                   # 実験ドキュメント
├── kaggle_notebooks/        # Kaggle環境専用ノートブック
│   ├── references/         # 参考ノートブック
│   │   ├── neurips-2025-open-polymer-challenge-tutorial.ipynb
│   │   └── open-polymer-prediction-2025.ipynb
│   ├── submission/         # 提出用ノートブック
│   │   └── neurips_polymer_advanced_ensemble/
│   │       ├── neurips_polymer_advanced_ensemble.ipynb
│   │       └── kernel-metadata.json
│   └── templates/          # 再利用可能テンプレート
│       ├── development/    # 開発用テンプレート
│       │   ├── eda_template.py          # データ探索
│       │   ├── feature_engineering_template.py  # 特徴量エンジニアリング
│       │   ├── model_comparison_template.py     # モデル比較
│       │   └── README.md                        # テンプレート使用方法
│       ├── baseline_submission.py       # ベースライン提出用
│       ├── complete_baseline_notebook.py # 完全ベースライン
│       └── submission_template.py       # Kaggle提出テンプレート
├── models/                  # モデル成果物
│   ├── trained/            # 訓練済みモデル
│   ├── checkpoints/        # モデルチェックポイント
│   └── submissions/        # 提出ファイル
├── reports/               # 分析レポート
│   ├── figures/          # 生成された図表
│   └── final/            # 最終レポート
├── scripts/              # ユーティリティスクリプト
│   ├── baseline_model.py          # ベースラインモデル
│   ├── prepare_kaggle_dataset.py  # Kaggle用モデル準備
│   └── train_model.py            # モデル訓練
├── src/                  # ソースコード
│   ├── data/            # データ処理
│   ├── features/        # 特徴量エンジニアリング
│   ├── models/          # モデル定義
│   ├── utils/           # ユーティリティ関数
│   └── visualization/   # プロット関数
├── tests/               # ユニットテスト
├── pyproject.toml       # UV依存関係管理
├── uv.lock             # 依存関係ロックファイル
└── Makefile            # 開発用コマンド
```

## はじめに

### 前提条件

- Python 3.8+
- uv (超高速Pythonパッケージマネージャー)

### インストール

1. uvをインストール:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. このリポジトリをクローン:
   ```bash
   git clone <repository-url>
   cd neurips-open-polymer-prediction-2025
   ```

3. uvで仮想環境を作成し、依存関係をインストール:
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/macOS
   # または .venv\Scripts\activate  # Windows
   uv sync
   ```

   または、Makefileを使用:
   ```bash
   make setup
   ```

### データセットアップ

1. Kaggleからコンペティションデータをダウンロード
2. データファイルを `data/raw/` に配置
3. データ準備を実行:
   ```bash
   uv run make data
   ```
   または
   ```bash
   make data
   ```

## 使用方法

### 開発ワークフロー

#### ローカル開発フェーズ
1. **ベースライン実験**: `experiments/polymer_prediction_baseline/run_experiment.sh` でベースライン実行
2. **探索的データ分析**: `kaggle_notebooks/templates/development/eda_template.py` から開始
3. **特徴量エンジニアリング**: `kaggle_notebooks/templates/development/feature_engineering_template.py` で特徴量を実装
4. **モデル比較**: `kaggle_notebooks/templates/development/model_comparison_template.py` で複数モデルを評価
5. **訓練**: `uv run scripts/train_model.py` または `make train` を使用

#### コードコンペ提出フェーズ
1. **ノートブック自動生成**: `scripts/create_kaggle_notebook.py` でPythonコードから.ipynb生成
2. **Kaggle自動アップロード**: Kaggle APIで直接ノートブックをアップロード
3. **Kaggle実行・提出**: アップロードされたノートブックをKaggleで実行・提出

```bash
# 簡単アップロード
./scripts/upload_baseline.sh

# カスタマイズアップロード
python scripts/create_kaggle_notebook.py \
    --input "kaggle_notebooks/templates/complete_baseline_notebook.py" \
    --title "My Baseline Model" \
    --public
```

### 利用可能なコマンド

```bash
make help          # 利用可能なコマンドを表示
make setup         # uvを使用した開発環境のセットアップ
make clean         # 一時ファイルのクリーンアップ
make lint          # コードリンティングの実行
make test          # ユニットテストの実行
make data          # データのダウンロードと準備
make train         # モデルの訓練
make submit        # 提出ファイルの生成
```

### uvコマンドの直接使用

```bash
uv add <package>           # パッケージを追加
uv remove <package>        # パッケージを削除
uv run <command>           # 仮想環境でコマンドを実行
uv sync                    # pyproject.tomlから依存関係を同期
uv lock                    # uv.lockファイルを更新
```

## 主要ライブラリ

- **データ処理**: pandas, numpy
- **機械学習**: scikit-learn, torch
- **化学**: rdkit-pypi (分子記述子用)
- **可視化**: matplotlib, seaborn
- **最適化**: optuna
- **実験管理**: wandb

## 実験管理 (WandB)

このプロジェクトではWeights & Biases (WandB)を使用して実験を管理します。

### WandBセットアップ

1. WandBアカウントを作成: https://wandb.ai/
2. APIキーを設定:
   ```bash
   uv run wandb login
   ```

### WandB使用方法

- **実験追跡**: すべての訓練実行、ハイパーパラメータ、メトリクスを自動記録
- **モデル比較**: 異なるモデルや設定の性能を視覚的に比較
- **アーティファクト管理**: 訓練済みモデルとデータセットのバージョン管理

### 基本的な使用例

```python
import wandb

# 実験開始
wandb.init(project="neurips-polymer-prediction")

# ハイパーパラメータをログ
wandb.config.update({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
})

# メトリクスをログ
wandb.log({"loss": loss, "accuracy": acc})

# モデルを保存
wandb.save("model.pth")
```

## コンペティション戦略

1. **データ理解**: ポリマー構造データと目的変数の性質を分析
2. **特徴量エンジニアリング**: 分子記述子とフィンガープリントを抽出
3. **モデル選択**: 様々なML手法を比較（RF、XGBoost、ニューラルネットワーク）
4. **検証**: 堅牢なクロスバリデーション戦略を実装
5. **アンサンブル**: より良いパフォーマンスのために複数モデルを組み合わせ
6. **実験管理**: WandBを活用した系統的な実験追跡と結果分析

## コードコンペティション要件

このコンペティションはコードコンペティション形式です：

### 提出要件
- **Kaggleノートブック**経由での提出必須
- **CPU/GPUノートブック**: 最大9時間の実行時間
- **インターネットアクセス**: 無効
- **外部データ**: 自由に利用可能な公開データ・事前訓練モデルは使用可
- **提出ファイル名**: `submission.csv` 必須

### 開発戦略
1. **ローカル開発**: uvとvenv環境で迅速な実験
2. **テンプレート活用**: `kaggle_notebooks/` の各種テンプレートを使用
3. **モデル事前訓練**: ローカルで訓練したモデルをKaggleデータセットとしてアップロード
4. **オフライン対応**: インターネット不要な実装
5. **実行時間最適化**: 9時間制限内での効率的な処理

### コードコンペ最適化機能
- **submission_template.py**: Kaggleノートブック用の包括的テンプレート
- **prepare_kaggle_dataset.py**: 訓練済みモデルのパッケージ化スクリプト
- **development templates**: EDA、特徴量エンジニアリング、モデル比較用テンプレート
- **メモリ最適化**: データ型最適化とメモリ使用量削減機能
- **実行時間管理**: 関数実行時間計測とタイムアウト対策

## 注記

- Kaggleノートブック用のオフライン対応コードを`kaggle_notebooks/`に準備
- 実行時間を考慮した効率的なモデル選択
- 提出ファイル名は必ず`submission.csv`とすること
- WandBなどのオンラインサービスはローカル開発時のみ使用
- コードコンペ用テンプレートは実用的でそのまま使用可能

## ライセンス

MIT License