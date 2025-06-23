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
├── experiments/             # 実験追跡
├── kaggle_submission_package/  # Kaggle提出ファイル
├── kaggle_upload/           # Kaggleデータセットアップロード用ファイル
├── models/                  # モデル成果物
│   ├── trained/            # 訓練済みモデル
│   ├── checkpoints/        # モデルチェックポイント
│   ├── kaggle_dataset/     # Kaggleデータセット用モデル
│   └── submissions/        # 提出ファイル
├── notebooks/              # Jupyterノートブック
│   ├── exploratory/       # EDAノートブック
│   ├── modeling/          # モデル開発
│   └── evaluation/        # モデル評価
├── reports/               # 分析レポート
│   ├── figures/          # 生成された図表
│   └── final/            # 最終レポート
├── scripts/              # ユーティリティスクリプト
├── src/                  # ソースコード
│   ├── data/            # データ処理
│   ├── features/        # 特徴量エンジニアリング
│   ├── models/          # モデル定義
│   ├── utils/           # ユーティリティ関数
│   └── visualization/   # プロット関数
└── tests/               # ユニットテスト
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

1. **探索的データ分析**: `notebooks/exploratory/` のノートブックから開始
2. **特徴量エンジニアリング**: `src/features/` で特徴量を実装
3. **モデル開発**: `src/models/` でモデルを作成
4. **訓練**: `uv run scripts/train_model.py` または `make train` を使用
5. **評価**: `notebooks/evaluation/` で結果を分析
6. **提出**: `make submit` で提出ファイルを生成

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
2. **ノートブック移植**: 完成したコードをKaggleノートブックに移植
3. **オフライン対応**: インターネット不要な実装
4. **実行時間最適化**: 9時間制限内での効率的な処理

## 注記

- Kaggleノートブック用のオフライン対応コードを`notebooks/`に準備
- 実行時間を考慮した効率的なモデル選択
- 提出ファイル名は必ず`submission.csv`とすること
- WandBなどのオンラインサービスはローカル開発時のみ使用

## ライセンス

MIT License