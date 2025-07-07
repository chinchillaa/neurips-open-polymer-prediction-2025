# NeurIPS Open Polymer Prediction 2025

Kaggleコンペティション「NeurIPS Open Polymer Prediction 2025」のソリューションリポジトリです。

## 🏆 コンペティション概要

- **目的**: 分子構造（SMILES）からポリマーの5つの物性を予測
- **タイプ**: コードコンペティション（9時間制限、オフライン実行）
- **評価指標**: 重み付き平均絶対誤差（wMAE）
- **ターゲット特性**: 
  - Tg（ガラス転移温度）
  - FFV（自由体積分率）
  - Tc（結晶化温度）
  - Density（密度）
  - Rg（回転半径）

## 🚀 クイックスタート

### 環境セットアップ
```bash
# uvのインストール（高速Pythonパッケージマネージャー）
curl -LsSf https://astral.sh/uv/install.sh | sh

# リポジトリのクローンと依存関係インストール
git clone <repository-url>
cd neurips-open-polymer-prediction-2025
uv venv
source .venv/bin/activate  # Linux/macOS
uv sync
```

### データ準備
```bash
# Kaggleからデータダウンロード
kaggle competitions download -c neurips-open-polymer-prediction-2025
unzip neurips-open-polymer-prediction-2025.zip -d data/raw/
```

## 📁 プロジェクト構造

```
neurips-open-polymer-prediction-2025/
├── data/                          # データファイル
│   ├── raw/                      # 生データ（train.csv, test.csv等）
│   ├── processed/                # 前処理済みデータ
│   └── external/                 # 外部データ
│
├── src/                           # ソースコード
│   ├── data/                     # データ処理モジュール
│   ├── features/                 # 特徴量エンジニアリング
│   ├── models/                   # モデル定義
│   ├── utils/                    # ユーティリティ関数
│   └── visualization/            # 可視化関連
│
├── experiments/                   # 実験管理
│   ├── baseline/                 # ベースライン実験
│   │   ├── scripts/             # 実験スクリプト
│   │   ├── config.yaml          # 実験設定
│   │   ├── results/             # 実験結果
│   │   │   ├── runs/           # 実行ごとの結果
│   │   │   ├── models/         # 訓練済みモデル
│   │   │   └── submissions/    # 提出ファイル
│   │   └── logs/               # ログファイル
│   │
│   ├── advanced_ensemble/        # アドバンスドアンサンブル実験
│   │   └── （同様の構造）
│   │
│   └── archive/                  # 過去の実験
│
├── notebooks/                     # Jupyter/Kaggleノートブック
│   ├── development/              # 開発・分析用ノートブック
│   │   └── templates/           # テンプレート
│   ├── kaggle/                  # Kaggle提出用
│   │   ├── active/             # 現在使用中のノートブック
│   │   │   ├── advanced_ensemble_v9/
│   │   │   └── baseline/
│   │   └── archive/            # 過去のバージョン
│   └── references/              # 参考ノートブック
│
├── models/                        # 共有モデル
│   ├── checkpoints/             # チェックポイント
│   └── pretrained/              # 事前学習済みモデル
│
├── docs/                          # ドキュメント
│   ├── guides/                   # 使い方ガイド
│   ├── experiments/              # 実験の説明
│   └── archive/                  # 古い文書
│
├── scripts/                       # 共通スクリプト
├── tests/                         # テストコード
├── config/                        # 設定ファイル
├── workflows/                     # ワークフロースクリプト
└── .artifacts/                    # 一時ファイル（gitignore対象）
    ├── wandb/                    # WandB関連
    ├── catboost_info/            # CatBoost関連
    └── tmp/                      # その他の一時ファイル
```

詳細は[STRUCTURE.md](STRUCTURE.md)を参照してください。

## 🔄 実験-Kaggleノートブック 1対1対応

| 実験名 | ローカル実験パス | Kaggleノートブックパス | 説明 |
|--------|-----------------|-------------------|------|
| **高度なアンサンブル** | `experiments/advanced_ensemble/` | `notebooks/kaggle/active/advanced_ensemble_v9/` | RDKit + XGBoost + CatBoost |
| **ベースライン** | `experiments/baseline/` | `notebooks/kaggle/active/baseline/` | 基本特徴量 + XGBoost |

### 対応の原則
- **同じ名前 = 同じ実験**: ディレクトリ名が一致
- **双方向変換**: ローカル ⇔ Kaggle の自動変換
- **設定共有**: `config.yaml`で実験設定を統一管理

## 🛠️ 主要コマンド

### 新しい実験の作成
```bash
cd workflows
python create_new_experiment.py my_new_experiment --description "新しい実験の説明"
```

### ローカル実験の実行
```bash
cd experiments/advanced_ensemble
python scripts/local_experiment.py
```

### Kaggleノートブックへの変換
```bash
cd workflows
python local_to_kaggle.py advanced_ensemble
```

### Kaggleへのアップロード
```bash
cd notebooks/kaggle/active/advanced_ensemble_v9
kaggle kernels push -p .
```

## 📊 現在の成果

### 提出済みノートブック
1. **[高度なアンサンブル V6](https://www.kaggle.com/code/tgwstr/neurips-polymer-advanced-ensemble-v6)**
   - RDKit分子記述子 + 複数アルゴリズムのアンサンブル
   - RDKit install whlデータセット使用

2. **[ベースライン V2](https://www.kaggle.com/code/tgwstr/polymer-prediction-baseline)**
   - 基本的な文字列特徴量 + XGBoost
   - 軽量・高速実行

## 🔧 技術スタック

- **言語**: Python 3.9+
- **パッケージ管理**: uv (超高速Pythonパッケージマネージャー)
- **主要ライブラリ**:
  - RDKit: 分子記述子計算
  - XGBoost, CatBoost: 勾配ブースティング
  - scikit-learn: 機械学習全般
  - WandB: 実験管理（ローカル実験用）

## 📝 開発ガイドライン

### コード規約
- **コメント**: 日本語優先（Kaggle提出用は英語も可）
- **命名規則**: 実験名は内容が分かる名前を使用
- **バージョン管理**: 実験毎にGitブランチを作成

### ワークフロー
1. **ローカル開発**: `experiments/`で実験開発・検証
2. **Kaggle変換**: `workflows/`のツールで自動変換
3. **提出**: Kaggle APIでアップロード・実行確認

## 🤝 貢献方法

1. Issueで議論
2. ブランチを作成（`feature/実験名`）
3. 実験実装とテスト
4. Pull Request作成

## 📚 参考資料

- [コンペティション公式ページ](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
- [docs/CODE_COMPETITION_GUIDE.md](docs/CODE_COMPETITION_GUIDE.md) - コードコンペ攻略ガイド
- [docs/KAGGLE_NOTEBOOK_WORKFLOW.md](docs/KAGGLE_NOTEBOOK_WORKFLOW.md) - Kaggleワークフロー詳細

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。