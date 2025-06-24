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

## 📁 プロジェクト構造（1対1対応設計）

```
neurips-open-polymer-prediction-2025/
├── experiments/                       # ローカル実験環境
│   ├── neurips_polymer_advanced_ensemble/  # ⟷ kaggle_notebooks/submission/neurips_polymer_advanced_ensemble/
│   │   ├── config.yaml               # 実験設定
│   │   ├── scripts/                  # ローカル実行スクリプト
│   │   ├── results/                  # 実験結果
│   │   └── README.md                 # 実験説明
│   │
│   └── polymer_prediction_baseline/   # ⟷ kaggle_notebooks/submission/polymer_prediction_baseline/
│       ├── config.yaml
│       ├── scripts/
│       ├── results/
│       └── README.md
│
├── kaggle_notebooks/                  # Kaggle提出用ノートブック
│   ├── submission/                    # 提出用（experiments/と1対1対応）
│   │   ├── neurips_polymer_advanced_ensemble/
│   │   │   ├── neurips_polymer_advanced_ensemble.ipynb
│   │   │   ├── kernel-metadata.json
│   │   │   └── install_dependencies.py
│   │   │
│   │   └── polymer_prediction_baseline/
│   │       ├── polymer_prediction_baseline.ipynb
│   │       └── kernel-metadata.json
│   │
│   ├── templates/                     # 再利用可能テンプレート
│   └── references/                    # 参考ノートブック
│
├── workflows/                         # 実験⇔Kaggle変換ツール
│   ├── local_to_kaggle.py            # ローカル → Kaggle変換
│   └── create_new_experiment.py      # 新実験セットアップ
│
├── data/                             # データディレクトリ
│   ├── raw/                          # 生データ
│   ├── processed/                    # 前処理済みデータ
│   └── external/                     # 外部データ
│
├── models/                           # モデル成果物
├── scripts/                          # ユーティリティスクリプト
├── src/                              # ソースコード
├── docs/                             # ドキュメント
└── tests/                            # テストコード
```

## 🔄 実験-Kaggleノートブック 1対1対応

| 実験名 | ローカル実験パス | Kaggleノートブックパス | 説明 |
|--------|-----------------|-------------------|------|
| **高度なアンサンブル** | `experiments/neurips_polymer_advanced_ensemble/` | `kaggle_notebooks/submission/neurips_polymer_advanced_ensemble/` | RDKit + XGBoost + CatBoost |
| **ベースライン** | `experiments/polymer_prediction_baseline/` | `kaggle_notebooks/submission/polymer_prediction_baseline/` | 基本特徴量 + XGBoost |

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
cd experiments/neurips_polymer_advanced_ensemble
python scripts/local_experiment.py
```

### Kaggleノートブックへの変換
```bash
cd workflows
python local_to_kaggle.py neurips_polymer_advanced_ensemble
```

### Kaggleへのアップロード
```bash
cd kaggle_notebooks/submission/neurips_polymer_advanced_ensemble
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