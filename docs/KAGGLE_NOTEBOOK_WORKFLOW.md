# Kaggleノートブック開発・提出ワークフロー

## 概要

NeurIPS Open Polymer Prediction 2025コンペティション用のKaggleノートブック開発から提出までの包括的なワークフローガイドです。自動化ツールの使用・非使用両方に対応しています。

## 開発ワークフロー選択肢

### 🔄 自動化ワークフロー（推奨）
```
Python コード → .ipynb 自動生成 → Kaggle API アップロード → Kaggle実行・提出
```
**メリット**: 効率的、エラー減少、バージョン管理容易

### ✋ 手動ワークフロー
```
Python コード → 手動.ipynb作成 → 手動アップロード → Kaggle実行・提出
```
**メリット**: 細かい制御、学習効果、ツール依存なし

## 必要な環境設定

### 共通設定

#### Kaggle API設定
1. [Kaggleアカウントの設定ページ](https://www.kaggle.com/account)でAPI Tokenをダウンロード
2. `~/.kaggle/kaggle.json` に配置
3. 権限設定: `chmod 600 ~/.kaggle/kaggle.json`

#### プロジェクト環境
```bash
# UVを使用した依存関係管理
uv sync

# または従来のpip
pip install pandas numpy scikit-learn xgboost catboost rdkit-pypi
```

### 自動化ツール使用時の追加設定
```bash
# 自動化ツール用パッケージ
uv add kaggle nbformat
# または
pip install kaggle nbformat
```

## 使用方法

### 1️⃣ 自動化ワークフロー

#### 基本的な使用方法
```bash
# 基本コマンド
python scripts/create_kaggle_notebook.py \
    --input "kaggle_notebooks/templates/complete_baseline_notebook.py" \
    --title "My Baseline Model" \
    --public

# データセット依存がある場合
python scripts/create_kaggle_notebook.py \
    --input "path/to/notebook.py" \
    --title "Advanced Model" \
    --datasets "username/dataset-name" \
    --competitions "neurips-open-polymer-prediction-2025"

# 既存ノートブックの更新
python scripts/create_kaggle_notebook.py \
    --input "path/to/notebook.py" \
    --title "My Baseline Model" \
    --update
```

#### 簡単アップロード（ベースライン）
```bash
# ベースラインノートブックの簡単アップロード
./scripts/upload_baseline.sh
```

### 2️⃣ 手動ワークフロー

#### Step 1: ローカル開発・テスト
```bash
# ローカル実験実行
python experiments/polymer_prediction_baseline/scripts/local_polymer_prediction.py

# クイックテスト
python experiments/polymer_prediction_baseline/tests/quick_test.py
```

#### Step 2: Kaggleノートブック手動作成
1. Kaggle Kernels画面で「New Notebook」作成
2. `kaggle_notebooks/templates/` からコードをコピー
3. 必要に応じてコードを調整・修正
4. セル分割とマークダウン追加

#### Step 3: 手動アップロード・実行
1. Kaggle Kernelで「Save Version」
2. 「Run All」でノートブック実行
3. 「Submit to Competition」で提出

## コード構造とベストプラクティス

### Pythonコードの構造化

#### 自動化ツール用：セクション分割
```python
# ============================================================================
# データ読み込みと前処理
# ============================================================================

# このセクションではデータの読み込みと基本的な前処理を行います
# - CSVファイルの読み込み
# - 欠損値の確認
# - データ型の最適化

import pandas as pd
import numpy as np

# データ読み込み
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
```

#### 手動作成用：コメント活用
```python
"""
=== データ読み込みと前処理 ===
このセクションではデータの読み込みと基本的な前処理を行います
"""

import pandas as pd
import numpy as np

# データ読み込み
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

print(f"訓練データ形状: {train.shape}")
print(f"テストデータ形状: {test.shape}")
```

### Kaggle環境対応コード

#### オフライン実行対応
```python
import sys
import os

# Kaggle環境判定
KAGGLE_ENV = '/kaggle/input' in sys.path[0] if sys.path else False

if KAGGLE_ENV:
    # Kaggle環境でのデータパス
    DATA_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
else:
    # ローカル環境でのデータパス
    DATA_PATH = 'data/raw/'
```

#### 依存関係インストール（Kaggle環境）
```python
# Kaggle環境でのRDKitインストール例
import subprocess
import sys

try:
    import rdkit
    print("✅ RDKit利用可能")
except ImportError:
    print("📦 RDKitをインストール中...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rdkit-pypi'])
    import rdkit
    print("✅ RDKitインストール完了")
```

## ファイル構造

```
kaggle_notebooks/
├── templates/              # 開発用テンプレート
│   ├── complete_baseline_notebook.py    # 完全ベースライン
│   ├── submission_template.py           # 提出用テンプレート
│   └── development/                     # 開発用分割テンプレート
│       ├── eda_template.py              # データ探索
│       ├── feature_engineering_template.py  # 特徴量
│       └── model_comparison_template.py      # モデル比較
├── submission/             # 提出用ノートブック（自動生成）
│   └── neurips_polymer_advanced_ensemble/
│       ├── neurips_polymer_advanced_ensemble.ipynb
│       └── kernel-metadata.json
└── references/             # 参考ノートブック
    ├── neurips-2025-open-polymer-challenge-tutorial.ipynb
    └── open-polymer-prediction-2025.ipynb
```

## 実行環境の選択肢

### ローカル開発環境
```bash
# 完全なローカル実行（WandB統合）
python experiments/polymer_prediction_baseline/scripts/local_polymer_prediction_with_wandb.py

# 基本ローカル実行
python experiments/polymer_prediction_baseline/scripts/local_polymer_prediction.py

# シェルスクリプト実行
./experiments/polymer_prediction_baseline/scripts/run_experiment.sh --install
```

### Kaggle環境
- **CPU環境**: 9時間制限、インターネット無効
- **GPU環境**: 9時間制限、GPU利用可能（必要に応じて）
- **TPU環境**: 特殊用途（このコンペでは通常不要）

## メリット・デメリット比較

### 🔄 自動化ワークフロー

**メリット**:
- ⚡ 高速な開発・デプロイサイクル
- 🚫 コピペエラーの排除
- 📝 一貫したフォーマット
- 🔄 バージョン管理との統合
- 📦 メタデータの自動生成

**デメリット**:
- 🛠️ 初期セットアップが必要
- 🔧 ツール依存
- 📚 学習コストあり

### ✋ 手動ワークフロー

**メリット**:
- 🎯 細かい制御が可能
- 📖 学習効果が高い
- 🆓 ツール依存なし
- 🔍 デバッグが容易

**デメリット**:
- ⏰ 時間がかかる
- 🐛 ヒューマンエラーのリスク
- 🔄 同期の手間
- 📊 バージョン管理の複雑さ

## トラブルシューティング

### 共通問題

1. **Kaggleデータアクセスエラー**
   ```
   解決方法: コンペティション参加確認、データセット存在確認
   ```

2. **依存パッケージエラー**
   ```
   解決方法: pip installまたはKaggle環境での明示的インストール
   ```

3. **メモリ不足エラー**
   ```
   解決方法: データサンプリング、特徴量削減、モデル簡素化
   ```

### 自動化ツール固有の問題

1. **Kaggle API認証エラー**
   ```
   解決方法: ~/.kaggle/kaggle.json の設置と権限確認
   ```

2. **ノートブック名の重複エラー**
   ```
   解決方法: --update オプションを使用するか、異なるタイトルを指定
   ```

3. **nbformat変換エラー**
   ```
   解決方法: pip install nbformat --upgrade
   ```

## 推奨開発フロー

### 初心者向け
1. 手動ワークフローで基本を理解
2. ローカル実験で機能検証
3. 手動でKaggleノートブック作成・提出
4. 慣れてきたら自動化ツールを導入

### 経験者向け
1. 自動化ワークフローを最初から活用
2. テンプレートを拡張・カスタマイズ
3. CI/CDパイプラインと統合
4. 複数バリエーションの並行開発

## 関連リソース

### プロジェクト内ファイル
- `scripts/create_kaggle_notebook.py` - 自動化メインスクリプト
- `scripts/upload_baseline.sh` - 簡単アップロード用
- `kaggle_notebooks/templates/` - 開発用テンプレート
- `experiments/polymer_prediction_baseline/` - ローカル実験環境

### 外部リソース
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Jupyter Notebook Format](https://nbformat.readthedocs.io/)
- [NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)

## 今後の拡張予定

- [ ] ノートブック実行状況の自動監視
- [ ] 提出ファイルの自動ダウンロード
- [ ] 複数バリエーションの一括アップロード
- [ ] テンプレートエンジンの統合
- [ ] CI/CDパイプラインとの統合
- [ ] 手動ワークフロー用のヘルパーツール