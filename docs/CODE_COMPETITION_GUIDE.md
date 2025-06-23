# コードコンペティション完全ガイド

## 概要

このガイドは、KaggleでのNeurIPS Open Polymer Prediction 2025コードコンペティションに参加するための包括的な手順を提供します。

## コンペティション形式

- **プラットフォーム**: Kaggle Notebooks
- **実行時間**: 最大9時間
- **インターネットアクセス**: 実行中は無効
- **必要な出力**: `submission.csv`
- **外部データ**: 許可（公開データセット、事前訓練済みモデル）

## 開発ワークフロー

### フェーズ1: ローカル開発

1. **環境セットアップ**
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

2. **探索的データ分析**
   ```bash
   python kaggle_notebooks/templates/development/eda_template.py
   ```

3. **特徴量エンジニアリング**
   ```bash
   python kaggle_notebooks/templates/development/feature_engineering_template.py
   ```

4. **モデル比較**
   ```bash
   python kaggle_notebooks/templates/development/model_comparison_template.py
   ```

5. **モデル訓練**
   ```bash
   make train
   # または
   uv run scripts/train_model.py
   ```

### フェーズ2: Kaggle準備

1. **アップロード用モデル準備**
   ```bash
   python scripts/prepare_kaggle_dataset.py
   ```

2. **モデルをKaggleデータセットとしてアップロード**
   ```bash
   cd kaggle_upload
   kaggle datasets create -p .
   ```

3. **提出用ノートブック作成**
   - `kaggle_notebooks/templates/submission_template.py` をコピー
   - 特定のモデルと特徴量に合わせて調整
   - Kaggleでノートブック形式に変換

### フェーズ3: 最終提出

1. **新しいKaggleノートブック作成**
2. **必要なデータセット追加**:
   - コンペティションデータ: `neurips-open-polymer-prediction-2025`
   - あなたのモデル: `your-username/neurips-polymer-models`
3. **コードの貼り付けと調整**
4. **実行テスト**（バージョン保存して実行）
5. **コンペティションに提出**

## コードテンプレート

### 1. EDAテンプレート (`eda_template.py`)
- データ読み込みと基本統計
- 欠損値分析
- 目的変数の分布
- SMILES文字列分析
- 特徴量相関分析

### 2. 特徴量エンジニアリングテンプレート (`feature_engineering_template.py`)
- 基本分子特徴量（原子数、結合数）
- 高度分子特徴量（複雑度指標）
- ポリマー固有特徴量（主鎖、側鎖）
- 統計的特徴量（比率、集約）

### 3. モデル比較テンプレート (`model_comparison_template.py`)
- 複数アルゴリズム比較
- クロスバリデーション評価
- ハイパーパラメータ最適化
- アンサンブル作成
- 性能追跡

### 4. 提出テンプレート (`submission_template.py`)
- データ読み込みから提出までの完全パイプライン
- メモリ最適化関数
- 実行時間監視
- エラーハンドリングとフォールバック

## 性能最適化

### メモリ最適化
```python
def reduce_mem_usage(df):
    """データフレームのメモリ使用量を削減"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    return df
```

### 実行時間監視
```python
def timer(func):
    """関数実行時間を計測するデコレータ"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} は {end - start:.2f} 秒かかりました")
        return result
    return wrapper
```

## よくある問題と解決策

### 問題: カーネルタイムアウト（9時間超過）
**解決策:**
- モデル複雑度を削減（推定器数減少、シンプルなモデル）
- 特徴量を減らす（特徴量選択）
- 早期停止を実装
- スクラッチ訓練ではなく事前訓練済みモデルを使用

### 問題: メモリエラー
**解決策:**
- `reduce_mem_usage()` 関数を使用
- データをチャンクで処理
- 未使用変数を削除（`del variable_name`）
- 全データ読み込みではなくジェネレータを使用

### 問題: インポートエラー
**解決策:**
- Kaggle環境ドキュメントを確認
- 代替ライブラリを使用（例：`xgboost`の代わりに`lightgbm`）
- フォールバック手法を実装

### 問題: ファイルパスエラー
**解決策:**
- 絶対パスを使用：`/kaggle/input/dataset-name/`
- データセット名を再確認
- 必要なファイルがすべて含まれているか確認

## ベストプラクティス

### 1. コード構成
- モジュール化された適切にドキュメント化されたコード
- コード重複を避けるため関数を使用
- 堅牢性のためエラーハンドリングを実装

### 2. モデル戦略
- シンプルなベースラインモデルから開始
- 段階的に複雑度を増加
- 常に動作するフォールバック解決策を持つ

### 3. 特徴量エンジニアリング
- ドメイン固有特徴量に焦点（分子記述子）
- 相互作用特徴量を作成
- 次元削減のため特徴量選択を使用

### 4. 検証戦略
- 堅牢なクロスバリデーションを使用
- 過学習を監視
- アウトオブフォールド予測で検証

### 5. 提出戦略
- 複数のモデルバージョンをテスト
- 何が有効かを追跡
- 最良のクロスバリデーションスコアを提出

## 最終チェックリスト

最終提出前に：

- [ ] コードが最初から最後まで正常に実行される
- [ ] 実行時間が9時間未満
- [ ] 出力ファイル名が`submission.csv`
- [ ] 提出形式がサンプル提出と一致
- [ ] コードにインターネット依存がない
- [ ] 必要なデータセットがすべてノートブックに追加済み
- [ ] エッジケース用のエラーハンドリングが実装済み
- [ ] メモリ使用量が最適化済み
- [ ] コードが適切にドキュメント化され整理済み

## トラブルシューティング

ノートブックが失敗した場合：

1. **ログを確認**して具体的なエラーメッセージを見る
2. **複雑度を削減**（特徴量減少、シンプルなモデル）
3. **メモリ最適化を追加**（パイプライン全体で）
4. **フォールバックを実装**（問題発生時用）
5. **段階的にテスト** - セクションをコメントアウトしてテスト

## 実験管理の活用

### ローカル実験環境
```bash
# WandB統合実験
python experiments/polymer_prediction_baseline/scripts/local_polymer_prediction_with_wandb.py

# 基本実験
python experiments/polymer_prediction_baseline/scripts/local_polymer_prediction.py

# クイックテスト
python experiments/polymer_prediction_baseline/tests/quick_test.py
```

### 実験結果の分析
- WandBダッシュボードでモデル性能比較
- 実験メタデータによる再現性確保
- Cross-Validation結果の詳細分析

## Kaggle固有の注意点

### 1. オフライン実行環境
```python
# Kaggle環境判定
import sys
KAGGLE_ENV = '/kaggle/input' in sys.path[0] if sys.path else False

if KAGGLE_ENV:
    DATA_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
else:
    DATA_PATH = 'data/raw/'
```

### 2. 依存関係管理
```python
# RDKitインストール例
import subprocess
import sys

try:
    from rdkit import Chem
    print("✅ RDKit利用可能")
except ImportError:
    print("📦 RDKitをインストール中...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rdkit-pypi'])
    from rdkit import Chem
    print("✅ RDKitインストール完了")
```

### 3. 実行時間管理
```python
import time
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("実行時間制限に達しました")

# 8時間でタイムアウト設定（9時間制限の余裕を持たせて）
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(8 * 60 * 60)  # 8時間

try:
    # メイン処理
    main_pipeline()
finally:
    signal.alarm(0)  # タイマー解除
```

## リソース

- [Kaggle Notebooks ドキュメント](https://www.kaggle.com/docs/notebooks)
- [コンペティション議論フォーラム](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion)
- [Kaggle Learn コース](https://www.kaggle.com/learn)

## 関連ファイル

プロジェクト内の関連ファイル：
- `experiments/polymer_prediction_baseline/` - ローカル実験環境
- `kaggle_notebooks/templates/` - 開発用テンプレート
- `scripts/create_kaggle_notebook.py` - 自動化ツール
- `docs/KAGGLE_NOTEBOOK_WORKFLOW.md` - ワークフロー詳細

## お問い合わせ

このガイドやプロジェクト構造に関する質問は、メインのREADME.mdを参照するか、リポジトリにissueを作成してください。