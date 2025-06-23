# Development Templates

このディレクトリには、コードコンペティション開発用のテンプレートが含まれています。

## テンプレート一覧

### 1. EDA Template (`eda_template.py`)
**データ探索と分析のための包括的フレームワーク**

- 基本的なデータ情報の表示
- 欠損値とデータ品質の分析
- ターゲット変数の分布分析
- SMILES文字列の特徴分析
- 特徴量間の相関分析

```bash
python eda_template.py
```

### 2. Feature Engineering Template (`feature_engineering_template.py`)
**ポリマー分子データ特有の特徴量抽出**

- 基本分子特徴量（原子数、結合数）
- 高度分子特徴量（複雑度指標）
- ポリマー特有特徴量（主鎖、側鎖分析）
- 統計的特徴量（比率、集約統計）
- パイプライン化されたFeatureEngineerクラス

```python
from feature_engineering_template import PolymerFeatureEngineer

fe = PolymerFeatureEngineer()
train_features = fe.fit_transform(train_df)
test_features = fe.transform(test_df)
```

### 3. Model Comparison Template (`model_comparison_template.py`)
**複数機械学習モデルの比較評価**

- 基本モデル（Ridge、RandomForest、GradientBoosting等）
- 高度モデル（XGBoost、LightGBM、CatBoost）
- クロスバリデーション評価
- ハイパーパラメータ最適化
- アンサンブルモデル作成

```python
from model_comparison_template import run_model_comparison

results = run_model_comparison(train_df, feature_cols, target_cols)
```

## 使用方法

### Step 1: データ探索
```bash
cd kaggle_notebooks/development
python eda_template.py
```

### Step 2: 特徴量エンジニアリング
```python
# Jupyter環境または.pyファイルで
from feature_engineering_template import PolymerFeatureEngineer

# データ読み込み
train = pd.read_csv('../../data/raw/train.csv')
test = pd.read_csv('../../data/raw/test.csv')

# 特徴量作成
fe = PolymerFeatureEngineer()
train_features = fe.fit_transform(train)
test_features = fe.transform(test)

print(f"Created {len(train_features.columns)} features")
```

### Step 3: モデル比較
```python
from model_comparison_template import run_model_comparison

# ターゲット列を指定（実際のデータに合わせて調整）
target_cols = ['target1', 'target2']  # 実際の列名に変更
feature_cols = [col for col in train_features.columns]

# モデル比較実行
results = run_model_comparison(train_features, feature_cols, target_cols)
```

## 注意事項

1. **データパス**: テンプレート内のデータパスは実際のデータ構造に合わせて調整してください
2. **ターゲット列**: 実際のコンペティションデータのターゲット列名に変更してください
3. **メモリ管理**: 大きなデータセットの場合は`reduce_mem_usage()`関数を活用してください
4. **実行時間**: Kaggleの9時間制限を考慮して、必要に応じてモデル数や計算量を調整してください

## Kaggle移植

これらのテンプレートで開発したコードは、`../templates/submission_template.py`を参考にしてKaggleノートブック形式に移植してください。

## カスタマイズ

- **新しい特徴量**: `feature_engineering_template.py`にドメイン固有の特徴量を追加
- **新しいモデル**: `model_comparison_template.py`に新しいアルゴリズムを追加
- **評価指標**: コンペティションの評価指標に合わせてメトリクスを調整

## トラブルシューティング

- **ImportError**: 必要なライブラリが不足している場合は`uv add <package>`でインストール
- **MemoryError**: `reduce_mem_usage()`関数の活用やデータのチャンク処理を検討
- **実行時間**: 計算量の多い処理は並列化やサンプリングで最適化