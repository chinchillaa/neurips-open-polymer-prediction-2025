# Polymer Prediction Baseline Experiment

## 概要

このディレクトリは、NeurIPS Open Polymer Prediction 2025コンペティション用のベースライン実験を管理します。Kaggle環境専用のノートブックをローカル環境で実行できるように変換したスクリプトが含まれています。

## ファイル構成

```
experiments/polymer_prediction_baseline/
├── README.md                           # このファイル
├── scripts/                           # 実行スクリプト
│   ├── local_polymer_prediction.py    # ローカル実行用メインスクリプト
│   ├── local_polymer_prediction_with_wandb.py  # WandB統合版
│   └── run_experiment.sh              # 実行シェルスクリプト
├── tests/                             # テスト用スクリプト
│   ├── quick_test.py                   # クイックテスト
│   ├── wandb_test.py                   # WandBテスト
│   └── online_wandb_test.py           # オンラインWandBテスト
├── experiments_results/               # 実験結果
│   └── [実験結果フォルダ]/            # 実行時に自動生成
│       ├── metadata.json             # 実験メタデータ
│       └── submission.csv            # 予測結果
└── wandb/                            # WandB実験ログ
    └── [WandB実験データ]
```

## 主な機能

### 🧬 分子特徴量生成
- **RDKit利用時**: 500+ 化学的特徴量（記述子、フィンガープリント）
- **RDKit非利用時**: 16種類の基本SMILES特徴量（フォールバック）

### 🤖 高度なアンサンブルモデル
- **XGBoost**: 各特性別最適化ハイパーパラメータ
- **CatBoost**: 高性能勾配ブースティング
- **Random Forest**: アンサンブルベース
- **Gradient Boosting**: 追加勾配ブースティング
- **K-NN**: 小データセット用

### 📊 実験管理
- **自動ディレクトリ作成**: タイムスタンプ付き実験フォルダ
- **メタデータ保存**: モデル性能、設定、実行時間
- **結果保存**: 予測ファイル、CVスコア

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
   pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn
   
   # 推奨（高精度予測用）
   pip install rdkit-pypi
   ```

### 実行

```bash
# プロジェクトルートから実行
cd neurips-open-polymer-prediction-2025

# 基本実験
python experiments/polymer_prediction_baseline/scripts/local_polymer_prediction.py

# WandB統合実験
python experiments/polymer_prediction_baseline/scripts/local_polymer_prediction_with_wandb.py

# シェルスクリプト実行
./experiments/polymer_prediction_baseline/scripts/run_experiment.sh
```

### 出力

実行後、以下が自動生成されます：

```
experiments/polymer_prediction_baseline/experiments_results/polymer_prediction_YYYYMMDD_HHMMSS/
├── metadata.json      # 実験設定・結果
├── submission.csv     # Kaggle提出用予測ファイル
```

## 設定オプション

スクリプト内の以下の定数で動作をカスタマイズ可能：

```python
SEED = 42                    # 再現性用ランダムシード
n_splits = 5                 # クロスバリデーションフォールド数
max_features = 500           # 特徴量選択上限
```

## 性能指標

- **評価指標**: wMAE（重み付き平均絶対誤差）
- **検証**: 5-Fold クロスバリデーション
- **特性別重み**: データ量と値域に基づく自動計算

## ログ出力例

```
🚀 実験開始: polymer_prediction_20250623_143022
✅ RDKit利用可能 - 高精度分子特徴量を使用
📂 ローカルデータ読み込み中...
🧬 分子特徴量生成中...
🤖 モデル訓練開始...
  Tg 用の高度なモデル訓練中...
  フォールド 1 XGB MAE: 45.234567
  フォールド 1 CatBoost MAE: 43.876543
🎯 推定重み付きMAE: 2.345678
⏱️  総実行時間: 15.34 分
```

## トラブルシューティング

### RDKitインストールエラー
```bash
# 代替インストール方法
conda install -c conda-forge rdkit
# または
pip install rdkit-pypi
```

### データファイルが見つからない
- `data/raw/`にKaggleデータセットを配置
- ファイル名が正確であることを確認

### メモリエラー
- 特徴量数を削減：`max_features = 200`
- モデル複雑度を下げる：`n_estimators`を削減

## 実験の比較・分析

実験結果は`metadata.json`に記録され、複数実験の比較が可能：

```json
{
  "experiment_name": "polymer_prediction_20250623_143022",
  "rdkit_available": true,
  "model_info": {
    "Tg": {"cv_score": 45.23, "model_type": "AveragingEnsemble"}
  },
  "estimated_wmae": 2.34
}
```

## 次のステップ

1. **特徴量エンジニアリング**: カスタム分子特徴量の追加
2. **ハイパーパラメータ最適化**: Optunaによる自動調整
3. **アンサンブル改善**: より高度な重み付け手法
4. **外部データ活用**: 追加の化学データベース統合