# プロジェクト構造

## ディレクトリ構造

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
└── .artifacts/                    # 一時ファイル（gitignore対象）
    ├── wandb/                    # WandB関連
    ├── catboost_info/            # CatBoost関連
    └── tmp/                      # その他の一時ファイル
```

## 実験の管理方法

1. **新しい実験の追加**
   ```bash
   mkdir -p experiments/new_experiment/{scripts,results/{runs,models,submissions},logs}
   ```

2. **実験の実行**
   - 各実験のscriptsディレクトリ内でスクリプトを実行
   - 結果は自動的にresults/runs/に日付付きで保存
   - ログはlogs/に出力

3. **最新結果へのアクセス**
   ```bash
   experiments/advanced_ensemble/results/runs/latest  # シンボリックリンク
   ```

## ノートブックの管理

- **開発用**: `notebooks/development/`
- **Kaggle提出用**: `notebooks/kaggle/active/`
- **過去バージョン**: `notebooks/kaggle/archive/`

## 一時ファイル

全ての一時ファイルは`.artifacts/`に集約され、gitignoreで除外されます。

## 依存関係の注意事項

### NumPyとRDKitの互換性
- **RDKit 2022.9.5**はNumPy 1.xでコンパイルされています
- **NumPy 2.0以降**を使用すると`AttributeError: _ARRAY_API not found`エラーが発生
- **解決方法**: `pip install "numpy<2"` または `uv pip install "numpy<2"`
- **推奨環境**:
  - numpy==1.26.4
  - rdkit-pypi==2022.9.5
  
これは特に分子特徴量を使用するadvanced_ensemble実験で重要です。