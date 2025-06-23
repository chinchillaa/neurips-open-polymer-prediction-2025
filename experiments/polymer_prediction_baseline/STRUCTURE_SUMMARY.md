# 整理されたディレクトリ構造

## ✅ 整理完了

### 📁 新しい構造

```
experiments/polymer_prediction_baseline/
├── README.md                           # メイン説明文書
├── STRUCTURE_SUMMARY.md               # この構造説明
├── scripts/                           # 実行スクリプト
│   ├── local_polymer_prediction.py    # メイン実験スクリプト
│   ├── local_polymer_prediction_with_wandb.py  # WandB統合版
│   └── run_experiment.sh              # 実行シェルスクリプト
├── tests/                             # テスト用スクリプト
│   ├── quick_test.py                   # クイックテスト
│   ├── wandb_test.py                   # WandBテスト
│   └── online_wandb_test.py           # オンラインWandBテスト
├── experiments_results/               # 実験結果
│   ├── polymer_prediction_20250623_175254/
│   ├── polymer_prediction_20250623_180023/
│   └── polymer_prediction_20250623_180044/
│       └── metadata.json
└── wandb/                            # WandB実験ログ
    ├── offline-run-20250623_175816-fpugsfy4/ (同期済み)
    ├── offline-run-20250623_180023-tus52zcg/ (同期済み)
    ├── offline-run-20250623_180044-fvv0uqk0/ (同期済み)
    └── run-20250623_180755-vtytijci/     (オンライン実験)
```

## 🎯 各ディレクトリの役割

### `scripts/` - 実行スクリプト
- **目的**: メインの実験実行用スクリプト
- **使用例**: 
  ```bash
  # 基本実験
  python experiments/polymer_prediction_baseline/scripts/local_polymer_prediction.py
  
  # WandB統合実験
  python experiments/polymer_prediction_baseline/scripts/local_polymer_prediction_with_wandb.py
  
  # シェルスクリプト実行
  ./experiments/polymer_prediction_baseline/scripts/run_experiment.sh
  ```

### `tests/` - テスト用スクリプト
- **目的**: 機能検証・クイックテスト
- **使用例**:
  ```bash
  # クイックテスト
  python experiments/polymer_prediction_baseline/tests/quick_test.py
  
  # WandBテスト
  python experiments/polymer_prediction_baseline/tests/wandb_test.py
  ```

### `experiments_results/` - 実験結果
- **目的**: 実験実行結果の保存
- **内容**: metadata.json, submission.csv等
- **自動生成**: スクリプト実行時に自動作成

### `wandb/` - WandB実験ログ
- **目的**: WandB実験追跡データ
- **内容**: オフライン・オンライン実験ログ
- **クラウド同期**: `wandb sync`で同期可能

## ✨ 整理による改善効果

1. **明確な役割分担**: scripts（実行）、tests（テスト）、results（結果）
2. **保守性向上**: 目的別にファイルが分類され、管理しやすい
3. **拡張性**: 新しいスクリプトやテストを適切な場所に配置可能
4. **WandB統合**: 実験追跡データが適切に管理される

## 🔧 パス修正内容

- スクリプト内の`PROJECT_ROOT`パス修正
- 実験結果保存先を`experiments_results/`に変更
- シェルスクリプトのパス参照更新

## ✅ 動作確認済み

- クイックテスト実行成功
- パス参照正常動作
- 実験結果保存先正常
- WandB実験管理機能正常