# Kaggle Notebooks - Code Competition

このディレクトリは、NeurIPS Open Polymer Prediction 2025 コードコンペティション用のノートブックとテンプレートを含んでいます。

## ディレクトリ構成

```
kaggle_notebooks/
├── development/           # 開発用テンプレート
│   ├── eda_template.py           # データ探索分析
│   ├── feature_engineering_template.py  # 特徴量エンジニアリング
│   ├── model_comparison_template.py     # モデル比較
│   └── README.md                        # 開発テンプレート使用方法
├── final_submission/      # 最終提出用ガイド
│   └── README.md                 # 提出手順とチェックリスト
└── templates/            # 再利用可能テンプレート
    └── submission_template.py    # Kaggle提出用テンプレート
```

## 開発ワークフロー

### Phase 1: ローカル開発
1. **データ探索**: `development/eda_template.py`でデータの特徴を理解
2. **特徴量設計**: `development/feature_engineering_template.py`で効果的な特徴量を作成
3. **モデル選択**: `development/model_comparison_template.py`で最適なモデルを特定

### Phase 2: Kaggle移植
1. **テンプレート活用**: `templates/submission_template.py`をベースに使用
2. **最適化**: メモリ使用量と実行時間を最適化
3. **テスト**: Kaggle環境での動作確認

### Phase 3: 最終提出
1. **チェック**: `final_submission/README.md`のチェックリストを確認
2. **提出**: Kaggleノートブックとして提出

## 主要機能

### 🔍 開発用テンプレート
- **包括的EDA**: データ品質、分布、相関分析
- **高度特徴量エンジニアリング**: ポリマー分子特有の特徴量抽出
- **モデル比較**: 複数アルゴリズムの自動比較とアンサンブル

### ⚡ コードコンペ最適化
- **メモリ最適化**: データ型最適化によるメモリ使用量削減
- **実行時間管理**: タイマー機能による処理時間監視
- **エラーハンドリング**: 堅牢性を高める例外処理

### 📋 提出支援
- **完全テンプレート**: すぐに使える提出用コードテンプレート
- **検証機能**: 提出ファイル形式の自動チェック
- **フォールバック**: エラー時の代替処理

## クイックスタート

### 1. データ探索から開始
```bash
cd development
python eda_template.py
```

### 2. 特徴量エンジニアリング
```python
from development.feature_engineering_template import PolymerFeatureEngineer

fe = PolymerFeatureEngineer()
train_features = fe.fit_transform(train_df)
```

### 3. モデル比較
```python
from development.model_comparison_template import run_model_comparison

results = run_model_comparison(train_df, feature_cols, target_cols)
```

### 4. Kaggle提出準備
```python
# templates/submission_template.pyをKaggleノートブックにコピー
# 必要に応じてカスタマイズ
```

## 重要な制約

- **実行時間**: 最大9時間
- **インターネット**: アクセス不可
- **メモリ**: 効率的な使用が必要
- **出力**: `submission.csv`必須

## 最適化のヒント

1. **データ型最適化**: `reduce_mem_usage()`関数を活用
2. **並列処理**: `n_jobs=-1`で計算を高速化
3. **早期終了**: 時間制限を考慮したモデル設定
4. **チャンク処理**: 大きなデータセットの分割処理

## トラブルシューティング

### よくある問題と解決策

- **メモリエラー**: データ型最適化とチャンク処理
- **実行時間超過**: モデル複雑度の削減と並列化
- **ライブラリエラー**: 代替実装やフォールバック処理
- **パスエラー**: Kaggle環境のパス確認

## 参考資料

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks)
- [Code Competition Guide](../docs/CODE_COMPETITION_GUIDE.md)
- [Project README](../README.md)

## 次のステップ

1. `development/README.md`で開発用テンプレートの詳細を確認
2. `final_submission/README.md`で提出手順を確認
3. 実際のデータでテンプレートをテスト
4. カスタマイズして自分の戦略に適応