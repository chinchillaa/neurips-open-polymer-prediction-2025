# Kaggle Submission ノートブック管理

## ディレクトリ構造

```
submission/
├── README.md                           # このファイル
├── neurips_polymer_advanced_ensemble_v9/   # 最新版（RDKit対応）
│   ├── kernel-metadata.json
│   └── neurips-polymer-advanced-ensemble-v2.py
├── polymer_prediction_baseline/        # ベースライン実装
│   ├── install_dependencies.py
│   ├── kernel-metadata.json
│   └── polymer_prediction_baseline.ipynb
└── archive/                           # 過去バージョンのアーカイブ
    └── old_versions/
        ├── neurips_polymer_advanced_ensemble/
        ├── neurips_polymer_advanced_ensemble_v2/
        ├── neurips_polymer_advanced_ensemble_v3/
        └── neurips_polymer_advanced_ensemble_v8/
```

## 最新版について

### neurips_polymer_advanced_ensemble_v9

- **特徴**: RDKit対応、100個の分子特徴量を使用
- **データセット**: `richolson/rdkit-install-whl`
- **モデル**: XGBoost, CatBoost, RandomForest, GradientBoosting, KNN のアンサンブル
- **予測対象**: Tg, FFV, Tc, Density, Rg の5特性

### 重要な設定

kernel-metadata.jsonでのRDKitデータセット指定:
```json
"dataset_sources": ["richolson/rdkit-install-whl"]
```

## バージョン履歴

- **v9**: RDKit正常動作版（現在の最新版）
- **v8**: RDKitパス修正版
- **v7**: Kaggleにアップロード成功
- **v3**: アンサンブル実装版
- **v2**: 初期実装版

## 使用方法

1. 最新版ディレクトリに移動
   ```bash
   cd neurips_polymer_advanced_ensemble_v9
   ```

2. Kaggleにアップロード
   ```bash
   kaggle kernels push
   ```

3. 実行状況を確認
   - https://www.kaggle.com/code/tgwstr/neurips-polymer-advanced-ensemble-v9