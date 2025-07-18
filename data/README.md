# データディレクトリ

このディレクトリには、NeurIPS Open Polymer Prediction 2025コンペティションのデータセットが格納されています。

## ディレクトリ構造

```
data/
├── raw/                    # 生データ（Kaggleからダウンロードした元データ）
│   ├── train.csv          # 訓練データ
│   ├── test.csv           # テストデータ
│   ├── sample_submission.csv  # 提出サンプル
│   └── train_supplement/  # 追加訓練データ
│       ├── dataset1.csv
│       ├── dataset2.csv
│       ├── dataset3.csv
│       └── dataset4.csv
├── processed/             # 前処理済みデータ
├── external/              # 外部データソース
│   ├── smiles-extra-data/ # dmitryuarov/smiles-extra-data
│   │   ├── JCIM_sup_bigsmiles.csv
│   │   ├── data_dnst1.xlsx
│   │   └── data_tg3.xlsx
│   └── tc-smiles/         # minatoyukinaxlisa/tc-smiles
│       └── Tc_SMILES.csv
└── README.md             # このファイル
```

## 各ディレクトリの説明

### raw/
- **用途**: Kaggleから直接ダウンロードした生データを保存
- **内容**:
  - `train.csv`: メインの訓練データセット（約691KB）
  - `test.csv`: 予測対象のテストデータセット（約263B）
  - `sample_submission.csv`: 提出形式のサンプル（約87B）
  - `train_supplement/`: 追加の訓練データセット

### processed/
- **用途**: 前処理・特徴量エンジニアリング後のデータを保存
- **内容**: スクリプトによって生成される加工済みデータ

### external/
- **用途**: コンペティション外部のデータソース（追加の特徴量として利用）
- **内容**: 
  - `smiles-extra-data/`: BigSMILES形式のポリマーデータとTg/DNSTデータ
  - `tc-smiles/`: TcとSMILESの対応データ

## データの取得方法

### 1. コンペティションデータ（raw/）

#### Kaggle CLIを使用する場合
```bash
# Kaggle APIの認証設定が必要
kaggle competitions download -c neurips-open-polymer-prediction-2025
unzip neurips-open-polymer-prediction-2025.zip -d raw/
```

#### Kaggle Hubを使用する場合
```python
import kagglehub

# データセットのダウンロード
path = kagglehub.dataset_download("neurips-open-polymer-prediction-2025")
```

### 2. 外部データセット（external/）

#### Kaggle CLIを使用する場合
```bash
# smiles-extra-dataのダウンロード
cd data/external
kaggle datasets download dmitryuarov/smiles-extra-data
unzip -o smiles-extra-data.zip -d smiles-extra-data/
rm smiles-extra-data.zip

# tc-smilesのダウンロード
kaggle datasets download minatoyukinaxlisa/tc-smiles
unzip -o tc-smiles.zip -d tc-smiles/
rm tc-smiles.zip
```

## データファイルの概要

### train.csv
- ポリマーの構造と物性に関する訓練データ
- 特徴量とターゲット変数を含む

### test.csv
- 予測対象のポリマーデータ
- ターゲット変数は含まれない

### train_supplement/
- 追加の訓練データセット
- dataset1-4.csvにはそれぞれ異なるソースのデータが含まれる

### external/smiles-extra-data/
- `JCIM_sup_bigsmiles.csv`: BigSMILES形式のポリマー構造データ
- `data_dnst1.xlsx`: DNST（密度）データセット
- `data_tg3.xlsx`: Tg（ガラス転移温度）データセット

### external/tc-smiles/
- `Tc_SMILES.csv`: Tc（結晶化温度）とSMILES構造の対応データ

## 注意事項

1. **Git管理について**
   - `raw/`ディレクトリ内の大きなCSVファイルは`.gitignore`に含まれています
   - データファイルは各自でKaggleからダウンロードしてください

2. **データの更新**
   - Kaggleでデータが更新された場合は、`raw/`ディレクトリを再ダウンロード
   - 前処理スクリプトを再実行して`processed/`を更新

3. **ストレージ管理**
   - 大きなデータファイルは定期的にクリーンアップ
   - 必要な処理済みデータのみを保持

## 関連スクリプト

- データの前処理: `scripts/prepare_kaggle_dataset.py`
- 特徴量エンジニアリング: 各実験ディレクトリ内のスクリプト