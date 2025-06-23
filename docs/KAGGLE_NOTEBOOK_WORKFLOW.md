# Kaggleノートブック自動化ワークフロー

## 概要

手動コピペから脱却し、Pythonコードから.ipynbファイルを自動生成してKaggle APIで直接アップロードする運用に変更しました。

## 新しいワークフロー

### 1. 従来の手動運用（廃止）
```
Python コード → 手動コピペ → Kaggle Notebook → 手動実行・提出
```

### 2. 新しい自動化運用
```
Python コード → .ipynb 自動生成 → Kaggle API アップロード → Kaggle実行・提出
```

## 必要な環境設定

### Kaggle API設定
1. Kaggleアカウントの設定ページでAPI Tokenをダウンロード
2. `~/.kaggle/kaggle.json` に配置
3. 権限設定: `chmod 600 ~/.kaggle/kaggle.json`

### 必要なパッケージ
```bash
pip install kaggle nbformat
```

## 使用方法

### 基本的な使用方法

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

### 簡単アップロード（ベースライン）

```bash
# ベースラインノートブックの簡単アップロード
./scripts/upload_baseline.sh
```

## コード構造の要件

### セクション分割
Pythonコードは以下の区切り文字で自動的にノートブックセルに分割されます：

```python
# ============================================================================
# セクションタイトル
# ============================================================================
```

### マークダウンセルの作成
セクション区切りの直後に `#` で始まるタイトルがある場合、マークダウンセルとして処理されます：

```python
# ============================================================================
# データ読み込みと前処理
# ============================================================================

# このセクションではデータの読み込みと基本的な前処理を行います
# - CSVファイルの読み込み
# - 欠損値の確認
# - データ型の最適化
```

## ファイル構造

```
kaggle_notebooks/
├── templates/          # Pythonテンプレート
│   ├── complete_baseline_notebook.py
│   └── submission_template.py
├── submission/         # 生成された.ipynbファイル
│   └── neurips_polymer_baseline_random_forest.ipynb
└── references/         # 参考ノートブック
    └── neurips-2025-open-polymer-challenge-tutorial.ipynb
```

## 自動生成されるファイル

### 1. Jupyterノートブック (.ipynb)
- Pythonコードから自動変換
- セルが適切に分割済み
- マークダウンとコードセルが混在

### 2. メタデータファイル (kernel-metadata.json)
```json
{
  "title": "NeurIPS Polymer Baseline - Random Forest",
  "id": "chinchillaa/neurips-polymer-baseline-random-forest",
  "licenses": [{"name": "CC0-1.0"}],
  "keywords": ["polymer", "prediction", "neurips", "machine-learning"],
  "datasets": [],
  "competitions": [{"source": "neurips-open-polymer-prediction-2025"}],
  "kernelType": "notebook",
  "isInternetEnabled": false,
  "language": "python",
  "enableGpu": false,
  "enableTpu": false
}
```

## メリット

### 🚀 開発効率の向上
- コピペエラーの排除
- バージョン管理の一元化
- 一括アップロード・更新

### 🔄 自動化による品質向上
- 一貫したフォーマット
- メタデータの自動生成
- 依存関係の自動設定

### 📝 保守性の向上
- ソースコードとノートブックの同期
- Git履歴での変更追跡
- 複数ノートブックの一括管理

## トラブルシューティング

### よくあるエラー

1. **Kaggle API認証エラー**
   ```
   解決方法: ~/.kaggle/kaggle.json の設置と権限確認
   ```

2. **ノートブック名の重複エラー**
   ```
   解決方法: --update オプションを使用するか、異なるタイトルを指定
   ```

3. **依存パッケージエラー**
   ```
   解決方法: pip install kaggle nbformat
   ```

## 今後の拡張予定

- [ ] ノートブック実行状況の自動監視
- [ ] 提出ファイルの自動ダウンロード
- [ ] 複数バリエーションの一括アップロード
- [ ] テンプレートエンジンの統合
- [ ] CI/CDパイプラインとの統合

## 関連ファイル

- `scripts/create_kaggle_notebook.py` - メインスクリプト
- `scripts/upload_baseline.sh` - 簡単アップロード用
- `kaggle_notebooks/templates/` - Pythonテンプレート
- `kaggle_notebooks/submission/` - 生成されたノートブック