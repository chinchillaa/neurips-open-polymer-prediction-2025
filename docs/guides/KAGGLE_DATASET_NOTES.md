# Kaggleデータセット設定メモ

## RDKit用Wheelファイル

### 正しいデータセット名

```json
"dataset_sources": ["richolson/rdkit-install-whl"]
```

### wheelファイルのパス

- データセットパス: `/kaggle/input/rdkit-install-whl/rdkit_wheel/`
- wheelファイル例: `rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`

### 使用方法

1. `kernel-metadata.json`に上記のデータセット名を追加
2. Pythonコード内で以下のパスをチェック:

   ```python
   rdkit_dataset = '/kaggle/input/rdkit-install-whl/rdkit_wheel'
   ```

### 確認済みバージョン

- v8: 正常にRDKitインストール確認
- RDKit利用により100個の分子特徴量を使用可能

### 注意事項

- データセット名のユーザー名部分は`richolson`（`chinchillaa`や`tgwstr`ではない）
- `--no-deps`オプションを使用してインストール（依存関係の競合を避けるため）