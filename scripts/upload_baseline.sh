#!/bin/bash

# ベースラインノートブックをKaggleにアップロードするスクリプト
# 使用方法: ./scripts/upload_baseline.sh

echo "🚀 NeurIPS Open Polymer Prediction 2025 - ベースラインノートブックアップロード"
echo "=" * 80

# プロジェクトルートに移動
cd "$(dirname "$0")/.."

# 必要なPythonパッケージの確認
echo "📦 必要なパッケージを確認中..."
python -c "import nbformat" 2>/dev/null || {
    echo "❌ nbformat が見つかりません。インストール中..."
    pip install nbformat
}

python -c "import kaggle" 2>/dev/null || {
    echo "❌ kaggle API が見つかりません。インストール中..."
    pip install kaggle
}

# Kaggle認証の確認
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "❌ Kaggle認証が設定されていません。"
    echo "   1. Kaggleアカウント設定から API Token をダウンロード"
    echo "   2. ~/.kaggle/kaggle.json に配置"
    echo "   3. chmod 600 ~/.kaggle/kaggle.json で権限設定"
    exit 1
fi

echo "✅ 環境チェック完了"

# ベースラインノートブックを生成・アップロード
echo ""
echo "📓 ベースラインノートブック生成・アップロード中..."

python scripts/create_kaggle_notebook.py \
    --input "kaggle_notebooks/templates/complete_baseline_notebook.py" \
    --title "NeurIPS Polymer Baseline - Random Forest" \
    --competitions "neurips-open-polymer-prediction-2025" \
    --public

echo ""
echo "🎉 アップロード完了!"
echo "📝 ノートブックURL: https://www.kaggle.com/code/chinchillaa/neurips-polymer-baseline-random-forest"