#!/bin/bash
# NeurIPS Polymer Advanced Ensemble Experiment実行スクリプト
# 
# 使用方法:
#   ./run_experiment.sh [オプション]
#
# オプション:
#   --help      このヘルプを表示
#   --install   依存関係をインストール
#   --rdkit     RDKitも含めてインストール
#   --wandb     WandB統合版で実行

set -e  # エラー時に停止

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ヘルプ表示
show_help() {
    cat << EOF
NeurIPS Polymer Advanced Ensemble Experiment実行スクリプト

使用方法:
    $0 [オプション]

オプション:
    --help      このヘルプを表示
    --install   依存関係をインストール
    --rdkit     RDKitも含めてインストール
    --wandb     WandB統合版で実行

例:
    $0                    # 実験実行
    $0 --install          # 依存関係インストール後実行
    $0 --rdkit           # RDKitも含めてインストール後実行
    $0 --wandb            # WandB統合版で実行

必要な環境:
    - Python 3.8+
    - data/raw/以下にKaggleデータセット
      ├── train.csv
      ├── test.csv
      └── sample_submission.csv
EOF
}

# 依存関係インストール
install_dependencies() {
    echo "📦 必須依存関係をインストール中..."
    pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn
    echo "✅ 必須依存関係インストール完了"
}

# RDKitインストール
install_rdkit() {
    echo "🧪 RDKitをインストール中..."
    if pip install rdkit-pypi; then
        echo "✅ RDKitインストール成功"
    else
        echo "⚠️  RDKitインストール失敗 - condaでの代替方法:"
        echo "   conda install -c conda-forge rdkit"
    fi
}

# データファイル確認
check_data_files() {
    local data_dir="$PROJECT_ROOT/data/raw"
    local missing_files=()
    
    if [[ ! -f "$data_dir/train.csv" ]]; then
        missing_files+=("train.csv")
    fi
    if [[ ! -f "$data_dir/test.csv" ]]; then
        missing_files+=("test.csv")
    fi
    if [[ ! -f "$data_dir/sample_submission.csv" ]]; then
        missing_files+=("sample_submission.csv")
    fi
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        echo "❌ 以下のデータファイルが見つかりません:"
        for file in "${missing_files[@]}"; do
            echo "   $data_dir/$file"
        done
        echo ""
        echo "Kaggleからデータをダウンロードして data/raw/ に配置してください:"
        echo "   https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data"
        exit 1
    fi
    
    echo "✅ データファイル確認完了"
}

# 実験実行
run_experiment() {
    echo "🚀 NeurIPS Polymer Advanced Ensemble Experiment実行開始"
    echo "📁 プロジェクトルート: $PROJECT_ROOT"
    echo "📁 実験ディレクトリ: $SCRIPT_DIR"
    echo ""
    
    # プロジェクトルートに移動
    cd "$PROJECT_ROOT"
    
    # データファイル確認
    check_data_files
    
    # Python環境確認
    if ! command -v python3 &> /dev/null; then
        echo "❌ python3が見つかりません"
        exit 1
    fi
    
    echo "🐍 Python版: $(python3 --version)"
    echo ""
    
    # 実験実行
    echo "▶️  実験スクリプト実行中..."
    if [[ "${USE_WANDB:-false}" == "true" ]]; then
        echo "📊 WandB統合モードで実行"
        python3 "$SCRIPT_DIR/local_experiment.py" --use-wandb
    else
        python3 "$SCRIPT_DIR/local_experiment.py"
    fi
    
    # 結果表示
    echo ""
    echo "🎉 実験完了!"
    echo "📊 結果は以下のディレクトリに保存されました:"
    latest_experiment=$(find "$SCRIPT_DIR/../experiments_results" -name "advanced_ensemble_*" -type d | sort | tail -1)
    if [[ -n "$latest_experiment" ]]; then
        echo "   $latest_experiment"
        echo ""
        if [[ -f "$latest_experiment/metadata.json" ]]; then
            echo "📋 実験サマリー:"
            if command -v jq &> /dev/null; then
                jq -r '.experiment_name, .rdkit_available, .estimated_wmae' "$latest_experiment/metadata.json" 2>/dev/null || cat "$latest_experiment/metadata.json"
            else
                grep -E '"experiment_name"|"rdkit_available"|"estimated_wmae"' "$latest_experiment/metadata.json" || echo "メタデータファイルが見つかりません"
            fi
        fi
    fi
}

# メイン処理
main() {
    case "${1:-}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --install)
            install_dependencies
            run_experiment
            ;;
        --rdkit)
            install_dependencies
            install_rdkit
            run_experiment
            ;;
        --wandb)
            USE_WANDB=true
            run_experiment
            ;;
        "")
            run_experiment
            ;;
        *)
            echo "❌ 不明なオプション: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# スクリプト実行
main "$@"