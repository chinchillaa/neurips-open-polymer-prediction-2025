# Kaggleオフライン環境用依存関係インストールスクリプト
# NeurIPS Open Polymer Prediction 2025 - Baseline用

import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

def install_package(package_name, pip_name=None):
    """パッケージを安全にインストール"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        # パッケージのインポートを試行
        __import__(package_name)
        print(f"✅ {package_name} は既にインストール済み")
        return True
    except ImportError:
        print(f"📦 {package_name} のインストール中...")
        try:
            # pipでインストール
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", pip_name, "--quiet"
            ])
            print(f"✅ {package_name} インストール完了")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {package_name} インストール失敗: {e}")
            return False

def main():
    """必要な依存関係をインストール"""
    print("🚀 Kaggleオフライン環境用依存関係インストール開始")
    print("==========================================================")
    
    # インストールが必要なパッケージリスト（ベースライン用は最小限）
    required_packages = [
        # 機械学習ライブラリ（通常は既にインストール済み）
        ("xgboost", "xgboost"),
        ("sklearn", "scikit-learn"),
        
        # データ処理ライブラリ（通常は既にインストール済み）
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    # 各パッケージのインストールを試行
    success_count = 0
    for package_name, pip_name in required_packages:
        if install_package(package_name, pip_name):
            success_count += 1
    
    print("==========================================================")
    print(f"📊 インストール結果: {success_count}/{len(required_packages)} パッケージ成功")
    
    print("\n🎯 依存関係インストール完了")
    print("次のセルでメインの予測コードを実行してください")

if __name__ == "__main__":
    main()