#!/usr/bin/env python3
"""
WandBテスト - NeurIPS Polymer Advanced Ensemble
Weights & Biases統合のテストスクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

def test_wandb_import():
    """WandBインポートテスト"""
    print("📊 WandBインポートテスト...")
    try:
        import wandb
        print(f"✅ WandB version: {wandb.__version__}")
        return True
    except ImportError:
        print("❌ WandBがインストールされていません")
        print("💡 インストール: pip install wandb")
        return False

def test_wandb_login():
    """WandBログイン状態確認"""
    print("\n🔐 WandBログイン状態確認...")
    try:
        import wandb
        if wandb.api.api_key:
            print("✅ WandB APIキーが設定されています")
            return True
        else:
            print("⚠️  WandB APIキーが未設定です")
            print("💡 ログイン: wandb login")
            return False
    except Exception as e:
        print(f"❌ ログイン確認エラー: {e}")
        return False

def test_offline_mode():
    """オフラインモードテスト"""
    print("\n💾 オフラインモードテスト...")
    try:
        import wandb
        import tempfile
        
        # 一時ディレクトリでテスト実行
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = tmpdir
            
            # テストラン作成
            run = wandb.init(
                project="test-project",
                name="test-run",
                config={"test": True}
            )
            
            # テストログ
            wandb.log({"test_metric": 1.0})
            
            # 終了
            wandb.finish()
            
            print("✅ オフラインモード正常動作")
            
            # 保存されたファイル確認
            wandb_files = list(Path(tmpdir).glob("**/*.wandb"))
            if wandb_files:
                print(f"✅ オフラインファイル作成: {len(wandb_files)}個")
            
            return True
            
    except Exception as e:
        print(f"❌ オフラインモードエラー: {e}")
        return False
    finally:
        # 環境変数クリーンアップ
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_DIR", None)

def test_config_integration():
    """設定ファイルとの統合テスト"""
    print("\n⚙️  設定ファイル統合テスト...")
    
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print("⚠️  config.yamlが見つかりません")
        return False
    
    try:
        import yaml
        import wandb
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # WandB設定の確認
        wandb_config = config.get("wandb", {})
        print(f"✅ WandBプロジェクト: {wandb_config.get('project', 'デフォルト')}")
        print(f"✅ 実験名: {wandb_config.get('name', 'デフォルト')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 設定統合エラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🧪 NeurIPS Polymer Advanced Ensemble - WandBテスト")
    print("=" * 50)
    
    # WandBインポート
    if not test_wandb_import():
        print("\n⚠️  WandBが利用できません")
        print("実験は実行可能ですが、実験管理機能は使用できません")
        return
    
    # ログイン状態
    logged_in = test_wandb_login()
    
    # オフラインモード
    test_offline_mode()
    
    # 設定統合
    test_config_integration()
    
    print("\n" + "=" * 50)
    if logged_in:
        print("✅ WandB環境は完全に整っています！")
        print("💡 WandB付き実験実行:")
        print("   ./scripts/run_experiment.sh --wandb")
    else:
        print("⚠️  WandBオフラインモードで実行可能です")
        print("💡 オンライン同期するには:")
        print("   wandb login")
        print("   wandb sync [オフラインディレクトリ]")

if __name__ == "__main__":
    main()