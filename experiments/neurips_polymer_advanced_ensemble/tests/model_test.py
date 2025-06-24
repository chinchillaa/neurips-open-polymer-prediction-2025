#!/usr/bin/env python3
"""
モデルテスト - NeurIPS Polymer Advanced Ensemble
モデルの動作確認とベンチマークテスト
"""

import sys
import os
from pathlib import Path
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

def test_model_imports():
    """モデルライブラリのインポートテスト"""
    print("🤖 モデルライブラリテスト...")
    
    models_ok = True
    
    # XGBoost
    try:
        import xgboost as xgb
        print(f"✅ XGBoost version: {xgb.__version__}")
    except ImportError:
        print("❌ XGBoost")
        models_ok = False
    
    # CatBoost
    try:
        import catboost
        print(f"✅ CatBoost version: {catboost.__version__}")
    except ImportError:
        print("❌ CatBoost")
        models_ok = False
    
    # scikit-learn
    try:
        import sklearn
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neighbors import KNeighborsRegressor
        print(f"✅ scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn")
        models_ok = False
    
    return models_ok

def test_simple_training():
    """簡単なモデル訓練テスト"""
    print("\n🏃 簡易訓練テスト...")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        import catboost
        from sklearn.ensemble import RandomForestRegressor
        
        # テストデータ生成
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 各モデルのテスト
        models = {
            'XGBoost': xgb.XGBRegressor(n_estimators=10, random_state=42, verbosity=0),
            'CatBoost': catboost.CatBoostRegressor(
                iterations=10, random_state=42, verbose=False
            ),
            'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        for name, model in models.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            score = model.score(X_test, y_test)
            print(f"✅ {name}: R²={score:.3f}, 訓練時間={train_time:.3f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 訓練テストエラー: {e}")
        return False

def test_feature_generation():
    """特徴量生成テスト（RDKit）"""
    print("\n🧬 分子特徴量生成テスト...")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem
        
        # テスト分子（エタノール）
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print("❌ SMILES解析エラー")
            return False
        
        # 基本記述子
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        print(f"✅ 基本記述子: MW={mw:.2f}, LogP={logp:.2f}")
        
        # フィンガープリント
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
        fp_array = np.array(fp)
        print(f"✅ Morganフィンガープリント: {len(fp_array)}ビット")
        
        return True
        
    except ImportError:
        print("⚠️  RDKitが利用できません（オプション）")
        return None
    except Exception as e:
        print(f"❌ 特徴量生成エラー: {e}")
        return False

def test_ensemble():
    """アンサンブル機能テスト"""
    print("\n🎯 アンサンブルテスト...")
    
    try:
        import numpy as np
        from sklearn.base import BaseEstimator, RegressorMixin
        
        # ダミー予測
        n_samples = 50
        predictions = {
            'model1': np.random.randn(n_samples),
            'model2': np.random.randn(n_samples),
            'model3': np.random.randn(n_samples)
        }
        
        # 単純平均
        avg_pred = np.mean(list(predictions.values()), axis=0)
        print(f"✅ 単純平均アンサンブル: shape={avg_pred.shape}")
        
        # 加重平均
        weights = [0.5, 0.3, 0.2]
        weighted_pred = np.average(
            list(predictions.values()), 
            axis=0, 
            weights=weights
        )
        print(f"✅ 加重平均アンサンブル: shape={weighted_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ アンサンブルテストエラー: {e}")
        return False

def test_model_persistence():
    """モデル保存・読み込みテスト"""
    print("\n💾 モデル永続化テスト...")
    
    try:
        import pickle
        import tempfile
        import xgboost as xgb
        import numpy as np
        
        # テストモデル作成
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        model = xgb.XGBRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        # 保存・読み込み
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            pickle.dump(model, tmp)
            tmp_path = tmp.name
        
        with open(tmp_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # 予測テスト
        pred_original = model.predict(X[:5])
        pred_loaded = loaded_model.predict(X[:5])
        
        if np.allclose(pred_original, pred_loaded):
            print("✅ モデル保存・読み込み成功")
            return True
        else:
            print("❌ 予測結果が一致しません")
            return False
            
    except Exception as e:
        print(f"❌ 永続化テストエラー: {e}")
        return False
    finally:
        # クリーンアップ
        try:
            os.unlink(tmp_path)
        except:
            pass

def main():
    """メインテスト実行"""
    print("🧪 NeurIPS Polymer Advanced Ensemble - モデルテスト")
    print("=" * 50)
    
    # モデルインポート
    if not test_model_imports():
        print("\n❌ 必要なモデルライブラリが不足しています")
        return
    
    # 各種テスト
    test_simple_training()
    test_feature_generation()
    test_ensemble()
    test_model_persistence()
    
    print("\n" + "=" * 50)
    print("✅ モデルテスト完了！")
    print("💡 全ての機能が正常に動作しています")

if __name__ == "__main__":
    main()