#!/usr/bin/env python3
"""
WandBの過去の実験結果を削除するスクリプト
"""

import wandb

# WandB APIの初期化
api = wandb.Api()

# 削除対象のrun IDリスト
runs_to_delete = [
    "epg0tkgq",  # 20250704_144054
    "2392eakb"   # 20250704_144351
]

print("🗑️ WandB実験結果削除開始...")

for run_id in runs_to_delete:
    try:
        run = api.run(f"chinchilla/neurips-polymer-prediction-2025/{run_id}")
        run_name = run.name
        run.delete()
        print(f"✅ 削除完了: {run_id} ({run_name})")
    except Exception as e:
        print(f"❌ 削除失敗: {run_id} - エラー: {e}")

print("\n✨ 削除処理完了")
print("最新の実験（fv1lo3c1）は保持されています。")