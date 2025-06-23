.PHONY: help setup install clean lint test data train submit format

help:
	@echo "利用可能なコマンド:"
	@echo "  setup     - uvを使用した開発環境のセットアップ"
	@echo "  install   - 依存関係のインストール"
	@echo "  clean     - 一時ファイルのクリーンアップ"
	@echo "  lint      - コードリンティングの実行"
	@echo "  format    - コードフォーマットの実行"
	@echo "  test      - テストの実行"
	@echo "  data      - データのダウンロードと準備"
	@echo "  train     - モデルの訓練"
	@echo "  submit    - 提出ファイルの生成"

setup:
	@echo "uvを使用して開発環境をセットアップ中..."
	uv venv
	@echo "依存関係をインストール中..."
	uv sync
	@echo "セットアップ完了。以下のコマンドで仮想環境をアクティベートしてください:"
	@echo "source .venv/bin/activate"

install:
	uv sync

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	uv run black --check .
	uv run isort --check-only .
	uv run flake8 .

format:
	uv run black .
	uv run isort .

test:
	uv run pytest tests/ -v --cov=src

data:
	uv run python scripts/download_data.py
	uv run python scripts/prepare_data.py

train:
	uv run python scripts/train_model.py

submit:
	uv run python scripts/generate_submission.py