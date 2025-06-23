"""
Kaggleノートブック自動生成・アップロードスクリプト
==================================================

このスクリプトはPythonコードから.ipynbファイルを生成し、
Kaggle APIを使用してKaggleに直接アップロードします。

使用方法:
    python scripts/create_kaggle_notebook.py --input complete_baseline_notebook.py --title "Baseline Model"

必要な設定:
    - kaggle.json がホームディレクトリに配置されていること
    - kaggle パッケージがインストールされていること
"""

import json
import argparse
import os
from pathlib import Path
import nbformat as nbf
import subprocess
import tempfile
import shutil

class NotebookGenerator:
    """Kaggleノートブック生成・アップロードクラス"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        
    def python_to_notebook(self, python_file: Path, output_file: Path = None):
        """PythonファイルをJupyterノートブックに変換"""
        print(f"Pythonファイルをノートブックに変換中: {python_file}")
        
        # Pythonファイルを読み込み
        with open(python_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 新しいノートブックを作成
        nb = nbf.v4.new_notebook()
        
        # コードをセルに分割（# ============================================================================ で区切り）
        sections = self._split_code_into_sections(content)
        
        for section in sections:
            if section['type'] == 'markdown':
                # マークダウンセル
                cell = nbf.v4.new_markdown_cell(section['content'])
            else:
                # コードセル
                cell = nbf.v4.new_code_cell(section['content'])
            
            nb.cells.append(cell)
        
        # ノートブックを保存
        if output_file is None:
            output_file = python_file.with_suffix('.ipynb')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        
        print(f"ノートブック生成完了: {output_file}")
        return output_file
    
    def _split_code_into_sections(self, content):
        """コードを論理的なセクションに分割"""
        sections = []
        lines = content.split('\n')
        current_section = []
        current_type = 'code'
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # セクション区切り線を検出
            if line.strip().startswith('# ' + '=' * 70):
                # 現在のセクションを保存
                if current_section:
                    sections.append({
                        'type': current_type,
                        'content': '\n'.join(current_section).strip()
                    })
                    current_section = []
                
                # 次の行がタイトルの場合、マークダウンとして処理
                if i + 1 < len(lines) and lines[i + 1].strip().startswith('#'):
                    # タイトル行を取得
                    title_line = lines[i + 1].strip()
                    
                    # docstringがある場合も含める
                    j = i + 2
                    desc_lines = []
                    
                    # 空行をスキップ
                    while j < len(lines) and lines[j].strip() == '':
                        j += 1
                    
                    # docstringまたはコメントブロックを探す
                    if j < len(lines):
                        if lines[j].strip().startswith('"""'):
                            # docstring処理
                            j += 1
                            while j < len(lines) and not lines[j].strip().endswith('"""'):
                                desc_lines.append(lines[j])
                                j += 1
                            if j < len(lines):
                                j += 1  # 終了の"""をスキップ
                        elif lines[j].strip().startswith('#'):
                            # コメントブロック処理
                            while j < len(lines) and lines[j].strip().startswith('#'):
                                desc_lines.append(lines[j].strip()[1:].strip())
                                j += 1
                    
                    # マークダウンセクションを作成
                    markdown_content = title_line
                    if desc_lines:
                        markdown_content += '\n\n' + '\n'.join(desc_lines)
                    
                    sections.append({
                        'type': 'markdown',
                        'content': markdown_content
                    })
                    
                    i = j - 1  # 次のループで正しい位置から開始
                    current_type = 'code'
                else:
                    current_type = 'code'
                
            else:
                current_section.append(line)
            
            i += 1
        
        # 最後のセクションを追加
        if current_section:
            sections.append({
                'type': current_type,
                'content': '\n'.join(current_section).strip()
            })
        
        return sections
    
    def create_kaggle_metadata(self, title: str, notebook_file: Path, 
                              dataset_sources: list = None, 
                              competition_sources: list = None):
        """Kaggleノートブック用のメタデータファイルを作成"""
        
        if dataset_sources is None:
            dataset_sources = []
        if competition_sources is None:
            competition_sources = ["neurips-open-polymer-prediction-2025"]
        
        metadata = {
            "title": title,
            "id": f"chinchillaa/{title.lower().replace(' ', '-')}",
            "licenses": [{"name": "CC0-1.0"}],
            "keywords": ["polymer", "prediction", "neurips", "machine-learning"],
            "datasets": [{"source": ds} for ds in dataset_sources],
            "competitions": [{"source": comp} for comp in competition_sources],
            "kernelType": "notebook",
            "isInternetEnabled": False,
            "language": "python",
            "enableGpu": False,
            "enableTpu": False
        }
        
        # メタデータファイルを作成
        metadata_file = notebook_file.parent / "kernel-metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"メタデータファイル作成: {metadata_file}")
        return metadata_file
    
    def upload_to_kaggle(self, notebook_file: Path, metadata_file: Path, 
                        is_private: bool = True, update_existing: bool = False):
        """Kaggleにノートブックをアップロード"""
        print(f"Kaggleにアップロード中: {notebook_file}")
        
        # Kaggle APIが利用可能かチェック
        try:
            result = subprocess.run(['kaggle', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"Kaggle API バージョン: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("エラー: Kaggle APIが利用できません。")
            print("pip install kaggle でインストールし、kaggle.jsonを設定してください。")
            return False
        
        # 一時ディレクトリでアップロード準備
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # ファイルをコピー
            temp_notebook = temp_dir / notebook_file.name
            temp_metadata = temp_dir / "kernel-metadata.json"
            
            shutil.copy2(notebook_file, temp_notebook)
            shutil.copy2(metadata_file, temp_metadata)
            
            # アップロードコマンドを実行
            try:
                if update_existing:
                    cmd = ['kaggle', 'kernels', 'version', '-p', str(temp_dir)]
                else:
                    cmd = ['kaggle', 'kernels', 'push', '-p', str(temp_dir)]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print("アップロード成功!")
                print(result.stdout)
                return True
                
            except subprocess.CalledProcessError as e:
                print(f"アップロードエラー: {e}")
                print(f"標準出力: {e.stdout}")
                print(f"標準エラー: {e.stderr}")
                return False
    
    def generate_and_upload(self, python_file: str, title: str, 
                           dataset_sources: list = None,
                           competition_sources: list = None,
                           is_private: bool = True,
                           update_existing: bool = False):
        """Python ファイルからノートブックを生成してアップロード"""
        print(f"ノートブック生成・アップロード開始: {title}")
        
        # パスを解決
        python_path = self.project_root / python_file
        if not python_path.exists():
            print(f"エラー: ファイルが見つかりません: {python_path}")
            return False
        
        # 出力ディレクトリを作成
        output_dir = self.project_root / "kaggle_notebooks" / "submission"
        output_dir.mkdir(exist_ok=True)
        
        # ノートブック生成
        notebook_file = output_dir / f"{title.lower().replace(' ', '_')}.ipynb"
        notebook_file = self.python_to_notebook(python_path, notebook_file)
        
        # メタデータ生成
        metadata_file = self.create_kaggle_metadata(
            title, notebook_file, dataset_sources, competition_sources
        )
        
        # Kaggleにアップロード
        success = self.upload_to_kaggle(
            notebook_file, metadata_file, is_private, update_existing
        )
        
        if success:
            print(f"✅ 完了: {title}")
            print(f"ノートブック: {notebook_file}")
            print(f"Kaggle URL: https://www.kaggle.com/code/chinchillaa/{title.lower().replace(' ', '-')}")
        
        return success

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Kaggleノートブック生成・アップロード')
    parser.add_argument('--input', required=True, help='入力Pythonファイル')
    parser.add_argument('--title', required=True, help='ノートブックタイトル')
    parser.add_argument('--datasets', nargs='*', help='依存データセット', default=[])
    parser.add_argument('--competitions', nargs='*', help='参加コンペティション', 
                       default=["neurips-open-polymer-prediction-2025"])
    parser.add_argument('--public', action='store_true', help='パブリックノートブック')
    parser.add_argument('--update', action='store_true', help='既存ノートブックを更新')
    
    args = parser.parse_args()
    
    # ノートブック生成・アップロード
    generator = NotebookGenerator()
    success = generator.generate_and_upload(
        python_file=args.input,
        title=args.title,
        dataset_sources=args.datasets,
        competition_sources=args.competitions,
        is_private=not args.public,
        update_existing=args.update
    )
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()