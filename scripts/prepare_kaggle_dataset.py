"""
Prepare Kaggle Dataset for Code Competition
==========================================

This script prepares trained models and dependencies for upload to Kaggle as a dataset.
The dataset will be used in the code competition submission notebook.
"""

import os
import shutil
import joblib
import pickle
from pathlib import Path
import zipfile
import json

class KaggleDatasetPreparer:
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.output_dir = self.project_root / "kaggle_upload"
        self.models_dir = self.project_root / "models" / "trained"
        
    def prepare_dataset(self):
        """Prepare complete dataset for Kaggle upload"""
        print("Preparing Kaggle dataset...")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Copy trained models
        self._copy_models()
        
        # Copy preprocessing artifacts
        self._copy_preprocessing_artifacts()
        
        # Create dataset metadata
        self._create_dataset_metadata()
        
        # Create upload instructions
        self._create_upload_instructions()
        
        print(f"Dataset prepared in: {self.output_dir}")
        
    def _copy_models(self):
        """Copy trained models to upload directory"""
        models_upload_dir = self.output_dir / "models"
        models_upload_dir.mkdir(exist_ok=True)
        
        # Copy from models directory
        for source_dir in [self.models_dir]:
            if source_dir.exists():
                for model_file in source_dir.glob("*"):
                    if model_file.is_file():
                        shutil.copy2(model_file, models_upload_dir)
                        print(f"Copied: {model_file.name}")
        
        # Create a combined models file if individual models exist
        self._create_combined_models_file(models_upload_dir)
        
    def _copy_preprocessing_artifacts(self):
        """Copy preprocessing artifacts (scalers, encoders, etc.)"""
        artifacts_dir = self.output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Look for common preprocessing artifacts
        artifact_patterns = [
            "scaler*.pkl", "scaler*.joblib",
            "encoder*.pkl", "encoder*.joblib", 
            "preprocessor*.pkl", "preprocessor*.joblib",
            "feature_names*.pkl", "feature_names*.joblib"
        ]
        
        for pattern in artifact_patterns:
            for artifact_file in self.project_root.rglob(pattern):
                if artifact_file.is_file():
                    shutil.copy2(artifact_file, artifacts_dir)
                    print(f"Copied artifact: {artifact_file.name}")
        
    def _create_combined_models_file(self, models_dir: Path):
        """Create a single file with all models for easy loading"""
        combined_models = {}
        
        # Load all model files
        for model_file in models_dir.glob("*.joblib"):
            model_name = model_file.stem
            try:
                model = joblib.load(model_file)
                combined_models[model_name] = model
                print(f"Added {model_name} to combined models")
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
        
        for model_file in models_dir.glob("*.pkl"):
            model_name = model_file.stem
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                combined_models[model_name] = model
                print(f"Added {model_name} to combined models")
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
        
        # Save combined models
        if combined_models:
            combined_file = models_dir / "all_models.pkl"
            with open(combined_file, 'wb') as f:
                pickle.dump(combined_models, f)
            print(f"Created combined models file: {combined_file}")
        
    def _create_dataset_metadata(self):
        """Create dataset-meta.json for Kaggle dataset"""
        metadata = {
            "title": "NeurIPS Polymer Prediction Models",
            "id": "your-username/neurips-polymer-models",
            "licenses": [{"name": "MIT"}],
            "keywords": ["neurips", "polymer", "prediction", "models"],
            "collaborators": [],
            "data": []
        }
        
        with open(self.output_dir / "dataset-metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
    def _create_upload_instructions(self):
        """Create instructions for uploading to Kaggle"""
        instructions = """
# Kaggle Dataset Upload Instructions

## 1. Install Kaggle API
```bash
pip install kaggle
```

## 2. Set up Kaggle API credentials
- Go to https://www.kaggle.com/account
- Create new API token
- Place kaggle.json in ~/.kaggle/

## 3. Create dataset
```bash
cd kaggle_upload
kaggle datasets create -p .
```

## 4. Update dataset (for subsequent uploads)
```bash
kaggle datasets version -p . -m "Updated models with improved performance"
```

## 5. Use in notebook
Add this dataset to your Kaggle notebook inputs:
- Dataset name: neurips-polymer-models
- Path in notebook: /kaggle/input/neurips-polymer-models/

## Files included:
- models/: Trained model files
- artifacts/: Preprocessing objects (scalers, encoders)
- all_models.pkl: Combined models file for easy loading

## Example usage in notebook:
```python
import pickle
with open('/kaggle/input/neurips-polymer-models/models/all_models.pkl', 'rb') as f:
    models = pickle.load(f)
```
"""
        
        with open(self.output_dir / "UPLOAD_INSTRUCTIONS.md", 'w') as f:
            f.write(instructions)
        
    def create_submission_package(self, notebook_name: str = "submission"):
        """Create a complete submission package"""
        package_dir = self.project_root / "kaggle_submission_package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy main submission notebook
        template_file = self.project_root / "kaggle_notebooks" / "templates" / "submission_template.py"
        if template_file.exists():
            shutil.copy2(template_file, package_dir / f"{notebook_name}.py")
        
        # Create submission instructions
        submission_instructions = f"""
# Kaggle Code Competition Submission

## Pre-submission Checklist:
1. Upload trained models as Kaggle dataset
2. Add dataset to notebook inputs
3. Test notebook execution (< 9 hours)
4. Verify output file is named 'submission.csv'
5. Ensure no internet dependencies

## Files:
- {notebook_name}.py: Main submission script
- Convert to notebook format in Kaggle

## Execution:
1. Create new Kaggle notebook
2. Add datasets: 
   - neurips-open-polymer-prediction-2025 (competition data)
   - neurips-polymer-models (your trained models)
3. Copy code from {notebook_name}.py
4. Submit notebook

## Notes:
- Maximum 9 hours execution time
- No internet access during execution
- Output must be 'submission.csv'
"""
        
        with open(package_dir / "SUBMISSION_INSTRUCTIONS.md", 'w') as f:
            f.write(submission_instructions)
        
        print(f"Submission package created in: {package_dir}")

def main():
    """Main execution"""
    preparer = KaggleDatasetPreparer()
    
    print("=" * 50)
    print("Kaggle Dataset Preparation")
    print("=" * 50)
    
    # Prepare dataset for upload
    preparer.prepare_dataset()
    
    # Create submission package
    preparer.create_submission_package()
    
    print("\n" + "=" * 50)
    print("Preparation completed!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Check kaggle_upload/ directory")
    print("2. Follow UPLOAD_INSTRUCTIONS.md")
    print("3. Use kaggle_submission_package/ for final submission")

if __name__ == "__main__":
    main()