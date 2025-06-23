# Code Competition Complete Guide

## Overview

This guide provides comprehensive instructions for participating in the NeurIPS Open Polymer Prediction 2025 code competition on Kaggle.

## Competition Format

- **Platform**: Kaggle Notebooks
- **Execution Time**: Maximum 9 hours
- **Internet Access**: Disabled during execution
- **Output Required**: `submission.csv`
- **External Data**: Allowed (public datasets, pre-trained models)

## Development Workflow

### Phase 1: Local Development

1. **Environment Setup**
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

2. **Exploratory Data Analysis**
   ```bash
   python kaggle_notebooks/templates/development/eda_template.py
   ```

3. **Feature Engineering**
   ```bash
   python kaggle_notebooks/templates/development/feature_engineering_template.py
   ```

4. **Model Comparison**
   ```bash
   python kaggle_notebooks/templates/development/model_comparison_template.py
   ```

5. **Model Training**
   ```bash
   make train
   # or
   uv run scripts/train_model.py
   ```

### Phase 2: Kaggle Preparation

1. **Prepare Models for Upload**
   ```bash
   python scripts/prepare_kaggle_dataset.py
   ```

2. **Upload Models as Kaggle Dataset**
   ```bash
   cd kaggle_upload
   kaggle datasets create -p .
   ```

3. **Create Submission Notebook**
   - Copy `kaggle_notebooks/templates/submission_template.py`
   - Adapt to your specific models and features
   - Convert to notebook format in Kaggle

### Phase 3: Final Submission

1. **Create new Kaggle notebook**
2. **Add required datasets**:
   - Competition data: `neurips-open-polymer-prediction-2025`
   - Your models: `your-username/neurips-polymer-models`
3. **Paste and adapt your code**
4. **Test execution** (save version and run)
5. **Submit to competition**

## Code Templates

### 1. EDA Template (`eda_template.py`)
- Data loading and basic statistics
- Missing value analysis
- Target variable distribution
- SMILES string analysis
- Feature correlation analysis

### 2. Feature Engineering Template (`feature_engineering_template.py`)
- Basic molecular features (atom counts, bonds)
- Advanced molecular features (complexity measures)
- Polymer-specific features (backbone, side chains)
- Statistical features (ratios, aggregations)

### 3. Model Comparison Template (`model_comparison_template.py`)
- Multiple algorithm comparison
- Cross-validation evaluation
- Hyperparameter optimization
- Ensemble creation
- Performance tracking

### 4. Submission Template (`submission_template.py`)
- Complete pipeline from data loading to submission
- Memory optimization functions
- Execution time monitoring
- Error handling and fallbacks

## Performance Optimization

### Memory Optimization
```python
def reduce_mem_usage(df):
    \"\"\"Reduce memory usage of dataframe\"\"\"
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    return df
```

### Execution Time Monitoring
```python
def timer(func):
    \"\"\"Decorator to time function execution\"\"\"
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f\"{func.__name__} took {end - start:.2f} seconds\")
        return result
    return wrapper
```

## Common Issues and Solutions

### Issue: Kernel Timeout (>9 hours)
**Solutions:**
- Reduce model complexity (fewer estimators, simpler models)
- Use fewer features (feature selection)
- Implement early stopping
- Use pre-trained models instead of training from scratch

### Issue: Memory Errors
**Solutions:**
- Use `reduce_mem_usage()` function
- Process data in chunks
- Delete unused variables (`del variable_name`)
- Use generators instead of loading all data

### Issue: Import Errors
**Solutions:**
- Check Kaggle environment documentation
- Use alternative libraries (e.g., `lightgbm` instead of `xgboost`)
- Implement fallback methods

### Issue: File Path Errors
**Solutions:**
- Use absolute paths: `/kaggle/input/dataset-name/`
- Double-check dataset names
- Verify all required files are included

## Best Practices

### 1. Code Organization
- Keep code modular and well-documented
- Use functions to avoid code repetition
- Implement error handling for robustness

### 2. Model Strategy
- Start with simple baseline models
- Gradually increase complexity
- Always have a working fallback solution

### 3. Feature Engineering
- Focus on domain-specific features (molecular descriptors)
- Create interaction features
- Use feature selection to reduce dimensionality

### 4. Validation Strategy
- Use robust cross-validation
- Monitor overfitting
- Validate on out-of-fold predictions

### 5. Submission Strategy
- Test multiple model versions
- Keep track of what works
- Submit your best cross-validation score

## Final Checklist

Before final submission:

- [ ] Code runs successfully from start to finish
- [ ] Execution time is under 9 hours
- [ ] Output file is named `submission.csv`
- [ ] Submission format matches sample submission
- [ ] No internet dependencies in the code
- [ ] All required datasets are added to notebook
- [ ] Error handling is implemented for edge cases
- [ ] Memory usage is optimized
- [ ] Code is well-documented and clean

## Troubleshooting

If your notebook fails:

1. **Check the logs** for specific error messages
2. **Reduce complexity** (fewer features, simpler models)
3. **Add memory optimization** throughout your pipeline
4. **Implement fallbacks** for when things go wrong
5. **Test incrementally** - comment out sections and test

## Resources

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks)
- [Competition Discussion Forum](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion)
- [Kaggle Learn Courses](https://www.kaggle.com/learn)

## Contact

For questions about this guide or the project structure, refer to the main README.md or create an issue in the repository.