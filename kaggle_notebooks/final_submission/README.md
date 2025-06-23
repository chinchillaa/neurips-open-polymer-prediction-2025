# Final Submission Guidelines

## Code Competition Workflow

### 1. Local Development → Kaggle Notebook Migration

1. **Develop locally** using the templates in `kaggle_notebooks/development/`
2. **Test your pipeline** with the full dataset 
3. **Optimize for execution time** (max 9 hours)
4. **Convert to Kaggle notebook format**

### 2. Pre-Submission Checklist

- [ ] All models trained and saved as Kaggle dataset
- [ ] No internet dependencies in final code
- [ ] Execution time < 9 hours (test locally)
- [ ] Output file named `submission.csv`
- [ ] All required libraries available in Kaggle environment
- [ ] Code is self-contained and reproducible

### 3. Kaggle Dataset Upload

Use `scripts/prepare_kaggle_dataset.py` to prepare models for upload:

```bash
cd /path/to/project
python scripts/prepare_kaggle_dataset.py
```

Follow the generated instructions in `kaggle_upload/UPLOAD_INSTRUCTIONS.md`

### 4. Final Submission Steps

1. **Create new Kaggle notebook**
2. **Add datasets**:
   - `neurips-open-polymer-prediction-2025` (competition data)
   - `your-username/neurips-polymer-models` (your trained models)
3. **Copy code** from `submission_template.py`
4. **Test execution** (save version, run)
5. **Submit to competition**

### 5. Execution Time Optimization Tips

- Use efficient data types (`float32` instead of `float64`)
- Implement memory optimization functions
- Use pre-trained models instead of training from scratch
- Parallelize computations where possible
- Monitor memory usage and clean up variables

### 6. Common Issues and Solutions

**Issue**: Kernel timeout (>9 hours)
- **Solution**: Reduce model complexity, use fewer features, or implement early stopping

**Issue**: Memory errors
- **Solution**: Use `reduce_mem_usage()` function, process data in chunks

**Issue**: Missing libraries
- **Solution**: Check Kaggle environment docs, use alternative implementations

**Issue**: File not found errors
- **Solution**: Double-check dataset paths, ensure all files are included

### 7. Backup Strategy

Always maintain multiple notebook versions:
- Development version (with comments and debugging)
- Optimized version (for final submission)
- Fallback version (simple baseline that works)

### 8. Final Validation

Before final submission:
1. Run notebook from scratch
2. Verify output format matches sample submission
3. Check execution time
4. Validate submission file integrity