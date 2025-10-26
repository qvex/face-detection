# Phase 2: Baseline Testing - Indian Face Datasets

## Overview
Phase 2 tests 4 pre-trained face recognition models on Indian faces to establish performance baselines before fine-tuning.

## Models Tested
1. **InsightFace Buffalo_L** - SOTA face recognition with ArcFace loss
2. **DeepFace VGG-Face** - Deep CNN trained on VGGFace dataset
3. **DeepFace Facenet512** - Google's FaceNet with 512-dim embeddings
4. **Dlib ResNet** - HOG-based face detection + ResNet embeddings

## Dataset Sources (Indian Faces Only)
1. **IMFDB** (Indian Movie Face Database)
   - Source: Kaggle
   - Images: 34,512
   - Size: ~3.5 GB
   - URL: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset

2. **IFExD** (Indian Face Expression Database)
   - Source: GitHub
   - Images: ~1,000+
   - Size: ~500 MB
   - URL: https://github.com/ravi0531rp/IFExD

## Step-by-Step Instructions

### Step 1: Download Indian Face Datasets

**Option A: Download IMFDB from Kaggle (Recommended)**

1. Create Kaggle account at https://www.kaggle.com
2. Generate API token:
   - Go to Account > Create New API Token
   - Download `kaggle.json`
3. Place `kaggle.json` in:
   - Windows: `C:\Users\<username>\.kaggle\`
   - Linux/Mac: `~/.kaggle/`
4. Run download script:
   ```bash
   venv\Scripts\python.exe scripts\download_indian_datasets.py
   ```
5. Select option 1 for IMFDB
6. Wait for download (~3.5 GB, 5-15 minutes)

**Option B: Clone IFExD from GitHub**

1. Ensure Git is installed
2. Run download script:
   ```bash
   venv\Scripts\python.exe scripts\download_indian_datasets.py
   ```
3. Select option 2 for IFExD
4. Wait for clone (~500 MB, 2-5 minutes)

### Step 2: Create Test Subset

Extract 100 images (10 individuals, 10 images each) for baseline testing:

```bash
venv\Scripts\python.exe scripts\create_test_subset.py
```

- Select dataset source (1=IMFDB, 2=IFExD)
- Script will randomly select 10 identities
- Output: `data/test/person_001/` through `data/test/person_010/`

### Step 3: Run Baseline Tests

Test all 4 models on the test subset:

```bash
venv\Scripts\python.exe scripts\run_baseline_tests.py
```

**Expected Runtime**: 10-20 minutes total
- InsightFace: 2-3 minutes (fastest, GPU-accelerated)
- DeepFace VGG-Face: 3-5 minutes
- DeepFace Facenet512: 3-5 minutes
- Dlib: 4-6 minutes

**Expected Output**:
```
BASELINE TEST RESULTS SUMMARY
======================================================================
Model                        Accuracy    Speed (ms)      FMR
----------------------------------------------------------------------
InsightFace_Buffalo_L            92.0%        45.23     8.0%
DeepFace_VGG-Face                85.5%       120.45    12.5%
DeepFace_Facenet512              88.2%        98.67    10.3%
DeepFace_Dlib                    82.1%       150.23    15.2%
======================================================================

Best Accuracy: InsightFace_Buffalo_L (92.0%)
Fastest Model: InsightFace_Buffalo_L (45.23ms)
```

## Metrics Explained

### Accuracy
Percentage of correctly identified faces compared to enrolled faces.
- **Target**: >90% on Indian faces
- **Calculation**: (Correct Matches / Total Tests) * 100

### Speed (ms per face)
Average time to process one face (detection + embedding extraction).
- **Target**: <100ms per face
- **GPU Accelerated**: InsightFace uses CUDA if available

### False Match Rate (FMR)
Percentage of incorrect matches (different person identified as enrolled person).
- **Target**: <10%
- **High FMR**: Model not discriminative enough, threshold too low

## Expected Results

### Typical Baseline Performance (Indian Faces)
Based on project requirements and SOTA models:

| Model | Accuracy | Speed | FMR | Notes |
|-------|----------|-------|-----|-------|
| InsightFace Buffalo_L | 88-93% | 30-50ms | 7-10% | Best overall, GPU accelerated |
| DeepFace VGG-Face | 82-87% | 100-150ms | 12-16% | Slower, less accurate |
| DeepFace Facenet512 | 85-90% | 80-120ms | 10-13% | Good balance |
| Dlib ResNet | 80-85% | 120-180ms | 14-18% | Slowest, lowest accuracy |

### Fine-Tuning Target
After fine-tuning on IMFDB + IFExD + custom data (Phase 6):
- **Accuracy**: 95%+ (5-8% improvement)
- **Speed**: <100ms (maintain or improve)
- **FMR**: <5% (reduce by half)

## Troubleshooting

### Kaggle API Issues
**Problem**: `ModuleNotFoundError: No module named 'kaggle'`
**Solution**: Install Kaggle API
```bash
venv\Scripts\python.exe -m pip install kaggle
```

**Problem**: `OSError: Could not find kaggle.json`
**Solution**: Download API token and place in correct directory (see Step 1)

**Problem**: `403 Forbidden`
**Solution**: Accept dataset rules on Kaggle website first

### Git Clone Issues
**Problem**: `'git' is not recognized`
**Solution**: Install Git from https://git-scm.com/

**Problem**: `Repository not found`
**Solution**: Check internet connection, verify URL is correct

### Model Loading Issues
**Problem**: `Failed to load InsightFace`
**Solution**: Model will auto-download on first run, ensure internet connection

**Problem**: CUDA out of memory
**Solution**: Close other GPU applications, reduce batch processing

### No Test Images Found
**Problem**: `Test directory not found`
**Solution**: Run `scripts/create_test_subset.py` first

## Next Steps

After completing Phase 2:

1. **Analyze Results**: Identify best performing baseline model
2. **Document Baseline**: Record metrics for comparison after fine-tuning
3. **Proceed to Phase 3**: Full dataset acquisition (IMFDB + IFExD)
4. **Phase 4**: Preprocessing pipeline design
5. **Phase 5**: Training setup with chosen model architecture
6. **Phase 6**: Fine-tuning on Indian face datasets

## Files Created

### Scripts
- `scripts/download_indian_datasets.py` - Download IMFDB and IFExD datasets
- `scripts/create_test_subset.py` - Extract 100-image test subset
- `scripts/run_baseline_tests.py` - Test 4 models and generate comparison report

### Data Structure
```
data/
├── raw/
│   ├── IMFDB/          # 34,512 images (if downloaded)
│   └── IFExD/          # ~1,000 images (if cloned)
└── test/               # 100 images for baseline testing
    ├── person_001/     # 10 images
    ├── person_002/     # 10 images
    ...
    └── person_010/     # 10 images
```

## Time Estimate

- **Dataset Download**: 10-30 minutes (IMFDB) or 5 minutes (IFExD)
- **Test Subset Creation**: 1-2 minutes
- **Baseline Testing**: 10-20 minutes
- **Total**: 20-50 minutes

## Success Criteria

Phase 2 is complete when:
- [ ] At least one Indian face dataset downloaded (IMFDB or IFExD)
- [ ] Test subset created (100 images, 10 individuals)
- [ ] All 4 baseline models tested successfully
- [ ] Results documented (accuracy, speed, FMR per model)
- [ ] Best model identified for fine-tuning in Phase 6

---

For questions or issues, refer to project-reqs.md for complete implementation plan.
