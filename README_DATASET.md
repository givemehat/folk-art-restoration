# Dataset Management (Git LFS & External Sources)

Due to GitHub's 100MB file limit, the full ~1.5 GB dataset and model checkpoints are managed using Git Large File Storage (LFS) or external hosting.

## 1. Using Git LFS (Recommended)
If the repository administrator has enabled LFS tracking for this repository:
```bash
# Install Git LFS
git lfs install

# Pull the large dataset and models
git lfs pull
```

## 2. Using the Downloader Script
If you prefer to reconstruct the dataset from public sources (Wikimedia Commons):
```bash
python data_utils/download_dataset.py --output_dir ../data/full --images_per_class 500
```
*Note: Depending on your network, you may encounter rate limits (429/403).*

## 3. Kaggle Dataset Integration (Recommended Alternative)
For reviewers or users without LFS, we recommend using the Kaggle **Indian Paintings Dataset**:
- **Kaggle**: [Indian Paintings Dataset](https://www.kaggle.com/datasets/ajg117/indian-paintings-dataset)

If you have the Kaggle API configured, you can download it directly:
```bash
kaggle datasets download -d ajg117/indian-paintings-dataset
unzip indian-paintings-dataset.zip -d data/full/
```

## Sample Dataset
A small, lightweight sample dataset (10-50 images) is committed directly to the repository under `data/sample/` for quick CI/CD testing and dry runs without needing LFS.
