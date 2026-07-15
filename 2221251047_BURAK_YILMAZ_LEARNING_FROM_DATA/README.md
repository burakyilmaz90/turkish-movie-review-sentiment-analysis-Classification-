# Sentiment Analysis Project

## How to Run (Notebook Order)

> Recommended: For each notebook, use **Kernel → Restart Kernel and Run All**.

### 1) Web Scraping
Run: `web_scraping.ipynb`  
Output: `raw_reviews.csv`

### 2) Data Preparation
Run: `Data_Prep.ipynb`  
Input: `raw_reviews.csv`  
Output: `processed_reviews.csv`

### 3) Feature Engineering
Run: `Feature_Eng.ipynb`  
Input: `processed_reviews.csv`  
Outputs:
- `X_features.pkl` (TF-IDF + custom features)
- `X_bow.pkl` (Bag-of-Words)
- `X_w2v.npy` (Word2Vec embeddings)
- `y_labels.pkl` (labels)

### 4) Modeling
Run: `Models.ipynb`  
Inputs: `X_features.pkl`, `X_bow.pkl`, `X_w2v.npy`, `y_labels.pkl`  
Output: `final_model.pkl`

### 5) Analysis & Evaluation
Run: `Analysis.ipynb`  
Inputs: model and feature files  
Outputs: learning curves, ROC-AUC plots, and error analysis.

## Environment Setup
Install required packages:

```bash
pip install -r requirements.txt
