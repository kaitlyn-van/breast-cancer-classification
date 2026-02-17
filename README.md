# 🧬 Breast Cancer Classification
A structured machine learning workflow for classifying breast tumors as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset.

This project demonstrates:
- Exploratory data analysis (EDA)
- Feature correlation analysis
- Model training
- Evaluation using accuracy, precision, ROC-AUC
- Confusion matrices
- ROC curves
- Train vs test comparison to assess overfitting

## 📊 Dataset
This project uses the Breast Cancer Wisconsin (Diagnostic) dataset.

**Source:**
UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset was accessed via:
```
sklearn.datasets.load_breast_cancer(as_frame=True)
```

## 🧪 Dataset Description
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

Each sample includes 30 real-valued features computed as the mean, standard error, and worst (largest value) of the following nucleus characteristics:
- Radius (mean distance from center to perimeter)
- Texture (standard deviation of grayscale values)
- Perimeter
- Area
- Smoothness (local variation in radius lengths)
- Compactness (perimeter² / area − 1)
- Concavity
- Concave points
- Symmetry
- Fractal dimension

Total samples: **569**
- 212 malignant
- 357 benign

Target encoding:
- 0 → Malignant
- 1 → Benign

## 📁 Project Structure
```
breast-cancer-classifier/
│
├── utils.py
├── breast_cancer_classification.ipynb
├── notebooks/
│   └── eda.ipynb
└── README.md
```

### 1. Run ```utils.py```
Contains helper functions for:
- Data loading
- Evaluation metrics
- Confusion matrix plotting
- ROC plotting

### 2. Run ```notebooks/eda.ipynb```
Exploratory data analysis:
- Pair plots
- Correlation heatmap

### 3. Run ```breast_cancer_classification.ipynb```
Model training and evaluation:
- Logistic Regression
- MLP Classifier
- Random Forest Classifier
- Confusion matrices
- ROC curves

## 📈 Results
| Model               | Accuracy | Precision | ROC-AUC |
|---------------------|----------|-----------|---------|
| Logistic Regression | 0.982    | 0.986     | 0.995   |
| Random Forest       | 0.956    | 0.959     | 0.992   |
| MLP                 | 0.895    | 0.969     | 0.985   |

**Train vs Test Accuracy:**
- Logistic Regression: 0.99 (train), 0.98 (test)
- Random Forest: 0.99 (train), 0.96 (test)
- MLP: 0.94 (train), 0.89 (test)
- No significant overfitting observed.

## 🔍 Observations
- Logistic regression achieved the highest ROC-AUC (0.995), indicating strong linear separability.
- Tree-based and neural network models did not significantly outperform logistic regression.

## ⚠️ Disclaimer

This project is for educational and research demonstration purposes only.
It is not intended for clinical decision-making.