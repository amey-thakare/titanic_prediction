

# Titanic Survival Prediction (Machine Learning Project)

##  Project Overview

This project predicts whether a passenger survived the Titanic disaster using Machine Learning algorithms.

The dataset used is the **Titanic dataset from Seaborn**.
Multiple classification models were implemented and compared based on accuracy and performance metrics.

---

##  Dataset Information

* Source: `seaborn.load_dataset("titanic")`
* Total records: **891 passengers**
* Target variable: `survived`
* Features used:

  * pclass
  * sex
  * age
  * sibsp
  * parch
  * fare
  * embarked
  * alone

---

##  Data Preprocessing Steps

* Dropped irrelevant columns:

  * deck
  * embark_town
  * alive
  * class
  * who
  * adult_male

* Filled missing values:

  * Age → replaced with mean
  * Removed rows with missing `embarked`

* Encoded categorical features:

  * sex → Label Encoding
  * embarked → Label Encoding

* Converted dataset to integer format

* Split dataset:

  * 80% Training
  * 20% Testing

* Applied StandardScaler (for KNN, SVM, Decision Tree)

---

##  Models Implemented

The following classification models were trained and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVM - RBF kernel)

---

##  Model Performance Comparison

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | 80.3%     |
| KNN                 | 79.2%     |
| Naive Bayes         | 77.5%     |
| Decision Tree       | 80.3%     |
| SVM                 | **81.4%** |

 **Best Performing Model: Support Vector Machine (SVM)**

Evaluation metrics used:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score

---

##  Sample Confusion Matrix (SVM)

```
[[91 18]
 [15 54]]
```

This shows strong performance in correctly classifying both survivors and non-survivors.

---

##  Technologies Used

* Python
* NumPy
* Pandas
* Seaborn
* Matplotlib
* Scikit-learn

---

## How to Run This Project

1. Clone the repository:

```
git clone https://github.com/yourusername/titanic_prediction.git
```

2. Install required libraries:

```
pip install numpy pandas seaborn matplotlib scikit-learn
```

3. Run the Jupyter Notebook:

```
jupyter notebook titanic_prediction.ipynb
```

---
