# Credit Risk Analysis

## Overview

Several supervised machine-learning algorithms were employed to determine which method would be the best predictor of credit card risk. Each of the methods had to contend with an imbalanced data set, since low-risk loans far outnumber high-risk ones.

## Software

The analysis made use of the following software:
- Python
  - Collections
  - Imbalanced-learn
    - combine
    - ensemble
    - metrics
    - over_sampling
    - under_sampling
  - Numpy
  - Pandas
  - PathLib
  - Scikit-learn
    - linear_model
    - metrics
    - model_selection

## Analysis

We analyzed date from over 68,000 loan applications from the first quarter of 2019. 75% of this data was used to train the algorithms to determine the risk level (high or low) of an application; the remaining 25% was used to test each model's predictive effectiveness after training.

Each model examined 95 features from the loans in its training and predicting process, and three statistics were considered: accuracy, precision, and recall. Precision and recall were further subdivided by high-risk and low-risk loan classifications, for a final tally of five considered statistics in total.

The meanings of the statistics are as follows:
- **Accuracy:** the overall proportion of predictions that match their loans' actual classification (high-risk or low-risk)
- **Precision:** for a given predicted class (high-risk or low-risk), the proportion of predicted loans whose actual class matches the prediction
- **Recall:** for a given *actual* loan class, the proportion of predictions on those loans that were correctly assessed

## Results

### Statistics by Method

The accuracy, precision, and recall scores from each of the machine learning tests are listed below (and the numeric results each link to screenshots from the corresponding test's output).

- Oversampling
  - Naïve Random Oversampling
    - Accuracy: [0.661](./Images/01_RO_accuracy.png)
    - Precision
      - High-Risk: [0.01](./Images/01_RO_class_rept.png)
      - Low-Risk: [1.00](./Images/01_RO_class_rept.png)
    - Recall
      - High-Risk: [0.72](./Images/01_RO_class_rept.png)
      - Low-Risk: [0.60](./Images/01_RO_class_rept.png)
  - SMOTE Oversampling
    - Accuracy: [0.658](./Images/02_SMOTE_accuracy.png)
    - Precision
      - High-Risk: [0.01](./Images/02_SMOTE_class_rept.png)
      - Low-Risk: [1.00](./Images/02_SMOTE_class_rept.png)
    - Recall
      - High-Risk: [0.62](./Images/02_SMOTE_class_rept.png)
      - Low-Risk: [0.69](./Images/02_SMOTE_class_rept.png)
- Undersampling
  - Cluster Centroids
    - Accuracy: [0.544](./Images/03_cluster_accuracy.png)
    - Precision
      - High-Risk: [0.01](./Images/03_cluster_class_rept.png)
      - Low-Risk: [1.00](./Images/03_cluster_class_rept.png)
    - Recall
      - High-Risk: [0.69](./Images/03_cluster_class_rept.png)
      - Low-Risk: [0.40](./Images/03_cluster_class_rept.png)
- Combination Over- and Undersampling
  - SMOTE-ENN
    - Accuracy: [0.671](./Images/04_SMOTEENN_accuracy.png)
    - Precision
      - High-Risk: [0.01](./Images/04_SMOTEENN_class_rept.png)
      - Low-Risk: [1.00](./Images/04_SMOTEENN_class_rept.png)
    - Recall
      - High-Risk: [0.77](./Images/04_SMOTEENN_class_rept.png)
      - Low-Risk: [0.57](./Images/04_SMOTEENN_class_rept.png)
- Ensemble
  - Random Forest Classifier
    - Accuracy: [0.789](./Images/05_random_forest_accuracy.png)
    - Precision
      - High-Risk: [0.03](./Images/05_random_forest_class_rept.png)
      - Low-Risk: [1.00](./Images/05_random_forest_class_rept.png)
    - Recall
      - High-Risk: [0.70](./Images/05_random_forest_class_rept.png)
      - Low-Risk: [0.87](./Images/05_random_forest_class_rept.png)
  - AdaBoost Classifier
    - Accuracy: [0.932](./Images/06_AdaBoost_accuracy.png)
    - Precision
      - High-Risk: [0.09](./Images/06_AdaBoost_class_rept.png)
      - Low-Risk: [1.00](./Images/06_AdaBoost_class_rept.png)
    - Recall
      - High-Risk: [0.92](./Images/06_AdaBoost_class_rept.png)
      - Low-Risk: [0.94](./Images/06_AdaBoost_class_rept.png)

\[Note that each of the above statistics is a rounded value. Thus, a displayed value of 1.00, for example, could actually represent any value ≥0.995.\]

### Maximum Statistics among Methods

The following list shows which method had the highest value for each of the assessed statistics among the methods employed:
- Accuracy: AdaBoost Classifier (0.932)
- Precision
  - High-Risk: AdaBoost Classifier (0.09)
  - Low-Risk: tie (all methods 1.00)
- Recall
  - High-Risk: AdaBoost Classifier (0.92)
  - Low-Risk: AdaBoost Classifier (0.94)

## Summary

### The Winner

The **AdaBoost Classifier** is the clear winner among the six methods employed in these tests, achieving the highest score among the methods in each of the five statistics. Further, it was the only method that achieved recall scores of more than 90% in both high- and low-risk categories.

#### Considerations

The AdaBoost Classifier's weakest statistic is its high-risk precision of 9% (0.09)—which is still triple its next-closest competitor's score in the same category. This relatively low score means that about 91% of loans the Classifier flags as high-risk actually aren't, which may cause many otherwise-acceptable loans to be denied. However, the inherent tension between precision and recall means that increasing precision in this area would necessarily reduce high-risk recall, which could cause the lender to issue more loans to high-risk borrowers than it might want to. Under the circumstances, it's probably better to deny more acceptable loans than to issue too many unknowingly high-risk ones.