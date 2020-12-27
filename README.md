# Distracted-Driver-Detection

# SML-Malaria-Detection

## Project Overview

This project is done as a part of `Machine Learning` Course.

Driving a car is a complex task, and it requires complete attention. Distracted driving is any activity that takes away the driver’s attention from the road. Approximately 1.35 million people die each year as a result of road traffic crashes.In this project our aim is to identify whether a driver is driving safely or indulged in distraction activities like texting, drinking etc.We show an in breadth & depth analysis of various features like **HOG, LBP, SURF, KAZE, pixel values** with feature reduction techniques **PCA, LDA** along with normalization techniques such as **min-max** over different classifiers such as **SVM XGBoost, Bagging, AdaBoost, K-Nearest Neighbors, Decision Trees** and compare their performance by tuning different hyperparameters. We evaluate the performance of these classifiers on metrics such as **Accuracy, Precision, Recall, F1 score and ROC**.

Project Poster can be found in [ML-Poster-Final.pdf](ML_Project_End_Term_PPT.pdf).

Project Report can be found in [ML_Project_EndTerm_Report.pdf](SML_Project_EndTerm_Report.pdf).

## Dataset

<div align="center"><img src="plots/dataset_vis.png" height='150px'/></div>

The dataset consists of 27,558 cell images; 13,780 images of infected and uninfected cells each and is taken from the official [NIH Website](https://ceb.nlm.nih.gov/repositories/malaria-datasets/).
You may also download it from [kaggle](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).

## Algorithm Used

<div align="center"><img src="plots/training_pipeline.png" /></div>

- Different combinations of feature sets were used, some of which are shown in Table 1 & 2 (**Ugly Duckling Theorem**) many other combinations were tried.
- Evaluated with different classifiers, model parameters were varied using **Grid Search** to find the best parameters (**No Free Lunch Theorem**).
- In PCA, number of components were preserved using **Elbow method over variance of PCA projected data** (Fig. 4).

## Evaluation Metrics and Results

Follwing are the results of the project: