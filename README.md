# Distracted-Driver-Detection

# SML-Malaria-Detection

## Project Overview

This project is done as a part of `Machine Learning` Course.

Driving a car is a complex task, and it requires complete attention. Distracted driving is any activity that takes away the driver’s attention from the road. Approximately 1.35 million people die each year as a result of road traffic crashes.In this project our aim is to identify whether a driver is driving safely or indulged in distraction activities like texting, drinking etc.We show an in breadth & depth analysis of various features like **HOG, LBP, SURF, KAZE, pixel values** with feature reduction techniques **PCA, LDA** along with normalization techniques such as **min-max** over different classifiers such as **SVM XGBoost, Bagging, AdaBoost, K-Nearest Neighbors, Decision Trees** and compare their performance by tuning different hyperparameters. We evaluate the performance of these classifiers on metrics such as **Accuracy, Precision, Recall, F1 score and ROC**.

Project Poster can be found in [ML-Poster-Final.pdf](ML_Project_End_Term_PPT.pdf).

Project Report can be found in [ML_Project_EndTerm_Report.pdf](ML_Project_End_Term_Report.pdf).

## Dataset
The dataset contains 22424 driver images in total download from [kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data). The dataset contains coloured images of size 640*480 pixels which are resized to 64*64 coloured images for training and testing pusposes.
Stratified splitting is used to split the dataset into 80:10 Training-Testing ratio. The training dataset is further split into 90:10 Training-Validation set.

The 10 classes to predict are:
- Safe driving 
- Texting(right hand) 
- Talking on the phone (right hand)
- Texting (left hand) 
- Talking on the phone (left hand)
- Operating the radio 
- Drinking 
- Reaching behind 
- Hair and makeup  
- Talking to passenger(s). 

<div align="center"><img src="plots/classes_imgs.jpg" height='150px'/></div>


## Algorithm Used

<div align="center"><img src="plots/training_pipeline.png" /></div>

- Different combinations of feature sets were used, some of which are shown in Table 1 & 2 (**Ugly Duckling Theorem**) many other combinations were tried.
- Evaluated with different classifiers, model parameters were varied using **Grid Search** to find the best parameters (**No Free Lunch Theorem**).
- In PCA, number of components were preserved using **Elbow method over variance of PCA projected data** (Fig. 4).

## Evaluation Metrics and Results

Follwing are the results of the project: