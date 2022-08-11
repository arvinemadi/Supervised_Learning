## Probability calibration for unbalanced data classification
Dataset: [Link](https://www.nature.com/articles/sdata201635.pdf)

Two probability calibration methods of Platt and Beta are used to calibrate the scores from three models. The three models tried for the binary classification in this unbalanced dataset are Random Forrest, a NN with three dense layers, and a Logistic Regression.

## Bayesian Neural Network
A simple BNN for multiclass classification is demonstrated.
Dataset: [Data](https://archive.ics.uci.edu/ml/datasets/wine)

Some of the theory behind BNN at high level is shown in the notebook. Reference given to [Link](https://arxiv.org/pdf/2007.06823.pdf).
BNN is a powerful tool to use when the amount of available training data is small and we want to avoid overfitting. Also the uncertainities in the predictions can be estimated which make them popular for predictions in critical applications.

## Feature Selection: manually or using trees
A simple example on manual feature selection by doing data exploration of numerical and categorical variables on a small dataset.
If the dataset or number of features are too big the manual approach can be difficult. Tree based methods such as random forrest can be used to score the features based on their important. The notebook shows a simple example on this small dataset.

## XGBoost
A simple example of using XGboost on two standard datasets from sklearn. XGBoost is a more recent advanced tree-based algorithm that can be used for regression, classification, and feature importance ranking/selection. Generally, it can perform better than random forrest.

Two issues with XGBoost:
- unlike random forrest, it may not be easy to do parallel processing in case of very large distributed data.
- many hyper-parameters.

## AutoGluon

[AutoGluon](https://auto.gluon.ai/stable/index.html) is an open-source AutoML framework that requires only a single line of Python to train highly accurate machine learning models on an unprocessed tabular dataset such as a CSV file. [Paper](https://arxiv.org/abs/2003.06505) [AWS blog](https://aws.amazon.com/blogs/opensource/machine-learning-with-autogluon-an-open-source-automl-library/)
It has the capability of using many different models and automatically stack and create and ensemble model from them. It could be a solution to many problems or at least the first step if the models make sense for the data for the problem. Good feature engineering before using the model could be critical.

Dataset: [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)

This is a quick test of AutoGluon on Sagemake and did not spent too much time on feature selection and cleaning.

## LSTM Model for Non-Linear Regression
A simple LSTM based model for non-linear regression is test on two datasets. The model is implemented with pytorch.
- First data is [AirPassenger dataset](https://www.kaggle.com/datasets/rakannimer/air-passengers)
- Second the model is test for prediction of non-linear motion in presense of noise.

It seems this type of models could be a good alternative for traditional estimators such as Kalman Filters.

I found [this](https://github.com/MohammadFneish7/Keras_LSTM_Diagram) illustration for LSTM very cool. The explanation should not be confused when using torch for implementing LSTM as the term are different than Keras.
