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
