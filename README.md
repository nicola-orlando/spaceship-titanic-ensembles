# Ensemble stacking for the spaceship-titanic Kaggle challenge

## Introduction

This notebook includes the implementation of a stacking ensemble model to make predictions on the spaceship-titanic dataset from the corresponding Kaggle challenge: https://www.kaggle.com/competitions/spaceship-titanic. 
The notebook entirely focus on the machine learning model implementation and leave aside all processes about data cleaning, feature extraction, EDA. I discussed these aspects in a previous notebook: https://www.kaggle.com/code/nicolaorlandowork/spaceship-titanic-everything-all-at-once. From this older notebook I define the following baseline setup in terms of features, NaNs and outliers treatment: 
1. Outliers are kept 
2. NaNs are removed from the training dataset while average imputation is used to remove NaNs from the testing data to be used at submission time. In the training set, if a NaN originates in one of the numerical columns ['HomePlanet','Cabin','Destination', 'PassengerId'] I use zero-imputation
3. The variables used are ['HomePlanet','CryoSleep','Destination','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Deck','Side','Group_size']

## Model definition 

The model I will use in a Stacked ensemble, the model setup is illustrated in the following figure. 
The stacked ensemble uses seven models to analyse the features of the dataset. Each model outputs a score as input of a second upstream model (XGBoost) that is then trained using the scores of each model of the ensemble. 
The training is performed for all models using a 5-fold cross validation method, with 40% allocated to the training set, and 30% for testing and validation set. 
In all cases, after the best hyperparameter is found the model is refit to the full training dataset. To be noted that each model is optimised individually instead of optimising the full ensemble as whole. 

### Models of the ensembles 

Here I list the models I used along with the hyperparameters I decided to tune. 

1. Logistic Regression 
- Regularisation parameter (C): 'C': [1,0.5,1.5], # Default is 1
- Solver (solver): 'solver': ['lbfgs','liblinear'], # Default is lbfgs
- Maximum number of iterations for the solver (max_iter): 'max_iter': [100,500] # Default is -1

2. Support Vector Classifier 
- Regularisation parameter (C): 'C': [1,0.5,1.5], # Default is 1
- Coefficient for the used kernel (Gaussian Radial Basis function, RBF, default in SVC): 'gamma': ['scale', 'auto'], # Default is scale
- shrinking: 'shrinking': [True, False] # Default is True

3. Bagged Decision Trees
- number of trees (n_estimators): 'n_estimators': [10,20,30], # Default is 10
- maximum number of data samples in each bagged sample (max_samples): 'max_samples': [0.3,0.6,1], # Default is 1
- maximum number of used features at each node split (max_features): 'max_features': [0.3,0.6,1] # Default is 1  

4. Random Forest Classifier
- number of trees in the forest (n_estimators): 'n_estimators': [50,100,150], # Default is 100
- node splitting criterion (criterion): 'criterion': ['gini', 'entropy', 'log_loss'], # Default is gini
- maximum depth (max_depth): 'max_depth': [None,3] # Default is None

5. Extreme Trees Classifier
- node splitting criterion (criterion): 'criterion': ['gini', 'entropy', 'log_loss'], # Default is gini
- maximum depth (max_depth): 'max_depth': [None,3] # Default is None
- minimum number of samples to further split a node (min_samples_split): 'min_samples_split': [2,4,8] # Default is 2
 
6. Ada Boosted Decision Trees Classifier 
- number of trees in the in the sequence (n_estimators): 'n_estimators': [10,20,30], # Default is 10
- Learning rate (learning_rate): 'learning_rate': [0.5,1,1.5] # Default is 1

7. Gradient Boosted Decision Trees Classifier 
- maximum depth (max_depth): 'max_depth': [None,3], # Default is None
- number of trees in the sequence (n_estimators): 'n_estimators': [50,100,150], # Default is 100
- Learning rate (learning_rate): 'learning_rate': [0.05,0.1,0.15], # Default is 0.1

I report the result of the hyperparameter scan in the table below, including the estimated accuracy on the 30% of the training sample allocated for testing (not the actual test sample for the submission in Kaggle). 

Model | Estimated accuracy | Best hyperparameters 
--- | --- | --- 
 model_logistic_regression    | 77.67 | {'C': 1, 'max_iter': 100, 'solver': 'lbfgs'}
 model_svc    | 78.64 | {'C': 1.5, 'gamma': 'scale', 'shrinking': True}
 model_bagging_trees    | 78.76 | {'max_features': 0.6, 'max_samples': 0.3, 'n_estimators': 20}
 model_random_forest    | 79.66 | {'criterion': 'gini', 'max_depth': None, 'n_estimators': 150}
 model_extra_trees    | 80.33 | {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 8}
 model_ada_boosted_trees    | 79.30 | {'learning_rate': 0.5, 'n_estimators': 30}
 model_gradient_boosted_trees    | 79.12 | {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
 
 ## Final model 
 
The terminal model will be a XGBoost model. The optimised hyperparameters are: 
- (eta): 'eta': [0.1*i for i in range(1, 6)], # Default is 0.3
- (gamma): 'gamma': [0.05*i for i in range(0, 3)], # Default is 0 
- (max_depth): 'max_depth': [4,5,6,7,8], # Default is 6 
- (max_leaves): 'max_leaves': [0,1,2] # Default is 0 
 The best fit model has 79.48% accuracy on the partitioned testing dataset and best hyperparameters {'eta': 0.1, 'gamma': 0.0, 'max_depth': 4, 'max_leaves': 0}. 
 
 ## Submission 
 
 At this stage all models of the ensemble are retrained. The final score I obtain on the testing dataset is 0.79588.  
