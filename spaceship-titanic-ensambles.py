#!python
import numpy as np
import pandas as pd
from pandas import read_csv

# Models zoo !
from xgboost import XGBClassifier # to be used for ensambles stacking
# Models of the ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier # Bagged Decision Trees
from sklearn.tree import DecisionTreeClassifier # Bagged Decision Trees
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import ExtraTreesClassifier # Extremely randomised trees
from sklearn.ensemble import GradientBoostingClassifier # GradientBoosting
from sklearn.ensemble import AdaBoostClassifier # AdaBoosting

# Training and performance check
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Other utilities 
from math import nan

# General options
dummy_optimisation = True
variables_to_use_test = ['HomePlanet','CryoSleep','Destination','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Deck','Side','Group_size']
variables_to_use_train = ['HomePlanet','CryoSleep','Destination','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Deck','Side','Group_size','Transported'] 

# Data to be manipulated 
data = read_csv('spaceship-titanic/train.csv')
# Data for testing 
data_test = read_csv('spaceship-titanic/test.csv')
# Data for submission
data_sub = read_csv('spaceship-titanic/test.csv')

# ------------------------------------------- Features analysis

# Starting with Cabin, Cabin is defined as deck/num/side, define then 3 new columns in the dataframe based on these 

data['Deck'] = data['Cabin'].apply(lambda x: x.split('/')[0] if(x==x) else nan)
data['Num'] = data['Cabin'].apply(lambda x: int(x.split('/')[1]) if(x==x) else nan)
data['Side'] = data['Cabin'].apply(lambda x: x.split('/')[2] if(x==x) else nan)

data_test['Deck'] = data_test['Cabin'].apply( lambda x: x.split('/')[0] if(x==x) else nan)
data_test['Num'] = data_test['Cabin'].apply( lambda x: int(x.split('/')[1]) if(x==x) else nan)
data_test['Side'] = data_test['Cabin'].apply( lambda x: x.split('/')[2] if(x==x) else nan)

# Now look at PassengerId, decompose this into two parts, a group code and group size (see definition of PassengerId)
data['Group'] = data['PassengerId'].apply(lambda x: int(x.split('_')[0]) if(x==x) else nan)
data['Group_size'] = data['PassengerId'].apply(lambda x: int(x.split('_')[1]) if(x==x) else nan)
data_test['Group'] = data_test['PassengerId'].apply(lambda x: int(x.split('_')[0]) if(x==x) else nan)
data_test['Group_size'] = data_test['PassengerId'].apply(lambda x: int(x.split('_')[1]) if(x==x) else nan)

# Cathegorical data 
data['VIP'] = data['VIP'].map({False: 0, True: 1})
data['CryoSleep'] = data['CryoSleep'].map({False: 0, True: 1})
data['Transported'] = data['Transported'].map({False: 0, True: 1})
data_test['VIP'] = data_test['VIP'].map({False: 0, True: 1})
data_test['CryoSleep'] = data_test['CryoSleep'].map({False: 0, True: 1})

# Convert other categorical data objects into numbers
categorical_data = ['HomePlanet','Cabin','Destination', 'PassengerId']
for cat_data_to_cnv in categorical_data :
  print ("Handling now data category "+cat_data_to_cnv)
  data[cat_data_to_cnv] = pd.Categorical(data[cat_data_to_cnv]).codes
  data_test[cat_data_to_cnv] = pd.Categorical(data_test[cat_data_to_cnv]).codes

# ------------------------------------------- Final selection of the training features 

data = data[variables_to_use_train]
data_test = data_test[variables_to_use_test]
    
for element in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    data[element]=data[element].fillna(0)
    data_test[element]=data_test[element].fillna(0)

# ------------------------------------------- NaNs treatment 

data = data.dropna()
data_test=data_test.fillna(data_test.mean())

# Convert categorical Deck/Side data objects into numbers so we canlcualte their correlation and use it in the training
categorical_data_cabin = ['Deck','Side']
for cat_data_to_cnv in categorical_data_cabin:
    if cat_data_to_cnv in list(data.columns):
        print ("Handling now data category "+cat_data_to_cnv)
        data[cat_data_to_cnv] = pd.Categorical(data[cat_data_to_cnv]).codes
        data_test[cat_data_to_cnv] = pd.Categorical(data_test[cat_data_to_cnv]).codes

# ------------------------------------------- Training with 5-fold cross validation, implement sequentially all models  

X = data.drop('Transported', axis = 1)
y = data['Transported'] 
print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

model_logistic_regression = LogisticRegression()
model_svc = SVC(probability=True)               
model_bagging_trees = BaggingClassifier(estimator=DecisionTreeClassifier(), bootstrap=True)
model_random_forest = RandomForestClassifier()
model_extra_trees = ExtraTreesClassifier(bootstrap=True)
model_ada_boosted_trees = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm='SAMME.R')
model_gradient_boosted_trees = GradientBoostingClassifier()

parameters_logistic_regression = {
    'C': [1,0.5,1.5], # Default is 1
    'solver': ['lbfgs','liblinear'], # Default is lbfgs
    'max_iter': [100,500] # Default is 100    
}

parameters_svc = {
    'C': [1,0.5,1.5], # Default is 1
    'gamma': ['scale', 'auto'], # Default is scale
    'shrinking': [True, False] # Default is True
}

parameters_bagging_trees = {
    'n_estimators': [10,20,30], # Default is 10
    'max_samples': [0.3,0.6,1], # Default is 1
    'max_features': [0.3,0.6,1] # Default is 1
}

parameters_random_forest = {
    'n_estimators': [50,100,150], # Default is 100
    'criterion': ['gini', 'entropy', 'log_loss'], # Default is gini
    'max_depth': [None,3] # Default is None
}

parameters_extra_trees = {
    'criterion': ['gini', 'entropy', 'log_loss'], # Default is gini
    'max_depth': [None,3], # Default is None
    'min_samples_split': [2,4,8] # Default is 2
}

parameters_ada_boosted_trees = {
    'n_estimators': [10,20,30], # Default is 10
    'learning_rate': [0.5,1,1.5] # Default is 1
    #'max_features': [0.3,0.6,1] # Default is sqrt(N_features)
}

parameters_gradient_boosted_trees = {
    'max_depth': [None,3], # Default is None
    'n_estimators': [50,100,150], # Default is 100
    'learning_rate': [0.05,0.1,0.15], # Default is 0.1
}

models_ensemble=[]
models_ensemble.append([model_logistic_regression,'model_logistic_regression',parameters_logistic_regression])      
models_ensemble.append([model_svc,'model_svc',parameters_svc])                                      
models_ensemble.append([model_bagging_trees,'model_bagging_trees',parameters_bagging_trees])                  
models_ensemble.append([model_random_forest,'model_random_forest',parameters_random_forest])                  
models_ensemble.append([model_extra_trees,'model_extra_trees',parameters_extra_trees])                      
models_ensemble.append([model_ada_boosted_trees,'model_ada_boosted_trees',parameters_ada_boosted_trees])          
models_ensemble.append([model_gradient_boosted_trees,'model_gradient_boosted_trees',parameters_gradient_boosted_trees])

parameters = {}
scores = []
ensemble_pars = []
# Train all models and store all predictions into the original dataframe 
for mod in models_ensemble:
    # Save the name of each score to be used later as column name
    print(X_train.head())
    scores.append(mod[1])
    cv = GridSearchCV(mod[0], mod[2], cv=5)
    cv.fit(X_train, y_train)
    y_pred = cv.best_estimator_.predict(X_test)
    predictions = [value for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print('Accuracy for model '+mod[1]+': %.2f%%' % (accuracy * 100.0))
    print('Best hyperparameters setting')
    print(cv.best_params_)
    ensemble_pars.append(cv.best_params_)
    
    y_prediction_training = cv.best_estimator_.predict_proba(X)
    y_prediction_train = [value[1] for value in y_prediction_training]
    data[mod[1]] = np.array(y_prediction_train)

# Model for the stacking ! We will have also to redefine the data so the training runs only on the scores of the classifiers in the ensemble
model = XGBClassifier()

X = data[scores] 
y = data['Transported'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

if dummy_optimisation:
    parameters = {
        'max_depth': [1,2,3] # Default is 6 
    }
else:
    parameters = {
        'eta': [0.1*i for i in range(1, 6)], # Default is 0.3
        'gamma': [0.05*i for i in range(0, 3)], # Default is 0 
        'max_depth': [4,5,6,7,8], # Default is 6 
        'max_leaves': [0,1,2] # Default is 0 
    }    
    
cv = GridSearchCV(model, parameters, cv=5)

cv.fit(X_train, y_train)

print(cv.best_estimator_)
y_pred = cv.best_estimator_.predict(X_test)
print('Prediction length = '+str(len(y_pred)))
print('Labels length = '+str(len(y_test)))
predictions = [value for value in y_pred]
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(list(X.columns))
print(cv.best_params_)

# ------------------------------------------- Final prediction

# Refit each model of the ensemble to the full training set and store scores in the testing dataset 

for index in range (0,len(models_ensemble)):
    print((models_ensemble[index])[0])
    print((models_ensemble[index])[1])
    print((models_ensemble[index])[2])

    model_ensemble = 0 
    if (models_ensemble[index])[1] == 'model_logistic_regression':
        model_ensemble = LogisticRegression(C=1,max_iter=100,solver='lbfgs')
    if (models_ensemble[index])[1] == 'model_svc':
        model_ensemble = SVC(probability=True,C=1.5,gamma='scale',shrinking=True)               
    if (models_ensemble[index])[1] == 'model_bagging_trees':
        model_ensemble = BaggingClassifier(estimator=DecisionTreeClassifier(), bootstrap=True,max_features=0.6,max_samples=0.3,n_estimators=20)
    if (models_ensemble[index])[1] == 'model_random_forest':
        model_ensemble = RandomForestClassifier(criterion='gini',max_depth=None,n_estimators=150)
    if (models_ensemble[index])[1] == 'model_extra_trees':
        model_ensemble = ExtraTreesClassifier(bootstrap=True,criterion='entropy',max_depth=None,min_samples_split=8)
    if (models_ensemble[index])[1] == 'model_ada_boosted_trees':
        model_ensemble = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm='SAMME.R',learning_rate=0.5,n_estimators=30)
    if (models_ensemble[index])[1] == 'model_gradient_boosted_trees':
        model_ensemble = GradientBoostingClassifier(learning_rate=0.1,max_depth=3,n_estimators=100)

    model_ensemble.fit(data[variables_to_use_train].drop('Transported',axis=1),data['Transported'])
    y_prediction_test = model_ensemble.predict_proba(data_test[variables_to_use_test])
    y_prediction_testing = [value[1] for value in y_prediction_test]    
    print('y_prediction_testing '+str(len(y_prediction_testing)))
    data_test[(models_ensemble[index])[1]] = np.array(y_prediction_testing)

# Refit the XGBoost model to the full training set
X_train = data[scores]
y_train = data['Transported'] 
model = XGBClassifier(eta=0.1,gamma=0.0,max_depth=4,max_leaves=0)
model.fit(X_train,y_train)

# Append scores of the XGBoost model to the full testing set
y_final_prediction = model.predict(data_test[scores])
predictions = [True if value== 1 else False for value in y_final_prediction]
data_submission = data_sub
data_submission['Transported']=pd.Series(predictions)
data_submission = data_submission[['PassengerId','Transported']]
data_submission.to_csv('submission.csv', index=False)
