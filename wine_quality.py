# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:26:27 2022

@author: OZAN

"""

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

df = pd.read_csv('WineQT.csv').dropna()
df

df['quality'].value_counts()

x = df.drop('quality', axis=1) #Feature matrix 
x = x.drop('Id', axis=1)       # To eliminate unnecessary column
y = df['quality']              # Label vector
df['quality'].value_counts()   #To see distribution of label (quantity number)

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

" Selection of Hiperparameters, manually (hidden_layer_sizes=(100,50,30),max_iter=300,activation=relu,solver=adam"

mlp_clf = MLPClassifier(hidden_layer_sizes=(100,50,30),max_iter = 300,activation = 'relu',solver = 'adam')

mlp_clf.fit(trainX_scaled, trainY)
y_pred = mlp_clf.predict(testX_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
fig = plot_confusion_matrix(mlp_clf, testX_scaled, testY, display_labels=mlp_clf.classes_)
fig.figure_.suptitle("Confusion Matrix for Wine Quality Dataset")
plt.show()
    
print(classification_report(testY, y_pred))

plt.plot(mlp_clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

" Hiperparameter Tuning using GridSearchCV"

param_grid = {
    'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
    'max_iter': [50, 100, 150],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','invscaling','adaptive'],
}

grid = GridSearchCV(mlp_clf, param_grid, n_jobs= -1, cv=5)
grid.fit(trainX_scaled, trainY)
print(grid.best_params_) 
grid_predictions = grid.predict(testX_scaled) 
print('Accuracy: {:.2f}'.format(accuracy_score(testY, grid_predictions)))

" We train the model using the selected hyperparameters. "

mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50),max_iter = 150,alpha= 0.0001,
                        activation = 'relu',solver = 'adam',learning_rate='invscaling')

mlp_clf.fit(trainX_scaled, trainY)
y_pred = mlp_clf.predict(testX_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
fig = plot_confusion_matrix(mlp_clf, testX_scaled, testY, display_labels=mlp_clf.classes_)
fig.figure_.suptitle("Confusion Matrix for Wine Quality Dataset")
plt.show()
    
print(classification_report(testY, y_pred))

plt.plot(mlp_clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

" SMOTE : Synthetic Minority Oversampling Technique || Data Augmentation Method (For Each Class has 500 datas) "

sm = SMOTE(random_state=42)

oversample = SMOTE(sampling_strategy = {5: 500, 6: 500, 7: 500, 4: 500, 8: 500, 3: 500})
X_smote, y_smote = oversample.fit_resample(x, y)

print(f'''Shape of X before SMOTE: {x.shape}
Shape of X after SMOTE: {X_smote.shape}''')

print('\nBalance of positive and negative classes (%):')
y_smote.value_counts(normalize=True) * 100

trainX, testX, trainY, testY = train_test_split(X_smote, y_smote, test_size = 0.2)

sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50),max_iter = 150,alpha= 0.0001,
                        activation = 'relu',solver = 'adam',learning_rate='invscaling')

mlp_clf.fit(trainX_scaled, trainY)

y_pred = mlp_clf.predict(testX_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

fig = plot_confusion_matrix(mlp_clf, testX_scaled, testY, display_labels=mlp_clf.classes_)
fig.figure_.suptitle("Confusion Matrix for Winequality Dataset")
plt.show()

print(classification_report(testY, y_pred))

plt.plot(mlp_clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

" SMOTE : Synthetic Minority Oversampling Technique || Data Augmentation Method (For Each Class has 1000 datas)"

sm = SMOTE(random_state=42)

oversample = SMOTE(sampling_strategy = {5: 1000, 6: 1000, 7: 1000, 4: 1000, 8: 1000, 3: 1000})
X_smote, y_smote = oversample.fit_resample(x, y)

print(f'''Shape of X before SMOTE: {x.shape}
Shape of X after SMOTE: {X_smote.shape}''')

print('\nBalance of positive and negative classes (%):')
y_smote.value_counts(normalize=True) * 100

trainX, testX, trainY, testY = train_test_split(X_smote, y_smote, test_size = 0.2)

sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50),max_iter = 150,alpha= 0.0001,
                        activation = 'relu',solver = 'adam',learning_rate='invscaling')

mlp_clf.fit(trainX_scaled, trainY)

y_pred = mlp_clf.predict(testX_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

fig = plot_confusion_matrix(mlp_clf, testX_scaled, testY, display_labels=mlp_clf.classes_)
fig.figure_.suptitle("Confusion Matrix for Winequality Dataset")
plt.show()

print(classification_report(testY, y_pred))

plt.plot(mlp_clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()