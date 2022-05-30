#import libraries
import tensorflow as tf

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import keras


from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras import regularizers


import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Basic libraries
import numpy as np
import pandas as pd
from scipy import stats


# Statistics, EDA, metrics libraries
from scipy.stats import normaltest, skew
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import boxcox, inv_boxcox

# Modeling libraries
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, cross_val_predict,  KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from scipy.stats import zscore
from itertools import combinations
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import IsolationForest
#import kmapper as km
from sklearn.cluster import KMeans
from sklearn.metrics import plot_roc_curve


import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('METABRIC_RNA_Mutation_Signature.csv', delimiter=',')

#Classification with All attributes
#features_to_drop = df.columns[520:]
features_to_drop = df.columns[52:]
df = df.drop(features_to_drop, axis=1)
all_categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
unwanted_columns = ['patient_id','death_from_cancer' ]
all_categorical_columns = [ele for ele in all_categorical_columns if ele not in unwanted_columns] 
dummies_df = pd.get_dummies(df.drop('patient_id',axis=1 ), columns= all_categorical_columns, dummy_na=True)
dummies_df.dropna(inplace = True)
dummies_df.to_csv('METABRIC_RNA_Mutation_Signature_Preprocessed.csv', encoding='utf-8', index=False)


# data splitting
X = dummies_df.drop( ['death_from_cancer','overall_survival'], axis=1)
y = dummies_df['overall_survival']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#All attributes - DL CNN 1 
# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):

    model = Sequential()
    model.add(Dense(25,input_shape=(114,), activation='relu', kernel_regularizer=regularizers.l1(1e-4)))
    #model.add(Dense(15,input_shape=(114,), activation='sigmoid', kernel_regularizer=regularizers.l1(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  
    #model.add(Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(1e-4)))
    model.add(Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(1e-4)))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics = ['accuracy'])

    # Pass several parameters to 'EarlyStopping' function and assign it to 'earlystopper'

earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')


# Fit model over 2000 iterations with 'earlystopper' callback, and assign it to history

history = model.fit(X_train, y_train, epochs = 200, validation_split = 0.15, verbose = 0, 
                    callbacks = [earlystopper])

y_test_pred = model.predict(X_test)

model.save('BreastCancer_DL.h5')
