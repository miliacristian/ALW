from numpy import argmax
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.testing import all_estimators
def l():
    print("prova")
    estimators = all_estimators()
    for name, class_ in estimators:
        if hasattr(class_, 'predict_proba'):
            print(name)