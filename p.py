from numpy import argmax
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.testing import all_estimators
def multiple_argument(*args):
    print(args)
    for arg in args:
        print(arg)
    return None
def multi_key_word_arg(l,**kwargs):
    for key, value in kwargs.items():
        print(key,value)
    return None


def g():
    # Contrive small dataset
    dataset = [[50, 30], [20, 90]]
    print(dataset)
    # Calculate min and max for each column
    minmax = dataset_minmax(dataset)
    print(minmax)


def l():
    print("prova")
    p='miao'
    multi_key_word_arg(p,b='blue',c='cyano',p='turc',)


