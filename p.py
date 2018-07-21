import numpy
from pandas import read_csv
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
    dataset = read_csv('pima-indians-diabetes.csv', header=None)
    # mark zero values as missing or NaN
    print(type(dataset))
    dataset[[1, 2, 3, 4, 5]] = dataset[[1, 2, 3, 4, 5]].replace(0, numpy.NaN)
    # count the number of NaN values in each column
    print(dataset.isnull().sum())

def l():
    print("prova")
    p='miao'
    multi_key_word_arg(p,b='blue',c='cyano',p='turc',)


