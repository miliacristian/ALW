import numpy
from pandas import read_csv
import __init__
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
    pass

def l():
    print("prova")
    p='miao'
    multi_key_word_arg(p,b='blue',c='cyano',p='turc',)



if __name__== '__main__':
    print("inizio")
    check_strategies(__init__.zoo,"mode")
    pass

