import numpy
from pandas import read_csv
import __init__

import pandas as pd
import seaborn as sns
import imgkit


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

    data = {('count', 's25'):
                {('2017-08-11', 'Friday'): 88.0,
                 ('2017-08-12', 'Saturday'): 90.0,
                 ('2017-08-13', 'Sunday'): 93.0},
            ('count', 's67'):
                {('2017-08-11', 'Friday'): 404.0,
                 ('2017-08-12', 'Saturday'): 413.0,
                 ('2017-08-13', 'Sunday'): 422.0},
            ('count', 's74'):
                {('2017-08-11', 'Friday'): 203.0,
                 ('2017-08-12', 'Saturday'): 227.0,
                 ('2017-08-13', 'Sunday'): 265.0},
            ('count', 's79'):
                {('2017-08-11', 'Friday'): 53.0,
                 ('2017-08-12', 'Saturday'): 53.0,
                 ('2017-08-13', 'Sunday'): 53.0}}

    table = pd.DataFrame.from_dict(data)
    table.sort_index(ascending=False, inplace=True)

    cm = sns.light_palette("seagreen", as_cmap=True)
    styled_table = table.style.background_gradient(cmap=cm)
    html = styled_table.render()

    imgkit.from_string(html, 'styled_table.jpeg')
