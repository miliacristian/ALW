import CSV,numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def print_dataset(X,Y):
    """
    Stampa features set X e label set Y
    :param X: features set
    :param Y: label set
    :return: None
    """
    print("Features set:\n",X)
    print("Label set:\n",Y)
    return None

def load_zoo_dataset():
    """
    :return:features set X del dataset zoo ,label set Y del dataset zoo
    """
    X, Y = CSV.read_csv('zoo.csv', skip_rows=31, skip_column_left=1)
    Y=CSV.convert_type_to_float(Y)
    X=CSV.convert_type_to_float(X)
    return X,Y

def load_balance_dataset():
    """
    :return:features set X del dataset zoo ,label set Y del dataset zoo
    """
    X, Y = CSV.read_csv('balance.csv', skip_rows=29,last_column_is_label=False)
    Y=CSV.convert_label_values(Y, ['L','B','R'], ['0', '1', '2'])
    X=CSV.convert_type_to_float(X)
    return X,Y

def load_seed_dataset():
    """
    :return:features set X del dataset zoo ,label set Y del dataset zoo
    """
    X, Y = CSV.read_csv('seeds.csv', skip_rows=2,delimiter='\t')
    Y=CSV.convert_type_to_float(Y)
    X=CSV.convert_type_to_float(X)
    return X,Y

def load_tris_dataset():
    """
    :return:features set X del dataset tris ,label set Y del dataset tris
    """
    X, Y = CSV.read_csv('tic_tac_toe.csv',skip_rows=17)
    Y = CSV.convert_label_values(Y, ['positive', 'negative'], [1, 0])
    for i in range(len(X[:,0])):
        for j in range(len(X[0,:])):
            if X[i][j]=='x':
                X[i][j]= numpy.float64(1)
            elif X[i][j]=='o':
                X[i][j] =numpy.float64(0)
            else:
                X[i][j]=numpy.float64(2)
    X=CSV.convert_type_to_float(X)
    return X,Y

def remove_row_with_label_L(X,Y,L):
    """
    Rimuove le labels L dal label set Y e le righe corrispondenti nel features set X
    :param X: features set
    :param Y: label set
    :param L: lista di label da rimuovere
    :return: X,Y con label rimosse
    """
    length=len(Y)
    i=0
    while i<length:
        if Y[i] in L:
            Y=numpy.delete(Y,i,axis=0)
            X=numpy.delete(X,i,axis=0)
            i=i-1
            length=length-1
        i = i + 1
    return X,Y

def remove_row_dataset(X,Y,A,B,step=1):
    """
    :param X: features set
    :param Y: label set
    :param A: int,indice prima riga da eliminare
    :param B: int,indice ultima riga-1 da eliminare
    :param step: int,quante righe saltare prima di eliminare la prossima riga
    :return: X,Y con righe da A,B eliminate con step step
    """
    temp_X=numpy.delete(X,numpy.s_[A:B:step],axis=0)#axis=0==riga,axis=1==colonna
    temp_Y = numpy.delete(Y, numpy.s_[A:B:step], axis=0)
    return temp_X,temp_Y

def remove_column_dataset(X,Y,A,B,step=1):
    """
    :param X: features set
    :param Y: label set
    :param A: int,indice prima colonna da eliminare
    :param B: int,indice ultima colonna-1 da eliminare
    :param step: int,quante colonne saltare prima di eliminare la prossima colonna
    :return: X,Y con colonne da A,B eliminate con step step
    """
    temp_X=numpy.delete(X,numpy.s_[A:B:step],axis=1)#axis=0==riga,axis=1==colonna
    temp_Y = numpy.delete(Y, numpy.s_[A:B:step], axis=1)
    return temp_X,temp_Y

def one_hot_encoding(Y):
    """
    Effettua il one hot encoding sul label set Y
    :param Y: label set Y
    :return: label set Y con one ho encoding
    """
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    #print(integer_encoded)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    # invert
    #inverted = label_encoder.inverse_transform(integer_encoded)
    return Y_onehot_encoded
