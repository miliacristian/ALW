import CSV,numpy

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

def load_tris_dataset():
    """
    :return:features set X del dataset tris ,label set Y del dataset tris
    """
    X, Y = CSV.read_csv('tic_tac_toe.csv')
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