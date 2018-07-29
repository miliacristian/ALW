import CSV, numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from math import sqrt
import os
from sklearn.preprocessing import Imputer
from __init__ import classification_datasets_dir, regression_dataset_dir
from math import floor


def print_dataset(X, Y):
    """
    Stampa features set X e label set Y
    :param X: features set
    :param Y: label set
    :return: None
    """
    print("Features set(" + str(len(X[:, 0])) + "," + str(len(X[0, :])) + "):\n", X)
    print("Label set(" + str(len(Y[:, 0])) + "," + str(len(Y[0, :])) + "):\n", Y)
    return None


def replace_value_in_column(X, list_column, value_to_replace, new_value):
    """
    :param X: features set
    :param list_column: list,lista delle colonne in cui sostituire i valori
    :param value_to_replace: valore da sostituire
    :param new_value: nuovo valore
    :return: X con valori sostituiti nelle colonne list column
    """
    for k in list_column:
        for i in range(len(X[:, 0])):
            if X[i][k] == value_to_replace:
                X[i][k] = new_value
    return X


def replace_value_in_row(X, list_row, value_to_replace, new_value):
    """
    :param X: features set
    :param list_row: list,lista delle righe in cui sostituire i valori
    :param value_to_replace: valore da sostituire
    :param new_value: nuovo valore
    :return: X con valori sostituiti nelle righe list row
    """
    for k in list_row:
        for j in range(len(X[0, :])):
            if X[k, j] == value_to_replace:
                X[k, j] = new_value
    return X


def replace_NaN_with_strategy(X, strategy):
    """

    :param X:
    :param strategy: must be 'mean' or 'median or 'most_frequent'
    :return:
    """
    imputer = Imputer(strategy=strategy)
    X = imputer.fit_transform(X)
    return X


def remove_row_with_Nan(X):
    """
    :param X: features set
    :return: X togliendo le righe che hanno almeno 1 valore NaN
    """
    X = X[~numpy.isnan(X).any(axis=1)]
    return X


def put_random_NaN_per_row(X, fraction_sample_missing_value, seed=100):
    """
    Put one Nan in a percent of random row of dataset
    :param X: dataset features
    :param fraction_sample_missing_value: percent of row with NaN
    :param seed:
    :return: the dataset with NaN
    """

    if fraction_sample_missing_value >= 1 or fraction_sample_missing_value < 0:
        print("invalid parameter fraction sample missing value")
        exit(1)
    count_nan = 0
    count_not_nan = 0
    rng = numpy.random.RandomState(seed)
    num_examples = len(X[:, 0])
    num_features = len(X[0, :])
    num_missing_example = floor(fraction_sample_missing_value * num_examples)
    list_index_missing_samples = rng.choice(num_examples, num_missing_example, replace=False)
    for i in list_index_missing_samples:
        rand_num = rng.randint(0, num_features)
        if numpy.isnan(X[i, rand_num]):
            count_nan = count_nan + 1
        else:
            X[i][rand_num] = numpy.nan
            count_not_nan = count_not_nan + 1
    print('count_nan', count_nan, 'count_not_nan', count_not_nan)
    return X


def put_random_NaN(X, fraction_missing_value, seed=100):
    """
    Put Nan in a percent of random component of dataset
    :param X: dataset features
    :param fraction_missing_value: percent of row with NaN
    :param seed:
    :return: the dataset with NaN
    """

    if fraction_missing_value >= 1 or fraction_missing_value < 0:
        print("invalid parameter fraction sample missing value")
        exit(1)
    rng = numpy.random.RandomState(seed)
    num_examples = len(X[:, 0])
    num_features = len(X[0, :])
    num_values = num_examples * num_features
    num_missing_example = floor(fraction_missing_value * num_values)
    list_index_missing_samples = rng.choice(num_values, num_missing_example, replace=False)
    for i in list_index_missing_samples:
        X[floor(i/num_features)][i - (floor(i/num_features) * num_features)] = numpy.nan

    return X


def load_zoo_dataset():
    """
    :return:features set X del dataset zoo ,label set Y del dataset zoo
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'zoo.csv', skip_rows=31, skip_column_left=1)
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    print(type(X))
    return X, Y


def load_balance_dataset():
    """
    :return:features set X del dataset balance ,label set Y del dataset balance
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'balance.csv', skip_rows=29, last_column_is_label=False)
    Y = CSV.convert_label_values(Y, ['L', 'B', 'R'], ['0', '1', '2'])
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_seed_dataset():
    """
    :return:features set X del dataset seed ,label set Y del dataset zoo
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'seeds.csv', skip_rows=2, delimiter='\t')
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_tris_dataset():
    """
    :return:features set X del dataset tris ,label set Y del dataset tris
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'tris.csv', skip_rows=17)
    Y = CSV.convert_label_values(Y, ['positive', 'negative'], [1, 0])
    for i in range(len(X[:, 0])):
        for j in range(len(X[0, :])):
            if X[i][j] == 'x':
                X[i][j] = numpy.float64(1)
            elif X[i][j] == 'o':
                X[i][j] = numpy.float64(0)
            else:
                X[i][j] = numpy.float64(2)
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_pima_indians_diabetes_dataset():
    """
    :return:features set X del dataset pima_indians_diabetes ,label set Y del dataset pima_indians_diabetes
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'pima_indians_diabetes.csv', skip_rows=18)
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    X = replace_value_in_column(X, [1, 2, 3, 4, 5], 0, numpy.nan)
    # numpy.isnan(X)
    return X, Y


def load_com_dataset():
    """
    :return:features set X del dataset com ,label set Y del dataset com
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + regression_dataset_dir + 'com.csv', skip_rows=20, delimiter='\t')
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_airfoil_dataset():
    """
    :return:features set X del dataset airfoil ,label set Y del dataset airfoil
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + regression_dataset_dir + 'airfoil_self_noise.csv', skip_rows=16, delimiter='\t')
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_classification_dataset(dataset):
    if dataset == 'tris':
        X, Y = load_tris_dataset()
    elif dataset == 'seed':
        X, Y = load_seed_dataset()
    elif dataset == 'balance':
        X, Y = load_balance_dataset()
    elif dataset == 'zoo':
        X, Y = load_zoo_dataset()
    elif dataset == 'indians':
        X, Y = load_pima_indians_diabetes_dataset()
    else:
        print("input must be 'tris' or 'seed' or 'balance' or 'zoo' or 'indians'.")
        exit(1)
    Y = one_hot_encoding(Y)
    return X, Y


def load_regression_dataset(dataset):
    if dataset == 'com':
        X, Y = load_com_dataset()
    elif dataset == 'airfoil':
        X, Y = load_airfoil_dataset()
    else:
        print("input must be 'com' or 'airfoil'.")
        exit(1)

    return X, Y


def remove_rows_with_NaN(X, Y):
    length = len(X[:, 0])
    i = 0
    while i < length:  # tutte le righe
        for j in range(len(X[0, :])):  # tutte le colonne
            if numpy.isnan(X[i][j]):
                Y = numpy.delete(Y, i, axis=0)
                X = numpy.delete(X, i, axis=0)
                i = i - 1
                length = length - 1
                break
        i = i + 1
    return X, Y


def remove_row_with_label_L(X, Y, L):
    """
    Rimuove le labels L dal label set Y e le righe corrispondenti nel features set X
    :param X: features set
    :param Y: label set
    :param L: lista di label da rimuovere
    :return: X,Y con label rimosse
    """
    length = len(Y)
    i = 0
    while i < length:
        if Y[i] in L:
            Y = numpy.delete(Y, i, axis=0)
            X = numpy.delete(X, i, axis=0)
            i = i - 1
            length = length - 1
        i = i + 1
    return X, Y


def remove_row_dataset(X, Y, A, B, step=1):
    """
    :param X: features set
    :param Y: label set
    :param A: int,indice prima riga da eliminare
    :param B: int,indice ultima riga-1 da eliminare
    :param step: int,quante righe saltare prima di eliminare la prossima riga
    :return: X,Y con righe da A,B eliminate con step step
    """
    temp_X = numpy.delete(X, numpy.s_[A:B:step], axis=0)  # axis=0==riga,axis=1==colonna
    temp_Y = numpy.delete(Y, numpy.s_[A:B:step], axis=0)
    return temp_X, temp_Y


def remove_column_dataset(X, Y, A, B, step=1):
    """
    :param X: features set
    :param Y: label set
    :param A: int,indice prima colonna da eliminare
    :param B: int,indice ultima colonna-1 da eliminare
    :param step: int,quante colonne saltare prima di eliminare la prossima colonna
    :return: X,Y con colonne da A,B eliminate con step step
    """
    temp_X = numpy.delete(X, numpy.s_[A:B:step], axis=1)  # axis=0==riga,axis=1==colonna
    temp_Y = numpy.delete(Y, numpy.s_[A:B:step], axis=1)
    return temp_X, temp_Y


def dataset_minmax(X):
    """
    Calcola il minimo e massimo per ogni features e ritorna una lista di coppie (min,max)
    :param X:features set
    :return: list,list max,min per ogni features
    """
    minmax = list()
    for i in range(len(X[0])):
        col_values = [row[i] for row in X]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


def normalize_dataset(X):
    """
    Normalizza i valori degli attributi di X
    :param X: features set
    :return: X normalizato,scaled_value =(value - min) / (max - min)
    """
    minmax = dataset_minmax(X)
    X_normalized = X.copy()
    for row in X_normalized:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return X_normalized


def column_means(X):
    """
    Calcola la media di ogni features
    :param X: features set
    :return: list,lista di medie per ogni features
    """
    means = [0 for i in range(len(X[0]))]
    for i in range(len(X[0])):  # per ogni attributo
        col_values = [row[i] for row in X]  # colonna dei valori di ogni attributo
        means[i] = sum(col_values) / float(len(X))
    return means


def column_stdevs(X, means):
    """
    Calcola la media di ogni features
    :param X: features set
    :param means:list,lista di medie per ogni features
    :return: list,lista di medie per ogni features
    """
    stdevs = [0 for i in range(len(X[0]))]
    for i in range(len(X[0])):
        variance = [pow(row[i] - means[i], 2) for row in X]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x / (float(len(X) - 1))) for x in stdevs]
    return stdevs


def standardize_dataset(X):
    """
    Standardizza il dataset X
    :param X: features set
    :return:X_ standardizzato,standard deviation = sqrt( (value_i - mean)^2 / (total_values-1))
    """
    means = column_means(X)
    stdevs = column_stdevs(X, means)
    X_std = X.copy()
    for row in X_std:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]
    return X_std


def one_hot_encoding(Y):
    """
    Effettua il one hot encoding sul label set Y
    :param Y: label set Y
    :return: label set Y con one ho encoding
    """
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    inverted = label_encoder.inverse_transform(integer_encoded)
    return Y_onehot_encoded


if __name__ == '__main__':
    X, Y = load_regression_dataset('com')
    print(X)
    print(Y)

    X, Y = load_regression_dataset('airfoil')
    print(X)
    print(Y)
