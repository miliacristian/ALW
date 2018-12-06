import CSV, numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from math import sqrt
import os
from sklearn.preprocessing import Imputer
from __init__ import classification_datasets_dir, regression_dataset_dir
from math import floor
import __init__


def print_dataset(X, Y):
    """
    Print features set X and label set Y
    :param X: features set
    :param Y: label set
    :return: None
    """
    print("Features set(" + str(len(X[:, 0])) + "," + str(len(X[0, :])) + "):\n", X)
    print("Label set(" + str(len(Y[:, 0])) + "," + str(len(Y[0, :])) + "):\n", Y)
    return None


def replace_value_in_column(X, list_columns, value_to_replace, new_value):
    """
    :param X: features set
    :param list_columns: list, list of columns where swap values
    :param value_to_replace: value to swap
    :param new_value: new value
    :return: X with values swapped in the columns list column
    """
    for k in list_columns:
        for i in range(len(X[:, 0])):
            if X[i][k] == value_to_replace:
                X[i][k] = new_value
    return X


def replace_value_in_row(X, list_rows, value_to_replace, new_value):
    """
    :param X: features set
    :param list_rows: list,list of rows where swap values
    :param value_to_replace: value to replace
    :param new_value: new value
    :return:X with values swapped in the rows list rows
    """
    for k in list_rows:
        for j in range(len(X[0, :])):
            if X[k, j] == value_to_replace:
                X[k, j] = new_value
    return X


def replace_NaN_with_strategy(X, strategy, list_mode_columns=[]):
    """
    :param X:features set
    :param strategy: must be 'mean' or 'median or 'most_frequent'
    :return:X with Nan values replaced by value conform with strategy
    """
    imputer = Imputer(strategy=strategy)
    imputer_mode = Imputer(strategy="most_frequent")
    for j in range(len(X[0])):
        if j in list_mode_columns:
            X[:, j] = imputer_mode.fit_transform(X[:, j].reshape(-1, 1))[:, 0]
        else:
            X[:, j] = imputer.fit_transform(X[:, j].reshape(-1, 1))[:, 0]
    return X


def get_list_mode_columns_by_dataset(dataset_name):
    """
    :param dataset_name:string,dataset name
    :return:list of column index where is possible to apply strategy mode for the dataset dataset_name
    """
    if dataset_name == __init__.auto:
        return [0, 5, 6]
    elif dataset_name == __init__.compress_strength:
        return [7]
    elif dataset_name == __init__.energy:
        return [4, 5, 6, 7]
    return []


def remove_row_with_Nan(X):
    """
    :param X: features set
    :return: X but removing all X's rows which contains almost one nan value
    """
    X = X[~numpy.isnan(X).any(axis=1)]
    return X


def put_random_NaN_per_row(X, fraction_sample_missing_value, seed=100):
    """
    Put one Nan in a percent of random row of dataset
    :param X: dataset features
    :param fraction_sample_missing_value: percent of row with NaN
    :param seed:seed number for pseudo number generator
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
    :param seed:sedd number for pseudo number generator
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
        X[floor(i / num_features)][i - (floor(i / num_features) * num_features)] = numpy.nan

    return X


def load_zoo_dataset():
    """
    :return:features set X of the dataset zoo ,label set Y of the dataset zoo
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'zoo.csv', skip_rows=31, skip_column_left=1)
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_balance_dataset():
    """
    :return:features set X of the  dataset balance ,label set Y of the dataset balance
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'balance.csv', skip_rows=29, last_column_is_label=False)
    Y = CSV.convert_label_values(Y, ['L', 'B', 'R'], ['0', '1', '2'])
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_seed_dataset():
    """
    :return:features set X of the dataset seed ,label set Y of the dataset zoo
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'seeds.csv', skip_rows=2, delimiter='\t')
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_tris_dataset():
    """
    :return:features set X of the dataset tris ,label set Y of the dataset tris
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


def load_eye_dataset():
    """
    :return:features set X of the dataset eye ,label set Y of the dataset eye
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'EEG_eye_state.csv', skip_rows=25)
    X = CSV.convert_type_to_float(X)
    Y = CSV.convert_type_to_float(Y)
    return X, Y


def load_page_dataset():
    """
    :return:features set X of the dataset page ,label set Y of the dataset page
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + classification_datasets_dir + 'page_block.csv', skip_rows=48, delimiter=' ')
    X = CSV.convert_type_to_float(X)
    Y = CSV.convert_type_to_float(Y)
    return X, Y


def load_com_dataset():
    """
    :return:features set X of the dataset com ,label set Y of the dataset com
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + regression_dataset_dir + 'concrete_compressive_strength.csv', skip_rows=20,
                        delimiter='\t')
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    Y.ravel()
    return X, Y


def load_airfoil_dataset():
    """
    :return:features set X of the dataset airfoil ,label set Y of the dataset airfoil
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + regression_dataset_dir + 'airfoil_self_noise.csv', skip_rows=16, delimiter='\t')
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_auto_dataset():
    """
    :return:features set X of the dataset auto ,label set Y of the dataset auto
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + regression_dataset_dir + 'auto_mpg.csv', skip_rows=24, delimiter=' ',
                        skip_column_right=1, last_column_is_label=False)
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_power_plant_dataset():
    """
    :return:features set X of the dataset power_plant ,label set Y of the dataset power_plant
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + regression_dataset_dir + 'combined_cycle_power_plant.csv', skip_rows=15)
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    return X, Y


def load_energy_efficiency_dataset():
    """
    :return:features set X of the dataset energy_efficency ,label set Y of the dataset energy_efficency
    """
    path = os.path.abspath('')
    X, Y = CSV.read_csv(path + regression_dataset_dir + 'energy_efficiency.csv', skip_rows=27, num_label=2,
                        delimiter='\t')
    Y = CSV.convert_type_to_float(Y)
    X = CSV.convert_type_to_float(X)
    # Y.ravel()
    return X, Y


def load_classification_dataset(dataset, multilabel=False):
    """
    Load clasification dataset dataset
    :param dataset:string, dataset name
    :param multilabel: boolean,true if dataset is multilabel,false otherwise
    :return: features set X and label set Y of the dataset dataset
    """
    if dataset == __init__.tris:
        X, Y = load_tris_dataset()
    elif dataset == __init__.seed:
        X, Y = load_seed_dataset()
    elif dataset == __init__.balance:
        X, Y = load_balance_dataset()
    elif dataset == __init__.zoo:
        X, Y = load_zoo_dataset()
    elif dataset == __init__.eye:
        X, Y = load_eye_dataset()
    elif dataset == __init__.page:
        X, Y = load_page_dataset()
    else:
        print("input must be 'tris' or 'seed' or 'balance' or 'zoo' or 'indians' or 'eye' or 'page'.")
        exit(1)
    Y = one_hot_encoding(Y)
    return X, Y


def load_regression_dataset(dataset, multilabel=False):
    """
    Load regression dataset dataset
    :param dataset:string, dataset name
    :param multilabel: boolean,true if dataset is multilabel,false otherwise
    :return: features set X and label set Y of the dataset dataset
    """
    if dataset == __init__.compress_strength:
        X, Y = load_com_dataset()
    elif dataset == __init__.airfoil:
        X, Y = load_airfoil_dataset()
    elif dataset == __init__.auto:
        X, Y, = load_auto_dataset()
    elif dataset == __init__.power_plant:
        X, Y = load_power_plant_dataset()
    elif dataset == __init__.energy:
        X, Y = load_energy_efficiency_dataset()
    else:
        print("input must be 'compressive_strength' or 'airfoil' or 'auto' or 'power_plant' or 'energy'.")
        exit(1)
    if not multilabel:
        Y = Y.ravel()
    return X, Y


def remove_rows_with_NaN(X, Y):
    """
    Removed rows which contains almost one element nan
    :param X: features set X
    :param Y: label set Y
    :return: X,Y with row Nan removed
    """
    length = len(X[:, 0])
    i = 0
    while i < length:  # all rows
        for j in range(len(X[0, :])):  # all column
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
    Remove labels L from label set Y and respective rows in the features set X
    :param X: features set
    :param Y: label set
    :param L: list of label to remove
    :return: X,Y with label removed
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
    :param A: int,index of first row to remove
    :param B: int,index of last row-1 to remove
    :param step: int,skip number
    :return: X,Y with rows from A to B with step step removed
    """
    temp_X = numpy.delete(X, numpy.s_[A:B:step], axis=0)  # axis=0==riga,axis=1==colonna
    temp_Y = numpy.delete(Y, numpy.s_[A:B:step], axis=0)
    return temp_X, temp_Y


def remove_column_dataset(X, Y, A, B, step=1):
    """
    Remove column of the dataset starting from column index A to column index B with skip column step
    :param X: features set
    :param Y: label set
    :param A: int,index of first column to remove
    :param B: int,index of last column-1 to remove
    :param step: int,skip number
    :return: X,Y with columns from A to B with step step removed
    """
    temp_X = numpy.delete(X, numpy.s_[A:B:step], axis=1)  # axis=0==row,axis=1==column
    temp_Y = numpy.delete(Y, numpy.s_[A:B:step], axis=1)
    return temp_X, temp_Y


def dataset_minmax(X):
    """
    Calculate min e max for each features and return list of couple (min,max)
    :param X:features set
    :return: list,list max,min for each features
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
    Normalize values of X's attributes
    :param X: features set
    :return: X normalized,scaled_value =(value - min) / (max - min)
    """
    minmax = dataset_minmax(X)
    X_normalized = X.copy()
    for row in X_normalized:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return X_normalized


def column_means(X):
    """
    Calculate mean for each featues
    :param X: features set
    :return: list,list of means for each features
    """
    means = [0 for i in range(len(X[0]))]
    for i in range(len(X[0])):  # per ogni attributo
        col_values = [row[i] for row in X]  # colonna dei valori di ogni attributo
        means[i] = sum(col_values) / float(len(X))
    return means


def column_stdevs(X, means):
    """
    Calculate standard deviation for each features
    :param X: features set
    :param means:list,list of means for each features
    :return: list,list of standard deviation for each features
    """
    stdevs = [0 for i in range(len(X[0]))]
    for i in range(len(X[0])):
        variance = [pow(row[i] - means[i], 2) for row in X]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x / (float(len(X) - 1))) for x in stdevs]
    return stdevs


def standardize_dataset(X):
    """
    Standardize the dataset X
    :param X: features set
    :return:X standardized,standard deviation = sqrt( (value_i - mean)^2 / (total_values-1))
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
    Do the one hot encoding on label set Y
    :param Y: label set Y
    :return: label set Y with one ho encoding
    """
    label_encoder = LabelEncoder()
    Y = Y.ravel()
    integer_encoded = label_encoder.fit_transform(Y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    inverted = label_encoder.inverse_transform(integer_encoded)
    return Y_onehot_encoded
