from sklearn.exceptions import UndefinedMetricWarning

#rende un classificatore multiclass funzionante anche per multilabel
from sklearn.multiclass import OneVsRestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import dataset
import scoringUtils
import numpy as np
from scoringUtils import hmean_scores
from __init__ import printValue
from __init__ import setting
from __init__ import model_settings_dir
from time import time
import warnings
import os
import main


def KNN_training(X, Y, k, scoring, seed, n_split, mean):
    """
    do the training of a KNN models with dataset X,Y and K_Fold_Cross_Validation
    :param X: feature set
    :param Y: label set
    :param scoring: dict of scoring used
    :param k: list of possibile neighbors
    :return: the best k
    """

    if printValue:
        print("Start training of KNN")
        start_time = time()

    best_k = None
    best_total_score = None
    for num_neighbors in k:
        model = KNeighborsClassifier(n_neighbors=num_neighbors, weights='distance')
        result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
        harmonic_mean = hmean_scores(scoring, result)  # funzione che da result calcola media armonica
        if best_total_score is None or best_total_score < harmonic_mean:
            best_total_score = harmonic_mean
            best_k = num_neighbors

    if printValue:
        print("End training of KNN after", time() - start_time, "s.")
    return best_k


def RANDOMFOREST_training(X, Y,list_n_trees,scoring,seed,n_split,mean):
    """
     do the training of a Random Forest with dataset X, Y and K_Fold_Cross_Validation. Trova il setting migliore iterando su
    tutte i valori possibili di max_features
    :param X: feature set
    :param Y: label set
    :param scoring: dict of scoring used
    :param n_trees: number of trees of the forest
    :return: best n_trees, best max_features
    """

    if printValue:
        print("Start training of Random Forest")
        start_time = time()

    best_n_trees = None
    best_max_features = None
    best_total_score = None

    for trees in list_n_trees:
        for max_features in range(1, len(X[0])+1):  # len(X[0])==numero features
            model=RandomForestClassifier(n_estimators=trees, random_state=seed, max_features=max_features)
            result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
            harmonic_mean = hmean_scores(scoring, result)  # funzione che da result calcola media armonica
            if best_total_score is None or best_total_score < harmonic_mean:
                best_total_score = harmonic_mean
                best_max_features = max_features
                best_n_trees = trees

    if printValue:
        print("End training of Random Forest after", time() - start_time, "s.")
    return best_n_trees, best_max_features


def SVC_training(X, Y, scoring, seed, n_split, mean):
    """
    do the training of a SVC with dataset X, Y and K_Fold_Cross_Validation. Find the best setting iterating on the type
    of kernel function use (linear, rbf, polynomial or sigmoid), iterating the parameter C in [0.001, 0.01, 0.1 ...
    100000, 1000000], the parameter degree in [1, 2, ..., 5, 6], the parameter gamma in [0.000000001, ..., 100, 1000],
    and using the parameter coef0 = 0.0
    :param X: feature set
    :param Y: label set
    :param scoring: dict of scoring use
    :return: the best parameter C, gamma and degree with the best kernel function
    """

    if printValue:
        print("Start training of SVC")
        start_time = time()

    best_kernel = None
    best_C = None
    best_gamma = None
    best_degree = None
    best_total_score = None
    # range in cui variano i parametri plausibili (lungo il training)
    C_range = np.logspace(-2, 2, 5)
    gamma_range = np.logspace(-5, 0, 6)
    degree_range = range(2, 4, 1)
    # range minori per testare funzionalità
    # C_range = np.logspace(-1, 1, 3)
    # gamma_range = np.logspace(-4, -1, 3)
    # degree_range = range(2, 3, 1)

    # case kernel is linear

    if printValue:
        print("Start training of SVC with kernel linear")
        start_time_linear = time()

    for C in C_range:
        model = OneVsRestClassifier(SVC(kernel='linear', random_state=seed,C=C))
        result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
        harmonic_mean = hmean_scores(scoring, result)
        if best_total_score is None or best_total_score < harmonic_mean:
            best_total_score = harmonic_mean
            best_C = C
            best_kernel = 'linear'

    if printValue:
        print("End training of SVC with kernel linear after", time() - start_time_linear, "s.")

    # case kernel is rbf

    if printValue:
        print("Start training of SVC with kernel rbf")
        start_time_rbf = time()

    for C in C_range:
        for gamma in gamma_range:
            model = OneVsRestClassifier(SVC(kernel='rbf', random_state=seed, C=C, gamma=gamma))
            result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
            harmonic_mean = hmean_scores(scoring, result)
            if best_total_score < harmonic_mean:
                best_total_score = harmonic_mean
                best_C = C
                best_gamma = gamma
                best_kernel = 'rbf'

    if printValue:
        print("End training of SVC with kernel rbf after", time() - start_time_rbf, "s.")

    # case kernel is polynomial

    if printValue:
        print("Start training of SVC with kernel polynomial")
        start_time_poly = time()

    for C in C_range:
        for degree in degree_range:
            for gamma in gamma_range:
                # if (C == 100 or C == 1000) and gamma == 10:
                #     continue
                if printValue:
                    print("Starting cycle with C =", C, "degree =", degree, "gamma =", gamma)
                    start_time_poly2 = time()
                model = OneVsRestClassifier(SVC(kernel='poly', random_state=seed, C=C, gamma=gamma, degree=degree))
                result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
                harmonic_mean = hmean_scores(scoring, result)
                if best_total_score < harmonic_mean:
                    best_total_score = harmonic_mean
                    best_C = C
                    best_gamma = gamma
                    best_degree = degree
                    best_kernel = 'poly'
                if printValue:
                    print("Ending in", time() - start_time_poly2)

    if printValue:
        print("End training of SVC with kernel polynomial after", time() - start_time_poly, "s.")

    # case kernel is sigmoid

    if printValue:
        print("Start training of SVC with kernel sigmoid")
        start_time_sigmoid = time()

    for C in C_range:
        for gamma in gamma_range:
                model = OneVsRestClassifier(SVC(kernel='sigmoid', random_state=seed, C=C, gamma=gamma))
                result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
                harmonic_mean = hmean_scores(scoring, result)
                if best_total_score < harmonic_mean:
                    best_total_score = harmonic_mean
                    best_C = C
                    best_gamma = gamma
                    best_kernel = 'sigmoid'

    if printValue:
        print("End training of SVC with kernel sigmoid after", time() - start_time_sigmoid, "s.")

    # setto valori numerici per evitare problemi nella lettura e conversione da file, tanto non verranno visti dal
    # costruttore del modello se sono ancora None a questo punto del codice
    if best_gamma is None:
        best_gamma = 0.0
    if best_degree is None:
        best_degree = 0

    if printValue:
        print("End training of SVC after", time() - start_time, "s.")

    return best_C, best_degree, best_gamma, best_kernel


def training(X, Y, name_models,  scoring, k=[5], list_n_trees=[10], seed=111, n_split=10, mean=True,
             file_name="best_setting.txt"):
    """
    Write on file the best setting for the specific dataset. Every line of file contains "the name of the parameter"
    + " " + "the value of parameter"
    :param file_name: name of file with the best setting
    :param mean: True if you want the mean in the K_fold_cross_validation, False if you want all the scores of the fold
    :param n_split: number of fold
    :param X: features set
    :param Y: label set
    :param name_models:list, lista nomi modelli
    :param scoring: dict,dizionario di scoring
    :param k: list,lista dei possibili k per KNN
    :param list_n_trees: list,lista dei possibili numeri di alberi per il random forest
    :param seed: int,seme per generatore pseudocasuale
    """
    path = os.path.abspath('')
    fl = open(path+model_settings_dir+file_name, "w")
    fl.writelines(["seed " + str(seed) + "\n", "n_split " + str(n_split) + "\n"])
    if 'RANDFOREST' in name_models:
        best_n_trees, best_max_features = RANDOMFOREST_training(X, Y, list_n_trees, scoring, seed, n_split, mean)
        fl.writelines(["best_n_trees " + str(best_n_trees) + "\n", "best_max_features " +
                       str(best_max_features) + "\n"])
    if 'KNN' in name_models:
        best_k = KNN_training(X, Y, k, scoring, seed, n_split, mean)
        fl.writelines(["best_k " + str(best_k) + "\n"])
    if 'SVC' in name_models:
        best_C, best_degree, best_gamma, best_kernel = SVC_training(X, Y, scoring, seed, n_split, mean)
        fl.writelines(["best_C " + str(best_C) + "\n","best_degree " + str(best_degree) + "\n", "best_gamma " +
                       str(best_gamma) + "\n", "best_kernel " + str(best_kernel) + "\n"])

    fl.close()


def build_models(name_models, file_name):
    """
    build the list of models using the setting specifying in file_name.
    :param name_models: name of models used
    :param file_name: name of settings file
    :return: the list of models training
    """

    models = {}
    path = os.path.abspath('')
    fl = open(path + model_settings_dir + file_name, "r")  # aggiungere path
    settings = {}
    while 1:
        line = fl.readline()
        line = line[:-1]
        if len(line) == 0:
            break
        parameter, value = str.split(line, " ")
        settings[parameter] = value

    if 'CART' in name_models:
        models['CART'] = DecisionTreeClassifier(random_state=int(settings["seed"]))

    if 'RANDFOREST' in name_models:
        models['RANDFOREST'] = RandomForestClassifier(random_state=int(settings["seed"]),
                                                      max_features=int(settings["best_max_features"]),
                                                      n_estimators=int(settings["best_n_trees"]))

    if 'KNN' in name_models:
        models['KNN'] = KNeighborsClassifier(n_neighbors=int(settings["best_k"]), weights='distance')

    if 'SVC' in name_models:
        models['SVC'] = OneVsRestClassifier(SVC(kernel=settings["best_kernel"], C=float(settings["best_C"]),
                                                gamma=float(settings["best_gamma"]),
                                                degree=int(settings["best_degree"])))

    fl.close()

    return models


if __name__ == '__main__':
    warnings.filterwarnings('always')
    seed = 100
    name_models = ['RANDFOREST', 'CART', 'KNN', 'SVC']
    dataset_name = 'balance'
    k_range = range(3, 21, 1)
    n_trees_range = range(5, 21, 1)

    X, Y, scoring, name_setting_file, name_radar_plot_file = main.case_NaN_dataset_classification(dataset_name, "mean", seed, 0.3)

    training(X, Y, name_models, scoring, k=k_range, list_n_trees=n_trees_range, seed=seed,
             n_split=10, mean=True, file_name=name_setting_file)
