# rende un classificatore multiclass funzionante anche per multilabel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import scoringUtils
import numpy as np
from scoringUtils import hmean_scores, total_score_regression
from __init__ import printValue
from __init__ import model_settings_dir, model_setting_test_dir
from time import time
import os
import main
import __init__


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
    best_scores = None
    for num_neighbors in k:

        if printValue:
            print("Start training of KNN with k =", num_neighbors)
            start_time_cycle = time()

        model = KNeighborsClassifier(n_neighbors=num_neighbors, weights='distance')
        result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
        harmonic_mean = hmean_scores(scoring, result)  # funzione che da result calcola media armonica
        if best_total_score is None or best_total_score < harmonic_mean:
            best_scores = result
            best_total_score = harmonic_mean
            best_k = num_neighbors

        if printValue:
            print("Ending KNN with k =", num_neighbors, "in ", time() - start_time_cycle)

    if printValue:
        print("End training of KNN after", time() - start_time, "s.")
    return best_k, best_scores, best_total_score


def RANDOMFOREST_training(X, Y, list_n_trees, scoring, seed, n_split, mean):
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
    best_scores = None
    best_total_score = None

    for trees in list_n_trees:
        for max_features in range(1, len(X[0]) + 1):  # len(X[0])==numero features

            if printValue:
                print("Start training of RF with num_trees =", trees, "and max_features =", max_features)
                start_time_cycle = time()

            model = RandomForestClassifier(n_estimators=trees, random_state=seed, max_features=max_features)
            result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
            harmonic_mean = hmean_scores(scoring, result)  # funzione che da result calcola media armonica
            if best_total_score is None or best_total_score < harmonic_mean:
                best_scores = result
                best_total_score = harmonic_mean
                best_max_features = max_features
                best_n_trees = trees

            if printValue:
                print("Ending RF with num_trees =", trees, "and max_features =", max_features, "in ",
                      time() - start_time_cycle)

    if printValue:
        print("End training of Random Forest after", time() - start_time, "s.")
    return best_n_trees, best_max_features, best_scores, best_total_score


def SVC_default_training(X, Y, scoring, seed, n_split, mean):
    """
    do the training of a SVC with dataset X, Y and K_Fold_Cross_Validation. Find the best setting iterating on the type
    of kernel function use (linear, rbf, polynomial or sigmoid)
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

    if printValue:
        print("Start training of default SVC")
        start_time_linear = time()
    model = OneVsRestClassifier(SVC(random_state=seed, max_iter=1000))
    result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
    harmonic_mean = hmean_scores(scoring, result)

    if printValue:
        print("End training of default SVC after", time() - start_time_linear, "s.")
    return best_C, best_degree, best_gamma, best_kernel


def SVC_training(X, Y, scoring, seed, n_split, mean):
    """
    do the training of a SVC with dataset X, Y and K_Fold_Cross_Validation. Find the best setting iterating on the type
    of kernel function use (linear, rbf, polynomial or sigmoid)
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
    best_scores = None
    best_total_score = None
    # range in cui variano i parametri plausibili (lungo il training)
    C_range = np.logspace(-2, 2, 5)
    gamma_range = np.logspace(-5, 0, 6)
    degree_range = range(2, 4, 1)
    # default parameter
    # C_range = [0.1, 1.0]
    # gamma_range = [0.01, 0.1, 1]
    # degree_range = [3]

    # case kernel is linear

    if printValue:
        print("Start training of SVC with kernel linear")
        start_time_linear = time()

    for C in C_range:

        if printValue:
            print("Starting cycle with C =", C)
            start_time_linear2 = time()

        model = OneVsRestClassifier(SVC(kernel='linear', random_state=seed, C=C, max_iter=1000))
        result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
        harmonic_mean = hmean_scores(scoring, result)
        if best_total_score is None or best_total_score < harmonic_mean:
            best_scores = result
            best_total_score = harmonic_mean
            best_C = C
            best_kernel = 'linear'

        if printValue:
            print("Ending in", time() - start_time_linear2)

    if printValue:
        print("End training of SVC with kernel linear after", time() - start_time_linear, "s.")

    # case kernel is rbf

    if printValue:
        print("Start training of SVC with kernel rbf")
        start_time_rbf = time()

    for C in C_range:
        for gamma in gamma_range:

            if printValue:
                print("Starting cycle with C =", C, "gamma =", gamma)
                start_time_rbf2 = time()

            model = OneVsRestClassifier(SVC(kernel='rbf', random_state=seed, C=C, gamma=gamma, max_iter=1000))
            result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
            harmonic_mean = hmean_scores(scoring, result)
            if best_total_score < harmonic_mean:
                best_scores = result
                best_total_score = harmonic_mean
                best_C = C
                best_gamma = gamma
                best_kernel = 'rbf'

            if printValue:
                print("Ending in", time() - start_time_rbf2)

    if printValue:
        print("End training of SVC with kernel rbf after", time() - start_time_rbf, "s.")

    # case kernel is polynomial

    if printValue:
        print("Start training of SVC with kernel polynomial")
        start_time_poly = time()

    for C in C_range:
        for degree in degree_range:
            for gamma in gamma_range:

                if C == 100 and gamma == 1:
                    continue

                if printValue:
                    print("Starting cycle with C =", C, "degree =", degree, "gamma =", gamma)
                    start_time_poly2 = time()

                model = OneVsRestClassifier(
                    SVC(kernel='poly', random_state=seed, C=C, gamma=gamma, degree=degree, max_iter=1000))
                result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
                harmonic_mean = hmean_scores(scoring, result)
                if best_total_score < harmonic_mean:
                    best_scores = result
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

            if printValue:
                print("Starting cycle with C =", C, "gamma =", gamma)
                start_time_sigmoid2 = time()

            model = OneVsRestClassifier(SVC(kernel='sigmoid', random_state=seed, C=C, gamma=gamma, max_iter=1000))
            result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
            harmonic_mean = hmean_scores(scoring, result)
            if best_total_score < harmonic_mean:
                best_scores = result
                best_total_score = harmonic_mean
                best_C = C
                best_gamma = gamma
                best_kernel = 'sigmoid'

            if printValue:
                print("Ending in", time() - start_time_sigmoid2)

    if printValue:
        print("End training of SVC with kernel sigmoid after", time() - start_time_sigmoid, "s.")

    # setto valori numerici per evitare problemi nella lettura e conversione da file, tanto non verranno visti dal
    # costruttore del modello se sono ancora None a questo punto del codice
    if best_gamma is None:
        best_gamma = 'auto'
    if best_degree is None:
        best_degree = 0

    if printValue:
        print("End training of SVC after", time() - start_time, "s.")

    return best_C, best_degree, best_gamma, best_kernel, best_scores, best_total_score


def KNR_training(X, Y, k, scoring, seed, n_split, mean):
    """
    do the training of a KNR models with dataset X,Y and K_Fold_Cross_Validation
    :param X: feature set
    :param Y: label set
    :param scoring: dict of scoring used
    :param k: list of possibile neighbors
    :return: the best k
    """

    if printValue:
        print("Start training of KNR")
        start_time = time()

    best_k = None
    best_total_score = None
    best_scores = None

    for num_neighbors in k:

        if printValue:
            print("Start training of KNR with k =", num_neighbors)
            start_time_cycle = time()

        model = KNeighborsRegressor(n_neighbors=num_neighbors, weights='distance')
        result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
        total_score = total_score_regression(scoring, result)  # funzione che da result calcola media armonica
        if best_total_score is None or best_total_score < total_score:
            best_scores = result
            best_total_score = total_score
            best_k = num_neighbors

        if printValue:
            print("Ending KNR with k =", num_neighbors, "in ", time() - start_time_cycle)

    if printValue:
        print("End training of KNR after", time() - start_time, "s.")

    return best_k, best_scores, best_total_score


def RANDOMFORESTRegressor_training(X, Y, list_n_trees, scoring, seed, n_split, mean):
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
    best_scores = None
    best_total_score = None

    for trees in list_n_trees:
        for max_features in range(1, len(X[0]) + 1):  # len(X[0])==numero features

            if printValue:
                print("Start training of RFRegressor with num_trees =", trees, "and max_features =", max_features)
                start_time_cycle = time()

            model = RandomForestRegressor(n_estimators=trees, random_state=seed, max_features=max_features)
            result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
            total_score = total_score_regression(scoring, result)  # funzione che da result calcola media armonica
            if best_total_score is None or best_total_score < total_score:
                best_scores = result
                best_total_score = total_score
                best_max_features = max_features
                best_n_trees = trees

            if printValue:
                print("Ending RF with num_trees =", trees, "and max_features =", max_features, "in ",
                      time() - start_time_cycle)

    if printValue:
        print("End training of Random Forest after", time() - start_time, "s.")
    return best_n_trees, best_max_features, best_scores, best_total_score


def SVR_training(X, Y, scoring, seed, n_split, mean, multilabel=False):
    """
    do the training of a SVR with dataset X, Y and K_Fold_Cross_Validation. Find the best setting iterating on the type
    of kernel function use (linear, rbf, polynomial or sigmoid)
    :param X: feature set
    :param Y: label set
    :param scoring: dict of scoring use
    :return: the best parameter C, epsilon, gamma and degree with the best kernel function
    """

    if printValue:
        print("Start training of SVR")
        start_time = time()

    best_kernel = None
    best_C = None
    best_eps = None
    best_gamma = None
    best_degree = None
    best_scores = None
    best_total_score = None
    # range in cui variano i parametri plausibili (lungo il training)
    C_range = np.logspace(-2, 2, 5)
    eps_range = [0.0, 0.1, 1]
    gamma_range = np.logspace(-5, 0, 6)
    degree_range = range(2, 4, 1)
    # default parameter
    # C_range = [1.0]
    # eps_range = [0.1]
    # gamma_range = ['auto']
    # degree_range = [3]

    # case kernel is linear

    if printValue:
        print("Start training of SVR with kernel linear")
        start_time_linear = time()

    for C in C_range:
        for eps in eps_range:

            if printValue:
                print("Starting cycle with C =", C, "eps =", eps)
                start_time_linear2 = time()

            if multilabel:
                model = MultiOutputRegressor(SVR(kernel='linear', C=C, epsilon=eps, max_iter=1000))
            else:
                model = SVR(kernel='linear', C=C, epsilon=eps, max_iter=1000)
            result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
            total_score = total_score_regression(scoring, result)
            if best_total_score is None or best_total_score < total_score:
                best_scores = result
                best_total_score = total_score
                best_C = C
                best_eps = eps
                best_kernel = 'linear'

            if printValue:
                print("Ending in", time() - start_time_linear2)

    if printValue:
        print("End training of SVR with kernel linear after", time() - start_time_linear, "s.")

    # case kernel is rbf

    if printValue:
        print("Start training of SVR with kernel rbf")
        start_time_rbf = time()

    for C in C_range:
        for eps in eps_range:
            for gamma in gamma_range:

                if printValue:
                    print("Starting cycle with C =", C, "eps =", eps, "gamma =", gamma)
                    start_time_rbf2 = time()

                if multilabel:
                    model = MultiOutputRegressor(SVR(kernel='rbf', C=C, gamma=gamma, epsilon=eps, max_iter=1000))
                else:
                    model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=eps, max_iter=1000)
                result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
                total_score = total_score_regression(scoring, result)
                if best_total_score < total_score:
                    best_scores = result
                    best_total_score = total_score
                    best_C = C
                    best_eps = eps
                    best_gamma = gamma
                    best_kernel = 'rbf'

                if printValue:
                    print("Ending in", time() - start_time_rbf2)

    if printValue:
        print("End training of SVR with kernel rbf after", time() - start_time_rbf, "s.")

    # case kernel is polynomial

    if printValue:
        print("Start training of SVR with kernel polynomial")
        start_time_poly = time()

    for C in C_range:
        for eps in eps_range:
            for degree in degree_range:
                for gamma in gamma_range:

                    if C == 100 and gamma == 1:
                        continue

                    if printValue:
                        print("Starting cycle with C =", C, "eps =", eps, "degree =", degree, "gamma =", gamma)
                        start_time_poly2 = time()

                    if multilabel:
                        model = MultiOutputRegressor(SVR(kernel='poly', C=C, gamma=gamma, degree=degree, max_iter=1000))
                    else:
                        model = SVR(kernel='poly', C=C, gamma=gamma, degree=degree, max_iter=1000)
                    result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
                    total_score = total_score_regression(scoring, result)
                    if best_total_score < total_score:
                        best_scores = result
                        best_total_score = total_score
                        best_C = C
                        best_eps = eps
                        best_gamma = gamma
                        best_degree = degree
                        best_kernel = 'poly'

                    if printValue:
                        print("Ending in", time() - start_time_poly2)

    if printValue:
        print("End training of SVR with kernel polynomial after", time() - start_time_poly, "s.")

    # case kernel is sigmoid
    if printValue:
        print("Start training of SVR with kernel sigmoid")
        start_time_sigmoid = time()

    for C in C_range:
        for eps in eps_range:
            for gamma in gamma_range:

                if printValue:
                    print("Starting cycle with C =", C, "eps =", eps, "gamma =", gamma)
                    start_time_sigmoid2 = time()

                if multilabel:
                    model = MultiOutputRegressor(SVR(kernel='sigmoid', C=C, gamma=gamma, epsilon=eps, max_iter=1000))
                else:
                    model = SVR(kernel='sigmoid', C=C, gamma=gamma, epsilon=eps, max_iter=1000)
                result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
                total_score = total_score_regression(scoring, result)
                if best_total_score < total_score:
                    best_scores = result
                    best_total_score = total_score
                    best_C = C
                    best_eps = eps
                    best_gamma = gamma
                    best_kernel = 'sigmoid'

                if printValue:
                    print("Ending in", time() - start_time_sigmoid2)

    if printValue:
        print("End training of SVR with kernel sigmoid after", time() - start_time_sigmoid, "s.")

    # setto valori numerici per evitare problemi nella lettura e conversione da file, tanto non verranno visti dal
    # costruttore del modello se sono ancora None a questo punto del codice
    if best_gamma is None:
        best_gamma = 'auto'
    if best_degree is None:
        best_degree = 0

    if printValue:
        print("End training of SVR after", time() - start_time, "s.")

    return best_C, best_eps, best_degree, best_gamma, best_kernel, best_scores, best_total_score


def training(X, Y, name_models, scoring, k=[5], list_n_trees=[10], seed=111, n_split=10, mean=True,
             file_name="best_setting.txt", multilabel=False):
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
    start_time = time()
    path = os.path.abspath('')
    fl = open(path + model_setting_test_dir + file_name, "w")
    fl.writelines(["seed " + str(seed) + "\n", "n_split " + str(n_split) + "\n"])
    if __init__.dec_tree in name_models:
        model = DecisionTreeClassifier(random_state=seed)
        best_scores = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, seed=seed, n_split=n_split,
                                                           mean=mean)
        best_total_score = scoringUtils.hmean_scores(scoring, best_scores)
        for key, value in best_scores.items():
            fl.writelines([str(key) + "_" + str(__init__.dec_tree) + " " + str(value) + "\n"])
        fl.writelines(["total_score_" + str(__init__.dec_tree) + " " + str(best_total_score) + "\n"])
    if __init__.rand_forest in name_models:
        best_n_trees, best_max_features, best_scores, best_total_score = \
            RANDOMFOREST_training(X, Y, list_n_trees, scoring, seed, n_split, mean)
        fl.writelines(["best_n_trees " + str(best_n_trees) + "\n", "best_max_features " +
                       str(best_max_features) + "\n"])
        for key, value in best_scores.items():
            fl.writelines([str(key) + "_" + str(__init__.rand_forest) + " " + str(value) + "\n"])
        fl.writelines(["total_score_" + str(__init__.rand_forest) + " " + str(best_total_score) + "\n"])
    if __init__.knn in name_models:
        best_k, best_scores, best_total_score = KNN_training(X, Y, k, scoring, seed, n_split, mean)
        fl.writelines(["best_k " + str(best_k) + "\n"])
        for key, value in best_scores.items():
            fl.writelines([str(key) + "_" + str(__init__.knn) + " " + str(value) + "\n"])
        fl.writelines(["total_score_" + str(__init__.knn) + " " + str(best_total_score) + "\n"])
    if __init__.svc in name_models:
        best_C, best_degree, best_gamma, best_kernel, best_scores, best_total_score = \
            SVC_training(X, Y, scoring, seed, n_split, mean)
        fl.writelines(["best_C " + str(best_C) + "\n", "best_degree " + str(best_degree) + "\n", "best_gamma " +
                       str(best_gamma) + "\n", "best_kernel " + str(best_kernel) + "\n"])
        for key, value in best_scores.items():
            fl.writelines([str(key) + "_" + str(__init__.svc) + " " + str(value) + "\n"])
        fl.writelines(["total_score_" + str(__init__.svc) + " " + str(best_total_score) + "\n"])
    if __init__.dec_tree_regressor in name_models:
        model = DecisionTreeRegressor(random_state=seed)
        best_scores = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, seed=seed, n_split=n_split,
                                                           mean=mean)
        best_total_score = scoringUtils.total_score_regression(scoring, best_scores)
        for key, value in best_scores.items():
            fl.writelines([str(key) + "_" + str(__init__.dec_tree_regressor) + " " + str(value) + "\n"])
        fl.writelines(["total_score_" + str(__init__.dec_tree_regressor) + " " + str(best_total_score) + "\n"])
    if __init__.rand_forest_regressor in name_models:
        best_n_trees, best_max_features, best_scores, best_total_score = \
            RANDOMFORESTRegressor_training(X, Y, list_n_trees, scoring, seed, n_split, mean)
        fl.writelines(["best_n_trees " + str(best_n_trees) + "\n", "best_max_features " +
                       str(best_max_features) + "\n"])
        for key, value in best_scores.items():
            fl.writelines([str(key) + "_" + str(__init__.rand_forest_regressor) + " " + str(value) + "\n"])
        fl.writelines(["total_score_" + str(__init__.rand_forest_regressor) + " " + str(best_total_score) + "\n"])
    if __init__.knr in name_models:
        best_k, best_scores, best_total_score = KNR_training(X, Y, k, scoring, seed, n_split, mean)
        fl.writelines(["best_k " + str(best_k) + "\n"])
        for key, value in best_scores.items():
            fl.writelines([str(key) + "_" + str(__init__.knr) + " " + str(value) + "\n"])
        fl.writelines(["total_score_" + str(__init__.knr) + " " + str(best_total_score) + "\n"])
    if __init__.svr in name_models:
        best_C, best_eps, best_degree, best_gamma, best_kernel, best_scores, best_total_score = \
            SVR_training(X, Y, scoring, seed, n_split, mean, multilabel=multilabel)
        fl.writelines(["best_C " + str(best_C) + "\n", "best_eps " + str(best_eps) + "\n", "best_degree " +
                       str(best_degree) + "\n", "best_gamma " + str(best_gamma) + "\n", "best_kernel " +
                       str(best_kernel) + "\n"])
        for key, value in best_scores.items():
            fl.writelines([str(key) + "_" + str(__init__.svr) + " " + str(value) + "\n"])
        fl.writelines(["total_score_" + str(__init__.svr) + " " + str(best_total_score) + "\n"])

    fl.close()
    print("End training after", time() - start_time, "s.")


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

    if __init__.dec_tree in name_models:
        models[__init__.dec_tree] = DecisionTreeClassifier(random_state=int(settings["seed"]))

    if __init__.rand_forest in name_models:
        models[__init__.rand_forest] = RandomForestClassifier(random_state=int(settings["seed"]),
                                                              max_features=int(settings["best_max_features"]),
                                                              n_estimators=int(settings["best_n_trees"]))

    if __init__.knn in name_models:
        models[__init__.knn] = KNeighborsClassifier(n_neighbors=int(settings["best_k"]), weights='distance')

    if __init__.svc in name_models:
        if settings["best_gamma"] == "'auto'":
            models[__init__.svc] = OneVsRestClassifier(SVC(kernel=settings["best_kernel"], C=float(settings["best_C"]),
                                                           degree=int(settings["best_degree"])))
        else:
            models[__init__.svc] = OneVsRestClassifier(SVC(kernel=settings["best_kernel"], C=float(settings["best_C"]),
                                                           gamma=float(settings["best_gamma"]),
                                                           degree=int(settings["best_degree"])))
    if __init__.dec_tree_regressor in name_models:
        models[__init__.dec_tree_regressor] = DecisionTreeRegressor(random_state=int(settings["seed"]))

    if __init__.rand_forest_regressor in name_models:
        models[__init__.rand_forest_regressor] = RandomForestRegressor(random_state=int(settings["seed"]),
                                                                       max_features=int(settings["best_max_features"]),
                                                                       n_estimators=int(settings["best_n_trees"]))

    if __init__.knr in name_models:
        models[__init__.knr] = KNeighborsRegressor(n_neighbors=int(settings["best_k"]), weights='distance')

    if __init__.svr in name_models:
        if settings["best_gamma"] == "'auto'":
            models[__init__.svr] = OneVsRestClassifier(SVR(kernel=settings["best_kernel"], C=float(settings["best_C"]),
                                                           degree=int(settings["best_degree"]),
                                                           epsilon=float(settings["best_eps"])))
        else:
            models[__init__.svr] = OneVsRestClassifier(SVR(kernel=settings["best_kernel"], C=float(settings["best_C"]),
                                                           gamma=float(settings["best_gamma"]),
                                                           degree=int(settings["best_degree"]),
                                                           epsilon=float(settings["best_eps"])))

    fl.close()

    return models


def read_setting(file_name):
    """
    Return the best scoring of a test
    :param name_models:
    :param file_name:
    :return: dict with the scores
    """

    path = os.path.abspath('')
    fl = open(path + model_settings_dir + file_name, "r")  # aggiungere path
    setting = {}
    while 1:
        line = fl.readline()
        line = line[:-1]
        if len(line) == 0:
            break
        parameter, value = str.split(line, " ")
        setting[parameter] = value

    return setting


def is_a_classification_dataset(dataset_name):
    if dataset_name in __init__.list_classification_dataset:
        return True
    return False


def is_a_multilabel_dataset(dataset_name):
    if dataset_name in __init__.list_multilabel_dataset:
        return True
    return False


def check_percentage(percentage):
    """
    Verifica che la percentuale percentuage è una percentuale compresa nella lista delle percentuali scelte
    :param percentage:float,compreso tra 0 e 1 (estremi inclusi)
    :return: None
    """
    if percentage > 1 or percentage < 0:
        print('percentage must be between 0 and 1')
        exit(1)
    if not percentage in __init__.percentuals_NaN:
        print('invalid percentage')
        exit(1)
    return None


def check_strategies(dataset_name, strategy):
    """
    Verifica che il dataset supporta la strategia strategy
    :param dataset_name:string,nome del dataset
    :param strategy: string,nome strategia
    :return: none
    """
    all_strategies = ['mean', 'eliminate_row', 'mode', 'median']
    strategies_balance = all_strategies
    strategies_eye = ['mean', 'eliminate_row', 'median']
    strategies_page = ['mean', 'eliminate_row', 'median']
    strategies_seed = ['mean', 'median']
    strategies_tris = ['eliminate_row', 'mode']
    strategies_zoo = ['mode']
    strategies_compress_strength = ['mean', 'eliminate_row', 'median']
    strategies_airfoil = ['mean', 'eliminate_row', 'median']
    strategies_auto = ['mean', 'eliminate_row', 'median']
    strategies_power_plant = ['mean', 'eliminate_row', 'median']
    strategies_energy = ['mean', 'eliminate_row', 'median']
    if dataset_name == __init__.balance:
        if not strategy in strategies_balance:
            print('invalid strategy for', __init__.balance)
            exit(1)
    elif dataset_name == __init__.eye:
        if not strategy in strategies_eye:
            print('invalid strategy for', __init__.eye)
            exit(1)
    elif dataset_name == __init__.page:
        if not strategy in strategies_page:
            print('invalid strategy for', __init__.page)
            exit(1)
    elif dataset_name == __init__.seed:
        if not strategy in strategies_seed:
            print('invalid strategy for', __init__.seed)
            exit(1)
    elif dataset_name == __init__.tris:
        if not strategy in strategies_tris:
            print('invalid strategy for', __init__.tris)
            exit(1)
    elif dataset_name == __init__.zoo:
        if not strategy in strategies_zoo:
            print('invalid strategy for', __init__.zoo)
            exit(1)
    elif dataset_name == __init__.compress_strength:
        if not strategy in strategies_compress_strength:
            print('invalid strategy for', __init__.compress_strength)
            exit(1)
    elif dataset_name == __init__.airfoil:
        if not strategy in strategies_airfoil:
            print('invalid strategy for', __init__.airfoil)
            exit(1)
    elif dataset_name == __init__.auto:
        if not strategy in strategies_auto:
            print('invalid strategy for', __init__.auto)
            exit(1)
    elif dataset_name == __init__.power_plant:
        if not strategy in strategies_power_plant:
            print('invalid strategy for', __init__.power_plant)
            exit(1)
    elif dataset_name == __init__.energy:
        if not strategy in strategies_energy:
            print('invalid strategy for', __init__.energy)
            exit(1)
    else:
        print('invalid dataset_name', dataset_name)
        exit(1)


mean = 'mean'
eliminate_row = 'eliminate_row'
median = 'median'
mode = 'mode'

if __name__ == '__main__':
    seed = 100
    name_models_classification = [__init__.rand_forest, __init__.dec_tree, __init__.knn, __init__.svc]
    name_models_regression = [__init__.rand_forest_regressor, __init__.dec_tree_regressor, __init__.knr, __init__.svr]
    dataset_name = __init__.energy
    classification = is_a_classification_dataset(dataset_name)
    multilabel = is_a_multilabel_dataset(dataset_name)
    k_range = range(3, 21, 1)
    n_trees_range = range(5, 21, 1)

    X, Y, scoring, name_setting_file, name_radar_plot_file = \
          main.case_full_dataset(dataset_name, standardize=False, normalize=True, classification=classification,
                           multilabel=multilabel)
    # X, Y, scoring, name_setting_file, name_radar_plot_file = \
    #     main.case_NaN_dataset(dataset_name, "median", seed, 0.05, classification=classification, multilabel=multilabel)

    if classification:
        training(X, Y, name_models_classification, scoring, k=k_range, list_n_trees=n_trees_range, seed=seed,
                 n_split=10, mean=True, file_name=name_setting_file, multilabel=multilabel)
    else:
        training(X, Y, name_models_regression, scoring, k=k_range, list_n_trees=n_trees_range, seed=seed,
                 n_split=10, mean=True, file_name=name_setting_file, multilabel=multilabel)
