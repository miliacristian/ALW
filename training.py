#rende un classificatore multiclass funzionante anche per multilabel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import scoringUtils
from scoringUtils import hmean_scores

def KNN_training(X, Y, k, scoring, seed, n_split, mean):
    """
    do the training of a KNN models with dataset X,Y and K_Fold_Cross_Validation
    :param X: feature set
    :param Y: label set
    :param scoring: dict of scoring used
    :param k: list of possibile neighbors
    :return: the best k
    """
    best_k = None
    best_total_score = None
    for num_neighbors in k:
        model = KNeighborsClassifier(n_neighbors=num_neighbors, weights='distance')
        result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
        harmonic_mean = hmean_scores(scoring, result)  # funzione che da result calcola media armonica
        if best_total_score is None or best_total_score < harmonic_mean:
            best_total_score = harmonic_mean
            best_k = num_neighbors
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
    best_n_trees = None
    best_max_features = None
    best_total_score = None
    for trees in list_n_trees:
        for max_features in range(1,len(X[0])+1):  # len(X[0])==numero features
            model=RandomForestClassifier(trees, random_state=seed, max_features=max_features)
            result = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=mean)
            harmonic_mean = hmean_scores(scoring, result)  # funzione che da result calcola media armonica
            if best_total_score is None or best_total_score < harmonic_mean:
                best_total_score = harmonic_mean
                best_max_features = max_features
                best_n_trees = trees
    return best_n_trees, best_max_features

def SVM_training(X, Y, scoring,seed,n_split,mean): #TODO
    pass


def training(X, Y, name_models,  scoring, k = [5], list_n_trees = [10], seed = 111, n_split=10, mean=True):
    """
    :param X: features set
    :param Y: label set
    :param name_models:list, lista nomi modelli
    :param scoring: dict,dizionario di scoring
    :param k: list,lista dei possibili k per KNN
    :param n_trees: list,lista dei possibili numeri di alberi per il random forest
    :param seed: int,seme per generatore pseudocasuale
    :return: dict model con key nome del modello e valori i modelli ottimali per il dataset X,Y
    """
    models = {}
    if ('RANDFOREST' in name_models):
        best_n_trees, best_max_features = RANDOMFOREST_training(X, Y, list_n_trees,scoring,seed,n_split,mean)
        models['RANDFOREST'] = RandomForestClassifier(best_n_trees, random_state=seed, max_features=best_max_features)
    if ('CART' in name_models):
        models['CART'] = DecisionTreeClassifier(random_state=seed)
    if ('KNN' in name_models):
        best_k = KNN_training(X, Y, k, scoring, seed, n_split, mean)
        models['KNN'] = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    if ('SVM' in name_models):  # solo per classificatore binario
        SVM_training(X, Y,scoring,mean,n_split)
        models['SVM'] = OneVsRestClassifier(SVC())
    return models
