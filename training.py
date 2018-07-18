from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def KNN_training(X, Y, k, scoring):
    """
    do the training of a KNN models with dataset X,Y and K_Fold_Cross_Validation
    :param X: feature set
    :param Y: label set
    :param scoring: dict of scoring used
    :param k: list of possibile neighbors
    :return: the best k
    """


def RANDOMFOREST_training(X, Y, n_trees, scoring):
    """
     do the training of a Random Forest with dataset X, Y and K_Fold_Cross_Validation. Trova il setting migliore iterando su
    tutte i valori possibili di max_features
    :param X: feature set
    :param Y: label set
    :param scoring: dict of scoring used
    :param n_trees: number of trees of the forest
    :return: best n_trees, best max_features
    """


def SVM_training(X, Y, scoring): #TODO
    pass


def training(X, Y, name_models, k = None, n_trees = None, seed = 111, scoring):
    #da richiamare i training precedenti presenti in name_models
    # ritorna lista di modelli
    models = {}
    if ('RANDFOREST' in name_models):
        best_n_trees, best_max_features = RANDOMFOREST_training(X, Y, n_trees)
        models['RANDFOREST'] = RandomForestClassifier(best_n_trees, random_state=seed, max_features=best_max_features)
    if ('CART' in name_models):
        models['CART'] = DecisionTreeClassifier(random_state=seed)
    if ('KNN' in name_models):
        best_k = KNN_training(X, Y, k)
        models['KNN'] = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    if ('SVM' in name_models):  # solo per classificatore binario
        SVM_training(X, Y)
        models['SVM'] = OneVsRestClassifier(SVC())

    return models
