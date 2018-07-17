import dataset
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scoringUtils
#rende un classificatore multiclass funzionante anche per multilabel
from sklearn.multiclass import OneVsRestClassifier


def list_models(names,num_tree=10,seed=1000, n_neighbors = 5):
    """""
    ritorna una lista di modelli
    :param names: string list
    :return: list models
    lista completa modelli:['RANDFOREST','CART','LR','LDA','KNN','NB','SVM']
    """
    models = []
    if('RANDFOREST' in names):
        models.append(('RANDFOREST',RandomForestClassifier(num_tree,random_state=seed)))
    # if('CART' in names):
    #      models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    if ('KNN' in names):
            models.append(('KNN', KNeighborsClassifier(n_neighbors= n_neighbors, weights='distance')))
    # if ('SVM' in names):#solo per classificatore binario
    #      models.append(('SVM', OneVsRestClassifier(SVC())))
    # if ('LR' in names):#solo per classificatore binario
    #     models.append(('LR', OneVsRestClassifier(LogisticRegression())))
    return models


if __name__=='__main__':
    # i dataset seed e balance non funzionano
    X, Y = dataset.load_tris_dataset()
    # Y = dataset.one_hot_encoding(Y)
    name_models = ['RANDFOREST', 'CART', 'LR', 'LDA', 'KNN', 'NB', 'SVM']
    results = []
    names = []
    seed = 100
    scoring = scoringUtils.create_dictionary_classification_scoring()
    for i in range(5, 10, 1):
        print(i)
        models = list_models(name_models, seed=1000, n_neighbors=int(i))
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10,shuffle=True,random_state=seed)
            scores=model_selection.cross_validate(model, X, Y, cv=kfold, scoring=scoring, return_train_score=True, n_jobs=1)
            results.append(scores)
            scoringUtils.print_scoring(name,scoring,scores,test=True,train=False,fit_time=True,score_time=True)



