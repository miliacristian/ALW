import dataset,__init__
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from training import training
import scoringUtils

#rende un classificatore multiclass funzionante anche per multilabel
from sklearn.multiclass import OneVsRestClassifier

if __name__=='__main__':
    X, Y = dataset.load_dataset('tris')
    print(len(X))
    exit(0)
    name_models = ['RANDFOREST', 'CART', 'KNN', 'SVM', 'LR']
    results = []
    names = []
    seed = 100
    scoring = scoringUtils.create_dictionary_classification_scoring()
    list_scores = []
    list_names = []
    models = training(X, Y, name_models,scoring,seed = seed,n_split=10,mean=True)
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10,shuffle=True,random_state=seed)
        scores=model_selection.cross_validate(model, X, Y, cv=kfold, scoring=scoring, return_train_score=True, n_jobs=1)
        results.append(scores)
        scoringUtils.print_scoring(name,scoring,scores,test=True,train=False,fit_time=True,score_time=True)
        list_scores.append(scores)
        list_names.append(name + str(i))
        print("KNN" + str(i) + "total score =", scoringUtils.hmean_scores(scoring, scores))
    scoringUtils.radar_plot(list_names, scoring, list_scores)