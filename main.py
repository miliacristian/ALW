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
    X, Y = dataset.load_dataset('balance')
    name_models = ['RANDFOREST', 'CART', 'KNN']
    results = []
    names = []
    seed = 100
    scoring = scoringUtils.create_dictionary_classification_scoring()
    list_scores = []
    list_names = []
    models = training(X, Y, name_models, scoring, k=range(8, 12, 1), list_n_trees=range(8, 12, 1), seed=seed,
                      n_split=10, mean=True)
    for name, model in models.items():
        scores = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split=10, seed=seed)
        results.append(scores)
        print(name)
        print(scores)
        # scoringUtils.print_scoring(name, scoring, scores, test=True, train=False, fit_time=True, score_time=True)
        list_scores.append(scores)
        list_names.append(name)
        print("total score =", scoringUtils.hmean_scores(scoring, scores))
    scoringUtils.radar_plot(list_names, scoring, list_scores)
