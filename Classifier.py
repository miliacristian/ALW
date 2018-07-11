import pandas,numpy
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

def classifier():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data" #url da dove prendere il dataset
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age','l', 'class'] #nomi features
    dataframe = pandas.read_csv(url, names=names) #dataset
    #il dataset deve essere numerico
    print("dataset:\n",dataframe)
    array = dataframe.values #matrice con tutti i dati,array[i]==riga i==esempio i
    X = array[:,0:9]#come array ma senza la colonna con le label
    Y = array[:,9] #array di label
    for i in range(len(Y)):
        if Y[i]=='positive':
            Y[i]=numpy.float64(1)
        else:
            Y[i]=numpy.float64(0)
    Y = Y.astype('float64')
    print(len(X))
    print(len(X[:,0]))#numero righe
    for i in range(len(X[:,0])):
        for j in range(len(X[0,:])):
            if X[i][j]=='x':
                X[i][j]= numpy.float64(1)
            elif X[i][j]=='o':
                X[i][j] =numpy.float64(0)
            else:
                X[i][j]=numpy.float64(2)
    print("Y:\n", Y)
    print("X:\n", X)
    print(type(X[0,0]))
    print(type(Y[0]))
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)#k-fold cross validation,senza il seed le metriche variano ad ogni esecuzione
    model = RandomForestClassifier(n_estimators=50,random_state=seed)#scelta del modello
    scoring = 'accuracy'#impostare la metrica
    """
    possibili scoring:
    ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']
    """
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring) #cv==cross validation
    print(results)
    print("Accuracy media=",results.mean())
    print("deviazione standard dell'Accuracy=",results.std())
    #print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
