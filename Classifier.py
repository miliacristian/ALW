import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

def classifier():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv" #url da dove prendere il dataset
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] #nomi features
    dataframe = pandas.read_csv(url, names=names) #dataset
    print("dataset:\n",dataframe)
    array = dataframe.values #matrice con tutti i dati,array[i]==riga i==esempio i
    X = array[:,0:8]#come array ma senza la colonna con le label
    Y = array[:,8] #array di label
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)#k-fold cross validation,senza il seed le metriche variano ad ogni esecuzione
    model = LogisticRegression()#scelta del modello
    scoring = 'accuracy'#impostare la metrica
    """
    possibili scoring:
    ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']
    """
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring) #cv==cross validation
    print("Accuracy media=",results.mean())
    print("deviazione standard dell'Accuracy=",results.std())
    #print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
