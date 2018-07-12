import CSV,numpy
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def list_models(names,num_tree=10,seed=1000):
    """
    ritorna una lista di modelli
    :param names: string list
    :return: list models
    lista completa modelli:['RANDFOREST','CART','LR','LDA','KNN','NB','SVM']
    """
    models = []
    if('RANDFOREST' in names):
        models.append(('RANDFOREST',RandomForestClassifier(num_tree,random_state=seed)))
    if('CART' in names):
        models.append(('CART', DecisionTreeClassifier()))
    if ('LR' in names):
        models.append(('LR', LogisticRegression()))
    if('LDA' in names):
        models.append(('LDA', LinearDiscriminantAnalysis()))
    if ('KNN' in names):
        models.append(('KNN', KNeighborsClassifier()))
    if('NB' in names):
        models.append(('NB', GaussianNB()))
    if ('SVM' in names):
        models.append(('SVM', SVC()))
    return models

if __name__=='__main__':
    X,Y=CSV.read_csv('tic_tac_toe.csv')
    Y=CSV.convert_label_values(Y,['positive','negative'],[1,0])
    print(Y)
    for i in range(len(X[:,0])):
        for j in range(len(X[0,:])):
            if X[i][j]=='x':
                X[i][j]= numpy.float64(1)
            elif X[i][j]=='o':
                X[i][j] =numpy.float64(0)
            else:
                X[i][j]=numpy.float64(2)
    X=CSV.convert_type_to_float(X)
    name_models=['RANDFOREST','CART','LR','LDA','KNN','NB','SVM']
    models=list_models(name_models)
    results = []
    names = []
    seed=70
    scoring = {'accuracy': 'accuracy',
               'precision_macro': 'precision_macro',
               'recall_micro': 'recall_macro'}
    #scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        scores=model_selection.cross_validate(model,X,Y,cv=kfold,scoring=scoring,return_train_score=True)
        #cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(scores)
        print(name,scoring['accuracy'],scores['test_accuracy'].mean(),scoring['precision_macro'],scores['test_precision_macro'].mean(),scoring['recall_micro'],scores['test_recall_micro'].mean())


