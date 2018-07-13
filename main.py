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
    """""
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
    if('NB' in names):
        models.append(('NB', GaussianNB()))
    if ('KNN' in names):
            models.append(('KNN', KNeighborsClassifier()))
    if ('SVM' in names):
        models.append(('SVM', SVC(probability=True)))
    return models

if __name__=='__main__':
    X,Y=CSV.read_csv('zoo.csv',skip_rows=8)
    print(X)
    exit(0)
    X,Y=CSV.read_csv('tic_tac_toe.csv')
    Y=CSV.convert_label_values(Y,['positive','negative'],[1,0])
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
    seed=7
    scoring = {'accuracy': 'accuracy',
               'precision': 'precision',
               'average_precision': 'average_precision',
               'precision_micro': 'precision_micro',
               'precision_macro': 'precision_macro',
               'precision_weighted': 'precision_weighted',
               #'precision_samples':'precision_samples',
               'recall': 'recall',
               'recall_macro': 'recall_macro',
               'recall_micro': 'recall_micro',
               'recall_weighted': 'recall_weighted',
               # 'recall_samples': 'recall_samples',
               'roc_auc': 'roc_auc',
                'neg_log_loss':'neg_log_loss',
                'f1': 'f1',
                'f1_macro':'f1_macro',
                'f1_micro': 'f1_micro',
               'f1_weighted':'f1_weighted'
               # 'f1_samples': 'f1_samples',
               }
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10,shuffle=True)
        scores=model_selection.cross_validate(model,X,Y,cv=kfold,scoring=scoring,return_train_score=True,n_jobs=1)
        results.append(scores)
        print(name,scoring['accuracy'],scores['test_accuracy'].mean(),scoring['precision_macro'],scores['test_precision_macro'].mean(),scoring['recall_micro'],scores['test_recall_micro'].mean(),scoring['precision'],scores['test_precision'].mean(),scoring['neg_log_loss'],scores['test_neg_log_loss'].mean())


