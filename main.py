import load_dataset
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#
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

def print_scoring(name_model,dict_name_scoring,dict_scores,test=True,train=False,fit_time=False,score_time=False):
    """
    Stampa tutti gli score presenti nel dizionario scoring del modello name_model
    :param name_model: string,nome modello es KNN
    :param dict_name_scoring:dictionary,dizionario nomi degli score
    :param dict_scores:dictionary, dizionario dei valori degli score
    :param test:boolean,se test==True stampa il test di tutte le score es test_accuracy,test precision
    :param train:boolean,se train==True stampa il train di tutte le score es train_accuracy,train precision
    :param fit_time:boolean,se fit_time==True stampa il tempo di fit
    :param score_time:boolena,se score_time==True stampa il tempo di score
    :return: None
    """
    print(name_model)
    if(fit_time):#stampa fit_time
        print('fit_time',dict_scores['fit_time'].mean(),end=' ')
    if (score_time):#stampa score_time
        print('score_time', dict_scores['score_time'].mean(),end=' ')
    for key, value in dict_name_scoring.items():
        if(test):#stampa test score
            print('test_'+key,dict_scores['test_'+value].mean(),end=' ')
        if(train):#stampa train_score
            print('train_' + key, dict_scores['train_' + value].mean(), end=' ')
    print()#serve per new line
    return None

def create_dictionary_of_scoring():
    scoring = {'accuracy': 'accuracy',
               'precision': 'precision',
               'average_precision': 'average_precision',
               'precision_micro': 'precision_micro',
               'precision_macro': 'precision_macro',
               'precision_weighted': 'precision_weighted',
               # 'precision_samples':'precision_samples',
               'recall': 'recall',
               'recall_macro': 'recall_macro',
               'recall_micro': 'recall_micro',
               'recall_weighted': 'recall_weighted',
               # 'recall_samples': 'recall_samples',
               'roc_auc': 'roc_auc',
               'neg_log_loss': 'neg_log_loss',
               'f1': 'f1',
               'f1_macro': 'f1_macro',
               'f1_micro': 'f1_micro',
               'f1_weighted': 'f1_weighted'
               # 'f1_samples': 'f1_samples',
               }
    return scoring

if __name__=='__main__':
    X,Y=load_dataset.load_seed_dataset()
    load_dataset.print_dataset(X,Y)
    name_models=['RANDFOREST','CART','LR','LDA','KNN','NB','SVM']
    models=list_models(name_models)
    results = []
    names = []
    seed=7
    scoring=create_dictionary_of_scoring()
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10,shuffle=True)
        scores=model_selection.cross_validate(model,X,Y,cv=kfold,scoring=scoring,return_train_score=True,n_jobs=1)
        results.append(scores)
        print_scoring(name,scoring,scores,test=True,train=True,fit_time=True,score_time=True)


