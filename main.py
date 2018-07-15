import load_dataset,numpy
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

def remove_row_with_label_L(X,Y,L):
    length=len(Y)
    for i in range(length):
        if Y[i]==L:
            Y=numpy.delete(Y,i,axis=0)
            X=numpy.delete(X,i,axis=0)
            i=i-1
            length=length-1
    return X,Y

def remove_row_dataset(X,Y,A,B,step=1):
    """
    :param X: features set
    :param Y: label set
    :param A: int,indice prima riga da eliminare
    :param B: int,indice ultima riga-1 da eliminare
    :param step: int,quante righe saltare prima di eliminare la prossima riga
    :return: X,Y con righe da A,B eliminate con step step
    """
    temp_X=numpy.delete(X,numpy.s_[A:B:step],axis=0)#axis=0==riga,axis=1==colonna
    temp_Y = numpy.delete(Y, numpy.s_[A:B:step], axis=0)
    return temp_X,temp_Y

def remove_column_dataset(X,Y,A,B,step=1):
    """
    :param X: features set
    :param Y: label set
    :param A: int,indice prima colonna da eliminare
    :param B: int,indice ultima colonna-1 da eliminare
    :param step: int,quante colonne saltare prima di eliminare la prossima colonna
    :return: X,Y con colonne da A,B eliminate con step step
    """
    temp_X=numpy.delete(X,numpy.s_[A:B:step],axis=1)#axis=0==riga,axis=1==colonna
    temp_Y = numpy.delete(Y, numpy.s_[A:B:step], axis=1)
    return temp_X,temp_Y


if __name__=='__main__':
    #il dataset seed non funziona
    X,Y=load_dataset.load_balance_dataset()
    X,Y=remove_row_dataset(X,Y,0,3)
    X,Y=remove_row_with_label_L(X,Y,2.0)
    load_dataset.print_dataset(X,Y)
    exit(0)
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


