import dataset
from sklearn import model_selection, metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import p
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
         models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    # if ('KNN' in names):
    #         models.append(('KNN', KNeighborsClassifier()))
    # if ('SVM' in names):#solo per classificatore binario
    #      models.append(('SVM', SVC(probability=True)))
    # if ('LR' in names):#solo per classificatore binario
    #     models.append(('LR', LogisticRegression()))
    # if('LDA' in names):#solo per classificatore binario
    #      models.append(('LDA', LinearDiscriminantAnalysis()))
    # if('NB' in names):#solo per classificatore binario
    #     models.append(('NB', GaussianNB()))




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
            print('test_'+key,(dict_scores['test_'+value].mean()),end=' ')
        if(train):#stampa train_score
            print('train_' + key, dict_scores['train_' + value].mean(), end=' ')
    print()#serve per new line
    return None

def create_dictionary_of_scoring():
    scoring = {'accuracy': 'accuracy',
               #'precision': 'precision' non usare per multilabel
                'average_precision': 'average_precision',
                'precision_micro': 'precision_micro',
                'precision_macro': 'precision_macro',
                'precision_weighted': 'precision_weighted',
                'precision_samples':'precision_samples',
                #'recall': 'recall', non usare per multilabel
                'recall_macro': 'recall_macro',
                'recall_micro': 'recall_micro',
                'recall_weighted': 'recall_weighted',
                 'recall_samples': 'recall_samples',
                 'roc_auc': 'roc_auc',
               #'neg_log_loss': 'neg_log_loss', non usare per multilabel
                #'f1': 'f1',non usare per multilabel
                'f1_macro': 'f1_macro',
                'f1_micro': 'f1_micro',
               'f1_weighted': 'f1_weighted',
               'f1_samples': 'f1_samples'
               }
    return scoring

def prova():
    L=[1,2,3,4,5,6,7,8,9,10]
    count=0
    lun=len(L)
    for l in range(lun):#il range non cambia anche se la lun cambia
        count=count+1
        print(lun)
        lun=lun-1
    print(count)


if __name__=='__main__':
    #i dataset seed e balance non funzionano
    X,Y=dataset.load_balance_dataset()
    #X,Y=remove_row_dataset(X,Y,0,3)
    dataset.print_dataset(X, Y)
    #X,Y=remove_row_with_label_L(X,Y,[1.0,2.0])
    #dataset.print_dataset(X, Y)
    #Y = dataset.one_hot_encoding(Y)
    name_models=['RANDFOREST','CART','LR','LDA','KNN','NB','SVM']
    models=list_models(name_models,seed=1000)
    results = []
    names = []
    seed=7
    # X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.3)
    # model=RandomForestClassifier(10,random_state=seed)
    # model.fit(X_train, Y_train)
    # predictions=model.predict(X_test)
    # neg_log_loss=metrics.log_loss(Y_test,predictions)
    # print(neg_log_loss)
    # print(predictions)
    # exit(0)
    scoring=create_dictionary_of_scoring()
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10,shuffle=True,random_state=seed)
        scores=model_selection.cross_validate(model,X,Y,cv=kfold,scoring=scoring,return_train_score=True,n_jobs=1)
        results.append(scores)
        print_scoring(name,scoring,scores,test=True,train=True,fit_time=True,score_time=True)


