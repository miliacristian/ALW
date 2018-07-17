from sklearn import metrics
from sklearn.metrics import make_scorer


def roc_auc_micro(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred, average="micro")


def roc_auc_weighted(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred, average="weighted")


def average_precision_micro(y_true, y_pred):
    return metrics.average_precision_score(y_true, y_pred, average="micro")


def average_precision_weighted(y_true, y_pred):
    return metrics.average_precision_score(y_true, y_pred, average="weighted")


def create_dictionary_classification_scoring():
    scoring = {
        'accuracy': 'accuracy',
        'precision_micro': 'precision_micro',
        # 'precision_weighted': 'precision_weighted',
        'recall_micro': 'recall_micro',
        # 'recall_weighted': 'recall_weighted', #uguale alla accuracy
        'roc_auc_micro': make_scorer(roc_auc_micro),
        # 'roc_auc_weighted': make_scorer(roc_auc_weighted),
        'f1_micro': 'f1_micro',
        # 'f1_weighted': 'f1_weighted',
        'average_precision_micro': make_scorer(average_precision_micro),
        # 'average_precision_weighted': make_scorer(average_precision_weighted),
               }
    return scoring


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
        print('fit_time', "{0:.6f}".format(dict_scores['fit_time'].mean()),end=' ')
    if (score_time):#stampa score_time
        print('score_time', "{0:.6f}".format(dict_scores['score_time'].mean()),end=' ')
    for key, value in dict_name_scoring.items():
        if(test):#stampa test score
            if type(value) is str:
                print('test_'+key,"{0:.6f}".format(dict_scores['test_'+ value].mean()),end=' ')
            else:
                print('test_' + key, "{0:.6f}".format(dict_scores['test_' + str(value)[12:-1]].mean()), end=' ')
        if(train):#stampa train_score
            if type(value) is str:
                print('train_' + key, "{0:.6f}".format(dict_scores['train_' + value].mean()), end=' ')
            else:
                print('train_' + key, "{0:.6f}".format(dict_scores['train_' + str(value)[12:-1]].mean()), end=' ')
    print() #new line
    return None