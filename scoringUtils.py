from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from scipy.stats import hmean
from sklearn import model_selection
from __init__ import radar_plot_dir
import warnings
import os


def roc_auc_micro(y_true, y_pred):
    """
    Chiama roc_auc_score con average=micro
    :param y_true:
    :param y_pred:
    :return:
    """
    return metrics.roc_auc_score(y_true, y_pred, average="micro")


def roc_auc_weighted(y_true, y_pred):
    """
    Chiama roc_auc_score con average=weighted
    :param y_true:
    :param y_pred:
    :return:
    """
    return metrics.roc_auc_score(y_true, y_pred, average="weighted")


def average_precision_micro(y_true, y_pred):
    """
    Chiama average_precision_score con average=micro
    :param y_true:
    :param y_pred:
    :return:
    """
    return metrics.average_precision_score(y_true, y_pred, average="micro")


def average_precision_weighted(y_true, y_pred):
    """
    Chiama average_precision_score con average=weigthed
    :param y_true:
    :param y_pred:
    :return:
    """
    return metrics.average_precision_score(y_true, y_pred, average="weighted")


def create_dictionary_classification_scoring():
    """
    Crea dizionario con key:'name_score' value:score da calcolare
    :return: dict,dizionario con tutti gli scoring da calcolare
    """
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


def create_dictionary_regression_scoring():
    """
    Crea dizionario con key:'name_score' value:score da calcolare
    :return: dict,dizionario con tutti gli scoring da calcolare
    """
    scoring = {
        'explained_variance': 'explained_variance',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_squared_log_error': 'neg_mean_squared_log_error',
        'neg_median_absolute_error': 'neg_median_absolute_error',
        'r2': 'r2',
               }
    return scoring


def radar_plot(name_models, dict_name_scoring, list_dict_scores, file_name="radar_plot", file_format=".png"):
    """
    Print and save the radar plot of the scoring

    :param file_name: name of plot file
    :param file_format: format of plot file
    :param name_models: list of str contains the name of the models used
    :param dict_name_scoring: dictionary contains the scoring of every models
    :param list_dict_scores: list of dictionary contains the scores result of k_fold for each models
    :return: None
    """
    fig = plt.figure()
    # Set data
    dict = {}
    for key, value in dict_name_scoring.items():
        if type(value) is str:
            dict[value] = []
        else:
            dict[str(value)[12:-1]] = []

    for i in range(len(list_dict_scores)):
        for key, value in dict_name_scoring.items():
            if type(value) is str:
                dict[value].append(list_dict_scores[i][value])
            else:
                dict[str(value)[12:-1]].append(list_dict_scores[i][str(value)[12:-1]])

    df = pd.DataFrame(dict)

    # ------- PART 1: Create background

    # number of variable
    categories = list(df)[:]
    N = len(categories)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # Initialise the spider plot
    #plt.suptitle(name_plot,loc='left')
    ax = plt.subplot(111, polar=True)
    plt.rc('axes', titlesize=25)
    # plt.title(name_plot,loc='right')
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.8, 0.83, 0.86, 0.89, 0.92, 0.95, 0.98],
               [str(i) for i in [0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.8, 0.83, 0.86, 0.89, 0.92, 0.95, 0.98]], color="grey", size=7)
    plt.ylim(0.6, 1)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable
    for i in range(len(name_models)):
        # loc[i] preleva la colonna i-esima dei valori del dataframe
        values = df.loc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=name_models[i])
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(bbox_to_anchor=(0.1, 0.2))

    # plt.show()
    path = os.path.abspath('')
    fig.savefig(path+radar_plot_dir+file_name+file_format)
    plt.close(fig)


def scores_to_list(dict_name_scoring, dict_scores):
    """
    Trasforma il dizionario degli scores in una lista di valori
    :param dict_name_scoring: dict,dizionario dei nomi degli scores
    :param dict_scores: dict,dizionario dei valori degli scores
    :return: list,lista degli scores
    """
    scores_list = []
    for key, value in dict_name_scoring.items():
            if type(value) is str:
                scores_list.append(dict_scores[value].mean())
            else:
                scores_list.append(dict_scores[str(value)[12:-1]].mean())
    return scores_list


def hmean_scores(dict_name_scoring, dict_scores):
    """
    Calcola la media armonica degli scores
    :param dict_name_scoring: dict,dizionario dei nomi degli scorer
    :param dict_scores: dict,dizionario dei valori degli scores
    :return: media armonica degli scores
    """

    if list(dict_scores.values()).__contains__(0.0):
        return 0
    return hmean(scores_to_list(dict_name_scoring, dict_scores))


def total_score_regression(dict_name_scoring, dict_scores):
    """
    Calcola il total score per la regressione
    :param dict_name_scoring: dict,dizionario dei nomi degli scorer
    :param dict_scores: dict,dizionario dei valori degli scores
    :return: total score
    """

    result = 0
    for i in scores_to_list(dict_name_scoring, dict_scores):
        result += i
    return result


def K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=True):
    """
    Esegue Kfold cross validation su X,Y usando il modello model dividendo il dataset in n_split fold
    :param model:  modello machine learning
    :param X:features set
    :param Y:label set
    :param scoring:dict,dizionario di scoring
    :param n_split: int, numero di test fold
    :param seed: int, seme generatore pseudocasuale
    :return: dictionary with key 'name metric' and value 'value mean of metric' or all values
    """
    warnings.filterwarnings('always')
    kfold = model_selection.KFold(n_splits=n_split, shuffle=True, random_state=seed)
    scores = model_selection.cross_validate(model, X, Y, cv=kfold, scoring=scoring, return_train_score=True, n_jobs=1)
    result = {}
    for name, value in scoring.items():
        if mean:
            if type(value) is str:
                result[name] = scores['test_' + value].mean()
            else:
                result[name] = scores['test_' + str(value)[12:-1]].mean()
        else:
            if type(value) is str:
                result[name] = scores['test_' + value]
            else:
                result[name] = scores['test_' + str(value)[12:-1]]
    return result


