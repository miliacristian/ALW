from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import matplotlib.text
import pandas as pd
from math import pi, sqrt
from scipy.stats import hmean
from sklearn import model_selection
from __init__ import radar_plot_classification_dir, radar_plot_regression_dir
import warnings
import os
from numpy import mean
import training
from __init__ import weight_explained_variance, weight_neg_mean_absolute_error, weight_neg_mean_squared_error, weight_r2


def roc_auc_micro(y_true, y_pred):
    """
    Call roc_auc_score function with average=micro
    :param y_true:
    :param y_pred:
    :return:
    """
    return metrics.roc_auc_score(y_true, y_pred, average="micro")


def roc_auc_weighted(y_true, y_pred):
    """
    Call roc_auc_score function with average=weighted
    :param y_true:
    :param y_pred:
    :return:
    """
    return metrics.roc_auc_score(y_true, y_pred, average="weighted")


def average_precision_micro(y_true, y_pred):
    """
    Call average_precision_score function with average=micro
    :param y_true:
    :param y_pred:
    :return:
    """
    return metrics.average_precision_score(y_true, y_pred, average="micro")


def average_precision_weighted(y_true, y_pred):
    """
    Call average_precision_score function with average=weigthed
    :param y_true:
    :param y_pred:
    :return:
    """
    return metrics.average_precision_score(y_true, y_pred, average="weighted")


def neg_mean_absolute_error_uniform_average(y_true, y_pred):
    """
    Call neg_mean_absolute_error function with multioutput='uniform_average'
    :param y_true:
    :param y_pred:
    :return:
    """

    return -metrics.mean_absolute_error(y_true, y_pred, multioutput='uniform_average')


def neg_median_absolute_error_uniform_average(y_true, y_pred):
    """
    Call neg_median_absolute_error function with multioutput='uniform_average'
    :param y_true:
    :param y_pred:
    :return:
    """

    # case monolabel
    if len(y_true[0]) == 1:
        return -metrics.median_absolute_error(y_true, y_pred)

    # case multilabel
    median_abs_error = []
    for i in range(len(y_true[0])):
        median_abs_error.append(metrics.median_absolute_error(y_true[i], y_pred[i]))
    return -mean(median_abs_error)


def neg_mean_squared_error_uniform_average(y_true, y_pred):
    """
    Call neg_mean_squared_error function with multioutput='uniform_average'
    :param y_true:
    :param y_pred:
    :return:
    """

    return -metrics.mean_squared_error(y_true, y_pred, multioutput='uniform_average')


def create_dictionary_classification_scoring():
    """
    Create dictionary with key:'name_score' and  value:score (to calcutate)
    :return: dict,dict with all classification's scoring to calculate
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
    Create dictionary with key:'name_score' and  value:score (to calcutate)
    :return: dict,dict with all regression's scoring to calculate
    """
    scoring = {
        'explained_variance': 'explained_variance',
        'neg_mean_absolute_error_uniform_average': make_scorer(neg_mean_absolute_error_uniform_average),
        'neg_mean_squared_error_uniform_average': make_scorer(neg_mean_squared_error_uniform_average),
        # 'neg_median_absolute_error_uniform_average': make_scorer(neg_median_absolute_error_uniform_average),
        'r2': 'r2',
    }
    return scoring


def radar_plot(name_models, dict_name_scoring, list_dict_scores, file_name="radar_plot_classification",
               file_format=".png", classification=True, title_radar_plot=""):
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
    t = matplotlib.text.Text
    plt.suptitle(title_radar_plot, x=0.01, horizontalalignment='left')
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
    # axis_grid = [0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.8, 0.83, 0.86, 0.89, 0.92, 0.95, 0.98]
    axis_grid = [round(0.55 + i * 0.05, 2) for i in range(10)]
    plt.yticks(axis_grid, [str(i) for i in axis_grid], color="grey", size=7)
    plt.ylim(0.5, 1)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable
    for i in range(len(name_models)):
        # loc[i] get column i that contains dataframe's values
        values = df.loc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=name_models[i])
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(bbox_to_anchor=(0.1, 0.2))

    # plt.show()
    path = os.path.abspath('')
    if classification:
        fig.savefig(path + radar_plot_classification_dir + file_name + file_format)
    else:
        fig.savefig(path + radar_plot_regression_dir + file_name + file_format)
    plt.close(fig)


def scores_to_list(dict_name_scoring, dict_scores):
    """
    Transform dict of scores into list of values
    :param dict_name_scoring: dict,dict with scorers' name
    :param dict_scores:  dict,dict with scorers'values
    :return: list,list of scores'values
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
    Calculate harmonic mean of scores
    :param dict_name_scoring: dict,dict with scorers' name
    :param dict_scores: dict,dict with scorers'values
    :return: harmonic mean of scores
    """

    if list(dict_scores.values()).__contains__(0.0):
        return 0
    return hmean(scores_to_list(dict_name_scoring, dict_scores))


def normalize_score(dict_scores):
    """
    Calculate normalize score from dict_scores
    :param dict_scores:dict,dict of scores
    :return:normalize score
    """
    min_value, max_value = -100, 1
    norm_score_dict = {}
    for key, value in dict_scores.items():
        normalize_value = (value - min_value) / (max_value - min_value)
        norm_score_dict[key] = normalize_value
    return norm_score_dict


def total_score_regression(dict_name_scoring, dict_scores):
    """
    Calculate total score for regression
    :param dict_name_scoring: dict,dict with scorers' name
    :param dict_scores: dict,dict with scorers'values
    :return: total score
    """

    print(dict_scores)
    # values = scores_to_list(dict_name_scoring, dict_scores)
    # min_value, max_value = -100, 1
    # result = 0
    # for e in values:
    #     normalize_value = (e - min_value) / (max_value - min_value)
    #     result += normalize_value
    if dict_scores['neg_mean_squared_error_uniform_average'] < 0:
        RMSE = -sqrt(-dict_scores['neg_mean_squared_error_uniform_average'])
    else:
        RMSE = sqrt(dict_scores['neg_mean_squared_error_uniform_average'])
    result = dict_scores['explained_variance'] * weight_explained_variance + \
             dict_scores['r2'] * weight_r2 + \
             RMSE * weight_neg_mean_squared_error + \
             dict_scores['neg_mean_absolute_error_uniform_average'] * weight_neg_mean_absolute_error
    return result


def getBestModel(name_models, file_name, classification):
    """
    Read from file the best model for this setting
    :param name_models:name models
    :param file_name:filename
    :return:best_model readed from file
    """
    settings_dict = training.read_setting(file_name, classification)
    best_model = ""
    if not settings_dict:
        return best_model
    best_score = -1000
    for model in name_models:
        if float(settings_dict["total_score_" + model]) > best_score:
            best_score = float(settings_dict["total_score_" + model])
            best_model = model
    return best_model


def K_Fold_Cross_validation(model, X, Y, scoring, n_split, seed, mean=True):
    """
    Execute Kfold cross validation on X;Y using model model dividing dataset into n_split fold
    :param model:  machine learning model
    :param X:features set
    :param Y:label set
    :param scoring:dict,dict of scoring
    :param n_split: int,number of test fold
    :param seed: int,seed for pseudonumber generator
    :return: dictionary with key 'name metric' and value 'value mean of metric' or all values
    """

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
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

