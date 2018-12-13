import os
import sys
import dataset
import training
import scoringUtils
from __init__ import setting
from __init__ import radar_plot, printValue, table_plot, seed
import __init__
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
import pandas as pd

def create_plot_istance(name_models, scoring, setting_file_name, name_plot_file, classification=True,
                        title_plot=""):
    """
    Create one plot (table or radar plot) for a set of models and best settings
    :param name_models: list with the name of models
    :param scoring: list of scores used
    :param setting_file_name: name of file contains the best settings
    :param name_plot_file: name of the output file
    :param classification: True if classification case, False if regression case
    :param title_plot: Title of table or radar plot
    :return: None
    """
    
    list_scores = []
    list_names = []
    setting = training.read_setting(setting_file_name, classification)
    for m in name_models:
        scores = {}
        for s in scoring:
            scores[s] = float(setting[s + "_" + m])
        list_names.append(m)
        list_scores.append(scores)

    if classification:
        scoringUtils.radar_plot(list_names, scoring, list_scores, file_name=name_plot_file,
                                classification=classification, title_radar_plot=title_plot)
    else:
        scoringUtils.table_plot(scoring, list_scores, list_names, title_table_plot=title_plot, file_name=name_plot_file)


def create_plot(dataset_names, name_models, classification=True):
    """
    Create all plot (table or radar) for a specific set of dataset
    :param dataset_names: list of dataset's name
    :param name_models: list of model's name
    :param classification: True if classification case, False if regression case
    :return: None
    """
    
    for dataset_name in dataset_names:
        for stand in [True, False]:
            for norm in [True, False]:
                if norm is True and stand is True:
                    continue
                if printValue:
                    print("Cycle with dataset =", dataset_name + ", standardize =", stand, "and normalize =", norm)
                X, Y, scoring, name_setting_file, name_plot_file, title_radar_plot = \
                    case_full_dataset(name_models, dataset_name, standardize=stand, normalize=norm,
                                      classification=classification,mode=__init__.result_analysis)
                create_plot_istance(name_models, scoring, name_setting_file, name_plot_file,
                                    classification=classification, title_plot=title_radar_plot)


def create_plot_NaN(dataset_names, name_models, percentuals_NaN):
    """
    Create a plot (table or radar) in the case of dataset with NaN value. In particular decides what strategies use to
    fill the dataset.
    :param dataset_names: list of dataset names 
    :param name_models: list of model used
    :param percentuals_NaN: percentage of NaN value in dataset
    :return: None
    """
    
    for dataset_name in dataset_names:
        if dataset_name == __init__.balance:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mean', 'eliminate_row', 'mode', 'median'])
        elif dataset_name == __init__.seed:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mean', 'median'])
        elif dataset_name == __init__.tris:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['eliminate_row', 'mode'])
        elif dataset_name == __init__.zoo:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mode'])
        elif dataset_name == __init__.page:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mean', 'eliminate_row', 'median'])
        elif dataset_name == __init__.eye:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mean', 'eliminate_row', 'median'])
        elif dataset_name == __init__.airfoil:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mean', 'eliminate_row', 'median'], classification=False)
        elif dataset_name == __init__.auto:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mean', 'eliminate_row', 'median'], classification=False)
        elif dataset_name == __init__.power_plant:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mean', 'eliminate_row', 'median'], classification=False)
        elif dataset_name == __init__.energy:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mean', 'eliminate_row', 'median'], classification=False)
        elif dataset_name == __init__.compress_strength:
            create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                  ['mean', 'eliminate_row', 'median'], classification=False)


def create_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN, strategies, classification=True):
    """
    Create a particolar istance of plot with strategies decides in function "create_plot_NaN".
    :param dataset_name: list of dataset name
    :param name_models: list of model's name
    :param percentuals_NaN: percentage of NaN value in dataset
    :param strategies: list of strategies use to fill dataset
    :param classification: True if classification case, False if regression case
    :return: None
    """
    
    for perc_NaN in percentuals_NaN:
        for strategy in strategies:
            if printValue:
                print("Cycle with dataset =", dataset_name + ", percentuals_NaN =", perc_NaN, "and strategy =",
                      strategy)
            X, Y, scoring, name_setting_file, name_radar_plot_file, title_radar_plot = \
                case_NaN_dataset(name_models, dataset_name, strategy=strategy, seed=seed, perc_NaN=perc_NaN,
                                 classification=classification,mode=__init__.result_analysis)
            create_plot_istance(name_models, scoring, name_setting_file, name_radar_plot_file,
                                classification=classification, title_plot=title_radar_plot)


def case_full_dataset(name_models, dataset_name, standardize=False, normalize=False, classification=True,
                      multilabel=False,mode='training'):
    """
    Do testing in the case of classification/regression with full dataset.
    :param dataset_name: name of dataset use
    :param standardize: is True if the dataset must be standardize
    :param normalize: is True if the dataset must be normalize
    :return: The dataset X, Y, the list of scoring for the test and the name of settings and radar plot files
    """

    if standardize is True and normalize is True:
        print("You cannot do normalization and standardization of dataset in the same test.", file=sys.stderr)
        exit(1)

    name_setting_file = dataset_name + setting
    if classification:
        name_plot_file = dataset_name + radar_plot
    else:
        name_plot_file = dataset_name + table_plot
    title_plot = "Dataset: " + dataset_name + "\n"

    if classification:
        X, Y = dataset.load_classification_dataset(dataset_name, multilabel=multilabel)
    else:
        X, Y = dataset.load_regression_dataset(dataset_name, multilabel=multilabel)

    if standardize:
        name_setting_file += __init__.standardize
        name_plot_file += __init__.standardize
        title_plot += "Strategy: standardize\n"
        X = dataset.standardize_dataset(X)

    if normalize:
        name_setting_file += __init__.normalize
        name_plot_file += __init__.normalize
        title_plot += "Strategy: normalize\n"
        X = dataset.normalize_dataset(X)

    name_setting_file += '.txt'

    if classification:
        scoring = scoringUtils.create_dictionary_classification_scoring()
    else:
        scoring = scoringUtils.create_dictionary_regression_scoring()
    if mode == __init__.result_analysis:
        if scoringUtils.getBestModel(name_models, name_setting_file, classification) == __init__.rand_forest:
            title_plot += "Best model: RF"
        elif scoringUtils.getBestModel(name_models, name_setting_file, classification) == __init__.rand_forest_regressor:
            title_plot += "Best model: RFRegressor"
        else:
            title_plot += "Best model: " + scoringUtils.getBestModel(name_models, name_setting_file, classification)
    return X, Y, scoring, name_setting_file, name_plot_file, title_plot


def case_NaN_dataset(name_models, dataset_name, strategy, seed=100, perc_NaN=0.1, classification=True,
                     multilabel=False,mode='training'):
    """
    Do testing in the case of classification with dataset having value NaN.
    :param dataset_name: name of dataset use
    :param strategy: must be 'eliminate_row' or 'mean' or 'mode' or 'median'. In particular it depends on the dataset:
    e.g. for balance dataset  --> all
         for seed dataset -->only  mean or median or eliminate_row are admitted
         for tris dataset -->only  mode or eliminate_row are admitted
         for zoo dataset --> only mode or eliminate_row are admitted
    :param seed:
    :param perc_NaN: percent of total entry of dataset that are setting randomly NaN
    :return: The dataset X, Y, the list of scoring for the test and the name of settings and radar plot files
    """
    list_mode_columns = dataset.get_list_mode_columns_by_dataset(dataset_name)
    training.check_strategies(dataset_name, strategy)
    training.check_percentage(perc_NaN)
    name_setting_file = dataset_name + setting
    if classification:
        name_plot_file = dataset_name + radar_plot
    else:
        name_plot_file = dataset_name + table_plot
    title_plot = "Dataset: " + dataset_name + "\n"
    title_plot += "Strategy: " + strategy + "\nPerc. of NaN: "

    if classification:
        X, Y = dataset.load_classification_dataset(dataset_name, multilabel=multilabel)
    else:
        X, Y = dataset.load_regression_dataset(dataset_name, multilabel=multilabel)
    name_setting_file += '_' + str(perc_NaN * 100) + '%_NaN'
    name_plot_file += '_' + str(perc_NaN * 100) + '%_NaN'
    title_plot += str(int(perc_NaN * 100)) + "%\n"
    X = dataset.put_random_NaN(X, perc_NaN, seed=seed)

    name_setting_file += '_' + strategy
    name_plot_file += '_' + strategy

    if strategy == 'eliminate_row':
        X, Y = dataset.remove_rows_with_NaN(X, Y)
    elif strategy == 'mean':
        X = dataset.replace_NaN_with_strategy(X, "mean", list_mode_columns)
    elif strategy == 'mode':
        X = dataset.replace_NaN_with_strategy(X, "most_frequent")
    elif strategy == 'median':
        X = dataset.replace_NaN_with_strategy(X, "median", list_mode_columns)
    else:
        print("Strategy parameter for dataset balance must be 'eliminate_row' or 'mean' or 'mode' or 'median'.",
              sys.stderr)
        exit(1)

    name_setting_file += '.txt'

    if classification:
        scoring = scoringUtils.create_dictionary_classification_scoring()
    else:
        scoring = scoringUtils.create_dictionary_regression_scoring()
    if mode==__init__.result_analysis:
        if scoringUtils.getBestModel(name_models, name_setting_file, classification) == __init__.rand_forest:
            title_plot += "Best model: RF"
        elif scoringUtils.getBestModel(name_models, name_setting_file, classification) == __init__.rand_forest_regressor:
            title_plot += "Best model: RFRegressor"
        else:
            title_plot += "Best model: " + scoringUtils.getBestModel(name_models, name_setting_file, classification)
    return X, Y, scoring, name_setting_file, name_plot_file, title_plot


def create_table_classification_analysis():
    """
    create a summary table of all best_model in all classification dataset with all strategies
    :return: create a csv file with the table name table_classification.csv
    """
    name_models_classification = [__init__.rand_forest, __init__.dec_tree, __init__.knn, __init__.svc]
    dataset_names_classification = [__init__.seed, __init__.tris, __init__.zoo, __init__.balance, __init__.eye,
                                    __init__.page]
    strategies = ['mean', 'eliminate_row', 'mode', 'median']
    l = os.listdir('./best_model_settings/classification_model_settings')
    table = {}
    for name in l:
        best_model = scoringUtils.getBestModel(name_models_classification, name, classification=True)
        for dataset in dataset_names_classification:
            if name.__contains__(dataset):
                dataset_name = dataset
        if name.__contains__('normalize'):
            strategy = "normalize"
        elif name.__contains__("standardize"):
            strategy = "standardize"
        elif name.__contains__('15.0%'):
            strategy = "15.0%_"
            for s in strategies:
                if name.__contains__(s):
                    strategy += s
        elif name.__contains__('5.0%'):
            strategy = "5.0%_"
            for s in strategies:
                if name.__contains__(s):
                    strategy += s
        elif name.__contains__('10.0%'):
            strategy = "10.0%_"
            for s in strategies:
                if name.__contains__(s):
                    strategy += s
        else:
            strategy = "full"
        if not dataset_name in table.keys():
            table[dataset_name] = {}
        table[dataset_name][strategy] = best_model

    f = open("temp_table_classification.csv", "w")
    all_strategies = ["full", "normalize", "standardize", "5.0%_eliminate_row", "5.0%_mean", "5.0%_median", "5.0%_mode",
                      "10.0%_eliminate_row", "10.0%_mean", "10.0%_median", "10.0%_mode",
                      "15.0%_eliminate_row", "15.0%_mean", "15.0%_median", "15.0%_mode"]
    f.write("classification")
    for s in all_strategies:
        f.write("," + s)
    for dataset in dataset_names_classification:
        f.write("\n" + dataset)
        for s in all_strategies:
            if s in table[dataset].keys():
                f.write("," + table[dataset][s])
            else:
                f.write(",")
    f.close()
    pd.read_csv('temp_table_classification.csv').T.to_csv('table_classification.csv', header=False)
    os.remove('temp_table_classification.csv')
    df = pd.read_csv('table_classification.csv')
    df = df.replace(np.nan, '', regex=True)
    trace = go.Table(
        header=dict(values=list(df.columns),
                    fill=dict(color='#C2D4FF'),
                    align=['center'] * 5),
        cells=dict(values=[df.classification, df.seed, df.tris, df.zoo, df.balance, df.eye, df.page],
                   fill=dict(color='#F5F8FF'),
                   align=['center'] * 5)
    )
    data = [trace]
    plot(data, filename='table_classification.html', image_filename='table_classification', image='jpeg',
         auto_open=False)

def create_table_regression_analysis():
    """
        create a summary table of all best_model in all regression dataset with all strategies
        :return: create a csv file with the table name table_regression.csv
        """
    name_models_regression = [__init__.rand_forest_regressor, __init__.dec_tree_regressor, __init__.knr, __init__.svr]
    strategies = ['mean', 'eliminate_row', 'median']
    dataset_names_regression = [__init__.airfoil, __init__.auto, __init__.power_plant, __init__.compress_strength,
                                __init__.energy]
    l = os.listdir('./best_model_settings/regression_model_settings')
    table = {}
    for name in l:
        best_model = scoringUtils.getBestModel(name_models_regression, name, classification=False)
        for dataset in dataset_names_regression:
            if name.__contains__(dataset):
                dataset_name = dataset
        if name.__contains__('normalize'):
            strategy = "normalize"
        elif name.__contains__("standardize"):
            strategy = "standardize"
        elif name.__contains__('15.0%'):
            strategy = "15.0%_"
            for s in strategies:
                if name.__contains__(s):
                    strategy += s
        elif name.__contains__('5.0%'):
            strategy = "5.0%_"
            for s in strategies:
                if name.__contains__(s):
                    strategy += s
        elif name.__contains__('10.0%'):
            strategy = "10.0%_"
            for s in strategies:
                if name.__contains__(s):
                    strategy += s
        else:
            strategy = "full"
        if not dataset_name in table.keys():
            table[dataset_name] = {}
        table[dataset_name][strategy] = best_model

    f = open("temp_table_regression.csv", "w")
    all_strategies = ["full", "normalize", "standardize", "5.0%_eliminate_row", "5.0%_mean", "5.0%_median",
                      "10.0%_eliminate_row", "10.0%_mean", "10.0%_median",
                      "15.0%_eliminate_row", "15.0%_mean", "15.0%_median"]
    f.write("regression")
    for s in all_strategies:
        f.write("," + s)
    for dataset in dataset_names_regression:
        f.write("\n" + dataset)
        for s in all_strategies:
            if s in table[dataset].keys():
                f.write("," + table[dataset][s])
            else:
                f.write(",")
    f.close()
    pd.read_csv('temp_table_regression.csv').T.to_csv('table_regression.csv', header=False)
    os.remove('temp_table_regression.csv')
    df = pd.read_csv('table_regression.csv')
    df = df.replace(np.nan, '', regex=True)
    trace = go.Table(
        header=dict(values=list(df.columns),
                    fill=dict(color='#C2D4FF'),
                    align=['center'] * 5),
        cells=dict(values=[df.regression, df.airfoil, df.auto, df.power_plant, df.compressive_strength, df.energy],
                   fill=dict(color='#F5F8FF'),
                   align=['center'] * 5)
    )
    data = [trace]
    plot(data, filename='table_regression.html', image_filename='table_regression', image='jpeg',
         auto_open=False)


if __name__ == '__main__':
    """
        It takes best models' settings from files in directories regression_model_settings and 
        classification_model_settings to create the radar_plots of all classification best models and the tables of 
        all regression best models.
    """

    # seed = 100
    # name_models_classification = [__init__.rand_forest, __init__.dec_tree, __init__.knn, __init__.svc]
    # dataset_names_classification = [__init__.seed, __init__.tris, __init__.zoo, __init__.balance, __init__.eye,
    #                                 __init__.page]
    # name_models_regression = [__init__.rand_forest_regressor, __init__.dec_tree_regressor, __init__.knr, __init__.svr]
    # dataset_names_regression = [__init__.airfoil, __init__.auto, __init__.power_plant, __init__.compress_strength,
    #                             __init__.energy]
    # strategies = ['mean', 'eliminate_row', 'mode', 'median']
    # percentuals_NaN = __init__.percentuals_NaN
    #
    # # # Classification
    # # create_plot(dataset_names_classification, name_models_classification, classification=True)
    # # create_plot_NaN(dataset_names_classification, name_models_classification, percentuals_NaN)
    #
    # # Regression
    # create_plot(dataset_names_regression, name_models_regression, classification=False)
    # create_plot_NaN(dataset_names_regression, name_models_regression, percentuals_NaN)

    create_table_classification_analysis()
    create_table_regression_analysis()
