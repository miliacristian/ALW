import sys
import warnings

import dataset
import training
import scoringUtils
from __init__ import setting
from __init__ import radar_plot, printValue


def testing(X, Y, models, scoring, seed, name_radar_plot_file):
    """
    Create a radar plot
    :param X: features dataset
    :param Y: label dataset
    :param models: list of models use already training
    :param scoring: list of metrics use
    :param seed:
    :param name_radar_plot_file: name of output file with radar plot
    """

    results = []
    list_scores = []
    list_names = []

    for name, model in models.items():
        scores = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split=10, seed=seed)
        results.append(scores)
        list_scores.append(scores)
        list_names.append(name)
        if printValue:
            print(name)
            print(scores)
            print("total score =", scoringUtils.hmean_scores(scoring, scores))
    scoringUtils.radar_plot(list_names, scoring, list_scores, file_name=name_radar_plot_file)


def case_full_dataset_classification(dataset_name, standardize=False, normalize=False):
    """
    Do testing in the case of classification with full dataset.
    :param dataset_name: name of dataset use
    :param standardize: is True if the dataset must be standardize
    :param normalize: is True if the dataset must be normalize
    :return: The dataset X, Y, the list of scoring for the test and the name of settings and radar plot files
    """

    if standardize is True and normalize is True:
        print("You cannot do normalization and standardization of dataset in the same test.", file=sys.stderr)
        exit(1)

    name_setting_file = dataset_name + setting
    name_radar_plot_file = dataset_name + radar_plot

    X, Y = dataset.load_classification_dataset(dataset_name)
    if printValue:
        dataset.print_dataset(X, Y)

    if standardize:
        name_setting_file += '_standardize'
        name_radar_plot_file += '_standardize'
        X = dataset.standardize_dataset(X)
        if printValue:
            dataset.print_dataset(X, Y)

    if normalize:
        name_setting_file += '_normalize'
        name_radar_plot_file += '_normalize'
        X = dataset.normalize_dataset(X)
        if printValue:
            dataset.print_dataset(X, Y)

    name_setting_file += '.txt'

    scoring = scoringUtils.create_dictionary_classification_scoring()

    return X, Y, scoring, name_setting_file, name_radar_plot_file


def case_NaN_dataset_classification(dataset_name, strategy, seed=100, perc_NaN=0.1):
    """
    Do testing in the case of classification with dataset having value NaN.
    :param dataset_name: name of dataset use
    :param strategy: must be 'eliminate_row' or 'mean' or 'mode' or 'median'. In particular it depends on the dataset:
                -   balance --> all
                -   seed --> mean or median or eliminate_row
                -   tris --> mode or eliminate_row
                -   zoo --> mode or eliminate_row
    :param seed:
    :param perc_NaN: percent of total entry of dataset that are setting randomly NaN
    :return: The dataset X, Y, the list of scoring for the test and the name of settings and radar plot files
    """

    name_setting_file = dataset_name + setting
    name_radar_plot_file = dataset_name + radar_plot

    X, Y = dataset.load_classification_dataset(dataset_name)
    name_setting_file += '_' + str(perc_NaN*100) + '%_NaN'
    name_radar_plot_file += '_' + str(perc_NaN*100) + '%_NaN'
    X = dataset.put_random_NaN(X, perc_NaN, seed=seed)
    if printValue:
        dataset.print_dataset(X, Y)

    if dataset_name == 'balance':
        if strategy == 'eliminate_row':
            name_setting_file += '_eliminate_row'
            name_radar_plot_file += '_eliminate_row'
            X, Y = dataset.remove_rows_with_NaN(X, Y)
        elif strategy == 'mean':
            name_setting_file += '_mean'
            name_radar_plot_file += '_mean'
            X = dataset.replace_NaN_with_strategy(X, "mean")
        elif strategy == 'mode':
            name_setting_file += '_mode'
            name_radar_plot_file += '_mode'
            X = dataset.replace_NaN_with_strategy(X, "most_frequent")
        elif strategy == 'median':
            name_setting_file += '_median'
            name_radar_plot_file += '_median'
            X = dataset.replace_NaN_with_strategy(X, "median")
        else:
            print("Strategy parameter for dataset balance must be 'eliminate_row' or 'mean' or 'mode' or 'median'.",
                  sys.stderr)
            exit(1)
    elif dataset_name == 'seed':
        if strategy == 'eliminate_row':
            name_setting_file += '_eliminate_row'
            name_radar_plot_file += '_eliminate_row'
            X, Y = dataset.remove_rows_with_NaN(X, Y)
        elif strategy == 'mean':
            name_setting_file += '_mean'
            name_radar_plot_file += '_mean'
            X = dataset.replace_NaN_with_strategy(X, "mean")
        elif strategy == 'median':
            name_setting_file += '_median'
            name_radar_plot_file += '_median'
            X = dataset.replace_NaN_with_strategy(X, "median")
        else:
            print("Strategy parameter for dataset seed must be 'eliminate_row' or 'mean' or 'median'.", sys.stderr)
            exit(1)
    elif dataset_name == 'tris':
        if strategy == 'eliminate_row':
            name_setting_file += '_eliminate_row'
            name_radar_plot_file += '_eliminate_row'
            X, Y = dataset.remove_rows_with_NaN(X, Y)
        elif strategy == 'mode':
            name_setting_file += '_mode'
            name_radar_plot_file += '_mode'
            X = dataset.replace_NaN_with_strategy(X, "most_frequent")
        else:
            print("Strategy parameter for dataset tris must be 'eliminate_row' or 'mode'.", sys.stderr)
            exit(1)
    elif dataset_name == 'zoo':
        if strategy == 'eliminate_row':
            name_setting_file += '_eliminate_row'
            name_radar_plot_file += '_eliminate_row'
            X, Y = dataset.remove_rows_with_NaN(X, Y)
        elif strategy == 'mode':
            name_setting_file += '_mode'
            name_radar_plot_file += '_mode'
            X = dataset.replace_NaN_with_strategy(X, "most_frequent")
        else:
            print("Strategy parameter for dataset zoo must be 'eliminate_row' or 'mode'.", sys.stderr)
            exit(1)
    else:
        print("dataset name must be seed, balance, tris, or zoo.", sys.stderr)
        exit(1)

    name_setting_file += '.txt'

    scoring = scoringUtils.create_dictionary_classification_scoring()

    return X, Y, scoring, name_setting_file, name_radar_plot_file


if __name__ == '__main__':
    warnings.filterwarnings('always')
    name_models = ['RANDFOREST', 'CART', 'KNN', 'SVC']
    dataset_names = ['seed', 'tris', 'zoo', 'balance','eye','page']
    seed = 100

    strategies = ['mean', 'eliminate_row', 'mode', 'median']
    percentuals_NaN = [0.05, 0.1, 0.15]

    for dataset_name in dataset_names:
        for stand in [True, False]:
            for norm in [True, False]:
                if norm is True and stand is True:
                    continue
                X, Y, scoring, name_setting_file, name_radar_plot_file = \
                    case_full_dataset_classification(dataset_name, standardize=stand, normalize=norm)
                models = training.build_models(name_models, name_setting_file)
                testing(X, Y, models, scoring, seed, name_radar_plot_file)

    for dataset_name in dataset_names:
        if dataset_name == "balance":
            for perc_NaN in percentuals_NaN:
                for strategy in strategies:
                    X, Y, scoring, name_setting_file, name_radar_plot_file = \
                        case_NaN_dataset_classification(dataset_name, strategy=strategy, seed=seed, perc_NaN=perc_NaN)
                    models = training.build_models(name_models, name_setting_file)
                    testing(X, Y, models, scoring, seed, name_radar_plot_file)
        if dataset_name == "seed":
            for perc_NaN in percentuals_NaN:
                for strategy in ["mean", "median"]:
                    X, Y, scoring, name_setting_file, name_radar_plot_file = \
                        case_NaN_dataset_classification(dataset_name, strategy=strategy, seed=seed, perc_NaN=perc_NaN)
                    models = training.build_models(name_models, name_setting_file)
                    testing(X, Y, models, scoring, seed, name_radar_plot_file)
        if dataset_name == "tris":
            for perc_NaN in percentuals_NaN:
                for strategy in ["mode", "eliminate_row"]:
                    X, Y, scoring, name_setting_file, name_radar_plot_file = \
                        case_NaN_dataset_classification(dataset_name, strategy=strategy, seed=seed, perc_NaN=perc_NaN)
                    models = training.build_models(name_models, name_setting_file)
                    testing(X, Y, models, scoring, seed, name_radar_plot_file)
        if dataset_name == "zoo":
            for perc_NaN in percentuals_NaN:
                for strategy in ["mode"]:
                    X, Y, scoring, name_setting_file, name_radar_plot_file = \
                        case_NaN_dataset_classification(dataset_name, strategy=strategy, seed=seed, perc_NaN=perc_NaN)
                    models = training.build_models(name_models, name_setting_file)
                    testing(X, Y, models, scoring, seed, name_radar_plot_file)
        if dataset_name == 'eye':
            for perc_NaN in percentuals_NaN:
                for strategy in ["eliminate_row", "mean", "median"]:
                    X, Y, scoring, name_setting_file, name_radar_plot_file = \
                        case_NaN_dataset_classification(dataset_name, strategy=strategy, seed=seed, perc_NaN=perc_NaN)
                    models = training.build_models(name_models, name_setting_file)
                    testing(X, Y, models, scoring, seed, name_radar_plot_file)
        if dataset_name == 'page':
            for perc_NaN in percentuals_NaN:
                for strategy in ["eliminate_row", "mean", "median"]:
                    X, Y, scoring, name_setting_file, name_radar_plot_file = \
                        case_NaN_dataset_classification(dataset_name, strategy=strategy, seed=seed, perc_NaN=perc_NaN)
                    models = training.build_models(name_models, name_setting_file)
                    testing(X, Y, models, scoring, seed, name_radar_plot_file)
