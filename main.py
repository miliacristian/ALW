import sys
import dataset
import training
import scoringUtils
from __init__ import setting
from __init__ import radar_plot, printValue
import __init__


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


def case_full_dataset(dataset_name, standardize=False, normalize=False, classification=True):
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

    if classification:
        X, Y = dataset.load_classification_dataset(dataset_name)
    else:
        X, Y = dataset.load_regression_dataset(dataset_name)
    if printValue:
        dataset.print_dataset(X, Y)

    if standardize:
        name_setting_file += __init__.standardize
        name_radar_plot_file += __init__.standardize
        X = dataset.standardize_dataset(X)
        if printValue:
            dataset.print_dataset(X, Y)

    if normalize:
        name_setting_file += __init__.normalize
        name_radar_plot_file += __init__.normalize
        X = dataset.normalize_dataset(X)
        if printValue:
            dataset.print_dataset(X, Y)

    name_setting_file += '.txt'

    if classification:
        scoring = scoringUtils.create_dictionary_classification_scoring()
    else:
        scoring = scoringUtils.create_dictionary_regression_scoring()

    return X, Y, scoring, name_setting_file, name_radar_plot_file


def case_NaN_dataset(dataset_name, strategy, seed=100, perc_NaN=0.1, classification=True):
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
    training.check_strategies(dataset_name, strategy)
    name_setting_file = dataset_name + setting
    name_radar_plot_file = dataset_name + radar_plot

    if classification:
        X, Y = dataset.load_classification_dataset(dataset_name)
    else:
        X, Y = dataset.load_regression_dataset(dataset_name)
    name_setting_file += '_' + str(perc_NaN * 100) + '%_NaN'
    name_radar_plot_file += '_' + str(perc_NaN * 100) + '%_NaN'
    X = dataset.put_random_NaN(X, perc_NaN, seed=seed)
    if printValue:
        dataset.print_dataset(X, Y)

    name_setting_file += '_' + strategy
    name_radar_plot_file += '_' + strategy

    if strategy == 'eliminate_row':
        X, Y = dataset.remove_rows_with_NaN(X, Y)
    elif strategy == 'mean':
        X = dataset.replace_NaN_with_strategy(X, "mean")
    elif strategy == 'mode':
        X = dataset.replace_NaN_with_strategy(X, "most_frequent")
    elif strategy == 'median':
        X = dataset.replace_NaN_with_strategy(X, "median")
    else:
        print("Strategy parameter for dataset balance must be 'eliminate_row' or 'mean' or 'mode' or 'median'.",
              sys.stderr)
        exit(1)

    name_setting_file += '.txt'

    if classification:
        scoring = scoringUtils.create_dictionary_classification_scoring()
    else:
        scoring = scoringUtils.create_dictionary_regression_scoring()

    return X, Y, scoring, name_setting_file, name_radar_plot_file


def create_radar_plot(dataset_names, name_models):
    for dataset_name in dataset_names:
        for stand in [True, False]:
            for norm in [True, False]:
                if norm is True and stand is True:
                    continue
                X, Y, scoring, name_setting_file, name_radar_plot_file = \
                    case_full_dataset(dataset_name, standardize=stand, normalize=norm)
                models = training.build_models(name_models, name_setting_file)
                testing(X, Y, models, scoring, seed, name_radar_plot_file)


def create_radar_plot_NaN(dataset_names, name_models, percentuals_NaN):
    for dataset_name in dataset_names:
        if dataset_name == __init__.balance:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        ['mean', 'eliminate_row', 'mode', 'median'])
        elif dataset_name == __init__.seed:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        ['mean', 'median'])
        elif dataset_name == __init__.tris:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        ['eliminate_row', 'mode'])
        elif dataset_name == __init__.zoo:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        ['mode'])
        elif dataset_name == __init__.page:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        ['mean', 'eliminate_row', 'median'])
        elif dataset_name == __init__.eye:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        ['mean', 'eliminate_row', 'median'])
        elif dataset_name == __init__.airfoil:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        [])
        elif dataset_name == __init__.auto:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        [])
        elif dataset_name == __init__.power_plant:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        [])
        elif dataset_name == __init__.energy:
            create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN,
                                        [])


def create_radar_plot_NaN_cycle(dataset_name, name_models, percentuals_NaN, strategies):
    for perc_NaN in percentuals_NaN:
        for strategy in strategies:
            X, Y, scoring, name_setting_file, name_radar_plot_file = \
                case_NaN_dataset(dataset_name, strategy=strategy, seed=seed, perc_NaN=perc_NaN)
            models = training.build_models(name_models, name_setting_file)
            testing(X, Y, models, scoring, seed, name_radar_plot_file)


if __name__ == '__main__':
    seed = 100
    name_models_classification = [__init__.rand_forest, __init__.dec_tree, __init__.knn, __init__.svc]
    dataset_names_classification = [__init__.seed, __init__.tris, __init__.zoo, __init__.balance, __init__.eye,
                                    __init__.page]
    name_models_regression = [__init__.rand_forest_regressor, __init__.dec_tree_regressor, __init__.knr, __init__.svr]
    dataset_names_regression = [__init__.airfoil, __init__.auto, __init__.power_plant, __init__.compress_strength,
                                __init__.energy]
    strategies = ['mean', 'eliminate_row', 'mode', 'median']
    percentuals_NaN = __init__.percentuals_NaN

    # Classification
    create_radar_plot(dataset_names_classification, name_models_classification)
    create_radar_plot_NaN(dataset_names_classification, name_models_classification, percentuals_NaN)

    # Regression
    create_radar_plot(dataset_names_regression, name_models_regression)
    create_radar_plot_NaN(dataset_names_regression, name_models_regression, percentuals_NaN)
