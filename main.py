import dataset
import training
import scoringUtils
from __init__ import setting
from __init__ import radar_plot
import numpy


import p

if __name__=='__main__':
    name_models = ['RANDFOREST', 'CART', 'KNN', 'SVC']
    dataset_name = 'indians'
    name_setting_file = dataset_name +setting
    name_radar_plot_file = dataset_name + radar_plot
    seed = 100
    results = []
    names = []
    list_scores = []
    list_names = []

    X, Y = dataset.load_dataset(dataset_name)
    dataset.print_dataset(X,Y)
    X=dataset.put_random_NaN(X,0.5,seed=100)
    print(X)
    X=dataset.remove_row_with_Nan(X)
    dataset.print_dataset(X,Y)
    exit(0)
    X=dataset.replace_NaN_with_strategy(X,"mean")
    X_norm = dataset.normalize_dataset(X)
    X_std = dataset.standardize_dataset(X)
    scoring = scoringUtils.create_dictionary_classification_scoring()
    models = training.build_models(name_models, name_setting_file)
    for name, model in models.items():
        scores = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split=10, seed=seed)
        results.append(scores)
        print(name)
        print(scores)
        list_scores.append(scores)
        list_names.append(name)
        print("total score =", scoringUtils.hmean_scores(scoring, scores))
    scoringUtils.radar_plot(list_names, scoring, list_scores, file_name=name_radar_plot_file)
