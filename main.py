import dataset
from training import training
import scoringUtils,p

if __name__=='__main__':
    name_models = ['RANDFOREST', 'CART', 'KNN']
    dataset_name='seed'
    seed = 100
    results = []
    names = []
    list_scores = []
    list_names = []

    X, Y = dataset.load_dataset(dataset_name)
    X_norm=dataset.normalize_dataset(X)
    X_std=dataset.standardize_dataset(X)
    scoring = scoringUtils.create_dictionary_classification_scoring()
    models = training(X, Y, name_models, scoring, k=range(8, 12, 1), list_n_trees=range(8, 12, 1), seed=seed,
                      n_split=10, mean=True)
    for name, model in models.items():
        scores = scoringUtils.K_Fold_Cross_validation(model, X, Y, scoring, n_split=10, seed=seed)
        results.append(scores)
        print(name)
        print(scores)
        list_scores.append(scores)
        list_names.append(name)
        print("total score =", scoringUtils.hmean_scores(scoring, scores))
    scoringUtils.radar_plot(list_names, scoring, list_scores,dataset_name)
