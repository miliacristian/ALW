# init file for constant values
printValue = True

# filenames
file_name_radar_plot = "radar_plot_classification"
setting = '_settings'
radar_plot = '_radar_plot'
table_plot = '_table_plot'
radar_plot_classification_dir = '/radar_plot_classification/'
table_plot_regression_dir = '/table_plot_regression/'
classification_datasets_dir = '/datasets/classification_datasets/'
regression_dataset_dir = '/datasets/regression_datasets/'
regression_model_settings_dir = '/best_model_settings/regression_model_settings/'
classification_model_settings_dir = '/best_model_settings/classification_model_settings/'
model_setting_test_dir = '/best_model_settings/model_setting_test/'

# models' names
rand_forest = 'RANDFOREST'
dec_tree = 'CART'
knn = 'KNN'
svc = 'SVC'
list_classification_model = [rand_forest, dec_tree, knn, svc]

rand_forest_regressor = 'RANDFORESTRegressor'
dec_tree_regressor = 'CARTRegressor'
knr = 'KNR'
svr = 'SVR'
list_regression_model = [rand_forest_regressor, dec_tree_regressor, knr, svr]

# datasets'names
seed = 'seed'
tris = 'tris'
zoo = 'zoo'
balance = 'balance'
eye = 'eye'
page = 'page'
list_classification_dataset = [seed, tris, zoo, balance, eye, page]

compress_strength = 'compressive_strength'
airfoil = 'airfoil'
auto = 'auto'
power_plant = 'power_plant'
energy = 'energy'
list_regression_dataset = [compress_strength, airfoil, auto, power_plant, energy]

standardize = '_standardize'
normalize = '_normalize'
percentuals_NaN = [0.05, 0.1, 0.15]
all_strategies = ['mean', 'eliminate_row', 'mode', 'median']
list_multilabel_dataset = [energy]

# weight regression metrics
weight_explained_variance = 20
weight_r2 = 20
weight_neg_mean_absolute_error = 3
weight_neg_mean_squared_error = 2
