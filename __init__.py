#init file for constant values
printValue = True

#filenames
file_name_radar_plot = "radar_plot_classification"
setting = '_settings'
radar_plot = '_radar_plot'
radar_plot_classification_dir = '/radar_plot_classification/'
radar_plot_regression_dir = '/radar_plot_regression/'
classification_datasets_dir = '/classification_datasets/'
regression_dataset_dir = '/regression_datasets/'
model_settings_dir = '/model_settings/'
model_setting_test_dir = '/model_setting_test/'

#models' names
rand_forest= 'RANDFOREST'
dec_tree='CART'
knn='KNN'
svc='SVC'
rand_forest_regressor = 'RANDFORESTRegressor'
dec_tree_regressor = 'CARTRegressor'
knr = 'KNR'
svr = 'SVR'

#datasets'names
seed='seed'
tris='tris'
zoo='zoo'
balance='balance'
eye='eye'
page='page'
list_classification_dataset=[seed,tris,zoo,balance,eye,page]

compress_strength= 'compressive_strength'
airfoil='airfoil'
auto='auto'
power_plant='power_plant'
energy='energy'
list_regression_dataset=[compress_strength,airfoil,auto,power_plant,energy]

#strategies
standardize='_standardize'
normalize='_normalize'
percentuals_NaN = [0.05, 0.1, 0.15]
list_multilabel_dataset = [energy]
