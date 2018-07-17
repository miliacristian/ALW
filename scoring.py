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
        'precision_weighted': 'precision_weighted',
        'recall_micro': 'recall_micro',
        'recall_weighted': 'recall_weighted',
        'roc_auc_micro': make_scorer(roc_auc_micro),
        'roc_auc_weighted': make_scorer(roc_auc_weighted),
        'f1_micro': 'f1_micro',
        'f1_weighted': 'f1_weighted',
        'average_precision_micro': make_scorer(average_precision_micro),
        'average_precision_weighted': make_scorer(average_precision_weighted),
               }
    return scoring