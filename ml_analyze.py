from sklearn.metrics import recall_score, precision_score


def global_metric_precision_score(y_true, y_pred):
    return precision_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0)


def global_metric_recall_score(y_true, y_pred):
    return recall_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0)


gmps = global_metric_precision_score
gmrs = global_metric_recall_score
