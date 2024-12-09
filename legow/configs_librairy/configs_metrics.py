METRICS_CLASSIFICATION_BINARY = [
    {"callable": "sklearn.metrics.accuracy_score"},
    {"callable": "sklearn.metrics.f1_score"},
    {"callable": "sklearn.metrics.precision_score"},
    {"callable": "sklearn.metrics.recall_score"},
]

METRICS_CLASSIFICATION_CONTINUOUS = [
    {"callable": "sklearn.metrics.roc_auc_score"},
]

METRICS_REGRESSION = [
    {"callable": "sklearn.metrics.mean_squared_error"},
    {"callable": "sklearn.metrics.mean_absolute_error"},
    {"callable": "sklearn.metrics.r2_score"},
]

METRICS_SURVIVAL = [
    {"callable": "sklearn.metrics.concordance_index"},
    {"callable": "sklearn.metrics.brier_score_loss"},
]
