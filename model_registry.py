from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVR


def get_models(task_type='classification'):
    if task_type == 'regression':
        return get_regression_models()
    return get_classification_models()


def get_classification_models():
    scaled_models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'LinearSVC': LinearSVC(),
        'SVC_RBF': SVC(kernel='rbf', probability=True),
        'SGDClassifier': SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3),
        'Perceptron': Perceptron(),
        'PassiveAggressive': PassiveAggressiveClassifier(max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }

    tree_models = {
        'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging': BaggingClassifier(random_state=42),
    }

    nb_models = {
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
    }

    models = {}
    for name, model in scaled_models.items():
        models[name] = make_pipeline(StandardScaler(), model)

    models.update(tree_models)
    models.update(nb_models)

    return models


def get_regression_models():
    scaled_models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
        'SVR_RBF': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    }

    tree_models = {
        'RandomForestRegressor': RandomForestRegressor(n_estimators=300, random_state=42),
        'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=300, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor(random_state=42),
    }

    models = {}
    for name, model in scaled_models.items():
        models[name] = make_pipeline(StandardScaler(), model)

    models.update(tree_models)
    return models
