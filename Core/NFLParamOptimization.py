import joblib
import maya
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

game_data_dir = '..\\Projects\\nfl\\NFL_Prediction\\Game Data\\'
other_dir = '..\\Projects\\nfl\\NFL_Prediction\\Other\\'


def create_model(best_features, games):
    evaluate_model_parameters(best_features, df=games)
    voting_classifier = get_voting_classifier(best_features, df=games)

    return voting_classifier


def evaluate_model_parameters(contributing_features, df=None):
    """
    Does a grid search on 6 different models to find the best parameters,
    evaluates each set of parameters on brier loss score and accuracy.

    :param contributing_features: The list of features to use in the models
    :param df: The data frame containing all games to train each model
    :return: Void
    """

    if df is None:
        # Get the data frame for all seasons
        df = pd.read_csv(game_data_dir + '20022018.csv')

    # Filter the columns to keep
    columns_to_keep = list()
    for feature in contributing_features:
        columns_to_keep.extend(list(filter(lambda f: f == feature, df.columns.values)))

    # Drop all other columns (except the home_victory label)
    columns_to_keep.extend(list(filter(lambda f: 'home_victory' in f, df.columns.values)))
    columns_to_drop = list(set(df.columns.values) - set(columns_to_keep))
    df = df.drop(columns=columns_to_drop)

    # Get the X and y frames
    feature_col_names = contributing_features
    predicted_class_name = ['home_victory']

    # Print the feature column names by order of importance
    print()
    print('Top ' + str(len(feature_col_names)) + ' features ranked by importance are:')
    for feature in feature_col_names:
        print(feature)
    print()

    # Get the feature and label data sets
    X = df[feature_col_names].values
    y = df[predicted_class_name].values

    # Standardize the X values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Using Stratified K-Fold cross validation with 5 folds
    skf = StratifiedKFold(n_splits=5)

    # Tune the parameters to find the best models based on brier loss and accuracy
    scores = ['brier_score_loss', 'f1']

    # Logistic Regression               -.21199     .73085
    tune_logistic_regression(X, y, skf, scores)

    # C Support Vector Classifier       -.21244     .73071
    tune_svc_classifier(X, y, skf, scores)

    # K Nearest Neighbors               -0.21621    .72284
    tune_k_nearest_neighbors(X, y, skf, scores)

    # Gaussian Naive Bayes              -0.22759    .73788
    tune_gauss_naive_bayes(X, y, skf, scores)

    # Random Forest                     -.21353     .73023
    tune_random_forest(X, y, feature_col_names, skf, scores)


def tune_logistic_regression(X, y, skf, scores):
    """
    Does a grid search over different parameters for a logistic regression model.
    Returns the model with the least brier loss and the most accurate model.

    :param X: The feature values
    :param y: The predicted class values
    :param skf: The stratified K fold
    :param scores: The metrics to evaluate hyper parameter tuning on
    :return: The best model for each metric
    """

    # Create a list of dicts to try parameter combinations over
    print('Logistic Regression')
    logistic_regression_parameters = [{'penalty': ['l2'],
                                       'tol': [1e-3, 1e-4, 1e-5],
                                       'C': np.logspace(-4, 4, 9, base=10),
                                       'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                                       'max_iter': range(900, 1100, 10),
                                       'class_weight': [None, 'balanced'],
                                       'random_state': [42]},
                                      {'penalty': ['l2'],
                                       'tol': [1e-3, 1e-4, 1e-5],
                                       'C': np.logspace(-4, 4, 9, base=10),
                                       'solver': ['saga'],
                                       'class_weight': [None, 'balanced'],
                                       'random_state': [42]},
                                      {'penalty': ['l1'],
                                       'tol': [1e-3, 1e-4, 1e-5],
                                       'C': np.logspace(-4, 4, 9, base=10),
                                       'solver': ['liblinear', 'saga'],
                                       'multi_class': ['auto'],
                                       'class_weight': [None, 'balanced'],
                                       'random_state': [42]}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(LogisticRegression(),
                           logistic_regression_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           return_train_score=True,
                           verbose=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'logistic_regression_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def tune_gauss_naive_bayes(X, y, skf, scores):
    """
    Does a grid search over different parameters for a gaussian naive bayes model.
    Returns the model with the least brier loss and the most accurate model.

    :param X: The feature values
    :param y: The predicted class values
    :param skf: The stratified K fold
    :param scores: The metrics to evaluate hyper parameter tuning on
    :return: The best model for each metric
    """

    # Create a list of dicts to try parameter combinations over
    print('Gaussian Naive Bayes')
    naive_bayes_parameters = [{'var_smoothing': [1 * 10 ** x for x in range(3, -20, -1)]}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(GaussianNB(),
                           naive_bayes_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           return_train_score=True,
                           verbose=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'gaussian_naive_bayes_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def tune_random_forest(X, y, feature_col_names, skf, scores):
    """
    Does a grid search over different parameters for a random forest model.
    Returns the model with the least brier loss and the most accurate model.

    :param X: The feature values
    :param y: The predicted class values
    :param feature_col_names: The list of feature column names
    :param skf: The stratified K fold
    :param scores: The metrics to evaluate hyper parameter tuning on
    :return: The best model for each metric
    """

    # Create a list of dicts to try parameter combinations over
    print('Random Forest')
    random_forest_parameters = [{'n_estimators': [500, 1000],
                                 'random_state': [42],
                                 'max_features': range(1, len(feature_col_names) + 1),
                                 'max_depth': range(1, 20)},
                                {'n_estimators': [500, 1000],
                                 'random_state': [42],
                                 'max_features': range(1, len(feature_col_names) + 1),
                                 'min_samples_split': range(2, 101)},
                                {'n_estimators': [500, 1000],
                                 'random_state': [42],
                                 'max_features': range(1, len(feature_col_names) + 1),
                                 'min_samples_leaf': range(1, 101)}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(RandomForestClassifier(),
                           random_forest_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           return_train_score=True,
                           verbose=2,
                           n_jobs=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'random_forest_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def tune_k_nearest_neighbors(X, y, skf, scores):
    """
    Does a grid search over different parameters for a K nearest neighbors model.
    Returns the model with the least brier loss and the most accurate model.

    :param X: The feature values
    :param y: The predicted class values
    :param skf: The stratified K fold
    :param scores: The metrics to evaluate hyper parameter tuning on
    :return: The best model for each metric
    """

    # Create a list of dicts to try parameter combinations over
    print('K Nearest Neighbors')
    k_neighbors_parameters = [{'n_neighbors': range(3, 71, 2),
                               'weights': ['uniform', 'distance'],
                               'algorithm': ['auto'],
                               'leaf_size': range(1, 31),
                               'p': [1, 2]}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(KNeighborsClassifier(),
                           k_neighbors_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           return_train_score=True,
                           verbose=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'knn_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def tune_svc_classifier(X, y, skf, scores):
    """
    Does a grid search over different parameters for a C support vector classification model.
    Returns the model with the least brier loss and the most accurate model.

    :param X: The feature values
    :param y: The predicted class values
    :param skf: The stratified K fold
    :param scores: The metrics to evaluate hyper parameter tuning on
    :return: The best model for each metric
    """

    # Create a list of dicts to try parameter combinations over
    print('C Support Vector Classifier')
    support_vector_parameters = [{'kernel': ['rbf', 'sigmoid'],
                                  'probability': [True],
                                  'C': [1 * 10 ** x for x in range(-1, 3)],
                                  'gamma': [1 * 10 ** x for x in range(-3, -8, -1)]},
                                 {'kernel': ['rbf', 'sigmoid'],
                                  'probability': [True],
                                  'C': [1 * 10 ** x for x in range(-1, 3)],
                                  'gamma': ['auto', 'scale']},
                                 {'kernel': ['linear'],
                                  'probability': [True],
                                  'C': [1 * 10 ** x for x in range(-1, 3)]}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(SVC(),
                           support_vector_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           return_train_score=True,
                           verbose=2,
                           n_jobs=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'svc_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def print_grid_search_details(clf, filename):
    """
    Prints the results of the grid search to a file and the console.

    :param clf: The classifier to print the details for
    :param filename: The name of the file to write to details to
    :return: Void
    """

    # Set the directory to write files to
    filename = other_dir + '7 Features\\Scores\\' + filename

    # Print the best parameter found in the search along with its score
    print('Best parameters set found on development set:')
    print('Best parameters set found on development set:', file=open(filename, 'a'))

    print('%0.5f (+/-%0.03f) for %r' % (clf.best_score_,
                                        clf.cv_results_['std_test_score'][clf.best_index_] * 2,
                                        clf.best_params_))
    print('%0.5f (+/-%0.03f) for %r' % (clf.best_score_,
                                        clf.cv_results_['std_test_score'][clf.best_index_] * 2,
                                        clf.best_params_), file=open(filename, 'a'))

    # Print the results of all parameter combinations, sorted from best score to worst
    print('\nGrid scores on development set:')
    print('\nGrid scores on development set:', file=open(filename, 'a'))

    test_means = clf.cv_results_['mean_test_score']
    test_stds = clf.cv_results_['std_test_score']
    train_means = clf.cv_results_['mean_train_score']
    train_stds = clf.cv_results_['std_train_score']
    results = list(zip(test_means,
                       test_stds,
                       train_means,
                       train_stds,
                       clf.cv_results_['params'],
                       clf.cv_results_['rank_test_score']))
    for test_mean, test_std, train_mean, train_std, params, rank in sorted(results, key=lambda tup: tup[5]):
        normal_dist = norm(train_mean, train_std)
        probability = 2 * min(normal_dist.cdf(test_mean), normal_dist.sf(test_mean))
        print('%0.5f (+/-%0.03f) for %r probability %0.03f' % (test_mean, test_std * 2, params, probability))
        print('%0.5f (+/-%0.03f) for %r probability %0.03f' % (test_mean, test_std * 2, params, probability),
              file=open(filename, 'a'))

    search = pd.DataFrame.from_dict(clf.cv_results_)
    search['probability'] = search.apply(
        lambda row: 2 * min(norm(row['mean_train_score'], row['std_train_score']).cdf(row['mean_test_score']),
                            norm(row['mean_train_score'], row['std_train_score']).sf(row['mean_test_score'])), axis=1)

    csv_filename = filename.replace('.txt', '.csv')
    search.to_csv(csv_filename, index=False)

    print()
    print('', file=open(filename, 'a'))


def get_best_knn():
    """
    Gets the SVC model that yielded the best result.

    :return: The best model
    """

    return KNeighborsClassifier(algorithm='kd_tree',
                                leaf_size=30,
                                n_neighbors=63,
                                p=1,
                                weights='uniform')


def get_best_logistic_regression():
    """
    Gets the logistic regression model that yielded the best result.

    :return: The best model
    """

    return LogisticRegression(C=0.04,
                              class_weight=None,
                              multi_class='auto',
                              penalty='l1',
                              random_state=42,
                              solver='saga',
                              tol=0.0001)


def get_best_svc():
    """
    Gets the SVC model that yielded the best result.

    :return: The best model
    """

    return SVC(C=10,
               gamma=0.001,
               kernel='sigmoid',
               probability=True)


def get_best_random_forest():
    """
    Gets the random forest model that yielded the best result.

    :return: The best model
    """

    return RandomForestClassifier(n_estimators=1000,
                                  max_features=4,
                                  max_depth=3,
                                  random_state=42)


def get_voting_classifier(contributing_features, df=None):
    """
    Creates a voting classifier based on the top 3 estimators,
    estimators are weighted by the normalized inverse of their respective briers.

    :param contributing_features: A list of features the best explain the variance
    :param df: A data frame containing the game data to train the estimators on
    :return: The trained voting classifier
    """

    if df is None:
        # Get the data frame for all seasons
        df = pd.read_csv(game_data_dir + '20022018.csv')

    # Drop all columns except for the most important features, and the predicted label
    columns_to_keep = list()
    for feature in contributing_features:
        columns_to_keep.extend(list(filter(lambda f: f == feature, df.columns.values)))

    columns_to_keep.extend(list(filter(lambda f: 'home_victory' in f, df.columns.values)))
    columns_to_drop = list(set(df.columns.values) - set(columns_to_keep))
    df = df.drop(columns=columns_to_drop)

    # Get a list of all the feature names
    feature_col_names = contributing_features
    predicted_class_name = ['home_victory']

    # Print the feature column names by order of importance
    print()
    print('Top ' + str(len(feature_col_names)) + ' features ranked by importance are:')
    for feature in feature_col_names:
        print(feature)

    # Get the feature and label data sets
    X = df[feature_col_names].values
    y = df[predicted_class_name].values

    # Standardize the X values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Pickle the scaler
    joblib.dump(scaler, other_dir + '7 Features\\2018Scaler.pkl')

    # Get the classification models
    logistic_regression = get_best_logistic_regression()
    svc = get_best_svc()
    random_forest = get_best_random_forest()

    # Create voting classifier from the 3 estimators, weighted by unit vector of the inverse of the briers, soft voting
    voting_classifier = VotingClassifier(estimators=[('Logistic Regression', logistic_regression),  # .21199, .73085
                                                     ('SVC', svc),                                  # .21244, .73071
                                                     ('Random Forest', random_forest)],             # .21353, .73023
                                         weights=[0.579149, 0.577922, 0.574972],
                                         voting='soft',
                                         flatten_transform=False)

    # Fit the voting classifier
    voting_classifier.fit(X, y.ravel())

    # Pickle the voting classifier
    joblib.dump(voting_classifier, other_dir + '7 Features\\2018VotingClassifier.pkl')

    return voting_classifier
