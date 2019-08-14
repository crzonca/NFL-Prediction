import statistics

import joblib
import maya
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

game_data_dir = '..\\Projects\\nfl\\NFL_Prediction\\Game Data\\'
other_dir = '..\\Projects\\nfl\\NFL_Prediction\\Other\\'


def evaluate_models(best_features, games):
    # TODO Break file into smaller pieces, suggest at very least breaking grooming, modelling, and evaluation
    evaluate_model_parameters(best_features, df=games)
    voting_classifier = get_voting_classifier(best_features, df=games)
    evaluate_2018_season()


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

    # Logistic Regression               -0.21185    66.557
    tune_logistic_regression(X, y, skf, scores)

    # C Support Vector Classifier       -0.21214    66.424
    tune_svc_classifier(X, y, skf, scores)

    # Random Forest                     -0.21281    66.314
    tune_random_forest(X, y, feature_col_names, skf, scores)

    # K Nearest Neighbors               -0.21657    65.345
    tune_k_nearest_neighbors(X, y, skf, scores)

    # Gaussian Naive Bayes              -0.22866    61.974
    tune_gauss_naive_bayes(X, y, skf, scores)


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
                                      {'penalty': ['elasticnet', 'l2'],
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
    k_neighbors_parameters = [{'n_neighbors': range(3, 141, 2),
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
    filename = other_dir + '7 Features No Outliers\\Scores\\' + filename

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

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    results = list(zip(means, stds, clf.cv_results_['params'], clf.cv_results_['rank_test_score']))
    for mean, std, params, rank in sorted(results, key=lambda tup: tup[3]):
        print('%0.5f (+/-%0.03f) for %r' % (mean, std * 2, params))
        print('%0.5f (+/-%0.03f) for %r' % (mean, std * 2, params), file=open(filename, 'a'))

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

    return LogisticRegression(C=0.06,
                              class_weight=None,
                              multi_class='ovr',
                              penalty='l1',
                              random_state=42,
                              solver='saga',
                              tol=0.001)


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
    joblib.dump(scaler, other_dir + '7 Features No Outliers\\2018Scaler.pkl')

    # Get the classification models
    logistic_regression = get_best_logistic_regression()
    svc = get_best_svc()
    random_forest = get_best_random_forest()

    # Create voting classifier from the 3 estimators, weighted by unit vector of the inverse of the briers, soft voting
    voting_classifier = VotingClassifier(estimators=[('Logistic Regression', logistic_regression),  # .21217, 66.519
                                                     ('SVC', svc),  # .21244, 66.386
                                                     ('Random Forest', random_forest)],  # .21353, 66.563
                                         weights=[0.57822, 0.57727, 0.57656],
                                         voting='soft',
                                         flatten_transform=False)

    # Fit the voting classifier
    voting_classifier.fit(X, y.ravel())

    # Pickle the voting classifier
    joblib.dump(voting_classifier, other_dir + '7 Features No Outliers\\2018VotingClassifier.pkl')

    return voting_classifier


def evaluate_2018_season():
    """
    Evaluates a voting classifier based on the 2018 season.

    :return: Void
    """

    # Set the directory to write files to
    filename = other_dir + '7 Features No Outliers\\Scores\\2018Confusion.txt'

    voting_classifier = joblib.load(other_dir + '7 Features No Outliers\\2017VotingClassifier.pkl')
    scaler = joblib.load(other_dir + '7 Features No Outliers\\2017Scaler.pkl')

    last_season = pd.read_csv(game_data_dir + '20022018.csv').values[-267:]
    last_season = pd.DataFrame(last_season)

    results = pd.DataFrame(columns=['rf_prob', 'svc_prob', 'lr_prob', 'vote_prob', 'outcome'])
    for game in last_season.values:
        home_victory = game[71]
        home_spread = game[7]
        elo_diff = game[240]
        average_scoring_margin_diff = game[285]
        win_pct_diff = game[239]
        average_touchdowns_diff = game[283]
        average_passer_rating_diff = game[282]
        average_total_yards_diff = game[255]

        game_features = (home_spread,
                         elo_diff,
                         average_scoring_margin_diff,
                         win_pct_diff,
                         average_touchdowns_diff,
                         average_passer_rating_diff,
                         average_total_yards_diff)

        # Convert the features to a data frame and scale it
        game = pd.DataFrame([game_features])
        game = scaler.transform(game)

        # Get the voting classifier probability
        vote_prob = voting_classifier.predict_proba(game)[0][1]

        # Get the individual estimator probabilities
        estimator_probs = voting_classifier.transform(game)
        lr_prob = estimator_probs[0][0][1]
        svc_prob = estimator_probs[1][0][1]
        rf_prob = estimator_probs[2][0][1]

        game_df = pd.DataFrame([[rf_prob, svc_prob, lr_prob, vote_prob, home_victory]],
                               columns=['rf_prob', 'svc_prob', 'lr_prob', 'vote_prob', 'outcome'])

        results = results.append(game_df)

    outcome = results['outcome']
    rf = results['rf_prob']
    svc = results['svc_prob']
    lr = results['lr_prob']
    vote = results['vote_prob']

    rf_brier = brier_score_loss(outcome, rf)
    print('Random Forest Brier Score Loss:', round(rf_brier, 4))
    print('Random Forest Brier Score Loss:', round(rf_brier, 4), file=open(filename, 'a'))

    svc_brier = brier_score_loss(outcome, svc)
    print('SVC Brier Score Loss:', round(svc_brier, 4))
    print('SVC Brier Score Loss:', round(svc_brier, 4), file=open(filename, 'a'))

    lr_brier = brier_score_loss(outcome, lr)
    print('Logistic Regression Brier Score Loss:', round(lr_brier, 4))
    print('Logistic Regression Brier Score Loss:', round(lr_brier, 4), file=open(filename, 'a'))

    vote_brier = brier_score_loss(outcome, vote)
    print('Voting Classifier Brier Score Loss:', round(vote_brier, 4))
    print('Voting Classifier Brier Score Loss:', round(vote_brier, 4), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    print('Random Forest:', round((.25 - rf_brier) * 26700, 2))
    print('SVC:', round((.25 - svc_brier) * 26700, 2))
    print('Logistic Regression:', round((.25 - lr_brier) * 26700, 2))
    print('Voting Classifier:', round((.25 - vote_brier) * 26700, 2))

    print('Random Forest:', round((.25 - rf_brier) * 26700, 2), file=open(filename, 'a'))
    print('SVC:', round((.25 - svc_brier) * 26700, 2), file=open(filename, 'a'))
    print('Logistic Regression:', round((.25 - lr_brier) * 26700, 2), file=open(filename, 'a'))
    print('Voting Classifier:', round((.25 - vote_brier) * 26700, 2), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    results.to_csv(other_dir + '7 Features No Outliers\\Scores\\2018Predictions.csv', index=False)

    rounded_rf = rf.apply(lambda row: round(row))
    rounded_svc = svc.apply(lambda row: round(row))
    rounded_lr = lr.apply(lambda row: round(row))
    rounded_vote = vote.apply(lambda row: round(row))

    print()
    print('Random Forest')
    print('-' * 120)
    print('Random Forest', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_rf, filename)

    print('SVC')
    print('-' * 120)
    print('SVC', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_svc, filename)

    print('Logistic Regression')
    print('-' * 120)
    print('Logistic Regression', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_lr, filename)

    print('Voting Classifier')
    print('-' * 120)
    print('Voting Classifier', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_vote, filename)

    # visualize_2018_season()


def get_metrics(y_true, y_pred, filename):
    """
    Creates a confusion matrix and prints detains about classification predictions.

    :param y_true: The actual game outcomes
    :param y_pred: The predicted game outcomes
    :param filename: The name of the file to write the reuslts to
    :return: Void
    """

    y_true = pd.to_numeric(y_true)
    outcome_counts = y_true.value_counts()
    outcome_positive = outcome_counts.loc[1]
    outcome_negative = outcome_counts.loc[0]
    total_games = outcome_positive + outcome_negative
    print('Actual home victories:', outcome_positive)
    print('Actual home defeats:', outcome_negative)
    print('Actual home victories:', outcome_positive, file=open(filename, 'a'))
    print('Actual home defeats:', outcome_negative, file=open(filename, 'a'))

    prevalence = outcome_positive / (outcome_positive + outcome_negative)
    print('Home victory prevalence:', round(prevalence * 100, 2),
          str(outcome_positive) + '/' + str(outcome_positive + outcome_negative))
    print()
    print('Home victory prevalence:', round(prevalence * 100, 2),
          str(outcome_positive) + '/' + str(outcome_positive + outcome_negative), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    predicted_counts = y_pred.value_counts()
    predicted_positive = predicted_counts.loc[1]
    predicted_negative = predicted_counts.loc[0]
    print('Predicted home victories:', predicted_positive)
    print('Predicted home defeats:', predicted_negative)
    print('Predicted home victories:', predicted_positive, file=open(filename, 'a'))
    print('Predicted home defeats:', predicted_negative, file=open(filename, 'a'))
    print()
    print('', file=open(filename, 'a'))

    confusion = confusion_matrix(y_true, y_pred)
    true_positive = confusion[1][1]
    false_positive = confusion[0][1]
    false_negative = confusion[1][0]
    true_negative = confusion[0][0]

    print('Correctly predicted home victories:', true_positive)
    print('Home victories predicted as defeats:', false_negative)
    print('Correctly predicted home defeats:', true_negative)
    print('Home defeats predicted as victories:', false_positive)
    print()

    print('Correctly predicted home victories:', true_positive, file=open(filename, 'a'))
    print('Home victories predicted as defeats:', false_negative, file=open(filename, 'a'))
    print('Correctly predicted home defeats:', true_negative, file=open(filename, 'a'))
    print('Home defeats predicted as victories:', false_positive, file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print('Prediction accuracy:', round(accuracy * 100, 2),
          str(true_positive + true_negative) + '/' + str(total_games))
    print()

    print('Prediction precision:', round(precision * 100, 2),
          str(true_positive) + '/' + str(predicted_positive))

    print('Prediction accuracy:', round(accuracy * 100, 2),
          str(true_positive + true_negative) + '/' + str(total_games), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    print('Prediction precision:', round(precision * 100, 2),
          str(true_positive) + '/' + str(predicted_positive), file=open(filename, 'a'))

    false_discovery = false_positive / predicted_positive
    print('False discovery rate:', round(false_discovery * 100, 2),
          str(false_positive) + '/' + str(predicted_positive))

    print('False discovery rate:', round(false_discovery * 100, 2),
          str(false_positive) + '/' + str(predicted_positive), file=open(filename, 'a'))

    negative_prediction = true_negative / predicted_negative
    print('Negative prediction rate:', round(negative_prediction * 100, 2),
          str(true_negative) + '/' + str(predicted_negative))

    print('Negative prediction rate:', round(negative_prediction * 100, 2),
          str(true_negative) + '/' + str(predicted_negative), file=open(filename, 'a'))

    false_omission = false_negative / predicted_negative
    print('False omission rate:', round(false_omission * 100, 2),
          str(false_negative) + '/' + str(predicted_negative))
    print()

    print('Prediction recall:', round(recall * 100, 2),
          str(true_positive) + '/' + str(outcome_positive))

    print('False omission rate:', round(false_omission * 100, 2),
          str(false_negative) + '/' + str(predicted_negative), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    print('Prediction recall:', round(recall * 100, 2),
          str(true_positive) + '/' + str(outcome_positive), file=open(filename, 'a'))

    miss_rate = false_negative / outcome_positive
    print('Miss rate:', round(miss_rate * 100, 2),
          str(false_negative) + '/' + str(outcome_positive))

    print('Miss rate:', round(miss_rate * 100, 2),
          str(false_negative) + '/' + str(outcome_positive), file=open(filename, 'a'))

    specificity = true_negative / outcome_negative
    print('Specificity:', round(specificity * 100, 2),
          str(true_negative) + '/' + str(outcome_negative))

    print('Specificity:', round(specificity * 100, 2),
          str(true_negative) + '/' + str(outcome_negative), file=open(filename, 'a'))

    fall_out = false_positive / outcome_negative
    print('Fall out:', round(fall_out * 100, 2),
          str(false_positive) + '/' + str(outcome_negative))
    print()

    print('Fall out:', round(fall_out * 100, 2),
          str(false_positive) + '/' + str(outcome_negative), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    positive_likelihood = recall / fall_out
    print('Positive likelihood ratio:', round(positive_likelihood, 4),
          str(round(recall * 100, 2)) + '/' + str(round(fall_out * 100, 2)))

    print('Positive likelihood ratio:', round(positive_likelihood, 4),
          str(round(recall * 100, 2)) + '/' + str(round(fall_out * 100, 2)), file=open(filename, 'a'))

    negative_likelihood = miss_rate / specificity
    print('Negative likelihood ratio:', round(negative_likelihood, 4),
          str(round(miss_rate * 100, 2)) + '/' + str(round(specificity * 100, 2)))
    print()

    print('Negative likelihood ratio:', round(negative_likelihood, 4),
          str(round(miss_rate * 100, 2)) + '/' + str(round(specificity * 100, 2)), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    diagnostic_odds = positive_likelihood / negative_likelihood
    print('Diagnostic odds ratio:', round(diagnostic_odds, 4),
          str(round(positive_likelihood, 4)) + '/' + str(round(negative_likelihood, 4)))

    print('Diagnostic odds ratio:', round(diagnostic_odds, 4),
          str(round(positive_likelihood, 4)) + '/' + str(round(negative_likelihood, 4)), file=open(filename, 'a'))

    f1 = f1_score(y_true, y_pred)
    print('F1 score:', round(f1, 4))
    print()
    print('F1 score:', round(f1, 4), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))


def visualize_2018_season():
    """
    Creates a visual representation of the prediction results for evaluating the 2018 season.

    :return: Void
    """
    predictions = pd.read_csv(other_dir + '7 Features No Outliers\\Scores\\2018Predictions.csv')

    for num, game in enumerate(predictions.values):
        vote_prob = int(round(game[3] * 100))
        home_victory = int(round(game[4] * 100))
        if home_victory == 0:
            vote_prob = 100 - vote_prob
            home_victory = 100

        brier = round((game[3] - game[4]) ** 2, 2)

        slider = ['.' for pct in range(100)]

        slider[49] = slider[49].replace('.', '') + '|'
        slider[vote_prob - 1] = slider[vote_prob - 1].replace('.', '') + 'V'
        slider[home_victory - 1] = slider[home_victory - 1].replace('.', '') + 'X'

        if vote_prob < 50:
            print('\033[31m' + str(num + 2).zfill(3) + ' ' + ''.join(slider) + ' ' + str(brier) + '\033[0m')
        else:
            print('\033[32m' + str(num + 2).zfill(3) + ' ' + ''.join(slider) + ' ' + str(brier) + '\033[0m')


def analyze_results():
    """
    Analyzes grid search results to determine best set of hyper parameters.

    :return: The sorted list of hyper parameters, sorted based on accuracy and loss
    """
    with open(other_dir + '7 Features No Outliers\\Scores\\logistic_regression_brier_score_loss.txt') as brier:
        with open(other_dir + '7 Features No Outliers\\Scores\\logistic_regression_accuracy.txt') as accuracy:
            brier_lines = brier.readlines()[4:]
            accuracy_lines = accuracy.readlines()[4:]

            tenth = int(len(brier_lines) * .1)
            if tenth > 10:
                brier_lines = brier_lines[:tenth]
                accuracy_lines = accuracy_lines[:tenth]

            briers = dict()
            for brier_line in brier_lines:
                brier_score = float(brier_line.split()[0])
                brier_params = brier_line.split(' for ')[-1]
                briers[brier_params] = brier_score

            accuracies = dict()
            for accuracy_line in accuracy_lines:
                accuracy_score = float(accuracy_line.split()[0])
                accuracy_params = accuracy_line.split(' for ')[-1]
                accuracies[accuracy_params] = accuracy_score

            brier_scores = briers.values()
            min_brier = min(brier_scores)
            max_brier = max(brier_scores)
            mean_brier = statistics.mean(brier_scores)
            brier_dev = statistics.stdev(brier_scores)

            for brier_key, brier_val in briers.items():
                # briers[brier_key] = (brier_val - mean_brier) / brier_dev
                briers[brier_key] = (brier_val - min_brier) / (max_brier - min_brier)

            accuracies_scores = accuracies.values()
            min_accuracy = min(accuracies_scores)
            max_accuracy = max(accuracies_scores)
            mean_accuracy = statistics.mean(accuracies_scores)
            accuracy_dev = statistics.stdev(accuracies_scores)

            for accuracy_key, accuracy_val in accuracies.items():
                # accuracies[accuracy_key] = (accuracy_val - mean_accuracy) / accuracy_dev
                accuracies[accuracy_key] = (accuracy_val - min_accuracy) / (max_accuracy - min_accuracy)

            params = dict()
            for brier_key, brier_val in briers.items():
                if accuracies.get(brier_key) is not None:
                    params[brier_key] = brier_val + accuracies.get(brier_key)

            sorted_params = sorted(params.items(), key=lambda kv: kv[1], reverse=True)
            return sorted_params
